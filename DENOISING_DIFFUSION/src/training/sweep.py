"""
src/training/sweep.py
=====================
U-Net training with early stopping + random hyperparameter sweep
(mentor direction, 2026-07-20 meeting):

  * random search first -- explore the space, gather statistics (a Bayesian
    sweep seeded with these runs can follow later; starting Bayesian risks
    honing in on the wrong region).
  * swept: base width, channel multipliers, learning rate, LR-scheduler
    patience, HybridLoss alpha (beta = 1 - alpha), beam conditioning on/off.
  * every run is scored on FIXED metrics (PSNR / SSIM / MSE on val) -- never
    on the swept loss, whose weights differ between runs.
  * early stopping: min/max epochs with patience on the fixed val PSNR.

`train_unet` is also used standalone for the single beam-conditioned run.
"""

import csv
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.utils.losses import HybridLoss


def _unwrap(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def _split_batch(batch, device, use_beam):
    """(dirty, clean[, beam]) -> tensors on device; beam None unless requested."""
    d, c = batch[0].to(device), batch[1].to(device)
    beam = batch[2].to(device) if (use_beam and len(batch) > 2) else None
    return d, c, beam


@torch.no_grad()
def val_metrics(model, val_loader, device, use_beam=False, arch: str = "unet"):
    """
    Fixed sweep metric: PSNR / SSIM / MSE on val, prediction clamped to [0,1].

    `arch` selects the forward adapter, so the *same* metric function scores every
    architecture. That matters: a comparison is only fair if the models differ and
    the measurement does not.
    """
    from pytorch_msssim import ssim as ssim_fn
    from src.training.architectures import forward_fn

    fwd = forward_fn(arch)
    model.eval()
    psnrs, ssims, mses = [], [], []
    for batch in val_loader:
        d, c, beam = _split_batch(batch, device, use_beam)
        pred, _ = fwd(model, d, beam)
        pred = pred.clamp(0, 1)
        mse = torch.mean((pred - c) ** 2, dim=(1, 2, 3))
        psnrs += (10 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))).cpu().tolist()
        ssims += ssim_fn(pred, c, data_range=1.0, size_average=False).cpu().tolist()
        mses += mse.cpu().tolist()
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
            "mse": float(np.mean(mses))}


def train_unet(
    train_ds,
    val_ds,
    device,
    *,
    base_channels: int = 32,
    channel_multipliers=(1, 2, 4),
    lr: float = 1e-3,
    alpha: float = 0.8,           # HybridLoss: alpha*MSE + (1-alpha)*(1-SSIM)
    batch_size: int = 32,
    use_beam: bool = False,
    min_epochs: int = 20,
    max_epochs: int = 100,
    patience: int = 5,            # early stop: epochs without val-loss improvement
    sched_patience: int = 5,      # ReduceLROnPlateau patience
    num_workers: int = 0,
    seed: int = 42,
    ckpt_path: Optional[str] = None,
    arch: str = "unet",
    latent_dim: int = 128,
    kl_weight: float = 0.0,
    verbose: bool = True,
):
    """
    Train one config with early stopping; return fixed metrics + history.

    Early stopping (mentor, 2026-07-20): train at least `min_epochs`, at most
    `max_epochs`, stop after `patience` epochs without val-loss improvement.
    Best-epoch weights are restored (and saved to `ckpt_path` if given) before
    the final fixed-metric evaluation.

    Args:
        arch: "unet" | "autoencoder" | "vae". Selects the model builder and
            forward adapter; everything else about the protocol is identical
            across architectures, which is what makes the scores comparable.
            Only the U-Net has ever been swept on this data, so the previously
            published AE/VAE numbers (Week-2, 64x64 continuum patches) are not a
            fair basis for an architecture claim.
        latent_dim: VAE only -- channels in the latent map.
        kl_weight: VAE only -- weight on the KL term. Ignored by architectures
            with no extra loss, so a single sweep driver can pass it blindly.

    Note the name kept for backwards compatibility: `train_unet` now trains any
    supported architecture. Renaming it would break the notebooks and the
    committed sweep CSVs that reference it.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.training.architectures import build_model, extra_loss_fn, forward_fn

    n_gpu = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    net = build_model(arch, base_channels=base_channels,
                      channel_multipliers=channel_multipliers,
                      use_beam=use_beam, latent_dim=latent_dim).to(device)
    fwd = forward_fn(arch)
    extra = extra_loss_fn(arch)
    model = torch.nn.DataParallel(net) if n_gpu > 1 else net

    criterion = HybridLoss(alpha=alpha, beta=1.0 - alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=sched_patience)

    pin = n_gpu > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    def run_epoch(loader, train):
        model.train(train)
        tot, n = 0.0, 0
        torch.set_grad_enabled(train)
        for batch in loader:
            d, c, beam = _split_batch(batch, device, use_beam)
            pred, aux = fwd(model, d, beam)
            total, _, _ = criterion(pred, c)
            # architecture-specific term (the VAE's KL); weight 0 leaves the
            # objective identical to the other architectures'
            if extra is not None and kl_weight:
                total = total + kl_weight * extra(aux)
            if train:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
            tot += total.item() * d.size(0)
            n += d.size(0)
        torch.set_grad_enabled(True)
        return tot / max(n, 1)

    best_val, best_epoch, best_state = float("inf"), -1, None
    epochs_no_improve = 0
    tr_hist, va_hist = [], []
    t_start = time.time()

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        scheduler.step(va)
        tr_hist.append(tr)
        va_hist.append(va)

        if va < best_val:
            best_val, best_epoch = va, ep
            best_state = {k: v.detach().clone() for k, v in _unwrap(model).state_dict().items()}
            epochs_no_improve = 0
            mark = " *best"
        else:
            epochs_no_improve += 1
            mark = ""
        if verbose:
            print(f"  ep {ep:>3} | train {tr:.4f} | val {va:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s){mark}",
                  flush=True)

        if ep >= min_epochs and epochs_no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {ep} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        _unwrap(model).load_state_dict(best_state)
        if ckpt_path:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            # `arch` is written so a checkpoint can be rebuilt without the caller
            # having to remember which architecture produced it. The legacy
            # "UNet" string is preserved for arch="unet" so existing checkpoints
            # and the loaders in notebooks 05/08 keep working unchanged.
            torch.save({"epoch": best_epoch, "model_state_dict": best_state,
                        "val_loss": best_val,
                        "arch": "UNet" if arch == "unet" else arch,
                        "arch_key": arch,
                        "base_channels": base_channels,
                        "channel_multipliers": list(channel_multipliers),
                        "alpha": alpha, "use_beam": use_beam,
                        "beam_dim": 4 if use_beam else 0,
                        "latent_dim": latent_dim, "kl_weight": kl_weight},
                       ckpt_path)

    metrics = val_metrics(model, val_loader, device, use_beam=use_beam, arch=arch)
    return {
        "model": _unwrap(model),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(tr_hist),
        "train_losses": tr_hist,
        "val_losses": va_hist,
        "wall_time_s": time.time() - t_start,
        **metrics,
    }


# --------------------------------------------------------------------------- #
# Random sweep                                                                #
# --------------------------------------------------------------------------- #
# Default search space. lr is log-uniform; alpha uniform; the rest choices.
SPACE = {
    "base_channels": [16, 32, 48, 64],
    "channel_multipliers": [(1, 2, 4), (1, 2, 2, 4), (1, 2, 4, 8)],
    "lr": (1e-4, 3e-3),           # log-uniform range
    "alpha": (0.5, 0.95),         # uniform range; beta = 1 - alpha
    "sched_patience": [3, 5, 8],
    "use_beam": [False, True],
}


def sample_config(rng: random.Random, space=None) -> dict:
    s = dict(SPACE, **(space or {}))
    lo, hi = s["lr"]
    a_lo, a_hi = s["alpha"]
    return {
        "base_channels": rng.choice(s["base_channels"]),
        "channel_multipliers": rng.choice(s["channel_multipliers"]),
        "lr": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "alpha": rng.uniform(a_lo, a_hi),
        "sched_patience": rng.choice(s["sched_patience"]),
        "use_beam": rng.choice(s["use_beam"]),
    }


def run_sweep(
    train_ds,
    val_ds,
    device,
    *,
    n_runs: int = 12,
    out_csv: str = "results/sweep_results.csv",
    seed: int = 42,
    space: Optional[dict] = None,
    batch_size: int = 32,
    min_epochs: int = 15,
    max_epochs: int = 60,
    patience: int = 5,
    num_workers: int = 0,
    verbose: bool = True,
):
    """
    Random hyperparameter sweep. Each run samples a config, trains with early
    stopping, and appends a CSV row immediately (crash-safe on Kaggle). Runs
    that OOM are retried at half batch (up to 2 halvings), else recorded failed.
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = ["run", "base_channels", "channel_multipliers", "lr", "alpha",
              "sched_patience", "use_beam", "batch_size", "best_epoch",
              "epochs_run", "best_val_loss", "psnr", "ssim", "mse", "wall_time_s"]
    new_file = not os.path.exists(out_csv)
    results = []

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()

        for i in range(n_runs):
            cfg = sample_config(rng, space)
            if verbose:
                print(f"\n=== sweep run {i+1}/{n_runs}: {cfg} ===", flush=True)
            bs = batch_size
            row = {"run": i, **cfg,
                   "channel_multipliers": "x".join(map(str, cfg["channel_multipliers"]))}
            for attempt in range(3):
                try:
                    res = train_unet(
                        train_ds, val_ds, device,
                        base_channels=cfg["base_channels"],
                        channel_multipliers=cfg["channel_multipliers"],
                        lr=cfg["lr"], alpha=cfg["alpha"],
                        sched_patience=cfg["sched_patience"],
                        use_beam=cfg["use_beam"], batch_size=bs,
                        min_epochs=min_epochs, max_epochs=max_epochs,
                        patience=patience, num_workers=num_workers,
                        seed=seed + i, verbose=verbose,
                    )
                    row.update({k: res[k] for k in
                                ("best_epoch", "epochs_run", "best_val_loss",
                                 "psnr", "ssim", "mse", "wall_time_s")})
                    row["batch_size"] = bs
                    del res["model"]
                    results.append({**cfg, **res})
                    break
                except RuntimeError as ex:
                    if "out of memory" not in str(ex).lower() or attempt == 2:
                        print(f"  run {i} FAILED: {ex}")
                        row.update({"batch_size": bs, "best_val_loss": "failed"})
                        break
                    torch.cuda.empty_cache()
                    bs //= 2
                    print(f"  [OOM] retrying at batch {bs}")
            writer.writerow(row)
            f.flush()

    if verbose and results:
        best = max((r for r in results if "psnr" in r), key=lambda r: r["psnr"])
        print(f"\nSweep done: {len(results)}/{n_runs} runs OK. "
              f"Best PSNR {best['psnr']:.3f} dB -> {out_csv}")
    return results


# --------------------------------------------------------------------------- #
# Multi-seed repeats                                                          #
# --------------------------------------------------------------------------- #
def repeat_config(
    train_ds,
    val_ds,
    device,
    *,
    n_seeds: int = 3,
    base_seed: int = 42,
    out_csv: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    tag: str = "config",
    verbose: bool = True,
    **config,
):
    """
    Train the SAME configuration under several seeds and report mean +/- std.

    Why this exists
    ---------------
    V7 and V9 were the identical configuration trained twice, and the Moment-2
    improvement moved from +18.4% to +2.5% on the same test cube -- pure run-to-run
    variance. That finding is what forced multi-cube evaluation, but every headline
    number since has still come from a SINGLE training run, so a claim like
    "37.11 dB beats V12's 32.95 dB" has no error bar on either side and could be
    seed noise. This runs a config `n_seeds` times so the comparison has one.

    What is varied, and what is not
    -------------------------------
    Only `train_unet`'s seed changes -- weight init, data order, and any
    augmentation stream. The cube split is seeded separately (`split_cubes`) and
    is deliberately held FIXED: re-splitting per seed would mix split variance
    into the estimate and stop the runs being comparable to V12, which used one
    fixed split. So the spread reported here is training variance on a fixed split,
    which is exactly the quantity the V7/V9 lesson is about.

    Args:
        n_seeds: number of repeats. 3 is the practical floor for a std to mean
            anything; the std is reported with ddof=1 and is itself noisy at n=3.
        base_seed: seeds used are base_seed, base_seed+1, ...
        out_csv: append one row per seed (crash-safe, same pattern as `run_sweep`).
        ckpt_dir: if given, each seed's best checkpoint is saved as
            `{ckpt_dir}/{tag}_seed{N}.pth`, so the moment-map protocol can be run
            per seed instead of only on the last model.
        tag: label used in the CSV and checkpoint filenames.
        **config: forwarded to `train_unet` (base_channels, lr, alpha, use_beam, ...).

    Returns:
        {"tag", "n_seeds", "rows": [per-seed dicts], "psnr": {"mean","std","values"},
         "ssim": {...}, "mse": {...}, "best_val_loss": {...}}

    Note that the model objects are dropped from the returned rows -- keeping
    `n_seeds` models alive is what makes a repeat loop run the GPU out of memory.
    """
    fields = ["tag", "seed", "best_epoch", "epochs_run", "best_val_loss",
              "psnr", "ssim", "mse", "wall_time_s"]
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    rows = []
    if verbose:
        print(f"=== repeating '{tag}' over {n_seeds} seeds "
              f"(split held fixed; only training seed varies) ===", flush=True)

    for k in range(n_seeds):
        seed = base_seed + k
        ckpt = os.path.join(ckpt_dir, f"{tag}_seed{seed}.pth") if ckpt_dir else None
        if verbose:
            print(f"\n--- {tag}: seed {seed} ({k+1}/{n_seeds}) ---", flush=True)
        res = train_unet(train_ds, val_ds, device, seed=seed, ckpt_path=ckpt,
                         verbose=verbose, **config)
        res.pop("model", None)          # free the GPU before the next seed
        row = {"tag": tag, "seed": seed,
               **{f: res[f] for f in fields if f in res}}
        rows.append(row)
        if out_csv:
            new_file = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new_file:
                    w.writeheader()
                w.writerow({k2: row.get(k2) for k2 in fields})
        if verbose:
            print(f"  seed {seed}: PSNR {res['psnr']:.4f} | SSIM {res['ssim']:.4f} "
                  f"| MSE {res['mse']:.6f}")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    out = {"tag": tag, "n_seeds": n_seeds, "rows": rows}
    for metric in ("psnr", "ssim", "mse", "best_val_loss"):
        vals = [r[metric] for r in rows if isinstance(r.get(metric), (int, float))]
        out[metric] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    if verbose:
        print(f"\n=== '{tag}' over {len(rows)} seeds ===")
        for metric in ("psnr", "ssim", "mse"):
            m = out[metric]
            print(f"  {metric.upper():<5} {m['mean']:.4f} +/- {m['std']:.4f}  "
                  f"(values: {', '.join(f'{v:.4f}' for v in m['values'])})")
    return out
