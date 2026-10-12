"""
Learned correction stage on top of 3D Wiener, scored with the wiggle scorecard (rc_r).

  wiener = 3D Wiener(dirty, a=2)                                  (blind: no clean data, wiggle_scorecard.wiener3d)
  out[c] = wiener[c] + UNet([dirty window, wiener window, noise-level map])    last conv zero-initialised, so the
                                                                   untrained model IS the Wiener cube

Why this shape (research.md 2026-10-02, PROGRESS.md 2026-10-08/09): 3D Wiener keeps the noise out but flattens or invents the
faint fingers; the best learned model so far (sg_k3_fresh) wins on unseen analytic disks but loses on high-SNR SG v2. Starting
from Wiener means the network cannot do worse than doing nothing by construction at step 0, and the spectral window gives it the
velocity context a one-channel model throws away.

Training data, fresh noise every round (never the same dirty cube twice): every training cube is clean (*) beam + beam (*) new
white noise at the cube's own measured noise level, the recipe that made the sg_synth pairs. Training cubes per fold: the 4
other hydro disks + 6 analytic disks (an01-an06); held out and never trained on: the fold's hydro disk, SG v2, analytic an07/an08.
Fixed step budget, no early stopping and no model selection on the holdout.

Loss: L1 to clean on three adjacent centres plus L1 on their central velocity difference (the line-centre shift is carried by
dI/dv), so a pixel-perfect-but-shifted line is penalised.

The spectral window is defined in km/s: neighbours sit at +-j*step channels with step = round(0.1 km/s / dv), so the 0.033 km/s
cubes (9019, SG v2) see the same velocity span as the 0.1 km/s ones.

Run (Kaggle, GPU):  see kaggle/wiggle_correction/kernel_wiggle_correction.py
Local smoke:        PYTHONPATH=. python3 experiments/train_wiggle_correction.py --smoke
"""
import os, sys, json, time, math, argparse
import numpy as np
import scipy.fft as sfft

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import wiggle_scorecard as ws
from src.models.unet import UNet

K = 3                                   # window half-width in channels at 0.1 km/s sampling
HYDRO = ["9074", "9025", "9015", "9032", "9019"]      # fold order: the hard disks first
ANALYTIC_TRAIN = ["an01", "an02", "an03", "an04", "an05", "an06"]
ANALYTIC_HOLD = ["an07", "an08"]
ALPHA = 2.0
dev = "cuda" if torch.cuda.is_available() else "cpu"


def step_of(dv):
    return max(1, int(round(0.1 / abs(dv))))


def offsets(dv):
    s = step_of(dv)
    return np.arange(-K, K + 1) * s, s


# ------------------------------------------------------------------ training cubes
class TrainCube:
    """clean + the fixed signal term beam (*) clean, and a fresh (dirty, wiener) draw on demand."""

    def __init__(self, name):
        d = ws.load(name)
        self.name, self.H = name, d["H"].astype(np.complex64)      # complex64: the 600 px cubes are 0.7 GB each
        self.clean = d["clean"].astype(np.float32)
        self.dv = float(d["velax"][1] - d["velax"][0])
        self.C, self.n = self.clean.shape[0], self.clean.shape[-1]
        self.conv = sfft.ifft2(sfft.fft2(self.clean, workers=-1) * self.H[None], workers=-1).real.astype(np.float32)
        res = d["dirty"][list(range(8)) + list(range(-8, 0))].astype(np.float64) - self.conv[list(range(8)) + list(range(-8, 0))]
        self.sigma = float(res.std())                     # measured post-beam noise of the provided pair (line-free channels)
        self.gain = float(np.sqrt(np.mean(np.abs(self.H) ** 2)))   # std of beam (*) N(0,1)
        self.ch_w = 0.1 + self.clean.reshape(self.C, -1).max(1) / self.clean.max()
        self.ch_w /= self.ch_w.sum()
        self.draw_ready = False
        del d

    def redraw(self, rng):
        noise = rng.normal(0.0, self.sigma / self.gain, self.clean.shape).astype(np.float32)
        noise = sfft.ifft2(sfft.fft2(noise, workers=-1) * self.H[None], workers=-1).real.astype(np.float32)
        dirty = self.conv + noise
        d = {"dirty": dirty, "H": self.H, "noise2d": ws.blind_noise2d(dirty, self.H)}
        w = ws.wiener3d(d, ALPHA)
        self.dirty, self.wiener = dirty, w
        self.s = float(w.max())
        self.sig_map = float(np.sqrt(d["noise2d"].mean()) / self.n / self.s)
        self.draw_ready = True


def make_batch(cubes, probs, batch, crop, rng):
    """(B*3, 2*(2K+1)+1, crop, crop) inputs, wiener centres, clean centres, all in the cube's own scale s."""
    X, Wc, Cc = [], [], []
    for _ in range(batch):
        cb = cubes[rng.choice(len(cubes), p=probs)]
        offs, step = offsets(cb.dv)
        c = int(rng.choice(cb.C, p=cb.ch_w))
        c = int(np.clip(c, step, cb.C - 1 - step))
        y0 = int(rng.integers(0, cb.n - crop + 1)); x0 = int(rng.integers(0, cb.n - crop + 1))
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        for cen in (c - step, c, c + step):
            nb = np.clip(cen + offs, 0, cb.C - 1)
            noise_ch = np.full((1, crop, crop), cb.sig_map, np.float32)
            X.append(np.concatenate([cb.dirty[nb][(slice(None),) + sl] / cb.s, cb.wiener[nb][(slice(None),) + sl] / cb.s, noise_ch]))
            Wc.append(cb.wiener[cen][sl] / cb.s)
            Cc.append(cb.clean[cen][sl] / cb.s)
    t = lambda a: torch.from_numpy(np.stack(a)).float().to(dev)
    return t(X), t(Wc), t(Cc)


def build_net(base):
    net = UNet(in_channels=2 * (2 * K + 1) + 1, out_channels=1, base_channels=base, channel_multipliers=(1, 2, 4, 8),
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, base), beam_dim=0).to(dev)
    nn.init.zeros_(net.conv_out.weight); nn.init.zeros_(net.conv_out.bias)      # untrained model == Wiener
    return net


def loss_fn(corr, wc, cc):
    pred = wc + corr[:, 0]                                    # (B*3, h, w)
    B = pred.shape[0] // 3
    p, c = pred.view(B, 3, *pred.shape[-2:]), cc.view(B, 3, *cc.shape[-2:])
    l_pix = (p - c).abs().mean()
    l_vel = ((p[:, 2] - p[:, 0]) - (c[:, 2] - c[:, 0])).abs().mean()
    return l_pix + l_vel, l_pix.detach(), l_vel.detach()


def train_fold(fold, cubes, args, out_dir, log):
    rng = np.random.default_rng(1234 + int(fold))
    torch.manual_seed(1234)
    net = build_net(args.base)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.rounds * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05, final_div_factor=30)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    hyd = [c for c in cubes if not c.name.startswith("an")]
    ana = [c for c in cubes if c.name.startswith("an")]
    probs = np.array([0.5 / len(hyd) if not c.name.startswith("an") else 0.5 / len(ana) for c in cubes])
    probs /= probs.sum()
    t0, step, hist = time.time(), 0, []
    for rnd in range(args.rounds):
        stale = [c for c in cubes if not c.draw_ready] or list(rng.choice(cubes, size=min(3, len(cubes)), replace=False))
        for c in stale:
            c.redraw(rng)
        net.train()
        for _ in range(args.steps):
            x, wc, cc = make_batch(cubes, probs, args.batch, args.crop, rng)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                corr = net(x, torch.zeros(x.shape[0], dtype=torch.long, device=dev), None)
            loss, lp, lv = loss_fn(corr.float(), wc, cc)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 25 == 0 or step == 1:
                with torch.no_grad():
                    base_p = (wc.view(-1, 3, *wc.shape[-2:]) - cc.view(-1, 3, *cc.shape[-2:])).abs().mean().item()
                hist.append((step, float(lp), float(lv), base_p))
        print(f"[{fold}] round {rnd + 1}/{args.rounds} step {step}/{total}  L1 {float(lp):.5f} (wiener {base_p:.5f})  "
              f"vel {float(lv):.5f}  {time.time() - t0:.0f}s", flush=True)
        # persist after every round (RULES.md #1), straight into the output dir
        torch.save({"model_state_dict": net.state_dict(), "base": args.base, "K": K, "alpha": ALPHA, "fold": fold,
                    "round": rnd + 1, "steps": step, "hist": hist}, os.path.join(out_dir, f"corr_fold{fold}.pth"))
    return net, {"minutes": (time.time() - t0) / 60, "steps": step, "hist": hist}


# ------------------------------------------------------------------ inference as a scorecard method
def make_method(net, bs):
    def run(d):
        w = ws.wiener3d(d, ALPHA)
        dirty = d["dirty"].astype(np.float32)
        C, n, _ = dirty.shape
        dv = float(d["velax"][1] - d["velax"][0])
        offs, _ = offsets(dv)
        s = float(w.max())
        sig = float(np.sqrt(np.mean(d["noise2d"])) / n / s)
        pad = (-n) % 8
        bsz = bs if n <= 320 else max(1, bs // 2)
        out = np.empty_like(w)
        net.eval()
        with torch.no_grad():
            for st in range(0, C, bsz):
                idx = np.arange(st, min(st + bsz, C))
                nb = np.clip(idx[:, None] + offs[None, :], 0, C - 1)
                x = torch.from_numpy(np.concatenate([dirty[nb] / s, w[nb] / s, np.full((len(idx), 1, n, n), sig, np.float32)], 1)).to(dev)
                if pad:
                    x = Fn.pad(x, (0, pad, 0, pad), mode="replicate")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    corr = net(x, torch.zeros(len(idx), dtype=torch.long, device=dev), None)
                out[idx] = w[idx] + corr.float()[:, 0, :n, :n].cpu().numpy() * s
        return out
    return run


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=",".join(HYDRO))
    ap.add_argument("--hydro", default=",".join(HYDRO), help="hydro disks available for training (a fold's own disk is always excluded)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300, help="optimiser steps per round")
    ap.add_argument("--batch", type=int, default=12, help="samples per step (x3 centres)")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-extra", default="sgv2,an07,an08")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/wiggle-correction/local"))
    ap.add_argument("--smoke", action="store_true", help="tiny model, few steps, one fold, one eval cube")
    args = ap.parse_args()
    if args.smoke:
        args.folds, args.rounds, args.steps, args.batch, args.crop, args.base = "9074", 2, 8, 2, 64, 8
        args.eval_extra = "an07"
        args.hydro = "9074,9015"
    os.makedirs(args.out, exist_ok=True)
    print(f"device {dev}  args {vars(args)}", flush=True)
    summary_path = os.path.join(args.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    cache = {}
    def get(name):
        if name not in cache:
            cache[name] = TrainCube(name)
        return cache[name]
    for fold in args.folds.split(","):
        try:
            t0 = time.time()
            cubes = [get(h) for h in args.hydro.split(",") if h != fold] + [get(a) for a in ANALYTIC_TRAIN]
            for c in cubes:
                c.draw_ready = False
            print(f"\n===== fold {fold}: train on {[c.name for c in cubes]} =====", flush=True)
            net, tr = train_fold(fold, cubes, args, args.out, None)
            ws.METHODS["wiener_corr"] = make_method(net, bs=8)
            fold_res = {"train": tr, "scores": {}}
            figdir = os.path.join(args.out, f"fold_{fold}"); os.makedirs(figdir, exist_ok=True)
            for name in [fold] + [x for x in args.eval_extra.split(",") if x]:
                fold_res["scores"][name] = ws.score_cube(name, ws.load(name), ["beam_only", "dirty", "wiener3d_a2", "wiener_corr"], figdir)
                json.dump({**summary, fold: fold_res}, open(summary_path, "w"), indent=1)
                m = fold_res["scores"][name]["methods"]
                print(f"[{fold}] {name}: rc_r  dirty {m['dirty']['rc_r']:.3f}  wiener {m['wiener3d_a2']['rc_r']:.3f}  "
                      f"CORR {m['wiener_corr']['rc_r']:.3f}  oracle {m['beam_only']['rc_r']:.3f} | "
                      f"line-centre median {m['wiener3d_a2']['dv_med_ms']:.0f} -> {m['wiener_corr']['dv_med_ms']:.0f} m/s", flush=True)
            summary[fold] = fold_res
            json.dump(summary, open(summary_path, "w"), indent=1)
            del net
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"[{fold}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception:                       # one failed fold must not cost the folds after it
            import traceback
            print(f"[{fold}] FOLD FAILED:\n{traceback.format_exc()}", flush=True)
            summary[fold] = {"failed": traceback.format_exc()[-1500:]}
            json.dump(summary, open(summary_path, "w"), indent=1)
            if dev == "cuda":
                torch.cuda.empty_cache()
    # table
    print("\nrc_r by fold (holdout hydro disk | SG v2 | an07 | an08):")
    for fold, fr in summary.items():
        if "failed" in fr:
            print(f"  fold {fold}  FAILED"); continue
        row = []
        for name, sc in fr["scores"].items():
            m = sc["methods"]
            row.append(f"{name}: {m['dirty']['rc_r']:.2f}/{m['wiener3d_a2']['rc_r']:.2f}/{m['wiener_corr']['rc_r']:.2f}")
        print(f"  fold {fold}  (dirty/wiener/corr)  " + "  ".join(row))


if __name__ == "__main__":
    main()
