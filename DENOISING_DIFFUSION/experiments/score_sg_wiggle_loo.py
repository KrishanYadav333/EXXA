"""
Does SG training recover the GI wiggle, across all 5 disks as genuine holdouts?

Extends score_sg_wiggle.py (which scored only run_9074, notebook 10's single holdout) to
all 5 leave-one-out folds from notebook 11. Each disk gets scored by the checkpoint that
never saw it during training, which is the entire point of leave-one-out: five honest
holdout measurements instead of one.

Every disk's true inclination is known from its .para file, so every fit holds inclination
fixed rather than free -- this sidesteps the mass-inclination degeneracy (PROGRESS.md
2026-09-04) instead of re-triggering it on the four 20-degree disks.

Run: PYTHONPATH=.. python3 experiments/score_sg_wiggle_loo.py
"""
import os, sys, math, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import torch
import torch.nn.functional as Fn
from astropy.io import fits

from src.models.unet import UNet
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude

SYN = "self-gravitating cube and dirty cube/sg_synth"
BASE, MULTS, SIZE, FRAC = 48, (1, 2, 4, 8), 256, 0.05
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# fold index -> (holdout run folder, true inclination from its .para)
FOLDS = [
    (0, "run_9015_00370_rt_00", 20.0),
    (1, "run_9019_00019_rt_00", 30.0),
    (2, "run_9025_00370_rt_00", 20.0),
    (3, "run_9032_00020_rt_00", 20.0),
    (4, "run_9074_00025_rt_00", 20.0),
]
WINNER_CKPT = "models/08-seeds/winner_aug_seed43.pth"


def denoise(path, cube):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UNet(in_channels=1, out_channels=1, base_channels=BASE, channel_multipliers=MULTS,
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, BASE), beam_dim=0).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not miss and not unexp, f"{path}: state dict mismatch"
    net.eval()
    C, H, W = cube.shape
    los = cube.reshape(C, -1).min(axis=1); his = cube.reshape(C, -1).max(axis=1)
    rng = his - los
    out = np.empty_like(cube, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, C, 8):
            blk = cube[s:s + 8].astype(np.float64)
            lo, hi = los[s:s + 8], his[s:s + 8]
            den = np.where((hi - lo) > 0, hi - lo, 1)[:, None, None]
            t = torch.from_numpy((blk - lo[:, None, None]) / den)[:, None].float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(b.shape[0]):
                out[s + k] = b[k] * rng[s + k] + los[s + k] if rng[s + k] > 0 else los[s + k]
    return out


all_rows = []
t_start = time.time()

for fold, run, true_incl in FOLDS:
    print("\n" + "=" * 70)
    print(f"FOLD {fold}  holdout={run}  true incl {true_incl} deg")
    print("=" * 70)

    d = f"{SYN}/{run}"
    with fits.open(f"{d}/{run}_clean.fits", memmap=True) as h:
        hdr, clean = h[0].header, h[0].data[:]
    with fits.open(f"{d}/{run}_dirty.fits", memmap=True) as h:
        dirty = h[0].data[:]
    velax = (hdr["CRVAL3"] + (np.arange(clean.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    au_per_px = abs(hdr["CDELT1"]) * 3600.0 * float(hdr["DIST_PC"])
    print(f"  {clean.shape}, dv {hdr['CDELT3']:.4f} km/s, {au_per_px:.3f} AU/px")

    m0 = generate_moment_maps("", data_velax=(clean.astype(np.float64), velax))[0]
    mask = signal_mask(m0, frac=FRAC)
    print(f"  mask: {mask.sum()} px ({100*mask.sum()/mask.size:.1f}% of field)")

    ARMS = {
        "frozen": WINNER_CKPT,
        "finetune": f"models/11-loo/loo{fold}_finetune.pth",
        "fresh": f"models/11-loo/loo{fold}_fresh.pth",
    }
    cubes = {"clean": clean.astype(np.float64), "dirty": dirty.astype(np.float64)}
    for name, path in ARMS.items():
        t0 = time.time()
        cubes[name] = denoise(path, dirty).astype(np.float64)
        print(f"    denoised {name} in {(time.time()-t0)/60:.1f} min")

    m1 = {}
    for name, cube in cubes.items():
        v0, _ = quadratic_moment1(cube, velax)
        m1[name] = v0 / 1000.0

    geom = fit_keplerian(m1["clean"], mask, au_per_px, fix_incl_deg=true_incl)
    print(f"  geometry (incl fixed {true_incl}): mstar={geom['mstar_msun']:.3f} "
          f"pa={geom['pa_deg']:.1f} vsys={geom['vsys']:.3f} at_bound={geom['mstar_at_bound']}")

    ref_resid = wiggle_residual(m1["clean"], geom)
    print(f"  {'method':10s} {'residRMS':>9s} {'raw r':>8s} {'resid r':>9s}")
    fold_row = dict(fold=fold, holdout=run, true_incl=true_incl,
                    mask_px=int(mask.sum()), geom=geom, methods={})
    for name in cubes:
        resid = wiggle_residual(m1[name], geom)
        amp = wiggle_amplitude(resid, mask)
        if name == "clean":
            print(f"  {name:10s} {amp['rms_kms']:9.3f} {'--':>8s} {'--':>9s}")
            fold_row["methods"][name] = dict(rms=amp["rms_kms"], raw_r=None, resid_r=None)
            continue
        ok = np.isfinite(m1["clean"][mask]) & np.isfinite(m1[name][mask])
        raw_r = float(np.corrcoef(m1["clean"][mask][ok], m1[name][mask][ok])[0, 1])
        ok2 = np.isfinite(ref_resid[mask]) & np.isfinite(resid[mask])
        resid_r = (float(np.corrcoef(ref_resid[mask][ok2], resid[mask][ok2])[0, 1])
                   if ok2.sum() > 10 else float("nan"))
        print(f"  {name:10s} {amp['rms_kms']:9.3f} {raw_r:8.4f} {resid_r:9.4f}")
        fold_row["methods"][name] = dict(rms=amp["rms_kms"], raw_r=raw_r, resid_r=resid_r)

    all_rows.append(fold_row)
    with open("results/self-gravitating/sg_loo_wiggle.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"  [saved {len(all_rows)}/5 folds, {(time.time()-t_start)/60:.0f} min elapsed]")

print("\n" + "=" * 70)
print("SUMMARY: resid r per fold (higher = wiggle preserved)")
print("=" * 70)
print(f"{'fold':>4} {'holdout':24s} {'dirty':>8} {'frozen':>8} {'finetune':>9} {'fresh':>8}")
for r in all_rows:
    m = r["methods"]
    print(f"{r['fold']:4d} {r['holdout'][:24]:24s} "
          f"{m['dirty']['resid_r']:8.3f} {m['frozen']['resid_r']:8.3f} "
          f"{m['finetune']['resid_r']:9.3f} {m['fresh']['resid_r']:8.3f}")

for name in ("dirty", "frozen", "finetune", "fresh"):
    vals = np.array([r["methods"][name]["resid_r"] for r in all_rows])
    print(f"{name:10s} mean {np.nanmean(vals):.3f} +/- {np.nanstd(vals, ddof=1):.3f}")

print("\nsaved -> results/self-gravitating/sg_loo_wiggle.json")
