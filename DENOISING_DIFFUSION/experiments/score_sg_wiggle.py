"""
Does SG training recover the GI wiggle, not just the pixel moments?

Notebook 10's headline (frozen negative on all moments, finetune positive on M1/M2) was
scored on M0/M1/M2 amplitude. That is not the same question as whether the kinematic
signature -- the Keplerian-subtracted residual -- comes back. This scores it directly.

The holdout disk (run_9074_00025_rt_00) has KNOWN true inclination, 20 deg, from its .para
file (see PROGRESS.md 2026-09-04). Fixing it sidesteps the mass-inclination degeneracy found
in that same entry rather than re-triggering it.

One shared geometry, fit on clean with inclination fixed, reused for every method
(compare_wiggles' own logic, applied here since compare_wiggles itself assumes free
inclination and this cube needs it fixed).

Run: PYTHONPATH=.. python3 experiments/score_sg_wiggle.py
"""
import os, sys, math, time

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

HOLD = "self-gravitating cube and dirty cube/sg_synth/run_9074_00025_rt_00"
TRUE_INCL = 20.0        # run_9074's .para: 1.0 Msun, 20 deg, 175.178 pc
ARMS = {
    "frozen":   "models/08-seeds/winner_aug_seed43.pth",
    "finetune": "models/10-sg/sg_finetune.pth",
    "fresh":    "models/10-sg/sg_fresh.pth",
}
BASE, MULTS, SIZE, FRAC = 48, (1, 2, 4, 8), 256, 0.05
dev = "cuda" if torch.cuda.is_available() else "cpu"


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


with fits.open(f"{HOLD}/run_9074_00025_rt_00_clean.fits", memmap=True) as h:
    hdr, clean = h[0].header, h[0].data[:]
with fits.open(f"{HOLD}/run_9074_00025_rt_00_dirty.fits", memmap=True) as h:
    dirty = h[0].data[:]
velax = (hdr["CRVAL3"] + (np.arange(clean.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
au_per_px = abs(hdr["CDELT1"]) * 3600.0 * float(hdr["DIST_PC"])
print(f"holdout {clean.shape}, dv {hdr['CDELT3']:.4f} km/s, {au_per_px:.3f} AU/px, "
      f"true incl {TRUE_INCL} deg (fixed), device {dev}")

m0 = generate_moment_maps("", data_velax=(clean.astype(np.float64), velax))[0]
mask = signal_mask(m0, frac=FRAC)
print(f"mask: {mask.sum()} px ({100*mask.sum()/mask.size:.1f}% of field)")

cubes = {"clean": clean.astype(np.float64), "dirty": dirty.astype(np.float64)}
for name, path in ARMS.items():
    t0 = time.time()
    cubes[name] = denoise(path, dirty).astype(np.float64)
    print(f"  denoised {name} in {(time.time()-t0)/60:.1f} min")

m1 = {}
for name, cube in cubes.items():
    v0, _ = quadratic_moment1(cube, velax)
    m1[name] = v0 / 1000.0

geom = fit_keplerian(m1["clean"], mask, au_per_px, fix_incl_deg=TRUE_INCL)
print(f"\nshared geometry (fit on clean, incl fixed at {TRUE_INCL}): "
      f"mstar={geom['mstar_msun']:.3f} pa={geom['pa_deg']:.1f} vsys={geom['vsys']:.3f} "
      f"mstar_at_bound={geom['mstar_at_bound']}")

ref_resid = wiggle_residual(m1["clean"], geom)
print(f"\n{'method':10s} {'residRMS':>9s} {'raw r':>8s} {'resid r':>9s}")
rows = {}
for name in cubes:
    resid = wiggle_residual(m1[name], geom)
    amp = wiggle_amplitude(resid, mask)
    if name == "clean":
        print(f"{name:10s} {amp['rms_kms']:9.3f} {'--':>8s} {'--':>9s}")
        rows[name] = dict(rms=amp["rms_kms"], raw_r=None, resid_r=None)
        continue
    ok = np.isfinite(m1["clean"][mask]) & np.isfinite(m1[name][mask])
    raw_r = float(np.corrcoef(m1["clean"][mask][ok], m1[name][mask][ok])[0, 1])
    ok2 = np.isfinite(ref_resid[mask]) & np.isfinite(resid[mask])
    resid_r = float(np.corrcoef(ref_resid[mask][ok2], resid[mask][ok2])[0, 1]) if ok2.sum() > 10 else float("nan")
    print(f"{name:10s} {amp['rms_kms']:9.3f} {raw_r:8.4f} {resid_r:9.4f}")
    rows[name] = dict(rms=amp["rms_kms"], raw_r=raw_r, resid_r=resid_r)

print("\ndone")
