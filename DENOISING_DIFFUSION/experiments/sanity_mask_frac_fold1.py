"""
Sanity check: is fold 1's near-ceiling wiggle result (everything within 0.006) a masking
artifact, or does it hold once the low-SNR halo is excluded?

Denoises fold 1's three models ONCE, then scores at three mask fracs, so this doesn't repeat
the fold's own ~80min denoise cost per frac tested.

Run: PYTHONPATH=.. python3 experiments/sanity_mask_frac_fold1.py
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

RUN = "run_9019_00019_rt_00"
D = f"self-gravitating cube and dirty cube/sg_synth/{RUN}"
TRUE_INCL = 30.0
BASE, MULTS, SIZE = 48, (1, 2, 4, 8), 256
FRACS = [0.05, 0.10, 0.15]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ARMS = {
    "frozen": "models/08-seeds/winner_aug_seed43.pth",
    "finetune": "models/11-loo/loo1_finetune.pth",
    "fresh": "models/11-loo/loo1_fresh.pth",
}


def denoise(path, cube):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UNet(in_channels=1, out_channels=1, base_channels=BASE, channel_multipliers=MULTS,
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, BASE), beam_dim=0).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not miss and not unexp
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


with fits.open(f"{D}/{RUN}_clean.fits", memmap=True) as h:
    hdr, clean = h[0].header, h[0].data[:]
with fits.open(f"{D}/{RUN}_dirty.fits", memmap=True) as h:
    dirty = h[0].data[:]
velax = (hdr["CRVAL3"] + (np.arange(clean.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
au_per_px = abs(hdr["CDELT1"]) * 3600.0 * float(hdr["DIST_PC"])
print(f"{RUN}: {clean.shape}, dv {hdr['CDELT3']:.4f} km/s, device {dev}")

cubes = {"clean": clean.astype(np.float64), "dirty": dirty.astype(np.float64)}
for name, path in ARMS.items():
    t0 = time.time()
    cubes[name] = denoise(path, dirty).astype(np.float64)
    print(f"  denoised {name} in {(time.time()-t0)/60:.1f} min")

m1 = {}
for name, cube in cubes.items():
    v0, _ = quadratic_moment1(cube, velax)
    m1[name] = v0 / 1000.0

m0 = generate_moment_maps("", data_velax=(clean.astype(np.float64), velax))[0]

print(f"\n{'frac':>5} {'mask%':>7} {'npx':>8}  {'dirty':>8} {'frozen':>8} {'finetune':>9} {'fresh':>8}")
for frac in FRACS:
    mask = signal_mask(m0, frac=frac)
    geom = fit_keplerian(m1["clean"], mask, au_per_px, fix_incl_deg=TRUE_INCL)
    ref_resid = wiggle_residual(m1["clean"], geom)
    vals = {}
    for name in ("dirty", "frozen", "finetune", "fresh"):
        resid = wiggle_residual(m1[name], geom)
        ok2 = np.isfinite(ref_resid[mask]) & np.isfinite(resid[mask])
        resid_r = float(np.corrcoef(ref_resid[mask][ok2], resid[mask][ok2])[0, 1]) if ok2.sum() > 10 else float("nan")
        vals[name] = resid_r
    print(f"{frac:5.2f} {100*mask.sum()/mask.size:6.1f}% {mask.sum():8d}  "
          f"{vals['dirty']:8.4f} {vals['frozen']:8.4f} {vals['finetune']:9.4f} {vals['fresh']:8.4f}"
          + ("  DEGEN" if geom["mstar_at_bound"] else ""))

print("\ndone")
