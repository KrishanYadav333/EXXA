"""
Sections 5-7 of notebook 07, run against the trained prior, with the corrected channel range.

The Kaggle run trained the prior successfully but scored it with only 10 channels (280-316),
which collapsed the Keplerian fit to its mstar bound and printed a meaningless "RECOVERY".
The line centre varies across channels 260-348 (that variation IS the disk rotation), so the
window has to cover it. Prior needs no retraining.

Run: PYTHONPATH=.. python3 ddrm_score.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Fn
from astropy.io import fits

from src.models.diffusion_unet import DiffusionUNet
from src.training.diffusion import (DenoisingDiffusion, data_transform,
                                    inverse_data_transform)
from src.training.ddrm import beam_transfer_function, ddrm_steps
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual

SG = "self-gravitating cube and dirty cube/kinematic_data_v2"
CKPT = "models/07-ddrm/ddrm_prior.pth"
CHANNELS = list(range(240, 361, 4))     # covers the 260-348 line-centre range
SIZE = 256
MSTAR_BOUND = 50.0
DDIM_STEPS = 50

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} | {len(CHANNELS)} channels")

ck = torch.load(CKPT, map_location=dev, weights_only=False)
cfg = ck["config"]
runner = DenoisingDiffusion(config=cfg, device=dev, checkpoint_path="/tmp/unused.pth")
runner._core.load_state_dict(ck["state_dict"])
if ck.get("ema_helper") and runner.use_ema:
    runner.ema_helper.load_state_dict(ck["ema_helper"])
    runner.ema_helper.ema(runner._core)      # sample from EMA weights
runner.model.eval()
print(f"prior: epoch {ck['epoch']}, val {ck['best_val_loss']:.2f}, "
      f"{ck['prediction_type']}-prediction, conditional={cfg.data.conditional}")

with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
    hdr = h[0].header
    sg_clean = np.stack([np.asarray(h[0].data[c], np.float32) for c in CHANNELS])
with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
    sg_dirty = np.stack([np.asarray(h[0].data[c], np.float32) for c in CHANNELS])
beam = fits.getdata("results/self-gravitating/dirty_beam_recovered_v2.fits").astype(np.float64)

transfer = beam_transfer_function(beam, (SIZE, SIZE), device=dev)
transfer = transfer / transfer.max()
print(f"transfer: {100*float((transfer > 0.01).float().mean()):.1f}% of modes above 1% gain")

restored = []
t0 = time.time()
for i in range(len(CHANNELS)):
    d = sg_dirty[i]
    lo, hi = float(d.min()), float(d.max())
    d01 = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    t = torch.from_numpy(d01)[None, None].float().to(dev)
    t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
    y = data_transform(t)
    sigma_y = float(hdr.get("RMS", 0.0013)) / max(hi - lo, 1e-12) * 2.0
    with torch.no_grad():
        xs, _ = ddrm_steps(y, list(range(0, runner.num_timesteps,
                                         runner.num_timesteps // DDIM_STEPS)),
                           runner.model, runner.betas, transfer, sigma_y=sigma_y,
                           prediction_type=ck["prediction_type"])
    out = inverse_data_transform(xs[-1].to(dev))
    out = Fn.interpolate(out, sg_clean.shape[-2:], mode="bilinear", align_corners=False)
    restored.append(out[0, 0].cpu().numpy() * (hi - lo) + lo)
    if i % 5 == 0:
        print(f"  {i}/{len(CHANNELS)}  ({time.time()-t0:.0f}s)", flush=True)
restored = np.stack(restored)
print(f"restored {restored.shape} in {(time.time()-t0)/60:.1f} min, "
      f"finite={np.isfinite(restored).all()}")

velax = ((hdr["CRVAL3"] + (np.array(CHANNELS) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0)
au_per_px = abs(hdr["CDELT1"]) * 3600.0 * hdr.get("DIST_PC", 140.0)
m0, _, _ = generate_moment_maps("", data_velax=(sg_clean.astype(np.float64), velax))
mask = signal_mask(m0, frac=0.02)

rows = {}
for tag, cube in (("clean", sg_clean), ("dirty", sg_dirty), ("DDRM", restored)):
    v0, _ = quadratic_moment1(cube.astype(np.float64), velax)
    m1 = v0 / 1000.0
    init = None if tag == "clean" else {k: rows["clean"]["geom"][k]
                                        for k in ("cx", "cy", "pa_deg", "incl_deg")}
    g = fit_keplerian(m1, mask, au_per_px, init=init)
    rows[tag] = dict(m1=m1, geom=g, resid=wiggle_residual(m1, g))

bad = [t for t, r in rows.items() if r["geom"]["mstar_msun"] > 0.9 * MSTAR_BOUND]
if bad:
    raise RuntimeError(f"degenerate Keplerian fit for {bad}; results would be meaningless")

print(f"\n{'':8s} {'mstar':>7s} {'incl':>6s} {'raw r':>8s} {'resid r':>9s}")
print(f"{'clean':8s} {rows['clean']['geom']['mstar_msun']:7.3f} "
      f"{rows['clean']['geom']['incl_deg']:6.1f}")
res_ddrm = None
for tag in ("dirty", "DDRM"):
    ok = np.isfinite(rows["clean"]["m1"][mask]) & np.isfinite(rows[tag]["m1"][mask])
    raw = float(np.corrcoef(rows["clean"]["m1"][mask][ok], rows[tag]["m1"][mask][ok])[0, 1])
    ok2 = np.isfinite(rows["clean"]["resid"][mask]) & np.isfinite(rows[tag]["resid"][mask])
    res = float(np.corrcoef(rows["clean"]["resid"][mask][ok2],
                            rows[tag]["resid"][mask][ok2])[0, 1])
    g = rows[tag]["geom"]
    print(f"{tag:8s} {g['mstar_msun']:7.3f} {g['incl_deg']:6.1f} {raw:8.4f} {res:9.4f}")
    if tag == "DDRM":
        res_ddrm = res

print(f"\n  beam-only floor : 0.116")
print(f"  DDRM            : {res_ddrm:.4f}")
print(f"  -> {'RECOVERY beyond the floor' if res_ddrm > 0.20 else 'NO recovery beyond the floor'}")

np.savez("experiments/ddrm_score_result.npz", mask=mask,
         **{f"{t}_{k}": v for t, r in rows.items() for k, v in
            (("m1", r["m1"]), ("resid", r["resid"]))})

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
vm = np.nanpercentile(np.abs(rows["clean"]["m1"][mask]), 98)
vr = np.nanpercentile(np.abs(rows["clean"]["resid"][mask]), 98)
for i, tag in enumerate(("clean", "dirty", "DDRM")):
    im = ax[0, i].imshow(np.where(mask, rows[tag]["m1"], np.nan), cmap="RdBu_r",
                         vmin=-vm, vmax=vm, origin="lower")
    ax[0, i].set_title(f"{tag}: M1"); plt.colorbar(im, ax=ax[0, i], fraction=0.046)
    im2 = ax[1, i].imshow(np.where(mask, rows[tag]["resid"], np.nan), cmap="RdBu_r",
                          vmin=-vr, vmax=vr, origin="lower")
    g = rows[tag]["geom"]
    ax[1, i].set_title(f"{tag}: residual (mstar={g['mstar_msun']:.2f})")
    plt.colorbar(im2, ax=ax[1, i], fraction=0.046)
plt.suptitle(f"DDRM restoration — residual r={res_ddrm:.3f} vs beam-only floor 0.116")
plt.tight_layout()
plt.savefig("results/self-gravitating/ddrm_restoration.png", dpi=130)
print("saved -> results/self-gravitating/ddrm_restoration.png")
