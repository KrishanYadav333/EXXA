"""
dirty / U-Net / DDRM / beam-only, all measured at IDENTICAL configuration.

Three errors on 2026-08-28 came from comparing wiggle correlations computed at different
channel samplings. The residual is acutely sensitive to it: at step 4 the CLEAN cube's own
residual RMS is 8x its step-1 value, because quadratic_moment1's 3-point parabola fit degrades
with coarse sampling, and that artifact is shared between cubes so they correlate spuriously.

So every method here is scored on the same channels, and the run reports at two samplings to
make the sensitivity visible rather than hidden behind one chosen number.

Run: PYTHONPATH=.. python3 wiggle_all_methods.py
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

from src.models.unet import UNet
from src.models.diffusion_unet import DiffusionUNet
from src.training.diffusion import DenoisingDiffusion, data_transform, inverse_data_transform
from src.training.ddrm import beam_transfer_function, ddrm_steps
from src.evaluation.forward_operator import apply_beam
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude

SG = "self-gravitating cube and dirty cube/kinematic_data_v2"
UNET_CKPT = "models/08-seeds/winner_aug_seed43.pth"
PRIOR_CKPT = "models/07-ddrm/ddrm_prior.pth"
BEAM = "results/self-gravitating/dirty_beam_recovered_v2.fits"
SIZE, MSTAR_BOUND = 256, 50.0
dev = "cuda" if torch.cuda.is_available() else "cpu"

with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
    hdr, cdata = h[0].header, h[0].data[:]
with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
    ddata = h[0].data[:]
beam = fits.getdata(BEAM).astype(np.float64)
AU = abs(hdr["CDELT1"]) * 3600.0 * hdr.get("DIST_PC", 140.0)


def unet_denoise(dirty):
    ck = torch.load(UNET_CKPT, map_location=dev, weights_only=False)
    import math
    net = UNet(in_channels=ck.get("in_channels", 1), out_channels=1,
               base_channels=ck["base_channels"], channel_multipliers=ck["channel_multipliers"],
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, ck["base_channels"]),
               beam_dim=ck.get("beam_dim", 0)).to(dev)
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    C, H, W = dirty.shape
    los = dirty.reshape(C, -1).min(axis=1); his = dirty.reshape(C, -1).max(axis=1)
    rng = his - los
    out = np.empty_like(dirty)
    with torch.no_grad():
        for s in range(0, C, 8):
            blk = dirty[s:s + 8]
            lo, hi = los[s:s + 8], his[s:s + 8]
            n = np.where((hi - lo)[:, None, None] > 0,
                         (blk - lo[:, None, None]) / np.where((hi - lo) > 0, hi - lo, 1)[:, None, None], 0)
            t = torch.from_numpy(n)[:, None].float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(b.shape[0]):
                out[s + k] = b[k] * rng[s + k] + los[s + k] if rng[s + k] > 0 else np.full((H, W), los[s + k])
    return out


def ddrm_restore(dirty):
    ck = torch.load(PRIOR_CKPT, map_location=dev, weights_only=False)
    r = DenoisingDiffusion(config=ck["config"], device=dev, checkpoint_path="/tmp/u.pth")
    r._core.load_state_dict(ck["state_dict"])
    if ck.get("ema_helper") and r.use_ema:
        r.ema_helper.load_state_dict(ck["ema_helper"]); r.ema_helper.ema(r._core)
    r.model.eval()
    tf = beam_transfer_function(beam, (SIZE, SIZE), device=dev); tf = tf / tf.max()
    out = []
    for i in range(dirty.shape[0]):
        d = dirty[i]; lo, hi = float(d.min()), float(d.max())
        d01 = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
        t = torch.from_numpy(d01)[None, None].float().to(dev)
        t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
        sy = float(hdr.get("RMS", 0.0013)) / max(hi - lo, 1e-12) * 2.0
        with torch.no_grad():
            xs, _ = ddrm_steps(data_transform(t),
                               list(range(0, r.num_timesteps, r.num_timesteps // 50)),
                               r.model, r.betas, tf, sigma_y=sy,
                               prediction_type=ck["prediction_type"])
        o = inverse_data_transform(xs[-1].to(dev))
        o = Fn.interpolate(o, d.shape, mode="bilinear", align_corners=False)
        out.append(o[0, 0].cpu().numpy() * (hi - lo) + lo)
    return np.stack(out)


for step in (1, 4):
    CH = list(range(240, 361, step))
    velax = (hdr["CRVAL3"] + (np.array(CH) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    clean = np.stack([np.asarray(cdata[c], np.float32) for c in CH])
    dirty = np.stack([np.asarray(ddata[c], np.float32) for c in CH])

    print("\n" + "=" * 78)
    print(f"CONFIG: channels 240-360 step {step}  ({len(CH)} channels, "
          f"dv = {abs(hdr['CDELT3'])*step:.3f} km/s)")
    print("=" * 78)

    t0 = time.time()
    cubes = {"clean": clean, "dirty": dirty,
             "beam-only": apply_beam(clean.astype(np.float64), beam),
             "U-Net": unet_denoise(dirty),
             "DDRM": ddrm_restore(dirty)}
    print(f"  (methods computed in {(time.time()-t0)/60:.1f} min)")

    m0, _, _ = generate_moment_maps("", data_velax=(clean.astype(np.float64), velax))
    mask = signal_mask(m0, frac=0.02)

    rows = {}
    for tag, cube in cubes.items():
        v0, _ = quadratic_moment1(cube.astype(np.float64), velax)
        m1 = v0 / 1000.0
        init = None if tag == "clean" else {k: rows["clean"]["g"][k]
                                            for k in ("cx", "cy", "pa_deg", "incl_deg")}
        g = fit_keplerian(m1, mask, AU, init=init)
        rows[tag] = dict(m1=m1, g=g, r=wiggle_residual(m1, g),
                         a=wiggle_amplitude(wiggle_residual(m1, g), mask))

    print(f"\n  {'method':12s} {'mstar':>7s} {'incl':>6s} {'residRMS':>9s} {'raw r':>8s} {'resid r':>9s} {'':>4s}")
    for tag in cubes:
        g, a = rows[tag]["g"], rows[tag]["a"]
        flag = " DEGEN" if g["mstar_msun"] > 0.9 * MSTAR_BOUND else ""
        if tag == "clean":
            print(f"  {tag:12s} {g['mstar_msun']:7.3f} {g['incl_deg']:6.1f} {a['rms_kms']:9.3f}"
                  f" {'--':>8s} {'--':>9s}{flag}")
            continue
        ok = np.isfinite(rows["clean"]["m1"][mask]) & np.isfinite(rows[tag]["m1"][mask])
        raw = float(np.corrcoef(rows["clean"]["m1"][mask][ok], rows[tag]["m1"][mask][ok])[0, 1])
        ok2 = np.isfinite(rows["clean"]["r"][mask]) & np.isfinite(rows[tag]["r"][mask])
        res = float(np.corrcoef(rows["clean"]["r"][mask][ok2], rows[tag]["r"][mask][ok2])[0, 1])
        print(f"  {tag:12s} {g['mstar_msun']:7.3f} {g['incl_deg']:6.1f} {a['rms_kms']:9.3f}"
              f" {raw:8.4f} {res:9.4f}{flag}")

    if step == 1:
        np.savez("experiments/wiggle_all_methods_step1.npz", mask=mask,
                 **{f"{t}_{k}": rows[t][k] for t in rows for k in ("m1", "r")})
        fig, ax = plt.subplots(2, 5, figsize=(23, 9))
        vm = np.nanpercentile(np.abs(rows["clean"]["m1"][mask]), 98)
        vr = np.nanpercentile(np.abs(rows["clean"]["r"][mask]), 98)
        for i, tag in enumerate(cubes):
            im = ax[0, i].imshow(np.where(mask, rows[tag]["m1"], np.nan), cmap="RdBu_r",
                                 vmin=-vm, vmax=vm, origin="lower")
            ax[0, i].set_title(f"{tag}: M1"); plt.colorbar(im, ax=ax[0, i], fraction=0.046)
            im2 = ax[1, i].imshow(np.where(mask, rows[tag]["r"], np.nan), cmap="RdBu_r",
                                  vmin=-vr, vmax=vr, origin="lower")
            ax[1, i].set_title(f"{tag}: residual (RMS {rows[tag]['a']['rms_kms']:.2f})")
            plt.colorbar(im2, ax=ax[1, i], fraction=0.046)
        plt.suptitle("GI wiggle, all methods at identical config (240-360, step 1)", fontsize=13)
        plt.tight_layout()
        plt.savefig("results/self-gravitating/wiggle_all_methods.png", dpi=120)
        print("\n  saved -> results/self-gravitating/wiggle_all_methods.png")
