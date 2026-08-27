"""
Score the best trained U-Net (winner_aug, seed 43) on the self-gravitating pair -- the
out-of-distribution test proposed 2026-08-21 and only run now.

Why this is a real OOD test, not another holdout cube: every training cube has clean AND
dirty already in Jy/beam (Phase 0's A = I). This pair does not -- lines.fits is Jy/pixel, an
unconvolved sky model, and dirty_cube.fits has a REAL dirty beam (51.55% negative pixels).
The model was trained to remove additive noise with no deconvolution mechanism; nothing here
predicts whether that generalises to removing a beam it never saw and was never conditioned
on.

Edge channels are NOT usable as a line-free baseline here, unlike the line-emission training
cubes. Channels 0-29 and 571-600 of lines.fits are byte-identical repeats (padding), not real
data, so continuum subtraction and bettermoments' estimate_RMS (which reads data[:N]/data[-N:]
literally) would both be corrupted by them. This script trims to the non-padded range
[30, 570] before doing anything.

Run: PYTHONPATH=.. python3 eval_self_gravitating.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
# numpy < 2.0 lacks np.trapezoid, which the installed bettermoments version calls.
# Local-environment shim only; Kaggle's numpy/bettermoments pairing already matches.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import torch
import torch.nn.functional as F
from astropy.io import fits

from src.models.unet import UNet
from src.evaluation.moment_maps import generate_moment_maps, moment_improvement

D = "self-gravitating cube and dirty cube/kinematic_data"
CKPT = "models/08-seeds/winner_aug_seed43.pth"
TARGET = 256
TRIM = (30, 571)          # [start, stop) -- excludes the repeated-padding blocks
BS = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def load_net(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNet(in_channels=ck.get("in_channels", 1), out_channels=1,
               base_channels=ck["base_channels"], channel_multipliers=ck["channel_multipliers"],
               time_emb_dim=128, num_res_blocks=2, groups=8, beam_dim=ck.get("beam_dim", 0))
    net.load_state_dict(ck["model_state_dict"])
    net.eval().to(device)
    print(f"loaded {path}: epoch {ck['epoch']}, val_loss {ck['val_loss']:.6f}")
    return net


def denoise_ood(dirty, net):
    """
    Per-channel dirty-scale normalisation (identical to `_minmax_norm_shared`, no continuum
    subtraction -- see module docstring for why), resize to TARGET, denoise, resize back.
    """
    C, H, W = dirty.shape
    los = dirty.reshape(C, -1).min(axis=1)
    his = dirty.reshape(C, -1).max(axis=1)
    rngs = his - los
    norm = np.zeros_like(dirty)
    nz = rngs > 0
    norm[nz] = (dirty[nz] - los[nz, None, None]) / rngs[nz, None, None]

    out = np.empty_like(dirty)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, C, BS):
            t = torch.from_numpy(norm[s:s + BS])[:, None].float().to(device)
            t256 = F.interpolate(t, (TARGET, TARGET), mode="bilinear", align_corners=False)
            tz = torch.zeros(t256.size(0), dtype=torch.long, device=device)
            pred = net(t256, tz, None)
            back = F.interpolate(pred, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(back.shape[0]):
                ch = s + k
                out[ch] = back[k] * rngs[ch] + los[ch] if rngs[ch] > 0 else np.full((H, W), los[ch], np.float32)
            if s % (BS * 10) == 0:
                print(f"  channel {s}/{C}  ({time.time() - t0:.0f}s elapsed)", flush=True)
    return out


def velax_from_header(hdr, n0, n1):
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    idx = np.arange(n0, n1)
    v_kms = crval + (idx + 1 - crpix) * cdelt   # FITS 1-indexed pixel convention
    return v_kms * 1000.0                        # bettermoments wants m/s


def main():
    net = load_net(CKPT)

    with fits.open(f"{D}/lines.fits", memmap=True) as h:
        clean_hdr = h[0].header
        clean = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)
    with fits.open(f"{D}/dirty_cube.fits", memmap=True) as h:
        dirty = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)

    C = clean.shape[0]
    print(f"trimmed cube: {C} channels [{TRIM[0]}:{TRIM[1]}) of 601, shape {clean.shape}")
    velax = velax_from_header(clean_hdr, *TRIM)
    print(f"velax range: {velax.min()/1000:.2f} to {velax.max()/1000:.2f} km/s")

    print("\ndenoising...")
    denoised = denoise_ood(dirty, net)

    print("\ncomputing moment maps (clean / dirty / denoised)...")
    m_clean = generate_moment_maps("", data_velax=(clean, velax))
    m_dirty = generate_moment_maps("", data_velax=(dirty, velax))
    m_den = generate_moment_maps("", data_velax=(denoised, velax))

    result = moment_improvement(m_clean, m_dirty, m_den)
    print("\n" + "=" * 60)
    print("OUT-OF-DISTRIBUTION RESULT -- winner_aug seed 43 on the")
    print("self-gravitating pair (never seen in training)")
    print("=" * 60)
    print(f"  signal pixels scored: {result['n_px']}")
    for m in ("M0", "M1", "M2"):
        print(f"  {m}: {result[m]:+7.1f}%  (unmasked: {result[m + '_all']:+7.1f}%)")
    print("=" * 60)

    np.savez("experiments/self_gravitating_ood_result.npz",
             clean_m0=m_clean[0], clean_m1=m_clean[1], clean_m2=m_clean[2],
             dirty_m0=m_dirty[0], dirty_m1=m_dirty[1], dirty_m2=m_dirty[2],
             den_m0=m_den[0], den_m1=m_den[1], den_m2=m_den[2], **result)
    print("saved -> experiments/self_gravitating_ood_result.npz")


if __name__ == "__main__":
    main()
