"""
Leaderboard for the SG v2 cube: untested/undertested checkpoints on the exact same protocol
(frac=0.05, channels 240-360, shared geometry fit on clean, mask=0.05) already used by
wiggle_all_methods.py, wiggle_domain_split.py and the rendering audit, so every number here
drops straight into that standing table.

Registry:
  winner_aug_seed43   (UNet k=0)        baseline: 0.7603
  winner_p10_seed44   (UNet k=0)        tests the early-stop/patience schedule as the only
                                         variable (same denoise_single path as winner_aug)
  winner_beam_seed42  (UNet k=0 + beam) tests beam conditioning out of its training
                                         distribution -- beam_features_of() on the SG dirty
                                         header returns [sin(2*BPA), cos(2*BPA), BMAJ*3600,
                                         BMIN*3600] = real, non-zero values (BMAJ 0.160",
                                         BMIN 0.100"), confirmed against the header directly,
                                         not a zero/no-op vector

DDPM K_AVG=1 is deliberately NOT in this script. Porting the exact DDIM sampling loop (EMA
weights, schedule reconstruction, posterior-mean-over-K-draws logic) is real work best done
inside 06-ddpm-line-emission.ipynb where that code already exists and is tested, rather than
half-reimplemented here.

Run: PYTHONPATH=.. python3 experiments/wiggle_leaderboard_sg.py
"""
import os, sys, math, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from src.data.fits_cube_dataset import beam_features_of
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, compare_wiggles
from wiggle_domain_split import (SG, SIZE, FRAC, MSTAR_BOUND, CH0, CH1, dev,
                                  load_winner_aug, denoise_single)

P10_CKPT = "models/08-seeds/winner_p10_seed44.pth"
BEAM_CKPT = "models/05-unet/winner_beam_seed42.pth"


def load_winner_p10():
    ck = torch.load(P10_CKPT, map_location=dev, weights_only=False)
    net = UNet(in_channels=ck.get("in_channels", 1), out_channels=1,
               base_channels=ck["base_channels"], channel_multipliers=ck["channel_multipliers"],
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, ck["base_channels"]),
               beam_dim=ck.get("beam_dim", 0)).to(dev)
    net.load_state_dict(ck["model_state_dict"], strict=True)
    net.eval()
    return net


def load_winner_beam(header):
    ck = torch.load(BEAM_CKPT, map_location=dev, weights_only=False)
    beam_dim = ck["beam_dim"]
    assert beam_dim == 4, f"expected beam_dim=4, checkpoint says {beam_dim}"
    net = UNet(in_channels=ck.get("in_channels", 1), out_channels=1,
               base_channels=ck["base_channels"], channel_multipliers=ck["channel_multipliers"],
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, ck["base_channels"]),
               beam_dim=beam_dim).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not miss and not unexp, f"winner_beam: state dict mismatch, miss={miss} unexp={unexp}"
    net.eval()
    vec = beam_features_of(header)
    print(f"    beam vector from SG header: {vec}")
    beam_vec = torch.from_numpy(vec).float().to(dev)
    return net, beam_vec


def denoise_beam(net, dirty, beam_vec, batch=8):
    """winner_beam's path: single channel + beam vector, resize to SIZE and back."""
    C, H, W = dirty.shape
    los = dirty.reshape(C, -1).min(axis=1); his = dirty.reshape(C, -1).max(axis=1)
    rng = his - los
    out = np.empty_like(dirty)
    with torch.no_grad():
        for s in range(0, C, batch):
            blk = dirty[s:s + batch]
            lo, hi = los[s:s + batch], his[s:s + batch]
            n = np.where((hi - lo)[:, None, None] > 0,
                         (blk - lo[:, None, None]) / np.where((hi - lo) > 0, hi - lo, 1)[:, None, None], 0)
            t = torch.from_numpy(n)[:, None].float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            b_batch = beam_vec[None, :].expand(t.size(0), -1)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), b_batch)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(b.shape[0]):
                out[s + k] = b[k] * rng[s + k] + los[s + k] if rng[s + k] > 0 else np.full((H, W), los[s + k])
    return out


def main():
    with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
        hdr, cdata = h[0].header, h[0].data[:]
    with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
        hdr_dirty, ddata = h[0].header, h[0].data[:]
    AU = abs(hdr["CDELT1"]) * 3600.0 * hdr.get("DIST_PC", 140.0)

    CH = list(range(CH0, CH1))
    velax = (hdr["CRVAL3"] + (np.array(CH) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    clean = np.stack([np.asarray(cdata[c], np.float32) for c in CH]).astype(np.float64)
    dirty = np.stack([np.asarray(ddata[c], np.float32) for c in CH]).astype(np.float64)

    print(f"device {dev} | {len(CH)} channels")

    t0 = time.time()
    wa = denoise_single(load_winner_aug(), dirty)
    print(f"  winner_aug_seed43 (baseline): {(time.time()-t0)/60:.2f} min")

    t0 = time.time()
    p10 = denoise_single(load_winner_p10(), dirty)
    print(f"  winner_p10_seed44:            {(time.time()-t0)/60:.2f} min")

    t0 = time.time()
    net_b, bvec = load_winner_beam(hdr_dirty)
    bm = denoise_beam(net_b, dirty, bvec)
    print(f"  winner_beam_seed42:           {(time.time()-t0)/60:.2f} min")

    cubes = {"clean": clean, "dirty": dirty,
             "winner_aug": wa, "winner_p10": p10, "winner_beam": bm}

    m0, _, _ = generate_moment_maps("", data_velax=(clean, velax))
    mask = signal_mask(m0, frac=FRAC)
    rows = {t: quadratic_moment1(c, velax)[0] / 1000.0 for t, c in cubes.items()}

    cmp = compare_wiggles(rows, mask, AU, reference="clean")
    g = cmp["clean"]["geom"]
    flag = " DEGEN" if g["mstar_msun"] > 0.9 * MSTAR_BOUND else ""
    print(f"\n  shared model: mstar={g['mstar_msun']:.3f} incl={g['incl_deg']:.1f}{flag}")

    print(f"\n  {'method':16s} {'residRMS':>9s} {'raw r':>8s} {'resid r':>9s}")
    for tag in cubes:
        rms = cmp[tag]["rms_kms"]
        if tag == "clean":
            print(f"  {tag:16s} {rms:9.3f} {'--':>8s} {'--':>9s}")
            continue
        ok = np.isfinite(rows["clean"][mask]) & np.isfinite(rows[tag][mask])
        raw = float(np.corrcoef(rows["clean"][mask][ok], rows[tag][mask][ok])[0, 1])
        print(f"  {tag:16s} {rms:9.3f} {raw:8.4f} {cmp[tag]['corr']:9.4f}")

    fig, ax = plt.subplots(2, len(cubes), figsize=(4.5 * len(cubes), 9))
    vm = np.nanpercentile(np.abs(rows["clean"][mask]), 98)
    vr = np.nanpercentile(np.abs(cmp["clean"]["residual"][mask]), 98)
    for i, tag in enumerate(cubes):
        im = ax[0, i].imshow(np.where(mask, rows[tag], np.nan), cmap="RdBu_r",
                             vmin=-vm, vmax=vm, origin="lower")
        ax[0, i].set_title(f"{tag}: M1"); plt.colorbar(im, ax=ax[0, i], fraction=0.046)
        im2 = ax[1, i].imshow(np.where(mask, cmp[tag]["residual"], np.nan), cmap="RdBu_r",
                              vmin=-vr, vmax=vr, origin="lower")
        ax[1, i].set_title(f"{tag}: residual (RMS {cmp[tag]['rms_kms']:.2f})")
        plt.colorbar(im2, ax=ax[1, i], fraction=0.046)
    plt.suptitle("SG v2 leaderboard: winner_aug vs winner_p10 vs winner_beam (frac=0.05)",
                 fontsize=13)
    plt.tight_layout()
    out = "results/self-gravitating/wiggle_leaderboard_sg.png"
    plt.savefig(out, dpi=120)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
