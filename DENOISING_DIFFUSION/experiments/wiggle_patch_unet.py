"""
Tests one specific hypothesis from PROGRESS.md (2026-09-11, resize round trip): does removing
the 600x600 -> 256 -> 600 resize from U-Net inference recover sharper structure on the wiggle
comparison? Same cube, same checkpoint, same mask, same shared Keplerian model as
wiggle_all_methods.py, only the U-Net's inference path changes -- resize (existing) vs
patch-based (this script) -- so the comparison isolates that one variable.

Patch-based: crop overlapping 256x256 tiles directly out of the native 600x600 cube (no
resize at all, a tile already matches the net's trained input size), run each tile through
the network unchanged, stitch with Hann-window overlap blending (accumulate weighted output
and weight separately, divide at the end -- standard tiled-inference blending, no dependence
on tile order and no seams).

Caveat this does NOT fix: `winner_aug_seed43.pth` was trained on line-emission cubes, whose
angular pixel scale (arcsec/px) differs from this self-gravitating cube's. Patch-based
inference removes the resize blur but does not close that separate physical-scale gap --
that is what Phase J's direct SG-training work exists to address. Read a difference here as
"is resize the dominant smoothing source", not as "does patching fully fix cross-domain use".

Run: PYTHONPATH=.. python3 experiments/wiggle_patch_unet.py
"""
import os, sys, math, time

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
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, compare_wiggles

SG = "self-gravitating cube and dirty cube/kinematic_data_v2"
UNET_CKPT = "models/08-seeds/winner_aug_seed43.pth"
SIZE, PATCH, OVERLAP, FRAC, MSTAR_BOUND = 256, 256, 64, 0.05, 50.0
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
    hdr, cdata = h[0].header, h[0].data[:]
with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
    ddata = h[0].data[:]
AU = abs(hdr["CDELT1"]) * 3600.0 * hdr.get("DIST_PC", 140.0)


def load_unet():
    ck = torch.load(UNET_CKPT, map_location=dev, weights_only=False)
    net = UNet(in_channels=ck.get("in_channels", 1), out_channels=1,
               base_channels=ck["base_channels"], channel_multipliers=ck["channel_multipliers"],
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, ck["base_channels"]),
               beam_dim=ck.get("beam_dim", 0)).to(dev)
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    return net


def resize_denoise(net, dirty, batch=8):
    """The existing method: whole-image resize down to SIZE, denoise, resize back up."""
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
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(b.shape[0]):
                out[s + k] = b[k] * rng[s + k] + los[s + k] if rng[s + k] > 0 else np.full((H, W), los[s + k])
    return out


def _tile_starts(length, patch, overlap):
    stride = patch - overlap
    starts = list(range(0, max(length - patch, 0) + 1, stride))
    if not starts or starts[-1] != length - patch:
        starts.append(length - patch)
    return sorted(set(starts))


def patch_denoise(net, dirty, batch=8):
    """Native-resolution tiled inference: crop, denoise, Hann-blend back, no resize anywhere."""
    C, H, W = dirty.shape
    ys = _tile_starts(H, PATCH, OVERLAP)
    xs = _tile_starts(W, PATCH, OVERLAP)
    hann = np.hanning(PATCH)
    hann[hann < 1e-3] = 1e-3  # keep image-edge tiles from getting near-zero weight
    window = np.outer(hann, hann).astype(np.float32)

    los = dirty.reshape(C, -1).min(axis=1); his = dirty.reshape(C, -1).max(axis=1)
    rng = np.where((his - los) > 0, his - los, 1.0)

    acc = np.zeros((C, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    tiles = [(y, x) for y in ys for x in xs]

    with torch.no_grad():
        for s in range(0, C, batch):
            centres = np.arange(s, min(s + batch, C))
            for (y, x) in tiles:
                crop = dirty[centres, y:y + PATCH, x:x + PATCH]
                n = (crop - los[centres, None, None]) / rng[centres, None, None]
                t = torch.from_numpy(n)[:, None].float().to(dev)
                p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)[:, 0].cpu().numpy()
                for j, c in enumerate(centres):
                    acc[c, y:y + PATCH, x:x + PATCH] += p[j] * window
            for c in centres:
                pass
            wsum[:] = 0.0
            for (y, x) in tiles:
                wsum[y:y + PATCH, x:x + PATCH] += window

    out = acc / wsum[None]
    return out * rng[:, None, None] + los[:, None, None]


def main():
    CH = list(range(240, 361, 1))
    velax = (hdr["CRVAL3"] + (np.array(CH) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    clean = np.stack([np.asarray(cdata[c], np.float32) for c in CH])
    dirty = np.stack([np.asarray(ddata[c], np.float32) for c in CH])

    net = load_unet()
    print(f"device {dev} | {len(CH)} channels | tiles per channel: "
          f"{len(_tile_starts(600, PATCH, OVERLAP))**2}")

    t0 = time.time()
    resize_out = resize_denoise(net, dirty)
    print(f"  resize-based denoise: {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    patch_out = patch_denoise(net, dirty)
    print(f"  patch-based denoise:  {(time.time()-t0)/60:.1f} min")

    cubes = {"clean": clean, "dirty": dirty, "U-Net (resize)": resize_out, "U-Net (patch)": patch_out}

    m0, _, _ = generate_moment_maps("", data_velax=(clean.astype(np.float64), velax))
    mask = signal_mask(m0, frac=FRAC)

    rows = {}
    for tag, cube in cubes.items():
        v0, _ = quadratic_moment1(cube.astype(np.float64), velax)
        rows[tag] = dict(m1=v0 / 1000.0)

    cmp = compare_wiggles({t: rows[t]["m1"] for t in rows}, mask, AU, reference="clean")
    g = cmp["clean"]["geom"]
    flag = " DEGEN" if g["mstar_msun"] > 0.9 * MSTAR_BOUND else ""
    print(f"\n  shared model (fit on clean): mstar={g['mstar_msun']:.3f} incl={g['incl_deg']:.1f}"
          f" pa={g['pa_deg']:.1f} vsys={g['vsys']:.3f}{flag}")

    print(f"\n  {'method':16s} {'residRMS':>9s} {'raw r':>8s} {'resid r':>9s}")
    for tag in cubes:
        rms = cmp[tag]["rms_kms"]
        if tag == "clean":
            print(f"  {tag:16s} {rms:9.3f} {'--':>8s} {'--':>9s}")
            continue
        ok = np.isfinite(rows["clean"]["m1"][mask]) & np.isfinite(rows[tag]["m1"][mask])
        raw = float(np.corrcoef(rows["clean"]["m1"][mask][ok], rows[tag]["m1"][mask][ok])[0, 1])
        print(f"  {tag:16s} {rms:9.3f} {raw:8.4f} {cmp[tag]['corr']:9.4f}")

    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    vm = np.nanpercentile(np.abs(rows["clean"]["m1"][mask]), 98)
    vr = np.nanpercentile(np.abs(cmp["clean"]["residual"][mask]), 98)
    for i, tag in enumerate(cubes):
        im = ax[0, i].imshow(np.where(mask, rows[tag]["m1"], np.nan), cmap="RdBu_r",
                             vmin=-vm, vmax=vm, origin="lower")
        ax[0, i].set_title(f"{tag}: M1"); plt.colorbar(im, ax=ax[0, i], fraction=0.046)
        im2 = ax[1, i].imshow(np.where(mask, cmp[tag]["residual"], np.nan), cmap="RdBu_r",
                              vmin=-vr, vmax=vr, origin="lower")
        ax[1, i].set_title(f"{tag}: residual (RMS {cmp[tag]['rms_kms']:.2f})")
        plt.colorbar(im2, ax=ax[1, i], fraction=0.046)
    plt.suptitle("U-Net inference: resize vs native-resolution patch (frac=0.05)", fontsize=13)
    plt.tight_layout()
    out_path = "results/self-gravitating/wiggle_patch_vs_resize.png"
    plt.savefig(out_path, dpi=120)
    print("\n  saved ->", out_path)


if __name__ == "__main__":
    main()
