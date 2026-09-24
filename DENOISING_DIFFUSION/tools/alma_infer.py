"""
Run trained U-Net denoisers on a REAL ALMA line cube (exoALMA / DSHARP fiducial images).

Jason (2026-09-12): "if you want to take the time to do inference on actual ALMA data, this is probably
the data that we're going to start with"; the fiducial line images, "just look at the image. So the dot image
dot fits"; "13CO is the one you want ... but 12CO is fine". There is no ground truth on a real cube, so this
measures what the model changes, not how far it is from the truth: noise reduction off the line, flux
conservation, and how far the velocity field moves.

Reproduces training's preprocessing exactly (src/data/fits_cube_dataset.py):
    crop -> subtract the cube's own continuum (mean of the first and last n channels)
         -> per-channel min-max by that channel's own (min, max)      [invertible]
         -> bilinear resize to the model's pixel grid
         -> model -> undo the resize -> undo the min-max
The one addition is the pixel-scale match. The 14 training simulations have a beam of 5.8 to 11.2 px at
256 px, median 7.9. exoALMA's circular 0.15" beam is 7.9 px at 19 mas/px, so the crop is resampled to that
(ALMA_PLAN.md section 4). Only 256 px models are run.

    ~/Projects/exxa-infer-venv/bin/python tools/alma_infer.py \\
        --cube alma_data/exoALMA/MWC_758_13CO_fiducial.image.fits \\
        --ckpt aug43=models/best_models/winner_aug_seed43.pth --out alma_out/mwc758
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

C_MS = 299792458.0


def load_cube(path, partial):
    from alma_simobserve import read_fits
    hdr, cube = read_fits(path, partial=partial)
    if cube.ndim == 4:
        cube = cube[0]
    return hdr, cube


def velocity_axis(hdr, n):
    """Radio-convention velocity of each plane in m/s (what bettermoments expects)."""
    nu0 = float(hdr.get("RESTFRQ") or hdr["RESTFREQ"])
    if hdr["CTYPE3"].startswith("FREQ"):
        nu = float(hdr["CRVAL3"]) + (np.arange(n) + 1 - float(hdr["CRPIX3"])) * float(hdr["CDELT3"])
        return (nu0 - nu) / nu0 * C_MS
    v = float(hdr["CRVAL3"]) + (np.arange(n) + 1 - float(hdr["CRPIX3"])) * float(hdr["CDELT3"])
    return v * (1000.0 if hdr.get("CUNIT3", "m/s").lower().startswith("km") else 1.0)


def crop_window(hdr, fov_arcsec):
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
    cell = abs(float(hdr["CDELT1"])) * 3600.0
    side = min(int(round(fov_arcsec / cell)), ny, nx)
    side -= side % 2
    y0, x0 = (ny - side) // 2, (nx - side) // 2
    return y0, x0, side, cell


def load_unet(path, device):
    import torch
    from src.models.unet import UNet
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck.get("beam_dim", 0) == 0 and ck.get("in_channels", 1) == 1, \
        f"{path}: beam or spectral-context checkpoint, needs inputs a bare cube does not have"
    net = UNet(in_channels=1, out_channels=1, base_channels=ck["base_channels"],
               channel_multipliers=ck["channel_multipliers"], time_emb_dim=128, num_res_blocks=2,
               groups=math.gcd(8, ck["base_channels"]), beam_dim=0).to(device)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net, ck


def denoise(net, planes, target, device, batch=4):
    """planes: (C, s, s) continuum-subtracted crop. Returns the denoised cube, same shape."""
    import torch
    import torch.nn.functional as F
    from src.training.architectures import forward_fn
    fwd = forward_fn("unet")
    C, s, _ = planes.shape
    out = np.empty_like(planes)
    with torch.no_grad():
        for c0 in range(0, C, batch):
            blk = planes[c0:c0 + batch]
            lo = blk.reshape(len(blk), -1).min(axis=1)
            hi = blk.reshape(len(blk), -1).max(axis=1)
            span = np.where(hi > lo, hi - lo, 1.0)
            x = (blk - lo[:, None, None]) / span[:, None, None]          # per-channel min-max
            t = torch.from_numpy(x.astype(np.float32))[:, None].to(device)
            t = F.interpolate(t, size=(target, target), mode="bilinear", align_corners=False)
            pred, _ = fwd(net, t, None)
            pred = F.interpolate(pred, size=(s, s), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            out[c0:c0 + batch] = pred * span[:, None, None] + lo[:, None, None]   # undo min-max
    return out


def summarize(name, raw, den, off, m0r, m1r, m0d, m1d, dv_kms):
    """What the model changed. Off-line rms = channels the cube says are line-free."""
    lines = [f"--- {name}"]
    rr, rd = raw[off].std(), den[off].std()
    lines.append(f"off-line rms   raw {rr * 1e3:7.3f}  denoised {rd * 1e3:7.3f}  mJy/beam   "
                 f"({(1 - rd / rr) * 100:+.1f}% noise removed)")
    lines.append(f"M0 flux total  raw {np.nansum(m0r):.4g}  denoised {np.nansum(m0d):.4g}  "
                 f"ratio {np.nansum(m0d) / np.nansum(m0r):.4f}   (1.0 = flux conserved)")
    bright = m0r > np.nanpercentile(m0r, 90)
    d1 = (m1d - m1r)[bright & np.isfinite(m1d) & np.isfinite(m1r)] / 1000.0
    lines.append(f"M1 shift on brightest 10% of pixels (km/s): median {np.median(d1):+.4f}  "
                 f"|median| {np.median(np.abs(d1)):.4f}  p95 |shift| {np.percentile(np.abs(d1), 95):.4f}  "
                 f"(channel width {dv_kms:.3f})")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=8.0, help="central field of view kept, arcsec")
    ap.add_argument("--pix-mas", type=float, default=19.0,
                    help="model pixel scale; 19 mas puts the 0.15 arcsec beam at 7.9 px, the training median")
    ap.add_argument("--n-edge", type=int, default=5, help="edge channels for continuum, as in training")
    ap.add_argument("--partial", action="store_true", help="accept a truncated (unfinished) download")
    ap.add_argument("--device", default="cuda", help="cuda on Kaggle, mps or cpu locally")
    ap.add_argument("--max-planes", type=int, default=0, help="keep only the first N planes (smoke tests)")
    ap.add_argument("--batch", type=int, default=4, help="channels per forward pass")
    a = ap.parse_args()

    import torch
    from astropy.io import fits
    from src.evaluation.moment_maps import generate_moment_maps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device = torch.device(a.device if ok.get(a.device, False) else "cpu")
    os.makedirs(a.out, exist_ok=True)

    hdr, cube = load_cube(a.cube, a.partial)
    if a.max_planes:
        cube = cube[:a.max_planes]
    C = cube.shape[0]
    y0, x0, side, cell = crop_window(hdr, a.fov)
    target = 8 * max(1, round(a.fov / (a.pix_mas / 1000.0) / 8))
    print(f"cube: {C} planes of {hdr['NAXIS3']}, {cell * 1e3:.1f} mas/px, beam {float(hdr['BMAJ']) * 3600:.3f}\" "
          f"| crop {side}x{side} px = {side * cell:.2f}\" -> model grid {target} px "
          f"({side * cell / target * 1e3:.1f} mas/px) | device {device}")

    raw = np.nan_to_num(np.asarray(cube[:, y0:y0 + side, x0:x0 + side], dtype=np.float32))
    n = max(1, min(a.n_edge, C // 2))
    if a.partial:
        print(f"!! partial cube: continuum and off-line rms use the FIRST {n} planes only "
              f"(the last planes present are not line-free). Development run, not a result.")
        edge = raw[:n]
        off = np.arange(0, n)
    else:
        edge = np.concatenate([raw[:n], raw[C - n:]])
        off = np.r_[0:n, C - n:C]
    cont = edge.mean(axis=0)
    csub = raw - cont
    velax = velocity_axis(hdr, C)
    dv_kms = abs(velax[1] - velax[0]) / 1000.0

    print("moments of the raw cube ...")
    m0r, m1r, m2r = generate_moment_maps(None, data_velax=(csub, velax), rms_n_channels=n)
    results = {"raw": (csub, (m0r, m1r, m2r))}
    report = []
    for spec in a.ckpt:
        label, path = spec.split("=", 1)
        net, ck = load_unet(path, device)
        print(f"{label}: base {ck['base_channels']} mult {ck['channel_multipliers']} loss {ck.get('loss_name')} "
              f"-> denoising {C} planes ...")
        den = denoise(net, csub, target, device, batch=a.batch)
        m0d, m1d, m2d = generate_moment_maps(None, data_velax=(den, velax), rms_n_channels=n)
        results[label] = (den, (m0d, m1d, m2d))
        report += summarize(label, csub, den, off, m0r, m1r, m0d, m1d, dv_kms)
        h = fits.Header()
        text = ("BUNIT", "CTYPE1", "CTYPE2", "CTYPE3", "CUNIT1", "CUNIT2", "CUNIT3", "SPECSYS")
        for k in text + ("BMAJ", "BMIN", "BPA", "RESTFRQ", "CDELT1", "CDELT2", "CDELT3",
                         "CRVAL1", "CRVAL2", "CRVAL3", "CRPIX3"):
            if k in hdr:
                h[k] = hdr[k] if k in text else float(hdr[k])
        h["CRPIX1"] = float(hdr["CRPIX1"]) - x0
        h["CRPIX2"] = float(hdr["CRPIX2"]) - y0
        h["HISTORY"] = f"denoised by {label} ({os.path.basename(path)}), continuum-subtracted, exxa alma_infer.py"
        fits.PrimaryHDU(den.astype(np.float32), header=h).writeto(os.path.join(a.out, f"{label}_denoised.fits"), overwrite=True)

    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))

    names = list(results)
    fig, ax = plt.subplots(len(names), 3, figsize=(13, 4.1 * len(names)), squeeze=False)
    m0ref = results["raw"][1][0]
    ext = [-side * cell / 2, side * cell / 2] * 2
    vsys = np.nanmedian(results["raw"][1][1][m0ref > np.nanpercentile(m0ref, 90)])
    for i, nm in enumerate(names):
        m0, m1, m2 = results[nm][1]
        for j, (m, t, cm, lim) in enumerate([
                (m0, "M0", "inferno", (0, np.nanpercentile(m0ref, 99.5))),
                (m1, "M1 (km/s)", "RdBu_r", (vsys - 4000, vsys + 4000)),
                (m2, "M2 (km/s)", "viridis", (0, 2500))]):
            sc = 1000.0 if j else 1.0
            im = ax[i, j].imshow(m / sc if j else m, origin="lower", cmap=cm, extent=ext,
                                 vmin=lim[0] / sc if j else lim[0], vmax=lim[1] / sc if j else lim[1])
            ax[i, j].set_title(f"{nm}: {t}", fontsize=10)
            fig.colorbar(im, ax=ax[i, j], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "moments.png"), dpi=110)

    pk = int(np.argmax(np.abs(csub).sum(axis=(1, 2))))
    fig, ax = plt.subplots(len(names) - 1, 3, figsize=(13, 4.2 * (len(names) - 1)), squeeze=False)
    v = np.percentile(np.abs(csub[pk]), 99.5)
    for i, nm in enumerate(names[1:]):
        d = results[nm][0][pk]
        for j, (im_, t, lim) in enumerate([(csub[pk], "raw", v), (d, nm, v), (d - csub[pk], f"{nm} - raw", v)]):
            h = ax[i, j].imshow(im_ * 1e3, origin="lower", cmap="RdBu_r", extent=ext, vmin=-lim * 1e3, vmax=lim * 1e3)
            ax[i, j].set_title(f"channel {pk} ({velax[pk] / 1000:.2f} km/s): {t}", fontsize=10)
            fig.colorbar(h, ax=ax[i, j], fraction=0.046, label="mJy/beam")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "channel.png"), dpi=110)
    print(f"wrote {a.out}/ (report.txt, moments.png, channel.png, <label>_denoised.fits)")


if __name__ == "__main__":
    main()
