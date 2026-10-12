"""
Wiggle check on Jason's five SHIPPED SG pairs (real dirty cubes, `_v3_extract`), 3D Wiener denoising only.

Clean and dirty carry the same beam in these pairs (PROGRESS.md 2026-09-04), so there is no beam to
undo: the filter is G = clip(1 - alpha * Pn / Pd, 0, 1) in (k_x, k_y, k_v), noise spectrum Pn from the
dirty cube's faintest channels, no clean data. Channels are windowed around the line for speed (same
window for every method). Scored like score_sg_wiggle_loo.py: shared Keplerian fit from clean,
true inclination fixed, signal_mask 0.05, resid_r vs clean.

Run: python3 experiments/wiggle_wiener3d_shipped.py
"""
import os, sys, json, glob, re, importlib.util, time
import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gw", os.path.join(ROOT, "src/evaluation/gi_wiggle.py"))
gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)
SRC = os.path.join(ROOT, "self-gravitating cube and dirty cube/_v3_extract/kinematic_data")
INCL = {"00019": 30.0, "15": 20.0, "25": 20.0, "32": 20.0, "74": 20.0}
ALPHAS = (1.0, 2.0, 4.0)


def quad_m1(cube, velax):
    idx = np.clip(np.argmax(cube, axis=0), 1, cube.shape[0] - 2)
    yy, xx = np.indices(idx.shape)
    ym, y0, yp = cube[idx - 1, yy, xx], cube[idx, yy, xx], cube[idx + 1, yy, xx]
    den = ym - 2 * y0 + yp
    off = np.clip(np.where(den != 0, 0.5 * (ym - yp) / den, 0.0), -0.5, 0.5)
    return velax[idx] + off * (velax[1] - velax[0])


rows = []
t0 = time.time()
for d in sorted(glob.glob(f"{SRC}/run_sg_*")):
    name = os.path.basename(d).rstrip(".")
    key = re.match(r"run_sg_(\d+)", name).group(1)
    key = "00019" if key == "00019" else key
    clean_p = glob.glob(f"{d}/*clean.fits")[0]; dirty_p = glob.glob(f"{d}/*dirty.fits")[0]
    with fits.open(clean_p, memmap=True) as h:
        hdr = h[0].header; cl = h[0].data[:].astype(np.float64)
    with fits.open(dirty_p, memmap=True) as h:
        di = h[0].data[:].astype(np.float64)
    C, n, _ = cl.shape
    velax_all = hdr["CRVAL3"] + (np.arange(C) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    velax_all = velax_all * (1.0 if abs(velax_all).max() < 50 else 1e-3)
    au = abs(hdr["CDELT1"]) * 3600.0 * float(hdr.get("DIST_PC", 140.0))
    spec_c = np.abs(cl).reshape(C, -1).max(1)
    on = np.where(spec_c > 0.02 * spec_c.max())[0]
    lo, hi = max(on.min() - 30, 0), min(on.max() + 30, C)
    cl, di, velax = cl[lo:hi], di[lo:hi], velax_all[lo:hi]
    Cw = cl.shape[0]
    diff = float(np.linalg.norm(di - cl) / np.linalg.norm(cl))

    kr = np.sqrt(np.fft.fftfreq(n)[None, :] ** 2 + np.fft.fftfreq(n)[:, None] ** 2)
    NB = 100; rb = np.minimum((kr / kr.max() * NB).astype(int), NB - 1); cnt = np.maximum(np.bincount(rb.ravel(), minlength=NB), 1)
    rad = lambda P: (np.bincount(rb.ravel(), P.ravel(), NB) / cnt)[rb]
    quiet = np.argsort(np.abs(di).reshape(Cw, -1).mean(1))[:max(8, Cw // 8)]
    Pn = rad(np.mean(np.abs(np.fft.fft2(di[quiet] - di[quiet].mean(axis=(1, 2), keepdims=True))) ** 2, axis=0))
    beam = fits.getdata(os.path.join(ROOT, "results/self-gravitating/dirty_beam_recovered_v2.fits")).astype(np.float64)
    beam /= beam.sum()
    pad = np.zeros((n, n)); hb = beam.shape[0] // 2; c0 = n // 2
    pad[c0 - hb:c0 + hb + 1, c0 - hb:c0 + hb + 1] = beam
    Hf = np.fft.fft2(np.fft.ifftshift(pad))
    Hs = np.where(np.abs(Hf) > 1e-6 * np.abs(Hf).max(), Hf, 1e-6 * np.abs(Hf).max())
    D = np.fft.fftn(di)
    Pd = np.stack([rad(p) for p in np.abs(D) ** 2]) / Cw
    cubes = {"dirty": di}
    for al in ALPHAS:
        G = np.clip(1 - al * Pn[None] / np.maximum(Pd, 1e-30), 0, 1)
        cubes[f"denoise_a{al}"] = np.real(np.fft.ifftn(D * G))
        cubes[f"deconv_a{al}"] = np.real(np.fft.ifftn(D * G / Hs[None]))
    del D, Pd

    m0 = cl.sum(0)
    mask = np.isfinite(m0) & (np.abs(m0) > 0.05 * np.abs(m0).max())
    m1c = quad_m1(cl, velax)
    geom = gw.fit_keplerian(m1c, mask, au, fix_incl_deg=INCL[key])
    ref = gw.wiggle_residual(m1c, geom)
    row = {"disk": name, "dirty_minus_clean_rel": diff, "mask_px": int(mask.sum()), "at_bound": bool(geom["mstar_at_bound"]),
           "ref_rms_kms": float(np.sqrt(np.nanmean(ref[mask] ** 2))), "methods": {}}
    for k, cube in cubes.items():
        r_ = gw.wiggle_residual(quad_m1(cube, velax), geom)
        ok = mask & np.isfinite(ref) & np.isfinite(r_)
        row["methods"][k] = {"resid_r": float(np.corrcoef(ref[ok], r_[ok])[0, 1]),
                             "psnr": float(20 * np.log10(cl.max() / np.sqrt(np.mean((cube - cl) ** 2))))}
    rows.append(row)
    print(f"{name:26s} diff {100*diff:4.1f}%  ref RMS {row['ref_rms_kms']:.3f} km/s  " +
          "  ".join(f"{k} r={v['resid_r']:.3f} psnr={v['psnr']:.1f}" for k, v in row["methods"].items()) + f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
    json.dump(rows, open(os.path.join(ROOT, "results/self-gravitating/wiener3d_shipped.json"), "w"), indent=1)

print("\nmean resid_r over disks:")
for k in rows[0]["methods"]:
    v = np.array([r["methods"][k]["resid_r"] for r in rows]); print(f"  {k:16s} {np.nanmean(v):.3f}")
