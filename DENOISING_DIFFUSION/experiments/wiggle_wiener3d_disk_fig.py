"""
Visual for one SG synth disk: clean vs dirty vs 3D Wiener (beam deconvolution + filtering along
velocity), channel maps and the Keplerian-subtracted velocity (GI wiggle).
Run: python3 experiments/wiggle_wiener3d_disk_fig.py [run_9015_00370_rt_00 20.0]
"""
import os, sys, importlib.util
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gw", os.path.join(ROOT, "src/evaluation/gi_wiggle.py"))
gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)

run = sys.argv[1] if len(sys.argv) > 1 else "run_9015_00370_rt_00"
incl = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
ALPHA = 2.0
d = f"{ROOT}/self-gravitating cube and dirty cube/sg_synth/{run}"
clean = fits.getdata(f"{d}/{run}_clean.fits").astype(np.float64)
dirty = fits.getdata(f"{d}/{run}_dirty.fits").astype(np.float64)
hdr = fits.getheader(f"{d}/{run}_clean.fits")
C, n, _ = clean.shape
velax = hdr["CRVAL3"] + (np.arange(C) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
au = abs(hdr["CDELT1"]) * 3600.0 * float(hdr["DIST_PC"])

beam = fits.getdata(f"{ROOT}/results/self-gravitating/dirty_beam_recovered_v2.fits").astype(np.float64)
beam /= beam.sum()
pad = np.zeros((n, n)); hb = beam.shape[0] // 2; c = n // 2
pad[c - hb:c + hb + 1, c - hb:c + hb + 1] = beam
H = np.fft.fft2(np.fft.ifftshift(pad))
Hs = np.where(np.abs(H) > 1e-6 * np.abs(H).max(), H, 1e-6 * np.abs(H).max())
kr = np.sqrt(np.fft.fftfreq(n)[None, :] ** 2 + np.fft.fftfreq(n)[:, None] ** 2)
NB = 100; rb = np.minimum((kr / kr.max() * NB).astype(int), NB - 1); cnt = np.maximum(np.bincount(rb.ravel(), minlength=NB), 1)
rad = lambda P: (np.bincount(rb.ravel(), P.ravel(), NB) / cnt)[rb]

Hm2 = np.abs(H) ** 2
quiet = np.argsort(np.abs(dirty).reshape(C, -1).mean(1))[:max(10, C // 8)]
Pq = np.mean(np.abs(np.fft.fft2(dirty[quiet] - dirty[quiet].mean(axis=(1, 2), keepdims=True))) ** 2, axis=0)
hi = (kr > 0.12) & (kr < 0.35) & (Hm2 > 1e-8 * Hm2.max())
Pn = rad(float(np.median(Pq[hi] / Hm2[hi])) * Hm2)
D = np.fft.fftn(dirty)
Pd = np.stack([rad(p) for p in np.abs(D) ** 2]) / C
G = np.clip(1 - ALPHA * Pn[None] / np.maximum(Pd, 1e-30), 0, 1)
wien = np.real(np.fft.ifftn(D * G / Hs[None]))


def quad_m1(cube):
    idx = np.clip(np.argmax(cube, axis=0), 1, cube.shape[0] - 2)
    yy, xx = np.indices(idx.shape)
    ym, y0, yp = cube[idx - 1, yy, xx], cube[idx, yy, xx], cube[idx + 1, yy, xx]
    den = ym - 2 * y0 + yp
    off = np.clip(np.where(den != 0, 0.5 * (ym - yp) / den, 0.0), -0.5, 0.5)
    return velax[idx] + off * (velax[1] - velax[0])


scale = 1.0 if abs(velax).max() < 50 else 1e-3
m1 = {k: quad_m1(v) * scale for k, v in (("clean", clean), ("dirty", dirty), ("wiener", wien))}
m0 = clean.sum(0)
mask = np.isfinite(m0) & (np.abs(m0) > 0.05 * np.abs(m0).max())
geom = gw.fit_keplerian(m1["clean"], mask, au, fix_incl_deg=incl)
res = {k: gw.wiggle_residual(v, geom) for k, v in m1.items()}
yy_, xx_ = np.indices(mask.shape)
rpx = np.hypot(yy_ - geom["cy"], xx_ - geom["cx"])
R_IN = 0.18 * np.percentile(rpx[mask], 95)           # hide the beam-smeared inner disk
outer = mask & (rpx > R_IN)
ok = mask & np.isfinite(res["clean"])
r = {k: float(np.corrcoef(res["clean"][ok], res[k][ok & np.isfinite(res[k])])[0, 1]) if k != "clean" else None for k in res}
psnr = {k: float(20 * np.log10(clean.max() / np.sqrt(np.mean((v - clean) ** 2)))) for k, v in (("dirty", dirty), ("wiener", wien))}

ch = int(np.argmax([clean[i].max() for i in range(C)]))
ys, xs = np.where(mask); pad_ = 15
sl = (slice(max(ys.min() - pad_, 0), ys.max() + pad_), slice(max(xs.min() - pad_, 0), xs.max() + pad_))

fig, ax = plt.subplots(3, 3, figsize=(15, 14.5))
for j, (k, t) in enumerate((("clean", "clean (target)"), ("dirty", "dirty"), ("wiener", "3D Wiener (no training)"))):
    cube = {"clean": clean, "dirty": dirty, "wiener": wien}[k]
    v = np.percentile(clean[ch][sl], 99.5)
    im = ax[0, j].imshow(cube[ch][sl], cmap="inferno", vmin=0, vmax=v, origin="lower")
    ax[0, j].set_title(f"{t}\nchannel map, v = {velax[ch]*scale:+.2f} km/s" + (f"\nPSNR {psnr[k]:.1f} dB" if k in psnr else ""), fontsize=12)
    lim = np.percentile(np.abs(res["clean"][outer & np.isfinite(res["clean"])]), 97)
    im2 = ax[1, j].imshow(np.where(outer, res[k], np.nan)[sl], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    ax[1, j].set_facecolor("#111")
    ax[1, j].set_title("GI wiggle, outer disk: M1 minus Keplerian (km/s)" + (f"\nr vs clean (whole disk) = {r[k]:.3f}" if r[k] is not None else ""), fontsize=12)
    if k != "clean":
        err = np.where(outer, res[k] - res["clean"], np.nan)
        ax[2, j].imshow(err[sl], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax[2, j].set_title("wiggle error (method - clean)", fontsize=12)
    else:
        spec_c = clean[:, mask].sum(1); ax[2, j].plot(velax * scale, spec_c / spec_c.max(), "k", label="clean")
        ax[2, j].plot(velax * scale, dirty[:, mask].sum(1) / spec_c.max(), "gray", label="dirty")
        ax[2, j].plot(velax * scale, wien[:, mask].sum(1) / spec_c.max(), "C0", label="Wiener")
        ax[2, j].legend(); ax[2, j].set_title("disk-integrated spectrum"); ax[2, j].set_xlabel("km/s")
for a in ax.ravel():
    if a.images: a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#111")
fig.suptitle(f"{run} (held-out SG disk, inclination {incl:.0f} deg): clean vs dirty vs 3D Wiener, alpha={ALPHA}", fontsize=14)
fig.tight_layout()
out = f"{ROOT}/results/self-gravitating/wiener3d_disk_{run}.png"
fig.savefig(out, dpi=100); print("saved", out, "r:", r, "psnr:", psnr)
