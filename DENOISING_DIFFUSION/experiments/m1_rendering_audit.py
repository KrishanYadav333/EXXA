"""
Rendering audit: is the smoothness in the M1 pixels or in the mask/contour rendering?

Mask, moment estimator and contouring are shared across every panel, so they cancel in
between-method comparisons and can only hide or quantize structure equally. They cannot
manufacture the gap between dirty's 0.862 and the U-Net's 0.760, which is computed on raw
arrays before any renderer runs. This audit therefore settles a PRESENTATION question -- are
the U-Net M1 ARRAYS themselves smoother than dirty's, with no mask and no contour involved --
and protects the figures' credibility, not the numbers' validity.

`clean` is the control. If clean stays sharp and the U-Net goes smooth through the same
estimator, same mask and same rendering, the renderer is exonerated.

Plot-independent sharpness on raw M1: gradient energy and Laplacian variance inside the shared
mask AND over all finite pixels (a mask artifact would make those two columns disagree for the
U-Net only), plus high-frequency spectral fraction of the mask-zeroed frame at periods shorter
than 10 px and 5 px. The mask-zeroing injects edge power, but identically for every cube, so
it cancels in the comparison.

Rendering matrix per cube: imshow nomask / imshow masked / contourf nomask / contourf masked
at 24 levels, then a 60-level masked pair (clean vs U-Net) to show level quantization is not
manufacturing the difference.

Note: grepped first -- `wiggle_all_methods.py`, `wiggle_patch_unet.py` and
`wiggle_domain_split.py` contain no `gaussian_filter` and no `contour` call at all. Every
published panel is a raw `imshow`. The imshow column here reproduces them exactly.

Run: PYTHONPATH=.. python3 experiments/m1_rendering_audit.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1
from wiggle_domain_split import SG, CH0, CH1, load_winner_aug, denoise_single

OUT = "results/self-gravitating"


def grad_energy(m1, sel):
    """Mean squared gradient magnitude over `sel`. Higher = more small-scale structure."""
    gy, gx = np.gradient(np.nan_to_num(m1))
    return float(np.mean(gx[sel] ** 2 + gy[sel] ** 2))


def sharpness(m1, mask):
    z = np.nan_to_num(m1)
    grad = grad_energy(m1, mask)

    inner = z[1:-1, 1:-1]
    lap = z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2] - 4 * inner
    lz = float(lap.var())

    zm = np.where(mask, z, 0.0)
    F = np.fft.fftshift(np.abs(np.fft.fft2(zm)) ** 2)
    n = m1.shape[0]
    f = np.fft.fftshift(np.fft.fftfreq(n))
    yy, xx = np.meshgrid(f, f, indexing="ij")
    rr = np.sqrt(yy ** 2 + xx ** 2)
    tot = F.sum()
    return grad, lz, float(F[rr > 1 / 10.0].sum() / tot), float(F[rr > 1 / 5.0].sum() / tot)


def main():
    with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
        hdr, cdata = h[0].header, h[0].data[:]
    with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
        ddata = h[0].data[:]
    CH = list(range(CH0, CH1))
    velax = (hdr["CRVAL3"] + (np.array(CH) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    clean = np.stack([np.asarray(cdata[c], np.float32) for c in CH]).astype(np.float64)
    dirty = np.stack([np.asarray(ddata[c], np.float32) for c in CH]).astype(np.float64)

    wa = denoise_single(load_winner_aug(), dirty)
    m0, _, _ = generate_moment_maps("", data_velax=(clean, velax))
    mask = signal_mask(m0, frac=0.05)

    cubes = {"clean": clean, "dirty": dirty, "winner_aug (resize)": wa}
    m1s = {t: quadratic_moment1(c, velax)[0] / 1000.0 for t, c in cubes.items()}

    print(f"{'cube':22s} {'gradE mask':>11s} {'gradE all':>11s} {'lapvar':>11s} "
          f"{'hf>1/10':>8s} {'hf>1/5':>8s}")
    stats = {}
    for t in cubes:
        g_m, lz, h10, h5 = sharpness(m1s[t], mask)
        g_a = grad_energy(m1s[t], np.isfinite(m1s[t]))
        stats[t] = (g_m, g_a)
        print(f"{t:22s} {g_m:11.3e} {g_a:11.3e} {lz:11.3e} {h10:8.4f} {h5:8.4f}")

    c_m, c_a = stats["clean"]
    print(f"\n  grad-energy ratio vs clean (masked): dirty {stats['dirty'][0]/c_m:.3f}, "
          f"U-Net {stats['winner_aug (resize)'][0]/c_m:.3f}")
    print(f"  grad-energy ratio vs clean (all px):  dirty {stats['dirty'][1]/c_a:.3f}, "
          f"U-Net {stats['winner_aug (resize)'][1]/c_a:.3f}")
    print("\n  Pre-registered reading: U-Net ratio far below dirty's in the RAW arrays means")
    print("  the smoothness is in the pixels and the renderer is exonerated. Ratios near")
    print("  equal means the figure code is the culprit and the panels must be reissued.")
    print("  Masked and all-pixel columns disagreeing for the U-Net ONLY would indicate a")
    print("  mask artifact rather than a real difference.")

    lev = 24
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    for i, t in enumerate(cubes):
        m1 = m1s[t]
        vm = np.nanpercentile(np.abs(m1[mask]), 99)
        mm = np.ma.masked_invalid(np.where(mask, m1, np.nan))
        ax[i, 0].imshow(m1, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
        ax[i, 0].set_title(f"{t}: imshow, no mask")
        ax[i, 1].imshow(mm, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
        ax[i, 1].set_title(f"{t}: imshow, masked  <- the published path")
        ax[i, 2].contourf(m1, levels=lev, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax[i, 2].set_title(f"{t}: contourf {lev}, no mask")
        ax[i, 3].contourf(mm, levels=lev, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax[i, 3].set_title(f"{t}: contourf {lev}, masked")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    plt.suptitle("M1 rendering audit: same arrays, four renderings. Rows share everything "
                 "but the cube.", fontsize=14)
    plt.tight_layout()
    p1 = f"{OUT}/m1_rendering_audit.png"
    plt.savefig(p1, dpi=110)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for j, t in enumerate(["clean", "winner_aug (resize)"]):
        m1 = m1s[t]
        vm = np.nanpercentile(np.abs(m1[mask]), 99)
        mm = np.ma.masked_invalid(np.where(mask, m1, np.nan))
        ax[j].contourf(mm, levels=60, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax[j].set_title(f"{t}: contourf 60 levels, masked")
        ax[j].set_xticks([]); ax[j].set_yticks([])
    plt.suptitle("60 levels: is level quantization hiding structure?", fontsize=13)
    plt.tight_layout()
    p2 = f"{OUT}/m1_rendering_audit_60lev.png"
    plt.savefig(p2, dpi=110)
    plt.close()
    print(f"\n  saved -> {p1}\n  saved -> {p2}")


if __name__ == "__main__":
    main()
