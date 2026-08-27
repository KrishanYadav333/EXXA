"""
Does the self-gravitating cube actually show a GI wiggle? Jason's reading list (Hall+2020/
2021/2022, Terry+2024, Speedie+2024) is all built around this one diagnostic: fit the disk's
Keplerian rotation, subtract it from the moment-1 map, look at the residual. A planet leaves
a localised kink; gravitational instability leaves a global "interlocking fingers" pattern.

DISTPC=140 pc in dirty_cube.fits matches Hall+2020's founding simulation exactly, so this
cube is very likely built on that same pipeline -- which means a wiggle should be there to
find.

Run: PYTHONPATH=.. python3 gi_wiggle_check.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import fit_keplerian, wiggle_residual, wiggle_amplitude

D = "self-gravitating cube and dirty cube/kinematic_data"
TRIM = (30, 571)   # excludes the byte-identical padding blocks at both ends


def velax_from_header(hdr, n0, n1):
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    idx = np.arange(n0, n1)
    return (crval + (idx + 1 - crpix) * cdelt) * 1000.0


def main():
    with fits.open(f"{D}/lines.fits", memmap=True) as h:
        hdr = h[0].header
        clean = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)
    with fits.open(f"{D}/dirty_cube.fits", memmap=True) as h:
        dhdr = h[0].header

    au_per_px = float(dhdr["PXAS"]) * float(dhdr["DISTPC"])
    print(f"scale: {au_per_px:.4f} AU/px (PXAS={dhdr['PXAS']}, DISTPC={dhdr['DISTPC']} pc)")

    velax = velax_from_header(hdr, *TRIM)
    m0, m1, m2 = generate_moment_maps("", data_velax=(clean, velax))
    # generate_moment_maps returns M1/M2 in the units of the velax fed to it -- m/s here,
    # per bettermoments' own convention. gi_wiggle.py works in km/s throughout (GM_SUN_OVER_AU
    # and the fit bounds are both km/s), so convert once, here, rather than silently mixing
    # units in the fit. That mismatch is what made the first version of this script pin
    # mstar at its upper bound and report a 683 km/s residual on a +/-9 km/s velocity axis.
    m1 = m1 / 1000.0
    mask = signal_mask(m0, frac=0.02)
    print(f"signal mask: {mask.sum()} px")

    geom = fit_keplerian(m1, mask, au_per_px)
    print("\nfitted disk geometry (clean cube):")
    for k, v in geom.items():
        print(f"  {k:12s} {v}")

    resid = wiggle_residual(m1, geom)
    amp = wiggle_amplitude(resid, mask)
    print(f"\nresidual over signal mask: RMS {amp['rms_kms']:.4f} km/s, "
          f"max |resid| {amp['max_abs_kms']:.4f} km/s")

    v_typical = float(np.nanmedian(np.abs(m1[mask] - geom["vsys"])))
    print(f"typical rotation speed in the mask: {v_typical:.3f} km/s "
          f"-> residual is {100*amp['rms_kms']/v_typical:.2f}% of rotation")

    resid_masked = np.where(mask, resid, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im0 = ax[0].imshow(np.where(mask, m1, np.nan), cmap="RdBu_r", origin="lower")
    ax[0].set_title("moment 1 (line-of-sight velocity)")
    plt.colorbar(im0, ax=ax[0], label="km/s")
    v = max(1e-6, np.nanpercentile(np.abs(resid_masked), 98))
    im1 = ax[1].imshow(resid_masked, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
    ax[1].set_title(f"M1 minus Keplerian fit\n(the GI wiggle, if present)")
    plt.colorbar(im1, ax=ax[1], label="km/s")
    im2 = ax[2].imshow(np.where(mask, m0, np.nan), cmap="inferno", origin="lower")
    ax[2].set_title("moment 0 (for reference)")
    plt.colorbar(im2, ax=ax[2])
    plt.suptitle(f"Self-gravitating cube: Keplerian residual, RMS {amp['rms_kms']:.3f} km/s "
                 f"({100*amp['rms_kms']/v_typical:.1f}% of rotation speed)")
    plt.tight_layout()
    out = "results/self-gravitating/gi_wiggle_clean.png"
    plt.savefig(out, dpi=130)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
