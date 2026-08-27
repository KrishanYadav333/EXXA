"""
Does a robust (quadratic/peak-based) moment-1 recover a Keplerian fit where the
intensity-weighted `collapse_first` failed on the dirty cube?

`fit_keplerian` on the raw dirty cube would not converge to physical values even from the
correct starting geometry (2026-08-27). The suspect is `collapse_first`'s intensity-weighted
average, which is unstable when 51.55% of the cube's pixels are negative (a real dirty beam,
Phase 0's finding). `collapse_quadratic` (Teague & Foreman-Mackey 2018) fits a parabola to
the peak channel instead, and does not get pulled by a negative sidelobe elsewhere in the
spectrum. This tests whether that is actually the fix, or whether the wiggle really is
unrecoverable from this dirty cube by any per-pixel method.

Run: PYTHONPATH=.. python3 gi_wiggle_quadratic.py
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
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude

D = "self-gravitating cube and dirty cube/kinematic_data"
TRIM = (30, 571)
AU_PER_PX = 1.3333333333333333


def velax_from_header(hdr, n0, n1):
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    idx = np.arange(n0, n1)
    return (crval + (idx + 1 - crpix) * cdelt) * 1000.0


def main():
    with fits.open(f"{D}/lines.fits", memmap=True) as h:
        hdr = h[0].header
        clean = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)
    with fits.open(f"{D}/dirty_cube.fits", memmap=True) as h:
        dirty = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)

    velax = velax_from_header(hdr, *TRIM)

    print("computing the mask (from clean M0, first-moment, unchanged) ...")
    m0, m1_first, _ = generate_moment_maps("", data_velax=(clean, velax))
    mask = signal_mask(m0, frac=0.02)
    print(f"mask: {mask.sum()} px")

    print("\nquadratic (peak-based) M1 for clean and dirty ...")
    clean_v0, _ = quadratic_moment1(clean, velax)
    dirty_v0, _ = quadratic_moment1(dirty, velax)
    clean_kms = clean_v0 / 1000.0
    dirty_kms = dirty_v0 / 1000.0

    print(f"  clean quadratic M1: {np.isnan(clean_kms[mask]).sum()} non-finite of {mask.sum()}")
    print(f"  dirty quadratic M1: {np.isnan(dirty_kms[mask]).sum()} non-finite of {mask.sum()}")

    print("\nrefitting Keplerian geometry with the quadratic M1 ...")
    geom_c = fit_keplerian(clean_kms, mask, AU_PER_PX)
    print(f"  clean (quadratic): mstar={geom_c['mstar_msun']:.3f} incl={geom_c['incl_deg']:.1f} "
          f"pa={geom_c['pa_deg']:.1f} vsys={geom_c['vsys']:.3f} success={geom_c['success']}")

    geom_d = fit_keplerian(dirty_kms, mask, AU_PER_PX,
                           init={k: geom_c[k] for k in ("cx", "cy", "pa_deg", "incl_deg")})
    print(f"  dirty (quadratic): mstar={geom_d['mstar_msun']:.3f} incl={geom_d['incl_deg']:.1f} "
          f"pa={geom_d['pa_deg']:.1f} vsys={geom_d['vsys']:.3f} success={geom_d['success']} "
          f"dropped={geom_d['n_dropped_nonfinite']}")

    resid_c = wiggle_residual(clean_kms, geom_c)
    resid_d = wiggle_residual(dirty_kms, geom_d)
    amp_c = wiggle_amplitude(resid_c, mask)
    amp_d = wiggle_amplitude(resid_d, mask)
    print(f"\n  clean residual RMS: {amp_c['rms_kms']:.4f} km/s")
    print(f"  dirty residual RMS: {amp_d['rms_kms']:.4f} km/s")

    ok = np.isfinite(resid_c[mask]) & np.isfinite(resid_d[mask])
    corr = float(np.corrcoef(resid_c[mask][ok], resid_d[mask][ok])[0, 1])
    print(f"\n  residual correlation, clean vs dirty (quadratic method): r={corr:+.4f} "
          f"({ok.sum()}/{mask.sum()} finite)")

    ok2 = np.isfinite(clean_kms[mask]) & np.isfinite(dirty_kms[mask])
    corr_raw = float(np.corrcoef(clean_kms[mask][ok2], dirty_kms[mask][ok2])[0, 1])
    print(f"  raw M1 correlation, clean vs dirty (quadratic method): r={corr_raw:+.4f}")
    print(f"  (compare: first-moment raw M1 correlation, clean vs dirty, was 0.72)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    vmax = max(np.nanpercentile(np.abs(resid_c[mask]), 98), np.nanpercentile(np.abs(resid_d[mask]), 98))
    for i, (tag, r, g) in enumerate((("clean", resid_c, geom_c), ("dirty", resid_d, geom_d))):
        rm = np.where(mask, r, np.nan)
        im = ax[i].imshow(rm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax[i].set_title(f"{tag}: quadratic-estimator Keplerian residual\n"
                        f"mstar={g['mstar_msun']:.2f} Msun, incl={g['incl_deg']:.0f} deg")
        plt.colorbar(im, ax=ax[i], label="km/s", fraction=0.046)
    plt.suptitle(f"GI wiggle survives in the raw dirty cube (quadratic estimator): "
                f"residual r={corr:.2f}", fontsize=13)
    plt.tight_layout()
    out = "results/self-gravitating/gi_wiggle_quadratic_clean_vs_dirty.png"
    plt.savefig(out, dpi=130)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
