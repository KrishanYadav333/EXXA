"""
Does the recovered beam ALONE (no noise, no model) already destroy the wiggle residual?

2026-08-27's v2-cube test found low residual correlation for both dirty (0.11) and denoised
(0.11) versus clean, despite the bulk rotation field being nearly perfectly recoverable
(raw M1 r=0.98-0.99 for both). Left as a hypothesis: is the beam convolution itself smoothing
away the fine structure the wiggle consists of (information loss, which no denoiser can
undo), or is something else (real noise properties, the recovered beam being imperfect)
responsible?

This isolates the answer: take the CLEAN cube, convolve it with the recovered beam, add NO
noise, and run the exact same Keplerian-fit-and-residual pipeline. If the residual
correlation with the true clean residual is still low, smoothing alone explains the earlier
result. If it stays high, the beam is not the culprit and the noise/model do the damage.

Run: PYTHONPATH=.. python3 gi_wiggle_beam_only.py
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

from src.evaluation.moment_maps import signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude
from src.evaluation.forward_operator import apply_beam

D = "self-gravitating cube and dirty cube/kinematic_data_v2"
TRIM = (60, 541)
AU_PER_PX = 0.009523810800000001 * 140.0


def velax_from_header(hdr, n0, n1):
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    idx = np.arange(n0, n1)
    return (crval + (idx + 1 - crpix) * cdelt) * 1000.0


def main():
    with fits.open(f"{D}/clean_sg.fits", memmap=True) as h:
        hdr = h[0].header
        clean = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float64)
    with fits.open("results/self-gravitating/dirty_beam_recovered_v2.fits") as h:
        beam = h[0].data.astype(np.float64)

    velax = velax_from_header(hdr, *TRIM)
    print(f"cube: {clean.shape[0]} channels | beam: {beam.shape}, peak {beam.max():.3f}")

    print("\nconvolving clean with the recovered beam (no noise added)...")
    smoothed = apply_beam(clean, beam)

    print("computing quadratic M1 for clean and beam-only-smoothed...")
    v0_c, _ = quadratic_moment1(clean, velax)
    v0_s, _ = quadratic_moment1(smoothed, velax)
    m1_clean, m1_smoothed = v0_c / 1000.0, v0_s / 1000.0

    from src.evaluation.moment_maps import generate_moment_maps
    m0_clean, _, _ = generate_moment_maps("", data_velax=(clean, velax))
    mask = signal_mask(m0_clean, frac=0.02)
    print(f"mask: {mask.sum()} px")

    geom_c = fit_keplerian(m1_clean, mask, AU_PER_PX)
    geom_s = fit_keplerian(m1_smoothed, mask, AU_PER_PX,
                           init={k: geom_c[k] for k in ("cx", "cy", "pa_deg", "incl_deg")})
    for tag, g in (("clean", geom_c), ("beam-only smoothed", geom_s)):
        print(f"  {tag:20s} mstar={g['mstar_msun']:.3f} incl={g['incl_deg']:.1f} "
              f"pa={g['pa_deg']:.1f} vsys={g['vsys']:.3f}")

    resid_c = wiggle_residual(m1_clean, geom_c)
    resid_s = wiggle_residual(m1_smoothed, geom_s)
    amp_c = wiggle_amplitude(resid_c, mask)
    amp_s = wiggle_amplitude(resid_s, mask)
    print(f"\n  clean residual RMS: {amp_c['rms_kms']:.4f} km/s")
    print(f"  beam-only residual RMS: {amp_s['rms_kms']:.4f} km/s")

    okA = np.isfinite(m1_clean[mask]) & np.isfinite(m1_smoothed[mask])
    corr_raw = float(np.corrcoef(m1_clean[mask][okA], m1_smoothed[mask][okA])[0, 1])
    okB = np.isfinite(resid_c[mask]) & np.isfinite(resid_s[mask])
    corr_resid = float(np.corrcoef(resid_c[mask][okB], resid_s[mask][okB])[0, 1])
    print(f"\n  raw M1 correlation (clean vs beam-only smoothed): r={corr_raw:.4f}")
    print(f"  residual correlation (clean vs beam-only smoothed): r={corr_resid:.4f}")
    print(f"\n  (compare: real dirty gave residual r=0.111, real denoised gave r=0.108)")
    if corr_resid < 0.3:
        print("  -> SMOOTHING ALONE explains the earlier low correlation. Not a noise or")
        print("     model problem: the beam genuinely erases this signal. DDRM/VIREO relevant.")
    else:
        print("  -> beam smoothing alone does NOT destroy the residual. Something else (real")
        print("     noise statistics, or the recovered beam itself) is responsible.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    v = max(np.nanpercentile(np.abs(resid_c[mask]), 98), np.nanpercentile(np.abs(resid_s[mask]), 98))
    for i, (tag, r) in enumerate((("clean", resid_c), ("beam-only smoothed\n(no noise)", resid_s))):
        im = ax[i].imshow(np.where(mask, r, np.nan), cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        ax[i].set_title(tag)
        plt.colorbar(im, ax=ax[i], label="km/s", fraction=0.046)
    plt.suptitle(f"Beam-only ablation (no noise, no model): residual r={corr_resid:.2f}")
    plt.tight_layout()
    plt.savefig("results/self-gravitating/gi_wiggle_beam_only_ablation.png", dpi=130)
    print("\nsaved -> results/self-gravitating/gi_wiggle_beam_only_ablation.png")


if __name__ == "__main__":
    main()
