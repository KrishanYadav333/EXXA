"""
Full test of the corrected self-gravitating cube (clean_sg.fits / dirty_sg.fits,
2026-08-27): the OOD moment-improvement metric AND the GI wiggle Keplerian fit, off one
denoising pass, since both need the same ~14-minute CPU inference.

Padding: channels 0-59 and 541-600 are byte-identical repeats (checked directly), wider
than the original cube's 0-29/571-600. Trims to [60, 541), 481 channels.

Run: PYTHONPATH=.. python3 test_v2_cube.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

from src.evaluation.moment_maps import generate_moment_maps, signal_mask, moment_improvement
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude
from eval_self_gravitating import load_net, denoise_ood, CKPT

D = "self-gravitating cube and dirty cube/kinematic_data_v2"
TRIM = (60, 541)
AU_PER_PX = 0.009523810800000001 * 140.0   # PXAS-equivalent (CDELT1 in arcsec) * DIST_PC


def velax_from_header(hdr, n0, n1):
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    idx = np.arange(n0, n1)
    return (crval + (idx + 1 - crpix) * cdelt) * 1000.0


def main():
    net = load_net(CKPT)

    with fits.open(f"{D}/clean_sg.fits", memmap=True) as h:
        hdr = h[0].header
        clean = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)
    with fits.open(f"{D}/dirty_sg.fits", memmap=True) as h:
        dirty = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)

    print(f"trimmed cube: {clean.shape[0]} channels [{TRIM[0]}:{TRIM[1]})")
    velax = velax_from_header(hdr, *TRIM)

    print("\ndenoising (one pass, reused for both tests below)...", flush=True)
    t0 = time.time()
    denoised = denoise_ood(dirty, net)
    print(f"done in {time.time()-t0:.0f}s")

    # ================= Test 1: standard moment-improvement OOD metric ==================
    print("\n" + "=" * 70)
    print("TEST 1: moment-improvement (M0/M1/M2), first-moment + clip, RULES.md #4 metric")
    print("=" * 70)
    m_clean = generate_moment_maps("", data_velax=(clean, velax))
    m_dirty = generate_moment_maps("", data_velax=(dirty, velax))
    m_den = generate_moment_maps("", data_velax=(denoised, velax))
    result = moment_improvement(m_clean, m_dirty, m_den)
    print(f"  signal pixels scored: {result['n_px']}")
    for m in ("M0", "M1", "M2"):
        print(f"  {m}: {result[m]:+7.1f}%  (unmasked: {result[m + '_all']:+7.1f}%)")

    # ================= Test 2: GI wiggle, quadratic estimator throughout ===============
    print("\n" + "=" * 70)
    print("TEST 2: GI wiggle -- quadratic (peak-fit) estimator, Keplerian fit + residual")
    print("=" * 70)
    mask = signal_mask(m_clean[0], frac=0.02)
    print(f"  mask: {mask.sum()} px")

    cubes = {"clean": clean, "dirty": dirty, "denoised": denoised}
    m1 = {}
    for tag, cube in cubes.items():
        v0, _ = quadratic_moment1(cube.astype(np.float64), velax)
        m1[tag] = v0 / 1000.0

    geoms, resids, amps = {}, {}, {}
    geoms["clean"] = fit_keplerian(m1["clean"], mask, AU_PER_PX)
    for tag in ("dirty", "denoised"):
        init = {k: geoms["clean"][k] for k in ("cx", "cy", "pa_deg", "incl_deg")}
        geoms[tag] = fit_keplerian(m1[tag], mask, AU_PER_PX, init=init)
    for tag, g in geoms.items():
        resids[tag] = wiggle_residual(m1[tag], g)
        amps[tag] = wiggle_amplitude(resids[tag], mask)
        v_typ = float(np.nanmedian(np.abs(m1[tag][mask] - g["vsys"])))
        print(f"  {tag:10s} mstar={g['mstar_msun']:.3f} incl={g['incl_deg']:.1f} "
              f"pa={g['pa_deg']:.1f} vsys={g['vsys']:.3f} success={g['success']} | "
              f"resid RMS {amps[tag]['rms_kms']:.3f} | typical v {v_typ:.3f}")

    corrs = {}
    for tag in ("dirty", "denoised"):
        okA = np.isfinite(m1["clean"][mask]) & np.isfinite(m1[tag][mask])
        rA = float(np.corrcoef(m1["clean"][mask][okA], m1[tag][mask][okA])[0, 1])
        okB = np.isfinite(resids["clean"][mask]) & np.isfinite(resids[tag][mask])
        rB = float(np.corrcoef(resids["clean"][mask][okB], resids[tag][mask][okB])[0, 1])
        corrs[tag] = (rA, rB)
        print(f"  {tag:10s} raw-M1 r={rA:.4f}  residual r={rB:.4f}")

    # ---- save everything: arrays + figure for both tests, per the "always visuals" rule ----
    np.savez("experiments/v2_cube_test_result.npz",
             mask=mask, moment_improvement=result,
             clean_m1=m1["clean"], dirty_m1=m1["dirty"], denoised_m1=m1["denoised"],
             clean_resid=resids["clean"], dirty_resid=resids["dirty"], denoised_resid=resids["denoised"],
             **{f"{t}_{k}": v for t, g in geoms.items() for k, v in g.items()
                if isinstance(v, (int, float, bool))})
    print("\nsaved -> experiments/v2_cube_test_result.npz")

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    vmax_m1 = max(np.nanpercentile(np.abs(m1[t][mask]), 98) for t in cubes)
    vmax_r = max(np.nanpercentile(np.abs(resids[t][mask]), 98) for t in cubes)
    for i, tag in enumerate(("clean", "dirty", "denoised")):
        im = ax[0, i].imshow(np.where(mask, m1[tag], np.nan), cmap="RdBu_r",
                             vmin=-vmax_m1, vmax=vmax_m1, origin="lower")
        ax[0, i].set_title(f"{tag}: quadratic M1")
        plt.colorbar(im, ax=ax[0, i], label="km/s", fraction=0.046)
        im2 = ax[1, i].imshow(np.where(mask, resids[tag], np.nan), cmap="RdBu_r",
                              vmin=-vmax_r, vmax=vmax_r, origin="lower")
        g = geoms[tag]
        ctxt = "" if tag == "clean" else f"\nresid r={corrs[tag][1]:.2f}, raw r={corrs[tag][0]:.2f}"
        ax[1, i].set_title(f"{tag}: residual (mstar={g['mstar_msun']:.2f}, incl={g['incl_deg']:.0f}deg){ctxt}")
        plt.colorbar(im2, ax=ax[1, i], label="km/s", fraction=0.046)
    plt.suptitle(f"Corrected self-gravitating cube (v2) -- winner_aug seed 43\n"
                f"moment improvement: M0 {result['M0']:+.1f}%  M1 {result['M1']:+.1f}%  M2 {result['M2']:+.1f}%",
                fontsize=12)
    plt.tight_layout()
    plt.savefig("results/self-gravitating/v2_cube_test.png", dpi=130)
    print("saved -> results/self-gravitating/v2_cube_test.png")


if __name__ == "__main__":
    main()
