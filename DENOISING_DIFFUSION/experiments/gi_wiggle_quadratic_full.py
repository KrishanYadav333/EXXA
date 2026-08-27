"""
Complete the clean/dirty/denoised comparison using the robust (quadratic) velocity
estimator throughout, not just for clean/dirty.

2026-08-27's clean-vs-dirty-vs-denoised comparison used bettermoments' intensity-weighted
`collapse_first`, which is unstable on the 51.55%-negative dirty cube and pinned the dirty
fit's mass at its bound. gi_wiggle_quadratic.py showed `collapse_quadratic` fixes that: the
dirty fit converges and its residual correlates with clean's at 0.92. This finishes the
three-way comparison the same way, so "does denoising help or hurt" is answered with the
correct tool rather than the broken one -- and so any earlier claim built on collapse_first's
denoised-cube numbers can be checked against something not suspected of the same failure.

Reuses denoise_ood from eval_self_gravitating.py rather than duplicating the inference code.
Costs a second ~8-minute CPU inference pass; the previous run only saved moment maps, not the
full denoised cube, so there is no way to get the raw channels back without rerunning it.

Run: PYTHONPATH=.. python3 gi_wiggle_quadratic_full.py
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

from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude
from eval_self_gravitating import load_net, denoise_ood, CKPT

D = "self-gravitating cube and dirty cube/kinematic_data"
TRIM = (30, 571)
AU_PER_PX = 1.3333333333333333


def velax_from_header(hdr, n0, n1):
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    idx = np.arange(n0, n1)
    return (crval + (idx + 1 - crpix) * cdelt) * 1000.0


def main():
    net = load_net(CKPT)

    with fits.open(f"{D}/lines.fits", memmap=True) as h:
        hdr = h[0].header
        clean = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)
    with fits.open(f"{D}/dirty_cube.fits", memmap=True) as h:
        dirty = np.asarray(h[0].data[TRIM[0]:TRIM[1]], dtype=np.float32)

    velax = velax_from_header(hdr, *TRIM)

    print("denoising (reused from the OOD eval)...", flush=True)
    t0 = time.time()
    denoised = denoise_ood(dirty, net)
    print(f"done in {time.time()-t0:.0f}s")

    print("\ncomputing the mask (from clean M0, first-moment, unchanged for comparability)...")
    m0, _, _ = generate_moment_maps("", data_velax=(clean, velax))
    mask = signal_mask(m0, frac=0.02)
    print(f"mask: {mask.sum()} px")

    print("\nquadratic (peak-based) M1 for all three...")
    cubes = {"clean": clean, "dirty": dirty, "denoised": denoised}
    m1 = {}
    for tag, cube in cubes.items():
        v0, _ = quadratic_moment1(cube, velax)
        m1[tag] = v0 / 1000.0
        n_bad = int((~np.isfinite(m1[tag][mask])).sum())
        print(f"  {tag:10s} non-finite in mask: {n_bad}/{mask.sum()}")

    print("\nrefitting Keplerian geometry (dirty/denoised init'd from clean's fit)...")
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
              f"resid RMS {amps[tag]['rms_kms']:.3f} km/s | typical v {v_typ:.3f} km/s")

    print("\ncorrelation with clean (quadratic method throughout):")
    print(f"  {'':10s} {'raw M1':>10s} {'residual':>10s}")
    corrs = {}
    for tag in ("dirty", "denoised"):
        okA = np.isfinite(m1["clean"][mask]) & np.isfinite(m1[tag][mask])
        rA = float(np.corrcoef(m1["clean"][mask][okA], m1[tag][mask][okA])[0, 1])
        okB = np.isfinite(resids["clean"][mask]) & np.isfinite(resids[tag][mask])
        rB = float(np.corrcoef(resids["clean"][mask][okB], resids[tag][mask][okB])[0, 1])
        corrs[tag] = (rA, rB)
        print(f"  {tag:10s} {rA:>10.4f} {rB:>10.4f}")

    # Save the arrays: an 858s inference run should never have to be repeated just to make a
    # figure or re-check a number later.
    np.savez("experiments/gi_wiggle_quadratic_full_result.npz",
             mask=mask,
             clean_m1=m1["clean"], dirty_m1=m1["dirty"], denoised_m1=m1["denoised"],
             clean_resid=resids["clean"], dirty_resid=resids["dirty"], denoised_resid=resids["denoised"],
             **{f"{t}_{k}": v for t, g in geoms.items() for k, v in g.items()
                if isinstance(v, (int, float, bool))},
             dirty_corr_raw=corrs["dirty"][0], dirty_corr_resid=corrs["dirty"][1],
             denoised_corr_raw=corrs["denoised"][0], denoised_corr_resid=corrs["denoised"][1])
    print("saved -> experiments/gi_wiggle_quadratic_full_result.npz")

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    vmax_m1 = max(np.nanpercentile(np.abs(m1[t][mask]), 98) for t in ("clean", "dirty", "denoised"))
    vmax_r = max(np.nanpercentile(np.abs(resids[t][mask]), 98) for t in ("clean", "dirty", "denoised"))
    for i, tag in enumerate(("clean", "dirty", "denoised")):
        m1_masked = np.where(mask, m1[tag], np.nan)
        im = ax[0, i].imshow(m1_masked, cmap="RdBu_r", vmin=-vmax_m1, vmax=vmax_m1, origin="lower")
        ax[0, i].set_title(f"{tag}: quadratic M1")
        plt.colorbar(im, ax=ax[0, i], label="km/s", fraction=0.046)

        r_masked = np.where(mask, resids[tag], np.nan)
        im2 = ax[1, i].imshow(r_masked, cmap="RdBu_r", vmin=-vmax_r, vmax=vmax_r, origin="lower")
        g = geoms[tag]
        corr_txt = "" if tag == "clean" else f"\nresid r={corrs[tag][1]:.2f}, raw r={corrs[tag][0]:.2f}"
        ax[1, i].set_title(f"{tag}: Keplerian residual (mstar={g['mstar_msun']:.2f} Msun, "
                           f"incl={g['incl_deg']:.0f} deg){corr_txt}")
        plt.colorbar(im2, ax=ax[1, i], label="km/s", fraction=0.046)
    plt.suptitle("GI wiggle, quadratic (peak-fit) estimator throughout -- winner_aug seed 43\n"
                 "top: moment-1  |  bottom: Keplerian-subtracted residual, per-cube best fit",
                 fontsize=13)
    plt.tight_layout()
    out = "results/self-gravitating/gi_wiggle_quadratic_clean_vs_dirty_vs_denoised.png"
    plt.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
