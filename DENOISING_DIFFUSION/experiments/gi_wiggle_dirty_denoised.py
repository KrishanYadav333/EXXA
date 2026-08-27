"""
Does the GI wiggle survive noise, and does the (already known to fail) U-Net destroy it?

Reuses the moment maps already computed in eval_self_gravitating.py's OOD run
(experiments/self_gravitating_ood_result.npz) rather than re-running the ~8-minute inference
pass. Same 541-channel range [30, 571), same winner_aug seed 43 checkpoint, same mask
(defined from the CLEAN M0, applied identically to all three -- moment_improvement's own
convention, so the comparison is apples to apples).

Run: PYTHONPATH=.. python3 gi_wiggle_dirty_denoised.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.moment_maps import signal_mask
from src.evaluation.gi_wiggle import fit_keplerian, wiggle_residual, wiggle_amplitude

AU_PER_PX = 1.3333333333333333   # PXAS(0.009524") * DISTPC(140 pc), from dirty_cube.fits

d = np.load("experiments/self_gravitating_ood_result.npz")
mask = signal_mask(d["clean_m0"], frac=0.02)
print(f"signal mask (from clean M0, shared across all three): {mask.sum()} px")

# Fit clean FIRST and use its converged geometry as the init for dirty/denoised, rather
# than an independent from-scratch guess for each. Physically this is the right thing to do:
# dirty and denoised are noisy/distorted versions of the SAME disk, so the true geometry is
# shared, and starting near it is both well-motivated and numerically far more stable than
# a crude unweighted-centroid guess. Without this, the dirty fit pinned mstar at its 50 Msun
# upper bound -- an optimiser failure from a bad starting point, not evidence about the data.
results = {}
order = ("clean", "dirty", "denoised")
clean_init = None
for tag, key in zip(order, ("clean_m1", "dirty_m1", "den_m1")):
    m1 = d[key] / 1000.0   # bettermoments convention: m/s in, m/s out
    geom = fit_keplerian(m1, mask, AU_PER_PX, init=clean_init)
    if tag == "clean":
        clean_init = {k: geom[k] for k in ("cx", "cy", "pa_deg", "incl_deg")}
    resid = wiggle_residual(m1, geom)
    amp = wiggle_amplitude(resid, mask)
    v_typ = float(np.nanmedian(np.abs(m1[mask] - geom["vsys"])))
    results[tag] = dict(geom=geom, resid=resid, amp=amp, v_typ=v_typ)
    print(f"\n=== {tag} ===")
    print(f"  mstar={geom['mstar_msun']:.3f} Msun  incl={geom['incl_deg']:.1f}deg  "
          f"pa={geom['pa_deg']:.1f}deg  vsys={geom['vsys']:.3f}  success={geom['success']}  "
          f"dropped {geom['n_dropped_nonfinite']}/{geom['n_dropped_nonfinite']+geom['n_fit']} "
          f"non-finite M1 pixels")
    print(f"  residual RMS {amp['rms_kms']:.4f} km/s | typical rotation {v_typ:.4f} km/s | "
          f"ratio {100*amp['rms_kms']/v_typ:.1f}%")

# Correlation of each residual against the CLEAN cube's own residual: does the pattern
# survive, or does dirty/denoised produce an unrelated pattern that happens to have similar
# amplitude? Amplitude alone cannot tell those apart; shape does.
clean_r = results["clean"]["resid"][mask].ravel()
print("\ncorrelation with the clean-cube residual (does the SAME pattern survive, not just")
print("similar amplitude):")
for tag in ("dirty", "denoised"):
    r = results[tag]["resid"][mask].ravel()
    ok = np.isfinite(clean_r) & np.isfinite(r)   # M1's own 0/0 leaves some pixels non-finite
    corr = float(np.corrcoef(clean_r[ok], r[ok])[0, 1])
    print(f"  {tag:10s} r = {corr:+.4f}  ({ok.sum()}/{ok.size} pixels finite in both)")

# Correlation of the RAW M1 maps (before any Keplerian subtraction), separate from the
# residual correlation above. The residual comparison is confounded when the fitted geometry
# itself differs wildly (denoised's mstar is 0.136 Msun against clean's 0.639): a near-flat
# model leaves the residual close to the raw map, so a high residual-correlation there could
# just mean "the raw velocity pattern survives", not "the wiggle specifically survives on top
# of a correctly recovered rotation curve". This separates the two claims.
print("\ncorrelation of the RAW M1 maps against clean (before Keplerian subtraction):")
clean_m1 = (d["clean_m1"] / 1000.0)[mask].ravel()
for tag, key in (("dirty", "dirty_m1"), ("denoised", "den_m1")):
    m1t = (d[key] / 1000.0)[mask].ravel()
    ok = np.isfinite(clean_m1) & np.isfinite(m1t)
    corr = float(np.corrcoef(clean_m1[ok], m1t[ok])[0, 1])
    print(f"  {tag:10s} r = {corr:+.4f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
vmax = max(np.nanpercentile(np.abs(results[t]["resid"][mask]), 98) for t in ("clean", "dirty", "denoised"))
for i, tag in enumerate(("clean", "dirty", "denoised")):
    r = np.where(mask, results[tag]["resid"], np.nan)
    im = ax[i].imshow(r, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax[i].set_title(f"{tag}\nRMS {results[tag]['amp']['rms_kms']:.2f} km/s")
    plt.colorbar(im, ax=ax[i], label="km/s", fraction=0.046)
plt.suptitle("GI wiggle residual: clean vs dirty vs denoised (winner_aug seed 43, same colour scale)")
plt.tight_layout()
out = "results/self-gravitating/gi_wiggle_clean_vs_dirty_vs_denoised.png"
plt.savefig(out, dpi=130)
print(f"\nsaved -> {out}")
