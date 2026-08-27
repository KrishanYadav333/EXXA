# The wiggle survives in the raw dirty cube -- with the right estimator

2026-08-27's dirty-cube Keplerian fit did not converge (mass pinned at its 50 Msun bound,
inclination collapsed to near-degenerate), even from the physically correct starting point.
Flagged as unresolved: does the wiggle information genuinely not survive in the dirty cube,
or was the failure the analysis method rather than the data?

## The suspect: `collapse_first` on 51.55%-negative data

`generate_moment_maps` computes M1 via bettermoments' `collapse_first`, an intensity-weighted
mean over the spectral axis. Phase 0 established the dirty cube is 51.55% negative pixels --
a genuine interferometric dirty beam, not a restored map. A negative "weight" in a
weighted-mean sum is not a small perturbation: it can flip the sign of the average or divide
by a total near zero, exactly the kind of instability that would pin a downstream fit at its
bounds.

`collapse_quadratic` (Teague & Foreman-Mackey 2018) fits a parabola to the peak channel and
its two neighbours instead of averaging the whole spectrum. A negative sidelobe elsewhere in
the spectrum cannot pull it off the true line centre. New in `src/evaluation/gi_wiggle.py`:
`quadratic_moment1`.

**Validated first, quantitatively, on synthetic data with an injected negative sidelobe**
(`tests/test_gi_wiggle.py` case 6): the naive intensity-weighted mean is pulled 1341 m/s off
the true line centre; the quadratic estimator is off by 21.5 m/s. 17 checks total.

## On the real cube, the fit converges

| | mstar (Msun) | incl (deg) | PA (deg) | vsys (km/s) | converged? |
|---|---|---|---|---|---|
| clean (quadratic) | 0.522 | 33.4 | 178.8 | 0.100 | yes |
| dirty (quadratic) | **0.580** | **27.5** | **175.2** | 0.083 | **yes** |

Compare to the first-moment dirty fit, which ran to mass=49.97 (its bound) and
inclination=1.8 (near face-on, degenerate). The quadratic fit lands within 11% of clean's
mass and 6 degrees of its inclination and PA, starting from the same clean-cube-informed init.

**Residual correlation, clean vs dirty: r = +0.92.** Raw M1 correlation: r = +0.77 (against
0.72 for the first-moment estimator on the same pair).

## What this settles

The wiggle IS recoverable from the raw, undenoised dirty cube -- the earlier non-convergence
was the estimator, not a genuine absence of signal. This is consistent with why the
literature (Hall+2021 in particular) works from individual channel maps or robust estimators
rather than the collapsed intensity-weighted moment-1, and it sharpens the earlier finding
about the trained U-Net: if the wiggle is present and recoverable in the raw data, and the
model destroys it anyway, that is a stronger statement about the model than "the signal may
have already been unrecoverable to begin with".

**Still running:** the same three-way comparison (clean/dirty/denoised) with the quadratic
estimator applied throughout, to check whether the earlier "denoising makes it worse"
conclusion (built on the less robust first-moment estimator for all three) survives the
better method too. See `gi_wiggle_quadratic_full.py` / `results/PROGRESS.md` for the outcome.
