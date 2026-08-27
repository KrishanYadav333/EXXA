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

## The full three-way comparison, quadratic estimator throughout

Completed by rerunning inference (858s) so the denoised cube's raw channels were available
for the same robust estimator, rather than relying on the first-moment maps already saved
from the 2026-08-27 OOD eval.

| | mstar (Msun) | incl (deg) | PA (deg) | vsys (km/s) |
|---|---|---|---|---|
| clean | 0.522 | 33.4 | 178.8 | 0.100 |
| dirty | 0.580 | 27.5 | 175.2 | 0.083 |
| denoised | 0.578 | **69.8** | **67.3** | **1.918** |

| | raw M1 correlation with clean | residual correlation with clean |
|---|---|---|
| dirty | **0.773** | **0.921** |
| denoised | **0.250** | **0.281** |

**This supersedes the 2026-08-27 first-moment comparison (0.72 / 0.53) with a sharper, more
decisive version of the same conclusion.** Under the validated, negative-sidelobe-robust
estimator, the gap between doing nothing and running the current model is larger, not
smaller: dirty stays at r=0.77-0.92, denoised drops to r=0.25-0.28.

**Do not read the denoised fit's mstar (0.578) as agreement with clean's (0.522).** Every
other parameter is essentially unrelated to the true geometry: inclination 69.8 deg against
33.4/27.5 for clean/dirty, position angle 67.3 deg against 178.8/175.2 (not an aliasing
artifact -- PA is reported mod 180, so this is a genuinely different orientation), and a
systemic velocity of 1.918 km/s against ~0.09-0.10 for the other two, which is unphysical for
the same disk. The optimiser found A local optimum for the denoised M1 field; it does not
resemble the disk's actual geometry in any of the four other parameters. The mass number is a
coincidence of that unrelated fit, not corroboration.

**Conclusion, now on the correct method for all three cubes:** the wiggle is real, large, and
recoverable from the raw undenoised dirty cube (r=0.92 residual correlation with truth). The
currently trained `winner_aug` checkpoint does not preserve it -- it produces a moment-1 field
whose best-fit disk geometry bears no resemblance to the true one. This is now the
authoritative comparison; the 2026-08-27 first-moment numbers should not be quoted going
forward.

Figure: `gi_wiggle_quadratic_clean_vs_dirty_vs_denoised.png` -- moment-1 and residual for all three, one colour scale per row.

## What the figure shows that the numbers don't

Clean and dirty's residual panels are visually near-identical: the same smooth two-lobe
pattern, same orientation, matching the 0.92 correlation. Denoised's residual is a
structurally different shape entirely, not just a noisier version of the same one.

Denoised's raw M1 panel (top right) is dominated by a large block of saturated colour where
the quadratic estimator has locked onto an extreme value across a big contiguous region --
visibly different from clean and dirty's smooth rotation gradient. That is consistent with
the model producing near-flat or garbage spectra there (the -86.5% M0 result from the same
day's OOD eval): a peak-finding estimator run on a spectrum with no real peak will lock onto
whichever channel has the most noise, and if that is systematically an edge channel across a
region, the result is exactly this kind of saturated block rather than scattered noise.
