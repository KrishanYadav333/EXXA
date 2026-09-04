# Progress log

Chronological record of runs, arrivals and bugs, newest first. Written at the time, per
RULES.md #11. `RUNS.md` maps a number to the run that produced it; this file records what
state the project is in and how it got here.

Entry format: **date | trigger | notebook** then what happened, the evidence, the
consequence. Triggers are `run`, `added` (a notebook downloaded into the repo), `bug`.

---

## 2026-09-04 | arrival + bug | September SG batch: unusable as training pairs, and it exposed
a degeneracy in the Keplerian fit

Jason sent more self-gravitating data (Drive, two zips, 3.7 GB) and said using SG data in
training would be a good idea. Extracted to
`self-gravitating cube and dirty cube/_v3_extract/`. Five new run folders plus the v2 pair
re-shipped, and MCFOST `.para` files alongside each, which turned out to matter more than the
cubes did.

**The pairs cannot train a denoiser as shipped.** Measured clean-vs-dirty difference at the
brightest channel, against the line-emission training set for scale:

| dataset | rmsdiff / rms_clean |
|---|---|
| line-emission training cubes | **0.41 - 0.57** |
| `run_sg_00019_rt_00019` | **0.0000** |
| `run_sg_74_00025_rt_00` | 0.0038 |
| `run_sg_15_00370_rt_rt00` | 0.0113 |
| `run_sg_32_00020_rt_00.` | 0.0265 |
| `run_sg_25_00370_rt_00` | 0.0708 |

The training set's dirty cubes differ from clean by ~50%; these differ by 0.4-7%. A network
trained on the latter learns the identity. `run_sg_00019`'s dirty is its clean: max absolute
difference 0.03125 against a 1e10 peak, a relative 3e-12, and that 1e10 Jy/beam scale is
itself wrong (v2's dirty peaks at 0.075). Nor are they a deconvolution task: clean and dirty
carry identical `BMAJ/BMIN` in every pair, so both sides already share a beam and there is no
operator between them. Phase 0 agrees (`NO_CONVOLUTION`), though the direct measurement above
is the evidence -- Phase 0 returned `A=nan`, which is a failed fit, not a verdict (RULES.md #8).

**What the `.para` files gave us is worth more: stated ground truth.** Stellar mass,
inclination, distance and grid size per run. Two configurations: 0.6 Msun / 30 deg / 140 pc,
and 1.0 Msun / 20 deg / 175.178 pc. The pixel scale they imply (600 AU over 301 px = 1.993
AU/px) matches what the code computes from the header, so the physical scale is confirmed
correct.

**First real validation of `fit_keplerian`, and it failed 3/5.** Until now the diagnostic had
exactly one check: 0.639 Msun recovered from the v2 cube against Hall+2020's 0.6, and that
0.6 came from a paper, not the data. Clean cubes, no noise, no beam, no model:

| run | true M | true incl | fitted M | fitted incl |
|---|---|---|---|---|
| `run_sg_00019` | 0.6 | 30 | 0.519 (-13%) | 31.8 |
| `run_sg_15` | 1.0 | 20 | **50.0 (bound)** | 3.3 |
| `run_sg_25` | 1.0 | 20 | 10.5 (+948%) | 6.8 |
| `run_sg_32` | 1.0 | 20 | 0.572 (-43%) | 22.6 |
| `run_sg_74` | 1.0 | 20 | **50.0 (bound)** | 2.2 |

**Cause: mass and inclination are nearly degenerate.** The line-of-sight velocity is
`sqrt(GM/r) sin(i) cos(theta)`, so the two enter as the single product `sqrt(M) sin(i)`. Only
the geometric deprojection, which stretches sky coordinates by `1/cos i`, separates them, and
that is a 6% effect at i = 20 deg against 15% at 30 deg. Hence the one disk at 30 deg
converging correctly (31.8 fitted against 30 true) while three at 20 deg collapsed toward
face-on and drove the mass to its bound to compensate.

Holding inclination at its stated value: -5.1% / +47% / +26% / -29% / still-pinned. Four of
five go from catastrophic to the right order of magnitude. The residual error is plausibly
physical rather than a bug, since these are *self-gravitating* disks and a point-mass
Keplerian model is the wrong model for one, but that is not demonstrated.

**Fixed in code.** `fit_keplerian` gains `m0=` (initial geometry from the emission's shape)
and `fix_incl_deg=` (hold inclination when it is known independently), and now returns
`mstar_at_bound` so a pinned fit cannot be quoted as a measurement. Case 8 in
`tests/test_gi_wiggle.py` covers it.

Also fixed a real but, as it turns out, *not causal* bug on the same line: the initial
geometry came from `disk_geometry_from_m0(np.ones_like(m1), mask)`, i.e. the shape of the
MASK rather than of the emission, which for a circular mask implies inclination ~0 and starts
the optimiser at the degenerate end of the valley. Passing the true M0 changed the results
barely at all (0.519 -> 0.519, 50 -> 50), so the degeneracy, not the initialisation, is what
breaks these fits. Worth fixing regardless.

**Which published numbers this touches.** The method comparison is safe: `compare_wiggles`
fits ONE geometry on the clean cube and subtracts that same model from every method, so a
mis-estimated absolute mass shifts all rows together and the ranking (beam preserves the
wiggle, U-Net degrades it, DDRM degrades it most) is unaffected. What weakens is the
standalone claim that recovering 0.639 Msun corroborates the cube's provenance: we now know
the fit is only trustworthy at favourable inclination, and even at 30 deg it is 5-13% off.
The v2 cube fitted 27-33 deg, in the favourable regime, so that number is not withdrawn --
but it should be quoted with the inclination caveat attached.

**`run_sg_74` fails even with inclination fixed** and has by far the smallest mask (2.2% of
the field, 2034 px) and the largest residual RMS. Separate problem, not yet diagnosed.

## 2026-08-29 | run + added | notebook 09 on Kaggle GPU: third independent confirmation

`09-wiggle-scoring.ipynb` run on GPU, first downloaded manually with outputs (exec counts
6-10, pulled commit `9f59221`), before its interactive session was ever committed on Kaggle,
so that run has no version number to attribute it to. Kaggle's GitHub integration then
auto-pushed a second, fresh-kernel run as `ce1b6ae` ("Kaggle Notebook | 09-wiggle-scoring |
Version 2", exec counts 1-5, pulled commit `7d73e2e`), which is the number this project uses
per RULES.md's own stated method: reconstruct the version from the push commit, not from
recollection. Both runs agree to the fourth decimal. Dataset resolved correctly at
`/kaggle/input/datasets/krishanyadav333/kaggle-wiggle-scoring-dataset/`.

**240-360 step 1 (121 channels), 7.2 min on GPU** (vs 140.9 min on CPU for the same config,
~19x): mstar=0.537, resid r: dirty 0.8907, beam-only 0.9198, U-Net 0.8040, DDRM 0.5833. Step 4
(31 channels), 1.8 min: same bunching-near-1.0 sampling artifact as every prior run of this
config, not a real finding.

**Third independent reproduction of the retraction's corrected table**, now across two
machines and both CPU and GPU: 0.891/0.920/0.804/0.583 (original), 0.892/0.920/0.805/0.587
(local CPU rerun), 0.891/0.920/0.804/0.583 (this run, rounded). No longer provisional in any
sense.

**Archived** at `results/09-wiggle-scoring/v2_2026-08-29_ce1b6ae/` and added to `RUNS.md`,
explicitly disambiguated from the unrelated pre-existing "09 -- Architecture comparison"
entry (same number, different notebook). Root `09-wiggle-scoring.ipynb` committed with its
outputs intact, matching how 07/08 already carry their own Kaggle-pushed outputs at root.

## 2026-08-28 | run | wiggle_all_methods.py confirms the retraction's corrected table

`compare_wiggles()` port of `experiments/wiggle_all_methods.py` (bug fix two entries below)
finished its first real run, both channel-sampling configs, 173 min total on CPU (the run it
replaces crashed at 111 min on a wrong key name, `vsys_kms` instead of `vsys`, fixed same
session).

**240-360 step 1 (121 channels), independent re-fit of the shared geometry (mstar 0.535 vs the
retraction entry's 0.538, same ballpark, not the same optimiser run):**

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean | 0.182 | -- | -- |
| dirty | 0.171 | 0.9928 | 0.8916 |
| beam-only | 0.169 | 0.9947 | 0.9204 |
| U-Net | 0.169 | 0.9874 | 0.8051 |
| DDRM | 0.303 | 0.9527 | 0.5869 |

Matches the RETRACTION entry's corrected table (0.920 / 0.891 / 0.804 / 0.583) to within
fit noise. **This is now a reproduced result, not a one-off.** Ordering holds: beam-only and
dirty close together, U-Net worse, DDRM worst by a wide margin.

**240-360 step 4 (31 channels)**: all five methods bunch at 0.987-0.999, which is the
sampling artifact the script's own docstring warns about, not a real finding -- coarse
sampling degrades `quadratic_moment1`'s parabola fit for every cube alike, and that shared
artifact correlates spuriously regardless of what happened upstream. Do not read "the
methods are indistinguishable" out of this config; step 1 is the one that resolves them.

Figure regenerated at `results/self-gravitating/wiggle_all_methods.png`, overwriting the
stale pre-fix version. `wiggle_all_methods_step1.npz` regenerated alongside it.

Kaggle GPU version (`09-wiggle-scoring.ipynb`) still worth finishing and running once the
kinematic-loss checkpoints exist, since a 173-minute CPU turnaround does not scale to
re-scoring four more gamma arms.

## 2026-08-28 | code + added | KinematicLoss and notebook 08, aimed at the corrected failure

Direct response to the corrected result two entries below: U-Net degrades the wiggle residual
(0.804 vs dirty's 0.891), DDRM degrades it worse (0.583). Built rather than deferred, at the
user's push -- the "needs 2 days" estimate did not hold up; the objective and its test took
about 90 minutes.

`src/utils/losses.py`: `spectral_moment1()`, a differentiable moment-1 (biased intensity-
weighted mean, same estimator as `collapse_first`; the bias is shared between prediction and
target so it cancels in the loss, unlike in the standalone diagnostic where `collapse_first`
was replaced for exactly this reason). `KinematicLoss(alpha, beta, gamma)` combines it with
MSE and SSIM. 15 checks in `tests/test_kinematic_loss.py`.

`08-kinematic-loss.ipynb` (new, 15 cells): trains `winner_aug`-configuration U-Net with
`out_channels=31` (k=15 stack, matching the line's measured ~37-channel FWHM) and
`kinematic_gamma` swept over [0.0, 0.1, 1.0, 10.0]. gamma=0 is a fresh control at the same
31-channel architecture, not notebook 05's single-channel `winner_aug` -- comparing against
05 directly would confound the loss change with the output-channel change.

**Not extending notebook 05**: its `denoise_cube`/`val_metrics`/moment-table path assumes
single-channel output throughout, and the wiggle score needs the self-gravitating cube, which
05 never loads. Scoring stays local (`experiments/wiggle_all_methods.py`).

Verified by extracting and executing the notebook's real cell sources against local data at
reduced scale (k=3, 64px, 1 epoch): data shapes match, velocity axis builds correctly from
`CDELT3`, both gamma=0 and gamma=1 train and checkpoint. Not real numbers, plumbing only.

**Not yet run on Kaggle GPU.** Success criterion: wiggle residual correlation above the
U-Net's 0.804 without losing the M0/PSNR gains `winner_aug` already has.

---

## 2026-08-17 | run + added | 05 Kaggle Version 24 — beam arm re-scored, the bug was real

First run with `denoise_cube()` passing the beam vector. No training.

```
winner_beam            M0        M1        M2
before (no vector)   -95.7%    +14.1%    -27.3%
after  (with it)      +9.6%    +63.2%    +20.9%
```

**M0 moves 105 points and M2 changes sign.** Four of five cubes are strongly positive on M0;
`run_0025_01000_rt_04` at −133.8% carries the whole deficit alone.

**Consequence:** beam conditioning is **not refuted**. It is positive on all three moments
and simply below the winner at one seed. The earlier reading, second-best PSNR paired with
the worst M0 in the table, was an artifact. That reading is in the submitted midterm blog,
which stays as written; corrected in RUNS.md and in the v24 README.

`winner_patch` PSNR 33.96 -> 34.98 on the common full-image set. Moments unchanged.

Kaggle auto-pushed again mid-commit. Checked before merging, per rule 2: no fix reverted,
all five probes intact. Notebook archived with outputs, per rule 10.

## 2026-08-17 | bug | 05 — beam arm scored without its beam vector

`denoise_cube()` never passed a beam vector, and `UNet.forward` ignores `beam=None`
silently (it asserts only in the opposite direction), so `winner_beam` was scored on moments
with its conditioning branch dead. Section 4 scored the same checkpoint *with* the beam and
got 38.71 dB, second best in the table; section 6 scored it *without* and got M0 −95.7%,
worst in the table.

**Caught by** reading `denoise_cube`'s call signature against the arm's `beam_dim=4`, during
a review, not by any test or assertion.

**Touches** RUNS.md's beam row (M0 −95.7 / M1 +14.1 / M2 −27.3, now retracted) and the
published blog, whose closing cites "second-best PSNR and worst M0 by a factor of two" as
evidence for the pixel-vs-science thesis. The blog is submitted and stays as is.

**Fixed** in the notebook: the vector now comes from the dirty header via `beam_features_of`,
a missing header raises instead of zeroing the conditioning, and stale beam rows are dropped
on resume so the fix cannot be masked by resumed values. **The arm still needs re-scoring.**
On a toy beam model the vector shifts the output by 0.025 mean absolute, so the two paths
were scoring different functions.

Same review found `winner_patch`'s PSNR measured on 64px crops while every other arm used
256px full images, putting 33.96 dB in a column of 37 to 39. Also fixed; PSNR now uses a
common full-image set.

## 2026-08-17 | run + added | 05 Kaggle Version 22 — artifact diagnostics complete

Sections 7 and 8 finished for the first time. No training, no new moment scores; the moment
table is identical to v20's. 300 validation channels analysed, and the per-channel CSV
finally saved.

Three readings, in the run's README: invented structure is a **threshold not a gradient**
(~25 channels at ~100% invented area, everything else near 0, all below SNR ~0.5); the
negative floor leak is **systematic** (`denoised.min` clusters near −0.10 and −0.155, no
channel near zero); and the worst channels are where the **metric stops meaning anything**
(SNR 0.0 scored against 20% of a clean peak that is itself a noise spike).

Kaggle auto-pushed during the commit and the push was rejected. Checked before merging: it
had **not** reverted any code fix, only re-added stored outputs. Rebased, kept the stripped
notebook. That push is also what revealed the run was Kaggle Version 22, not the v21 it had
been filed as; folder renamed.

## 2026-08-16 | bug | 05 — section 7 CSV write discarded the whole diagnostic

`channel_artifacts()` returns `n_background_px`; the hand-kept `fieldnames` list did not
include it, and `DictWriter` raises only at write time. All 300 channels were analysed and
printed, then thrown away at the last line.

**Fixed** by deriving columns from the data. `n_background_px` is the field that should never
have been dropped: it is the denominator RULES.md #8 exists to make you check.

## 2026-08-14 | run + added | 05 v20 — first U-Net scores on the DDPM's metric

Zero arms trained. 12 checkpoints restored from 08, 3 from v19's Output, `winner_patch`
scored from stored weights, 15 rows over 6 arms. The whole session went to section 6.

**Why it mattered:** every U-Net number before it was on the raw or clip-only metric while
the DDPM was on mask + clip, so no cross-family comparison had been like for like. Best
U-Net arm beats the DDPM by 85 percentage points on M0 with a spread twenty times tighter.

Two negative results, both kept: beam conditioning worst on M0 (later retracted, see above),
64px patches at M0 −40.8%.

**Notebook not archived.** Lost. Kaggle did not auto-push this version and the local copy was
stripped before the gap was noticed. Part of why RULES.md #10 exists.

## 2026-08-27 | finding | robust estimator: wiggle recoverable raw, model destroys it more decisively than first thought

Two follow-ons to the same day's GI wiggle work, both from asking whether `collapse_first`
(intensity-weighted mean) was the wrong tool for a cube that is 51.55% negative pixels.

**Part 1: does the wiggle survive in the raw dirty cube, or did `collapse_first` just fail to
find it?** Swapped to `collapse_quadratic` (Teague & Foreman-Mackey 2018) --
`quadratic_moment1` in `src/evaluation/gi_wiggle.py` -- which fits a parabola to the peak
channel rather than averaging the whole spectrum, so a negative sidelobe elsewhere cannot pull
it off the true line centre. Validated on synthetic data with an injected negative sidelobe
before trusting it on anything real: naive weighted mean pulled 1341 m/s off the true centre,
quadratic estimator off by 21.5 m/s. 17 checks in `tests/test_gi_wiggle.py`.

On the real dirty cube the fit now converges (mstar 0.580 Msun against clean's 0.522,
inclination 27.5 against 33.4 degrees) where it previously ran to its 50 Msun bound. **The
wiggle is genuinely recoverable from the raw, undenoised dirty cube.** Residual correlation
with clean: 0.92.

**Part 2: does "denoising makes it worse" survive the same correct method?** Reran inference
(858s) to get the denoised cube's raw channels, since only its first-moment maps had been
saved. Full three-way comparison, quadratic throughout:

| | mstar (Msun) | incl (deg) | PA (deg) | vsys (km/s) | raw M1 r | residual r |
|---|---|---|---|---|---|---|
| clean | 0.522 | 33.4 | 178.8 | 0.100 | -- | -- |
| dirty | 0.580 | 27.5 | 175.2 | 0.083 | 0.773 | 0.921 |
| denoised | 0.578 | 69.8 | 67.3 | 1.918 | 0.250 | 0.281 |

**This supersedes the earlier first-moment comparison (0.72/0.53) with a sharper version of
the same conclusion, not a different one.** The gap between doing nothing and running the
model is larger under the correct method: dirty holds at r=0.77-0.92, denoised falls to
r=0.25-0.28. Denoised's mstar (0.578) looks close to clean's by coincidence; every other fit
parameter is essentially unrelated to the true geometry (inclination off by ~40 degrees,
position angle a genuinely different orientation not an aliasing artifact, systemic velocity
off by nearly 2 km/s). The optimiser found a local optimum with no physical resemblance to the
real disk.

**The 2026-08-27 first-moment numbers (0.72/0.53) should not be quoted going forward; this
comparison is now the authoritative one.**

## 2026-08-27 | finding | denoising makes the disk's kinematics WORSE than doing nothing

Extended the GI wiggle check to `dirty_cube.fits` and the `winner_aug` seed-43 denoised
output already computed in the OOD eval, reusing its saved moment maps.

**Clean, unconfounded result, no Keplerian fit needed: correlation of raw moment-1 with the
true clean M1 is 0.72 for the untouched dirty cube and 0.53 for the denoised one.** The
model degrades recoverable kinematics below doing nothing. Matches the OOD moment-improvement
result from earlier the same day (M0 -86.5%, M2 -168.3%) with an independent metric.

**Two `fit_keplerian` bugs found and fixed along the way**, both from using the same crude,
data-independent initial guess for every cube:

- Dirty's M1 has NaN pixels (moment-1's own 0/0, since `mask` is defined from clean's M0, not
  the dirty cube's own signal quality). `fit_keplerian` used to propagate that straight into
  the fit bounds and crash with "x0 is infeasible". Now drops non-finite samples before
  fitting and reports how many. 3 new regression cases (15 total in
  `tests/test_gi_wiggle.py`).
- Even with NaNs handled, dirty's fit still ran mass to its 50 Msun bound with a
  near-degenerate inclination, from the same fixed init used for every cube regardless of
  data quality. Re-fit using CLEAN's own converged geometry as the init for dirty/denoised
  (physically correct: same disk, so the true geometry is shared) -- and dirty STILL does not
  converge. That is now a real finding rather than an optimiser artifact: a naive per-pixel
  Keplerian fit to the raw dirty cube's moment-1 does not work. Consistent with why the
  literature (Hall+2021 in particular) works from individual channel maps rather than the
  collapsed M1, which is more noise-sensitive.

**A caveat worth keeping:** denoised's fit converges, to a disk at about a fifth of the true
mass with almost no real rotation (0.034 km/s median deviation vs 0.635 km/s truth). Its
residual correlates with clean's residual at 0.82, which looks like "the wiggle survived" --
it should not be read that way, since a near-flat fitted model leaves the residual close to
the raw M1 map itself. The raw-M1 correlation (0.53) is the number not confounded this way,
and it already answers the question in the opposite direction.

**Settled:** `winner_aug` should not be used to prepare a cube for GI wiggle analysis; it
removes kinematic information. **Not settled:** whether a better (channel-map-based) fit
could recover something from the raw dirty cube where the moment-1 fit failed.

Write-up: `results/self-gravitating/gi_wiggle_dirty_vs_denoised.md`, figure
`gi_wiggle_clean_vs_dirty_vs_denoised.png`.

## 2026-08-27 | finding | GI wiggle diagnostic built and run: a real, large, coherent residual, provenance corroborated

Jason redirected the priority: work properly on the self-gravitating data, and pointed at
five papers (Speedie+2024 Nature, Terry+2024 A&A, Hall+2021 MNRAS, Hall+2022 ApJL,
Hall+2020 ApJ). All five are built around one diagnostic, the "GI wiggle": fit the disk's own
Keplerian rotation, subtract it from the moment-1 map, and look at the residual.
`DISTPC=140 pc` in `dirty_cube.fits` matches Hall+2020's founding simulation exactly.

New module `src/evaluation/gi_wiggle.py`: geometry init from M0's image moments, a
least-squares Keplerian fit (centre, PA, inclination, vsys, stellar mass) against the moment-1
field, and the residual. Validated on synthetic data before running on real cubes: a pure
Keplerian case recovers every parameter exactly with zero residual; a known injected m=2
perturbation is recovered at r=0.85 correlation, with the caveat that amplitude is inflated
because the geometric fit itself absorbs some of the injected signal.

**Fitted on the clean cube (`lines.fits`, non-padded range [30,571)):** stellar mass
**0.639 Msun**, with no mass information anywhere in either header -- recovered purely from
the velocity field. That is close enough to Hall+2020's 0.6 Msun founding simulation that it
is independent corroboration of the DISTPC match: this cube is very likely built on that exact
pipeline.

**The residual: RMS 1.34 km/s against 0.64 km/s of typical local rotation** -- larger than the
rotation itself -- and it does not decay with radius (checked to r=100 px against a median
disk radius of 130 px, ruling out a 1/sqrt(r) fit artifact near the centre). One annulus
(r 120-140 px) shows a clean single-period (m=1) sinusoid in azimuth with essentially no
scatter. Large, coherent, global: consistent with what the papers describe.

**Not yet claiming a confirmed GI wiggle detection.** A pure radius-independent m=1 residual
is also what a slightly imperfect flat thin-disk geometric model produces on its own, and the
field is actively debating exactly this degeneracy (arXiv:2510.05601 argues infall can mimic
a wiggle in AB Aurigae, the same object Speedie+2024 reads as gravitational instability). What
is established: a validated pipeline, a large real residual, and a stellar-mass recovery that
independently corroborates the cube's provenance.

Not yet done: run the fit on `dirty_cube.fits` and on a denoised cube (does the wiggle survive
noise and the currently-failing U-Net); check the residual's radial structure against the
literature's "interlocking fingers" description, not just its azimuthal one; a proper
Fourier-decomposed amplitude if a q estimate is wanted, rather than the RMS/max proxy built
here.

Write-up: `results/self-gravitating/gi_wiggle_check.md`, figure `gi_wiggle_clean.png`.

## 2026-08-27 | run | first OOD test: the trained U-Net fails on a real dirty beam

`experiments/eval_self_gravitating.py`, `winner_aug` seed 43, no retraining. The out-of-
distribution test proposed 2026-08-21 and only now run: the model had never actually been
scored against the self-gravitating pair before this.

Result over the non-padded range (541 channels, see below): **M0 -86.5%, M1 -10.8%,
M2 -168.3%**, signal-masked. Every moment is negative -- the denoised cube is FURTHER from
truth than the raw dirty cube, not just less improved. Figure at
`results/self-gravitating/ood_moment_comparison.png`, write-up at
`results/self-gravitating/ood_eval.md`.

**Why this is expected and still worth having measured.** Every training cube has `A = I`
(Phase 0), so the model was trained to remove additive noise with no mechanism for inverting
a beam. Applying it to a cube with a REAL dirty beam (recovered 2026-08-25, peak 0.911,
-2.8% sidelobes) does not degrade gracefully, it actively damages the signal. That is now
measured rather than assumed, and it is the first concrete evidence of the domain gap between
this project's training regime and what real ALMA data will look like.

**Also found in building the eval:** channels 0-29 and 571-600 of `lines.fits` are
byte-identical repeats within each block, confirmed with `np.array_equal`, not a real
line-free baseline. That breaks both the training pipeline's continuum-subtraction assumption
and `bettermoments.estimate_RMS`, which reads `data[:N]`/`data[-N:]` literally. The eval trims
to the non-padded [30, 571) range and skips continuum subtraction rather than apply it to
padding.

## 2026-08-28 | code | velocity-aware objective, built to target the measured failure

The corrected comparison shows every model degrades the GI wiggle (beam-only 0.920, dirty
0.891, U-Net 0.804, DDRM 0.583). The U-Net's damage has an identifiable cause: it optimises
MSE + SSIM, pixel accuracy, while the wiggle is a sub-channel velocity perturbation that
per-channel smoothing shifts. Nothing in the objective ever asked it to preserve velocity
structure. This adds that.

`KinematicLoss` in `src/utils/losses.py`:

    L = alpha*MSE + beta*(1-SSIM) + gamma*|M1(pred) - M1(target)|

with `spectral_moment1` a differentiable intensity-weighted mean velocity over the channel
axis. It is not an unbiased M1 and does not need to be: it is computed identically on
prediction and target, so a finite-window bias cancels in the difference. What it must be is
sensitive to a line-centre shift, which it is.

Measured properties (`tests/test_kinematic_loss.py`, 15 checks): M1 tracks a shifted line,
is invariant to amplitude scaling (it measures velocity, not flux), is finite on empty
spectra, and the loss separates a velocity shift (0.0855) from a pure amplitude change
(1.1e-08) by seven orders of magnitude. `gamma=0` contributes nothing to the total while
still reporting the velocity term for monitoring.

Three pieces of plumbing this needed, all in code the existing arms share:
- `FITSChannelDataset(stack_target=True)` returns all 2k+1 CLEAN channels rather than the
  centre alone; a moment-1 penalty needs a line profile, not one slice. Default unchanged.
- `_build_unet(out_channels=N)` and `train_unet(out_channels=N)`: the model has to predict a
  stack for its M1 to be computable.
- `train_unet(kinematic_gamma=..., velax_kms=...)`, recorded in the checkpoint.

**Window size matters and is measured, not guessed:** the line FWHM is ~37 channels at a
typical bright spaxel, so a useful stack is ~31 channels (`n_neighbors=15`), not the 3 the
spectral-context arms used.

**Not yet trained.** Needs a Kaggle run: `n_neighbors=15, stack_target=True, out_channels=31,
kinematic_gamma` swept. The honest test is whether it improves the wiggle correlation above
the U-Net's 0.804 WITHOUT giving up the M0 and PSNR gains, since a model that only preserves
velocity by refusing to denoise would be useless.

## 2026-08-28 | RETRACTION | the wiggle comparisons were methodologically broken; the beam never erased the wiggle

Running the U-Net alongside DDRM at matched configuration exposed the flaw behind a chain of
wrong conclusions: **each cube was fitted its OWN Keplerian model before its residual was
compared to the truth's.** The residual then means something different per cube, so the
comparison measures fit disagreement, not the wiggle.

Beam-convolved clean data vs truth, across five channel ranges at step 1:

| range | own-fit r | shared-model r |
|---|---|---|
| 240-360 | 0.117 | 0.920 |
| 230-370 | 0.998 | 0.999 |
| 250-350 | 0.153 | 0.998 |
| 200-400 | 0.949 | 0.998 |
| 60-540 | 0.116 | 0.997 |

The own-fit column is noise. That instability was visible all along and went unchecked.

**WITHDRAWN: "the beam alone erases the GI wiggle" (2026-08-27).** Under a shared model the
beam-convolved cube correlates at **0.92-0.999** with truth and its residual RMS (0.168)
matches the truth's (0.182). The beam preserves the wiggle. **This finding motivated the whole
DDRM effort.**

**WITHDRAWN: all three DDRM verdicts issued on 2026-08-28** ("no recovery", "destroys signal",
and an unpublished "recovers"). All used per-method fits; two also compared across different
channel samplings.

**Corrected result** (240-360 step 1, one shared model from clean):

| method | resid r | resid RMS |
|---|---|---|
| clean (ref) | -- | 0.182 |
| beam-only | 0.920 | 0.168 |
| dirty | 0.891 | 0.170 |
| U-Net | 0.804 | 0.169 |
| DDRM | 0.583 | 0.303 |

The beam preserves the wiggle; noise costs a little; the U-Net degrades it; DDRM degrades it
most and inflates the residual amplitude 1.7x. The ordering "do nothing > U-Net > DDRM" holds,
so the qualitative claim that models hurt the kinematics survives. What does not survive is
the beam being the culprit, or DDRM ever having recovered anything.

**Fixed in code:** `compare_wiggles()` fits the reference once and subtracts that model from
everyone; regression case 7 covers it. Also fixed `fit_keplerian` raising "x0 is infeasible"
when its default initial guess fell outside its own bounds -- unreachable when an explicit
`init` is passed, which is why every prior test missed it.

**Lesson:** three conclusions were published in one day from a statistic whose instability was
measurable from the start. Validate a comparison metric against its own free parameters before
reading any result from it.

Write-up: `results/self-gravitating/RETRACTION_wiggle_methodology.md`.

## 2026-08-28 | correction | the DDRM comparison was invalid; DDRM DESTROYS signal, not merely fails to recover it

Published a DDRM result an hour ago comparing 0.1277 against a 0.116 floor and calling it "no
recovery". **The comparison was invalid and the conclusion understated.**

DDRM was scored on 31 channels at step 4; the 0.116 floor came from 481 channels at step 1.
Different configurations. Measured at DDRM's OWN configuration the beam-only floor is
**0.938**, so DDRM at 0.128 is far BELOW it: it destroys structure that survives the beam
untouched.

**Root cause: the wiggle residual is acutely sensitive to velocity sampling.**

| step | dv (km/s) | clean resid RMS | dirty resid RMS | resid r |
|---|---|---|---|---|
| 1 | 0.033 | 0.182 | 1.341 | 0.115 |
| 2 | 0.067 | 0.185 | 1.355 | 0.117 |
| 4 | 0.133 | **1.436** | 1.418 | **0.997** |
| 8 | 0.267 | 1.741 | 1.689 | 0.997 |

At step 4 the CLEAN cube's residual jumps 8x. That is parabola-fit error in
`quadratic_moment1` (it fits the peak channel and two neighbours), not signal, and being a
deterministic sampling artifact it is SHARED between clean and dirty -- hence the spurious
0.997.

**This also retracts a correction I made earlier today.** I attributed the 2026-08-27 value of
0.111 to "a 10-channel window collapsing the fit". Wrong: that value came from 481 channels at
step 1 and is sound for its configuration. The discrepancy was the sampling step, not the
window length.

**What stands:** the prior trained properly; DDRM's fitted geometry is physically incoherent
(8.2 Msun against 0.56, inclination 8 against 32 degrees); the beam passes 1.3% of modes above
1% gain; DDRM is not usable for this problem on this data. The qualitative conclusion is
unchanged and in fact stronger.

**Methodological rule going forward:** a GI wiggle correlation is only comparable to another at
the SAME channel range AND step. Three separate errors today came from comparing across
configurations. Record the configuration with every value.

## 2026-08-28 | finding | DDRM does not recover the wiggle: 0.128 against a 0.116 floor

Prior trained on Kaggle (60 epochs, unconditional, v-prediction, loss 8106.7 -> 16.5).
Restoration and scoring rerun locally on 31 channels (240-360), the range that actually covers
the line-centre variation.

| | mstar (Msun) | incl (deg) | raw M1 r | residual r |
|---|---|---|---|---|
| clean | 0.564 | 31.6 | -- | -- |
| dirty | 0.618 | 29.4 | 0.9939 | **0.9970** |
| DDRM | 8.205 | 8.0 | 0.9572 | **0.1277** |

**Beam-only floor 0.116, DDRM 0.1277. No recovery.** This is outcome 2 of the two written into
`ddrm_feasibility.md` BEFORE the prior was trained: the prior hallucinates plausible structure
that is not the truth.

Three things make this a real negative rather than a failed run: the prior trained properly
and was still improving at epoch 60; the signal WAS recoverable, since with a proper channel
window the dirty cube's own residual correlates at 0.997 with truth; and the failure is
physically diagnostic, with DDRM's fitted geometry badly wrong (8.2 Msun against 0.56,
inclination 8 deg against 32) while its raw M1 correlation stays high at 0.957. It produced a
disk-shaped velocity field whose implied physics is incoherent.

**Also corrects an earlier number.** The 2026-08-27 entries recorded the dirty cube's residual
correlation as 0.111 and read it as "the wiggle is not clearly recoverable from the dirty cube
in the first place". That was the 10-channel window collapsing the Keplerian fit. With the
correct window it is **0.997**: the wiggle IS strongly recoverable from the dirty cube. The
beam-only ablation (0.116) still stands, since it used the full 481-channel range.

Likely explanation, known in advance: the beam passes only 1.3% of Fourier modes above 1% of
peak gain, so DDRM had to invent ~99% of the spectrum. Measurement consistency is a weak
constraint when the instrument measured almost nothing.

Write-up `results/self-gravitating/ddrm_result.md`, figure `ddrm_restoration.png`.

## 2026-08-28 | run + bug | DDRM prior trained; the scoring cell reported a degenerate fit as "RECOVERY"

**Prior trained successfully.** 60 epochs, 115 min on T4 x2, 16.9M params unconditional, loss
8106 -> 17.7 train / 20.5 val, still improving at the end. Checkpoint persisted (271 MB).
That part of notebook 07 works.

**The scoring is wrong and its output must not be quoted.** It printed:

```
          mstar    incl    raw r   resid r
dirty    49.999    1.3     0.9766   0.7871
DDRM     50.000    1.5     0.8632   0.4578
  DDRM: 0.4578 -> RECOVERY
```

`mstar = 50.000` is `fit_keplerian`'s upper BOUND and `incl ~ 1.3 deg` is face-on degenerate:
both fits failed, so the correlations between them mean nothing. The giveaway is `dirty`
scoring 0.787 here against 0.111 in the validated 2026-08-27 run on the same cube.

**Cause: too few channels.** The notebook restored 10 channels (280-316). Measured, the line
CENTRE varies from channel 260 to 348 across the disk -- that variation IS the rotation.
`quadratic_moment1` finds each spectrum's peak channel, so with a narrower window most
spaxels' peaks land on the window edge, the velocity field flattens, and the fit runs to its
bound. Confirmed directly:

| channels | mstar | incl |
|---|---|---|
| 10 (280-316), as run | **50.000 (bound)** | 1.1 |
| 61 (240-360, step 2) | 0.529 | 33.1 |
| 481 (60-541), validated | 0.572 | 32.0 |

Fixed: `CHANNELS` now spans 240-360, and the scoring cell **raises** if any fit hits 90% of
the mstar bound rather than printing a verdict from it. A degenerate fit is a failure, not a
measurement, and it should never again be reportable as a result.

Also fixed `collect_outputs()` missing its required `patterns` argument, which errored in the
final cell.

**The DDRM question remains unanswered.** The restoration itself ran (10 channels, finite
output); only the scoring was invalid. Rerunning sections 5-7 with the corrected channel range
will answer it -- the prior does not need retraining.

## 2026-08-28 | bug | notebook 07's bootstrap pointed DATA_DIR at the wrong Dataset

First Kaggle run of notebook 07 failed immediately:
`FileNotFoundError: No valid cube pairs found under .../self-gravitating-v2`.

My bug. The bootstrap globbed `/kaggle/input/**/*dirty*.fits`, which matches
**`dirty_sg.fits`** from the self-gravitating Dataset, so `DATA_DIR` resolved to that Dataset
instead of the line-emission cubes. Notebook 06 used the stricter `*_dirty.fits` pattern,
which would not have matched; I dropped it when writing 07.

Fixed by locating the data through the `run_<id>_<step>_rt_<pp>` folder structure
`split_cubes` actually requires, and picking the directory containing the most such folders.
Verified locally: selects `Line Emission Data` with its 14 run folders, excludes the
self-gravitating Dataset.

**Then executed the notebook's real cell sources end to end** at reduced scale (64px, 1
epoch, 3 channels) against local cubes, rather than the separate dry-run script used before.
All stages run: data (56 train / 16 val), unconditional training (checkpoint written), DDRM
restoration (finite, correct shape), and scoring. The notebook's own relative paths
(`../results/...`, `../results/checkpoints/...`) resolve correctly from `notebooks/`, which is
where the Kaggle bootstrap `chdir`s.

Also hoisted the transfer-function build out of the per-channel loop (identical every
iteration) and dropped an unused `dirty_rs` list.

**Do not read anything into the dry run's "RECOVERY" line.** With a 1-epoch prior on 56
images at 64px, both fits are degenerate (mstar 0.001, inclination ~88 degrees) and the
numbers are noise. The plumbing is what was being checked.

## 2026-08-27 | code | DDRM dry run: caught an oversized-beam bug before it reached Kaggle

`experiments/ddrm_dryrun.py` runs notebook 07's whole pipeline at 64px on real cubes (2 cubes,
3 epochs) purely to catch plumbing failures. It immediately earned its keep: stage 3 crashed
with a broadcast error because `beam_transfer_function` assumed the beam is smaller than the
target grid, and the recovered beam is 129px against a 64px grid.

Worth noting the notebook would NOT have crashed at its 256px setting -- the 129px beam fits
there -- but the same code path would have mis-placed the beam silently, which is the worse
outcome. Fixed by cropping an oversized beam symmetrically about its centre. Verified the peak
gain is identical (~355) at 64/128/256/600px, and regression case 4b checks a centred delta
gives flat unit transfer at every grid size.

After the fix all four stages run on real data: data loading, unconditional training (loss
972.6 -> 920.9 in 10s), checkpointing (70 MB), DDRM sampling (finite output), and GI wiggle
scoring. The restoration numbers at this scale are meaningless by construction and are not
recorded as results.

One number IS worth carrying forward: the transfer function on the real beam passes **1.3% of
modes above 1% gain**, matching the independent measurement in `ddrm_feasibility.md`. That is
the same hard constraint from two different code paths.

**Notebook 07 is now safe to run on Kaggle.**

## 2026-08-27 | code | DDRM notebook (07), plus a checkpointing bug that broke every config

Notebook `07-ddrm-restoration.ipynb`: trains an unconditional diffusion prior over ~2800 clean
Jy/beam channel maps, then restores `dirty_sg.fits` with DDRM using the recovered beam, scored
on the GI wiggle residual against the beam-only floor of 0.116.

**Found a real bug in `DotDict.__getattr__` that broke checkpointing for the WHOLE project,
not just DDRM.** It returned `None` for any missing key, dunders included. `pickle` probes for
`__reduce_ex__`/`__getstate__`, got `None`, and tried to call it -- surfacing as
`TypeError: 'NoneType' object is not callable` from inside `torch.save`, with nothing in the
traceback pointing at `DotDict`. Any `save_checkpoint` call would hit this. Fixed by raising
`AttributeError` for dunder lookups while keeping the `None`-for-missing-key behaviour the
codebase relies on.

Three smaller fixes to make the unconditional path work at all, all in code the conditional
DDPM shares:
- `noise_estimation_loss` hardcoded the conditional concat; now branches on channel count.
- `_epoch_loss` unpacked `for x, _ in loader` and took `x[:, 1:]` for the clean channel, which
  is EMPTY for single-channel input. Both handled.
- Notebook API corrected against the real signatures: `n_epochs` not `epochs`,
  `train_losses`/`val_losses` not `train_loss`/`val_loss`.

Verified end to end on CPU: unconditional training runs, loss decreases (214.7 -> 191.6),
checkpoint writes, and `ddrm_steps` produces finite output. All five existing DDPM/architecture
tests still pass. `tests/test_ddrm.py` now has 5 cases including the pickling regression.

**Not yet run on Kaggle.** The prior needs GPU training.

## 2026-08-27 | finding | ablation: the BEAM alone erases the wiggle, not noise, not the denoiser

Tested the hypothesis left open by the v2-cube run. Convolved the CLEAN cube with the
recovered beam, added no noise, ran no model, then ran the identical Keplerian-fit pipeline.

| | residual RMS (km/s) | residual r vs clean |
|---|---|---|
| clean (reference) | 1.394 | -- |
| **beam-only, no noise, no model** | **0.174** | **0.116** |
| real dirty | 0.176 | 0.111 |
| real denoised | 0.170 | 0.108 |

**The beam alone reproduces the entire effect** -- 0.116 against the real dirty cube's 0.111
and denoised's 0.108, statistically indistinguishable. Raw M1 correlation stays at 0.988,
matching the real cubes' 0.98-0.99, so bulk rotation survives smoothing just as it does in
the real data.

**This settles the v2 cube's open question.** The wiggle is destroyed by the instrument
response before noise or denoising exist. The earlier result was never a model failure on
this cube -- there was nothing left to preserve. A denoiser cannot fix this in principle:
removing noise cannot restore what a convolution erased.

**It also sharpens the DDRM/VIREO case from "worth trying" to "the indicated approach".** A
measurement-consistency prior reconstructs structure the instrument did not measure, which is
categorically different from denoising, and the forward operator it needs is already
recovered and saved (`dirty_beam_recovered_v2.fits`).

Note this does NOT retract the original (wrong-script) cube's finding, where denoised raw-M1
correlation fell to 0.25 against dirty's 0.77 -- that was a genuine degradation on different
data. Here both sit at 0.98.

Caveat: uses the recovered beam, whose held-out validation on this cube was 0.80 rather than
the original pair's 0.994. The agreement with the real dirty cube is close enough that it is
clearly capturing the dominant effect, but a more accurate beam could smooth slightly
differently.

Figure: `results/self-gravitating/gi_wiggle_beam_only_ablation.png`. Write-up:
`results/self-gravitating/beam_only_ablation.md`.

## 2026-08-27 | run | corrected cube tested: a different failure mode, not a repeat

`winner_aug` seed 43 on `clean_sg.fits`/`dirty_sg.fits`, trimmed [60,541) (this cube's
padding is wider: 0-59 and 541-600, checked directly). One denoising pass fed both the
standard moment-improvement metric and the GI wiggle Keplerian fit.

**Moment improvement: M0 +7.5%, M1 -62.3%, M2 -79.5%** (signal-masked). Different from the
original cube's uniformly catastrophic M0 -86.5% / M1 -10.8% / M2 -168.3% -- here M0 is
genuinely positive, M1/M2 are not. Mixed, not uniformly bad.

**GI wiggle, quadratic estimator: all three fitted geometries agree closely** (mstar 0.572 /
0.579 / 0.500 Msun, inclination 32.0 / 30.7 / 32.3 deg, PA and vsys likewise close). No
unphysical local optimum on this cube's denoised fit, unlike the original's.

**Raw M1 correlates very well with truth for both dirty (0.986) and denoised (0.979). Residual
correlation is low for both (0.111 / 0.108), and nearly identical between them.** This is not
"denoising destroys the wiggle" -- it is "the wiggle is not clearly recoverable from the dirty
cube here, and denoising does not meaningfully change that either way." Clean's residual is
large and structured (RMS 1.39); dirty and denoised's are both nearly flat (RMS 0.18/0.17,
~8x smaller) -- their own Keplerian fits absorb almost all their variance.

**A hypothesis, not yet tested:** the beam convolution itself, not noise or denoising, may be
smoothing out the fine structure the wiggle consists of on this cube -- an information-loss
problem rather than a noise problem, which a plain denoiser cannot fix and which is exactly
the case DDRM/VIREO's measurement-consistency prior is designed for.

Figure + data: `results/self-gravitating/v2_cube_test.png` / `experiments/v2_cube_test_result.npz`.
Full write-up: `results/self-gravitating/v2_cube_test.md`.

## 2026-08-27 | run + bug | corrected self-gravitating pair downloaded, and a real Phase 0 numerical bug found

Jason's replacement cube (`clean_sg.fits`/`dirty_sg.fits`, same Drive folder) downloaded and
checked. Different SHA-256 from the originals, same 601x600x600 shape -- confirmed genuinely
different data, not a resend.

**Different setup from the first pair.** Both cubes are now `BUNIT=JY/BEAM` with identical
BMAJ/BMIN/BPA in both headers (the original had no beam info on its dirty cube at all).
Negative pixels in dirty: 19.25%, against the original's 51.55%. Simulation-recipe header
keys (DISTPC/HACNTR/TRKLEN/NTIME/DECDEG) are gone, replaced by RMS/PBCOR/SEED.

**Found and fixed a real numerical bug in Phase 0's fitter.** First run: `A=0.000`, Gaussian
match RMS in the billions, fitted sigma 0.81 px against the header's 5.63 px -- looked like a
data anomaly. Checked the raw spectra before believing it (RULES.md #8): `P_dirty` runs
~120,000x `P_clean` at the lowest k, confirmed visually
(`kinematic_data_v2_amplitude_check.png` -- same physical structure in both, dirty ~110x
brighter at peak). The least-squares solve in `fit_forward_model` was never tested at this
dynamic range and returned garbage. Fixed with column scaling before the solve (Jacobi/Ruiz
preconditioning); regression case 11 verifies a real ~1e10 power-scale factor no longer
breaks it, recovering both A and sigma within a few percent of the truth.

**Phase 0, after the fix: `NON_GAUSSIAN_CONVOLUTION`.** Fitted beam sigma 6.27 px against the
header's 5.63 px -- 11% agreement, "consistent". A real convolution, genuinely a
deconvolution problem, same category as the original self-gravitating pair (before it turned
out to be the wrong script). Figure: `phase0_v2_cube.png`.

**Beam recovery works, less cleanly than the original pair.** Peak 0.991, shallow -1.6%
sidelobe, but held-out validation gives correlation 0.80 (against the original's 0.994) with
60% residual. Reported honestly rather than smoothed over: likely the ~100x amplitude scale
factor between clean and dirty is not perfectly constant channel to channel, which a single
beam averaged across 12 fit channels cannot correct for. Not yet confirmed. Figure + data:
`dirty_beam_recovered_v2.png` / `.fits`.

**Not yet done:** the GI wiggle Keplerian fit and OOD-style denoising test, run on the
original pair, have not been repeated on this corrected one. Natural next step.

Write-up: `results/self-gravitating/kinematic_data_v2.md`.

## 2026-08-27 | correction | Jason: the self-gravitating cube was made with the wrong script

Reply to the 2026-08-27 email (thread "Doubt", jason.terry47@gmail.com, 13:34): "it turns out
there's a reason it's different: I made it with a completely different script. That's my
bad." He has sent an updated version via the same Drive folder link
(folders/1V33FGbjb8JsaQYSbnYkSEcrSSx-b2kfv), which per his note has "a lot more channels than
the other ones" but should otherwise match in structure.

**Not yet downloaded.** The Drive API cannot list this folder's contents for this account
(same limitation hit on 2026-08-07); it needs a human to open the link first. Whoever picks
this up next: download the new pair, confirm channel count and BUNIT on both files the same
way the first version was checked, before assuming anything else carries over.

**What this means for everything dated 2026-08-21 through 2026-08-27 under
`results/self-gravitating/`:** all of it -- the recovered dirty beam (peak 0.911, FWHM 5px),
the OOD moment result (M0 -86.5%, M2 -168.3%), the GI wiggle Keplerian fits (mstar 0.522 to
0.639 Msun, matching Hall+2020's 0.6 Msun), the quadratic-estimator three-way comparison
(0.92/0.28 residual correlation) -- was measured on the cube Jason now says used the wrong
script. None of it is retracted; it may still describe that specific (if unintended) cube
correctly. But none of it should be presented as the definitive characterisation of "the
self-gravitating disk" going forward, since a corrected version exists and has not yet been
examined.

**What is NOT affected:** every piece of general-purpose code built along the way --
`src/evaluation/gi_wiggle.py` (Keplerian fit, quadratic estimator, both validated on
synthetic data independent of this cube), `estimate_beam_from_pair` in
`forward_operator.py`, `plot_phase0_report` -- is reusable on the corrected cube unchanged.
The methods are not in question, only which cube they were pointed at today.

## 2026-08-25 | finding | the dirty beam is measured, so DDRM no longer waits on the mentor

Both datasets are now on the local machine: the line-emission set at
`DENOISING_DIFFUSION/Line Emission Data/` (14 cube pairs, 7.5 GB, splitting 7/2/5 exactly as
the notebook does) and the self-gravitating pair extracted alongside it.

I had recorded that a DDRM arm needs the PSF image from Jason. **That was wrong.** With both a
clean and a dirty cube in hand the operator can be measured from the data:
`B = <D conj(C)> / <|C|^2>`, now `estimate_beam_from_pair` in
`src/evaluation/forward_operator.py`.

| recovered beam | |
|---|---|
| peak | 0.9109, at the exact centre pixel |
| FWHM | 5 px |
| deepest sidelobe | -2.77% of peak, ring from r ~ 26 px |
| total flux, full field | ~0 |

Peak near 1 with zero net flux is the definition of an interferometric dirty beam, and the
negative ring is exactly what Phase 0's `non_gaussian_convolution` verdict was detecting. Two
independent cross-checks agree: Phase 0's Gaussian fit gave sigma 2.07 px (FWHM 4.9 px)
against the 5 px measured directly, and convolving `lines.fits` with the recovered beam
reproduces `dirty_cube.fits` at **correlation 0.9939 on channels 150-450, none of which were
used to fit it**, leaving a 16.6% residual that is the noise term.

Saved as `results/self-gravitating/dirty_beam_recovered.fits` (72 KB, committed with a
gitignore exception since it is a result and re-deriving it needs the 1.7 GB cubes). Note the
129 px crop sums to 26.5 rather than 0 because it truncates the outer negative bowl; anything
needing flux conservation should re-derive at full size.

**What this changes.** "Ask Jason for the PSF" is off the DDRM critical path. What still
blocks a DDRM arm is data volume: one pair cannot train a model, so Jason's "I'll give you
more later" is the ask that matters, not the PSF.

**What it does not change.** DDRM and VIREO's consistency term remain refuted for the
line-emission training data, where `A = I`. Nothing here contradicts that; the two datasets
are simply different problems.

## 2026-08-21 | finding | Jason's self-gravitating pair IS a deconvolution problem, unlike the training data

The cubes Jason shared on 2026-08-07 were downloaded on 08-08 and never opened. They are in
`DENOISING_DIFFUSION/self-gravitating cube and dirty cube/` (gitignored, 1.5 GB zip, two
601x600x600 cubes). Opening them changes the DDRM/VIREO conclusion for this data.

| | training cubes | Jason's new pair |
|---|---|---|
| clean | `BUNIT=JY/BEAM` | **`BUNIT=JY/PIXEL`** |
| dirty | `BUNIT=JY/BEAM` | `BUNIT=Jy/beam` |
| operator between them | none, `A = I` | **a real convolution** |

`lines.fits` is an unconvolved sky model: 0.00% negative pixels, minimum +41.55.
`dirty_cube.fits` is a genuine dirty image: **51.55% of its pixels are negative**, its minimum
is -0.34 of its maximum, and its total flux is ~0. That is the signature of a non-deconvolved
interferometric map, sidelobes pushing flux below zero with no zero-spacing baseline to carry
the total. A restored map convolved with a Gaussian would show neither.

Phase 0 on the pair returns `non_gaussian_convolution` with a fitted amplitude
**A = 293.5**, which is the beam area in pixels, exactly the Jy/beam-against-Jy/pixel factor
the report warns about. The Gaussian fit itself is poor (residual 0.66, RMS 10.7), which is
what a real sidelobed beam should do to a Gaussian model, so the label is right but the fitted
sigma of 2.07 px is not to be trusted.

**What this changes.** DDRM and VIREO's data-consistency term are refuted **for the
line-emission training data**, where both sides are already beam-convolved. They are not
refuted in general, and this pair is precisely the setup they were designed for. The v26
entry's "the physics-informed line is closed" applies to the data trained on so far, not to
this.

**What it would take.** This is a pivot, not a bolt-on. Training on this regime needs (1) more
than one such pair, and Jason said "I'll give you more later", and (2) the **dirty beam / PSF
image**, because with 51% negative pixels a Gaussian `A` would enforce badly wrong constraints.
Both are concrete asks rather than open questions.

Also worth noting: no header beam. `dirty_cube.fits` carries no BMAJ/BMIN/BPA at all, so
`beam_features_of` returns a zero vector and `beam_kernel_of` returns None on this cube. Any
beam-conditioned arm would silently run unconditioned on it.

## 2026-08-21 | run | 05 v26 — the band-limit idea is refuted, and spectral context is a pixel win only

First run with all three post-v25 fixes, so the first to show the spectral arms in the summary
table and the first to measure out-of-band error. Archived at
`results/05-unet-line-emission/v26_2026-08-21_df38336/`.

**The physics-informed line is now closed, on measurements rather than for lack of time.**

Median out-of-band excess **0.5x**, 1.1% of residual power above the beam cutoff. Excess below
1 means the model's error is if anything smoother than the truth, so a band-limit penalty
would be aimed at nothing. The invented structure is **beam-scale, not sharp** — the worse of
the two possibilities, because a beam-scale blob is exactly what a real detection looks like
and no band constraint can separate them.

With Phase 0's `A = I`, all three are struck: DDRM, VIREO-lite's data-consistency term, and
the band-limit variant. Each was killed by a number, not by the calendar.

**Spectral context, three runs per arm at seed 42:**

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner_k1 | 41.26 ± 0.92 | 21.0 ± 9.3 | 63.4 ± 4.9 | 33.0 ± 16.3 |
| winner_k2 | 41.81 ± 0.55 | 16.6 ± 2.1 | 67.6 ± 6.1 | 23.8 ± 12.3 |
| winner_aug (3 seeds) | 39.30 ± 0.46 | 29.2 ± 7.2 | 74.0 ± 2.0 | 55.0 ± 13.9 |

About +2.4 dB of PSNR, and it **loses M0 and M2 to augmentation**. Three runs each, so this is
not the seed noise that made v25's reading unreliable. `winner_aug` is still the best arm on
the metric that matters.

That divergence is itself a result worth reporting: the two spectral arms are the clearest
case yet in this project of PSNR and moment reliability moving in opposite directions.

Unlike v25's, this version's push **preserved** the committed cells rather than reverting them.

## 2026-08-21 | run + bug | 05 v25, and the error bars in this project are too small

Second run of `winner_k1` and `winner_k2` at seed 42, same code, and it does not reproduce
the first. Archived at `results/05-unet-line-emission/v25_2026-08-21_889cd44/`.

| arm, seed 42 | first run | v25 | difference |
|---|---|---|---|
| k1 PSNR | 41.948 | 40.219 | 1.73 dB |
| k1 best epoch | 25 | 18 | |
| k1 M2 | +46.2% | +14.8% | 31.4 pp |
| k2 PSNR | 41.199 | 42.269 | 1.07 dB |

A third partial run of k1 hit 41.058 at epoch 22, so three fixed-seed runs span 40.219 to
41.948, standard deviation **0.865 dB**. `winner_aug`'s spread across three different SEEDS
is **0.455 dB**. Re-running one seed varies more than changing the seed.

Cause is GPU nondeterminism, not a notebook bug: `torch.manual_seed` does not fix cuDNN's
algorithm selection, and the T4 x2 setup splits batches across devices.

**Which published numbers this touches.** Every per-arm figure in this project is a single
run, so every band quoted as a seed spread is really a run spread and understates the total.
The 3-seed bands are the least affected because three runs at three seeds do sample it. The
1-seed rows, `winner_beam` (+9.6 / +63.2 / +20.9), `winner_patch`, and now both spectral arms,
have no error bar at all and should not be compared against anything at this resolution.

**Withdrawn:** the reading from the first spectral run that k1 lifts M1 and M2 beyond the
seed spread. v25's k1 gives M1 58.5 against the baseline's 55.6 and M2 14.8 against 6.0, both
well inside the spread. **What survives** is the PSNR gain over the un-augmented baseline,
40.2 to 41.9 against 37.5, consistent across all three runs.

**Also: Kaggle reverted committed work again (RULES.md #2, third time).** Commit `df38336`
pushed the kernel's notebook over `84b6cb6`, deleting 2248 lines: the CONFIGS-derived moment
table, the out-of-band check, and the section 7 centre-channel fix. Restored in `201b4ef`.
v25 itself ran at `889cd44`, before all three, so it has no out-of-band measurement and its
summary table again omits the spectral arms. The VIREO question is still open.

## 2026-08-20 | code | the spectral-context arms are wired into 05 and ready to run

Two new arms in notebook 05, `winner_k1` and `winner_k2`, identical to `sweep_winner` except
that the input is the channel plus k neighbours along velocity. This is the Phase 3a item and,
after Phase 0, the only physics-informed lead still standing.

Wired end to end rather than only in `src/`:

- `train_unet(n_neighbors=k)` forwards to `build_model`, and the checkpoint now records
  `in_channels`, so a resumed run rebuilds the right shape instead of assuming 1.
- Cell 12 builds the k=1 and k=2 views and asserts their centre channel is byte-identical to
  the k=0 item, so the arms stay a clean one-change ablation.
- Cell 14's score-without-retraining path reads `in_channels` from the checkpoint and scores a
  k>0 arm on its own val view, because a 2k+1-channel model cannot be fed the 1-channel
  loader at all.
- `denoise_cube` builds the neighbour stack for section 6, with neighbours normalised by the
  CENTRE channel's (min,max) and the cube's ends clamped, matching the dataset exactly.

That last one is where a silent bug would have lived. `denoise_cube` keeps a per-channel
`norm` array, and slicing it for neighbours would have given each its own scale, erasing the
relative amplitude along velocity, which is the whole signal these arms add. It would have
trained on one representation and scored on another with nothing raising, exactly the shape of
the `winner_beam` failure. `tests/test_denoise_cube_spectral.py` executes the notebook's real
`denoise_cube` source against a spy model and compares its input tensors to the dataset's:
8/8 channels identical, and the deliberately-wrong version is flagged 8/8, so the check can
fail (RULES.md #8).

Verified by training a k=1 arm for one epoch on synthetic cubes: trains, writes
`in_channels=3`, and the resume path rebuilds it and reproduces the same PSNR.

**Not yet run on real data.** Suggest k=1 and k=2 at one seed against `sweep_winner` at the
same seed. Expected effect is on M1 and M2 specifically; if they do not move, the
per-channel-independence explanation for their weakness is wrong and worth revisiting.

## 2026-08-20 | finding | the clean cubes are already beam-convolved

Run 4 came back `indeterminate` on all four cubes, and the diagnostic dump answered the
question outright.

**Both cubes are `BUNIT=JY/BEAM`.** A model or sky image would be Jy/PIXEL. Jy/beam means the
beam has already been applied, to the CLEAN cube as well as the dirty one. The spectra agree:
`P_clean` for run_0006 falls 1650 -> 4.6 -> 0.0069 -> 1e-6 -> 4e-11, thirteen orders by
k = 0.104, exactly where the header's 7.83 px beam cuts off, and then flattens onto the
float32 floor near 1e-12. An unconvolved model image would keep following a power law.

**So the answer to Phase 0 is that there is no operator BETWEEN the two cubes.** The pair is
`dirty = clean + noise`, with the same beam already inside both sides. The ratio behaves
accordingly: it climbs 1.009 -> 2.42 -> 8.79 -> 41.7 -> 58.6 and never dips, which is
`1 + N/P_sky` with `P_sky` falling.

Two code faults this exposed, both now fixed and both regression-tested:

- **The band included the float32 floor.** Past k ~ 0.10 neither spectrum is physical, yet
  their ratio settles near a plausible 2.2. Mixing that with the real region, where the ratio
  climbs to 58, left no (A, N) able to fit either, which is what produced sigma = 0 with
  residuals of 0.42 to 0.89. `_measurement_band` now cuts on the CLEAN floor as well as the
  dirty one.
- **The noise was modelled as white.** In a Jy/beam map the noise has been through the beam
  too, so its spectrum is beam-shaped; a flat term implied N running from 16 down to 2e-9
  across the band. Both a white and a beam-shaped term are now fitted together, so neither
  the caller nor the code has to guess which kind a cube has.

Also fixed a `KeyError: 'indeterminate'` in the notebook cell's verdict-explanation dict.

**What this means for the plan.** DDRM has nothing to invert: the instrument response is not
between the network's input and its target. VIREO-lite's image-plane data-consistency term is
dead for the same reason, since `A = I` makes it `||pred - dirty||`, which would train the
model toward the noise. Both are struck, and not for lack of time.

This is a result worth stating in the final blog rather than a dead end: **the problem here is
denoising, not deconvolution.** The network is learning to remove beam-correlated noise from
an already-convolved map. It also reframes the Friday questions: the useful ask is no longer
the PSF image but whether an unconvolved (Jy/pixel) sky cube exists, because that is what
would turn this into a deconvolution problem and put DDRM and VIREO back on the table.

## 2026-08-20 | bug | Phase 0 run 3: the input was half empty channels

`non_gaussian_convolution` 4/4 again, and this time visibly broken: best-fit beam **0.00 px**
with gain 1.00x, meaning the fit found no beam at all, yet it still landed in the convolution
branch and printed Gaussian RMS values up to 7e32.

**Two faults.**

1. **Input.** `phase0_from_fits` took evenly spaced channels via `np.linspace(0, n-1, 8)`,
   which includes channel 0 and channel n-1. Those are the extreme high-velocity ends, which
   the mentor's own sampling note (2026-06-18) calls "mostly continuum with little signal".
   Clean power there is near zero, so P_d/P_c explodes for reasons unrelated to any beam, and
   averaging them in with real channels left a model that fits neither way: no-beam residual
   0.5556 / 0.8029 / 0.8657 / 0.8992. Channels are now ranked by clean standard deviation and
   the line-bright ones used.
2. **Logic hole.** A fit landing on sigma = 0 found no beam, so it can never be a convolution.
   It now returns `indeterminate` with the reason, instead of falling through to a
   Gaussian-vs-sidelobe comparison against a flat transfer.

`phase0_diagnostics()` added and wired into the notebook cell: on an indeterminate verdict it
now dumps shapes, BUNIT, header keys compared between clean and dirty, the radial spectra as
a table, and the per-channel clean std. Three wrong verdicts came from a summary statistic
hiding what the spectra were doing, so a surprising answer should no longer be guessed at.

**Phase 0 still unanswered**, fourth run pending. Published numbers touched: none.

## 2026-08-20 | bug | Phase 0 got two more wrong verdicts before the method was right

Three versions of the discriminator, three confident wrong answers on the real cubes. Nothing
was recorded from any of them, and no published number is affected, but the pattern is worth
keeping: each failure came from choosing a normalisation and then testing a threshold against
it, rather than fitting the thing being measured.

1. **Raw ratio dips below 1?** Breaks when the two maps are on different intensity scales.
   Real cubes returned `no_convolution` 4/4 with minima 1.010 / 1.024 / 1.138 / 3.962.
2. **Normalise the ratio by its own low-k level, take the minimum over the band?** For a
   monotonically rising ratio the lowest bin is by construction below the median of the
   lowest bins, so a dip appears where nothing was suppressed. Real cubes flipped to
   `non_gaussian_convolution` 4/4, with `k_at_min` pinned at the lowest bin, 0.004, on every
   one, and Gaussian fit RMS of 1.59 to 3.47. It was reporting non-Gaussian because the fit
   had failed, not because it found sidelobes.
3. **Does a Gaussian beam beat the no-beam model?** A sidelobed beam is not Gaussian, so
   neither model fits, the comparison ties, and a real convolution reads `no_convolution`.

**What replaced them.** Fit the forward model directly,
`P_d(k) = A exp(-4 pi^2 sigma^2 k^2) P_c(k) + N`, and ask whether its no-beam version is
adequate ON ITS OWN. `A` is a free parameter, so intensity scale cannot mislead it; sigma is
recovered rather than inferred from a threshold. Given sigma the model is linear in (A, N),
so a scan over sigma with a linear solve at each step is exact and needs no optimiser.

Separation on synthetic cases is two orders of magnitude, so the threshold is not delicate:

| case | no-beam residual |
|---|---|
| additive noise, x1 to x25 | 0.0008 to 0.0107 |
| sidelobed beam | 0.8938 |
| Gaussian beam, sigma 1 to 3 | 2.86 to 3.21 |

Recovered sigma is exact to 0.3% and unchanged across a 200x range of intensity scale. All
three failures are regression cases in `tests/test_forward_operator.py`.

**Phase 0 is still unanswered.** Needs a third run.

## 2026-08-20 | bug | Phase 0's first real run was wrong: the check was not scale-invariant

First run on the project's cubes returned `no_convolution` on 4/4, which would have killed
DDRM and VIREO-lite outright. The numbers gave it away: minima of **1.010, 1.024, 1.138 and
3.962**. If `A = I` truly held, `dirty = clean + noise` puts every cube at ~1.00. A 3.96 means
that cube's dirty map carries 15.7x the clean power at the largest scales, which additive
noise cannot do, and a 1.01-to-3.96 spread is not one operator measured four times.

**Cause.** The verdict tested whether the raw ratio `P_d/P_c` dips below 1, which assumes
clean and dirty share an intensity scale. A dirty or restored map is conventionally Jy/BEAM
and a model image Jy/PIXEL, differing by the beam area in pixels, order 200 for these
headers. Any such factor multiplies the whole ratio and lifts a real convolution above 1.

**Reproduced before fixing.** Same synthetic Gaussian blur, four intensity scales, old code:

| scale | verdict | min |
|---|---|---|
| x1 | gaussian_convolution | 0.060 |
| x5 | non_gaussian_convolution | 0.291 |
| x25 | **no_convolution** | 1.456 |
| x200 | **no_convolution** | 11.824 |

The observed 1.01-3.96 sits inside that failure range.

**Fix.** Normalise the ratio by its own low-k level before the dip test. A convolution kernel
has unit sum so |B(k)| -> 1 as k -> 0, meaning the low-k level estimates the scale factor by
itself; dividing it out leaves the shape, and the shape is what separates the two cases. The
factor is now reported as `dirty/clean low-k amplitude`, flagged when it is far from 1, so a
units mismatch is visible rather than silently steering the verdict.

Six regression cases added, x1 through x200 convolved plus rescaled additive. Recovered beam
sigma is now 2.97 px across a 200x range of scales.

**Second look, same day: the scale factor was probably not the story.** Jy/beam against
Jy/pixel predicts a factor near 200; the cubes showed 1.01 to 3.96. And on all four the
minimum sat in the LOWEST k bin, so the ratio never fell anywhere, and dividing out a factor
that small cannot change that. The re-run will most likely return `no_convolution` again.

What was actually missing is a check that the measurement could have SEEN a beam. The band
ends where the dirty spectrum sinks into its own noise floor; the header's beam (sigma ~6.5
px) does not suppress anything below k ~ 0.033. A null from a band that stops short of that
is uninformative, not negative. `phase0_report` now reports the band's reach against the
header's own rolloff and returns `indeterminate` rather than `no_convolution` when the band
was too narrow to decide. RULES.md #8, applied to this diagnostic itself.

**Published numbers touched: none.** The bad verdict was never recorded anywhere; it was
caught in the same session it was produced. **Phase 0 is unanswered** and needs a re-run,
which will now say either `no_convolution` with the band coverage to back it, or
`indeterminate`.

## 2026-08-19 | code | 2.5D spectral context, the one Phase 3 item nothing gates

`FITSChannelDataset(n_neighbors=k)` now emits a `(2k+1, H, W)` dirty tensor, the sampled
channel plus k neighbours each side along velocity, with the clean target still the centre
channel alone. `_build_unet(n_neighbors=k)` sets `in_channels` to match; `UNet` already took
`in_channels`, so there was no model change to make.

**Why this one and not VIREO-lite.** VIREO-lite turns out to be gated by Phase 0 as well, not
just DDRM: if `A = I` the data-consistency term collapses to `||pred - dirty||`, which would
push the model toward the noisy input. Spectral continuity is gated by nothing, and the plan
already called it the highest-value item in the document. M0 is a spectral sum and scores
~+70%; M1 and M2 are spectral shape statistics and lag, because every channel is denoised
independently and nothing uses the axis those two are computed over.

22 checks in `tests/test_spectral_context.py`, and the existing 24-test data pipeline still
passes. `n_neighbors=0` reproduces the old items bit-for-bit, asserted with `torch.equal`
rather than a tolerance, so no existing result changes meaning.

Two decisions in it would have silently destroyed the point if made the other way: neighbours
share the CENTRE channel's scale (per-neighbour normalisation would erase relative amplitude
along velocity, which IS the added signal), and the cube's ends clamp rather than wrap (the
first and last channels are the line-free high-velocity ends and are unrelated).

**Not yet run.** The notebook builds its own datasets and cells do not sync from git
(RULES.md #2), so an arm needs `n_neighbors` threaded through 05 by hand.

## 2026-08-19 | code | Phase 0's forward-operator check exists, verdict still unknown

`src/evaluation/forward_operator.py` + `tests/test_forward_operator.py`. Settles the gate in
PHYSICS_INFORMED_PLAN.md that decides whether DDRM gets built: is `dirty = clean (*) beam +
noise`, or just `dirty = clean + noise`? In the second case `A = I` and DDRM collapses into
the conditional DDPM that already exists.

Discriminator is the radially averaged power-spectrum ratio, which dips below 1 only if
something suppressed spatial frequencies. Three verdicts, because a convolution by a real
dirty beam is not the same finding as a convolution by a Gaussian: the first needs the PSF
image from Jason before `A` can be written down.

Also lands `pixel_scale_arcsec` and `beam_kernel_of`, reading `CDELT`, which no code in
`src/` read before. `beam_features_of` takes BPA/BMAJ/BMIN only, which describes a beam in
angular units but cannot build a kernel in pixels.

Verified against three synthetic cases with known operators, 14 checks. Two wrong versions
were caught by them and both are now regression cases: a signal band set as a fraction of
peak power, which cuts off at k = 0.04 on a red spectrum and makes every beam look Gaussian;
and reading the verdict off the noise-subtracted ratio, whose floor estimate manufactures a
dip where nothing was suppressed.

**Nothing measured yet.** The check needs FITS data, which lives on Kaggle. Until it runs,
DDRM stays unbuilt, per the plan's own gate.

## 2026-08-14 | run | 05 v19 — crashed at 4.0 h, no moment scores

`DeadKernelError` at 14413s, inside Kaggle's 12 h limit, so a crash rather than a timeout.
`ConnectionResetError: [Errno 104] Connection reset by peer` in
`multiprocessing/resource_sharer.py`, after `winner_patch` early-stopped at epoch 26 with its
epoch times drifting 111 to 128s where every other arm held flat near 90s. Memory pressure,
most likely `FlatPatchDataset.__getitem__` decoding each image once per patch, 8x redundant.

Section 6 never ran, so the run produced **no moment scores at all**. Three checkpoints
trained and survived in the Output; `winner_patch`'s metric row did not, because the kernel
died between its last epoch and `val_metrics`.

**Consequence:** 05 gained a resume path (`_import_prior_nb05`), a score-without-retrain
branch for a checkpoint with no metric row, `RUN_NATIVE600 = False`, and a guard so section 5
cannot `NameError` when every arm resumes. **Notebook not archived. Lost.**

## Earlier

See `RUNS.md` for the full per-notebook index back to 05 v12 and 06 v11. This log starts at
the point the project began losing information that the run folders alone did not capture.
