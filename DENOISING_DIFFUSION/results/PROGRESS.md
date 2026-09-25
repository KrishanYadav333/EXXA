# Progress log

Chronological record of runs, arrivals and bugs, newest first. Written at the time, per
RULES.md #11. `RUNS.md` maps a number to the run that produced it; this file records what
state the project is in and how it got here.

Entry format: **date | trigger | notebook** then what happened, the evidence, the
consequence. Triggers are `run`, `added` (a notebook downloaded into the repo), `bug`.

---

## 2026-09-25 | fix | notebook 16 figures: every checkpoint, core disk sheets only (~34 images per case for 34 checkpoints)

Paging alone gave ~75 images per case (228 for the quick run). A first cut limited the disk sheets to the top 16 checkpoints by M0; that dropped 18 checkpoints from the images, against the standing
requirement that every checkpoint has its disk figures, and was reverted the same day. **Default now:** EVERY checkpoint on the core disk sheets (M0, M1, M2, M1 error, 8 checkpoints per page with clean and dirty on
every page) and the classic wiggle pages (5 per page), plus the single figures (spectra, radial/power, calibration, integrated spectrum, error histogram, ensemble, top-k detail): 5x4 + 7 + 7 = 34 per case, so
about 108 for the quick run with its 6 dashboards. The channel, sharpness, invented-structure, M0-error and residual-sheet kinds return with `FIG_EXTRA = True` (about 4x); `FIG_TOP = n` limits the disk sheets to
the best n. The defaults are in `src/`, so a stale Kaggle cell still gets them. Figures only; no published number touched.

---

## 2026-09-25 | fix | notebook 16's contact sheets tiled all 34 checkpoints in one figure; now 8 per page

**What was wrong.** Every M0/M1/M2, error, channel, wiggle, sharpness and invented-structure sheet put clean, dirty and ALL checkpoints on one figure at 7 columns, so each disk was about
1/8 of an 11 inch page and the fine structure (the thing the sheets exist to show) was not legible. **How it was caught.** The user looked at the sheets from the first Kaggle run against the
one-checkpoint-per-figure images from the older notebooks. **Fix.** `contact_sheet` pages: 8 checkpoints per page, 4 columns, 3 inch tiles, the reference tiles (clean, dirty, or the error to beat)
repeated at the top of every page (the error, channel-error and invented-structure sheets open each page with clean and dirty on their own scales, then the error to beat), one shared colour scale across pages (`..._p01.png` ...; a sheet that fits keeps its old name). The notebook display cell shows page 1 of each and the classic wiggle
pages 1 and 2; every page is in the Output. A test builds 34 panels and asserts 5 pages. **Published numbers it touches:** none (figures only). The first run's Output still has the old
unpaged sheets; the next run, or a local rebuild from the downloaded `nb16_maps/*.npz` with `cf.build_all`, produces the paged ones.

---

## 2026-09-25 | run | notebook 16 first Kaggle run (quick): 34 checkpoints, 3 cases, 102 rows in 49 min; 320 px is NOT inherently bad, 600 px does not help

Archived as `results/16-checkpoint-evaluation/v_pending_2026-09-25_quick_6e4e139/` (Kaggle version number unknown). Line-emission means are over **2 cubes** (`rt_00`, `rt_01`), sd across cubes:
**not comparable to the 5-cube 05 tables**; the harness reproduces the published v20 row for `sweep_winner_aug_seed43` on `rt_00` (M0 +31.3 / M1 +77.2 / M2 +84.6), so it is scoring consistently.
A display cell raised after all scoring and stopped `collect_outputs`; scoring/figures are in the Output, the CSV must be downloaded. Fixed.

**Resolution (ask 2), 2-cube means M0 / M1 / M2:**
| model | M0 | M1 | M2 |
|---|---|---|---|
| aug at 256, 3 seeds (mean) | 30.4 | 80.0 | 51.6 |
| aug at 480 | 44.5 | 81.0 | 68.9 |
| earlier 320 px arm `winner_res320_seed42` | 44.8 | 80.9 | 67.3 |
| aug at 320 `winner_aug_res320_seed43` | -39.6 (sd 105) | 24.2 | -56.9 |
| 600 px: plain / aug / p10 | 5.3 / -1.7 / -13.4 | 60.7 / 60.6 / 67.6 | -23.4 / 45.3 / -6.1 |

**Corrects the 05 v47 reading ("320 is clearly worse").** A different 320 px model, `winner_res320_seed42`, scores like 480 (44.8 / 80.9 / 67.3), so 320 px is not inherently worse:
`winner_aug_res320_seed43` is one bad training run (it fails on one cube by -114). Two 320 px models, one good and one bad, is a seed spread, not a resolution effect. 480 is level with or slightly above
256 (M0 +14 is inside the 256 px seed spread of about 11 to 30 depending on the cube pair). **600 px is worse than 256 on M1 (61 to 68 against 71 to 84) and M0**, with lower PSNR (38 to 39 against 40 to 42) and it costs 74 s per cube per checkpoint against 21: the extra resolution is not recovered.
One seed per resolution arm.

**Loss controls (p10 source, 2 cubes) M0 / M1 / M2:** hybrid control 39.3 / 79.3 / 70.5; mae 35.5 / 75.4 / 84.2; wavelet 35.6 / 79.8 / 75.1; starlet 45.2 / 78.6 / 81.2; gradient 30.7 / 81.9 / 77.8.
Same shape as the 5-cube reading: only starlet (M0 +6, M2 +11) and mae (M2 +14) sit above the control, and only on some moments; one seed. Not established.

**kin_gamma0 family (stack_kin, 31 neighbour channels):** `kin_gamma0_starlet_ft` is the best line-emission checkpoint here on all three moments (50.4 / 84.1 / 72.8, PSNR 47.3; M0 sd across the 2 cubes 19.6) and `kin_gamma0_mae_ft` next (46.6 / 80.0 / 49.8). The base `kin_gamma0` is -23.1 on M0, and `kin_gamma0_mae_fresh` is a failed training (M0 -266.6, sharpness ratio 173). **A best-of-34 pick from two cubes; the 5-cube run must confirm it.**
**PSNR vs the moments:** Spearman across checkpoints +0.66 (M0), +0.73 (M1), +0.81 (M2). That is dominated by the failed arms (`sg_k3_fresh`, `kin_gamma0_mae_fresh`); the top PSNR arm is also the top M0 arm here, unlike the 05 result where `winner_k1` had the best PSNR and a poor M2. Do not read either as PSNR ranking or not ranking from this alone.

**SG v2 (cross-domain, 1 cube) is a suspect result (RULES.md #8), not a finding.** Every line-emission model has M1 -40 to -98 and M2 -23 to -93, and every wiggle gain is negative (worse than dirty, consistent with the earlier
"every model loses to dirty on the wiggle here"). But `sg_k3_fresh`, trained on SG data, scores M0 -66, M1 -57, M2 -123 on it, and is among the worst; SG v2 is a different cube from its training family and its amplitude is rescaled by `match_amplitude`, so a domain gap and a harness artefact are both possible.
Needs the per-cube figures and `amp_scale` from the CSV before it is quoted.
**Speed:** measured 21 s (256 px U-Net), 33 s (stack_kin), 74 s (600 px) per row: a full run (about 66 supported checkpoints x 7 cases, plus slower DDPM rows) is now about 4 to 6 h, not the 15 to 30 h guessed earlier.
**Published numbers it touches:** the 2026-09-25 v47 entry and BLOG_PROGRESS.md section 2 said 320 px was worse; both corrected.

---

## 2026-09-25 | run | 05 v47: 320 and 480 px scored at their own size; ask 2 answered for the aug recipe, one seed each

Clean scoring-only session (no training, 38 checkpoints restored, fixes live at `499030b`). Means over 5 holdout cubes, clipped + signal-masked, one seed per resolution arm
against a 256 px baseline whose spread is across 3 seeds:

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| aug at 256 (3 seeds) | 39.30 | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| aug at 320 | 37.36 | -17.8 | 48.0 | -3.9 |
| aug at 480 | 40.18 | 39.7 | 73.5 | 67.3 |

**Reading.** 320 is clearly worse: M1 26 points and M2 59 points under the baseline, and 2 of 5 cubes fail outright (M0 -114 and -114). 480 is level with 256: M1 the same,
M0 +10 and M2 +12, both inside one seed spread of the baseline (7.2 and 13.9), so **no established gain from less downsampling at 480**. The 320 failure being worse than 256
while 480 is not is not monotonic in resolution, which points at this one seed (or the 320 recipe) rather than at resolution itself; a second seed would settle it. Native 600 px
is not in 05 (gated, never trained); the only 600 px models are notebook 14's three, scored in notebook 16.
**Published numbers it touches:** the 2026-09-25 rows for `winner_aug_res320`/`res480` (scored at 256 in v42/v43) are superseded; BLOG_PROGRESS.md section 2 updated. Notebook 16 had a
second version of the same defect for 14's arms (label `sweep_winner_600` read as 256); fixed in `_train_size`.
**Archived:** `results/05-unet-line-emission/v47_2026-09-25_773b0ce/`. Versions 45 and 46 are not in git.

---

## 2026-09-25 | added | notebook 16 set to a QUICK first run: 3 cases, ~25 checkpoints

Full inventory (~65 scoreable checkpoints) x 7 cases is ~455 rows; the only timing is the local CPU preview (230 to 275 s per row), so a full run is a guess of 15 to 30 h
(2 to 4 sessions). Cell 6 now has `PROFILE = 'quick'` (2 line-emission cubes + SG v2) and an `INCLUDE` regex for the checkpoints the loss, resolution and
best-model decision needs: the loss arms `_p10_ft` and `_fresh`, both hybrid controls, aug/p10/non-aug seed baselines, `winner_beam`, `res320`, `res480`, the three
600 px arms, `kin_gamma0` (+ mae/starlet), `sg_k3_fresh`. ~25 checkpoints x 3 cases ~ 75 rows, guessed at 2.5 to 5 h. **Unmeasured until the first Kaggle session
prints `wall_s`.** Not included on purpose: DDPM/DDRM, the older 08/10/11/12 families, `winner_k1/k2`, patch. A later session sets `INCLUDE = None`, `PROFILE = 'full'`
and resumes from this Output. No published number touched.

---

## 2026-09-25 | run + bug | 05 v44 FAILED (CUDA OOM scoring res480), but the control arms ran: much of the fine-tune gain is extra training, not the loss

**Run.** Fresh Kaggle import, cells current (`1b1ee7f`). Trained `winner_hybrid_ft` (from aug, PSNR 39.574) and `winner_hybrid_p10_ft` (from p10,
PSNR 40.253), 30 epochs each, both persisted. Then the scoring step died: `OutOfMemoryError: Tried to allocate 3.96 GiB` in cell 18, scoring
`winner_aug_res480`. **Cause:** the 2026-09-25 per-arm-size fix resized the image but kept `BS = 32` channels per forward, which fits at 256 and 320
and not at 480 (or 600). **Fix:** `bs = max(1, int(BS*(TARGET_SIZE/size)**2))`, now in `tools/reapply_05_fixes.py`. **Published numbers it touches:**
none; `res480` and `native600` still have no valid moment row.

**The control arms answer the confound flagged on 2026-09-25 (RULES.md #4).** Same source, same 30 epochs at 0.1x lr, ORIGINAL hybrid loss. Means over
the 5 holdout cubes, one seed each, against a baseline whose spread is across 3 seeds:

| arm | M0 | M1 | M2 |
|---|---|---|---|
| aug baseline (3 seeds) | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| **control: hybrid, from aug** | **39.7** | **80.9** | **69.9** |
| mae / wavelet / starlet / gradient, from aug | 26.3 / 38.8 / 42.6 / 32.1 | 77.4 / 76.1 / 80.5 / 78.7 | 70.5 / 70.0 / 81.1 / 77.3 |
| p10 baseline (3 seeds) | 33.5 +/-9.6 | 70.7 +/-6.7 | 31.8 +/-11.1 |
| **control: hybrid, from p10** | **40.4** | **76.2** | **68.4** |
| mae / wavelet / starlet / gradient, from p10 | 43.5 / 41.4 / 44.5 / 41.7 | 79.6 / 78.1 / 77.1 / 80.3 | 83.8 / 73.1 / 79.6 / 70.5 |

**Reading.** Extra training alone (no new loss) moves aug by +10.5 / +6.9 / +14.9 pp and p10 by +6.9 / +5.5 / +36.6 pp. From aug, no new loss beats that
control except starlet on M2 (81.1 vs 69.9) and starlet on M0 (42.6 vs 39.7, inside noise); MAE is BELOW it on M0 (26.3 vs 39.7). From p10 the losses sit
at or above the control by 0 to +4 on M0, 0 to +4 on M1, and +2 to +15 on M2. So **most of the `_ft` gain in the 2026-09-25 table was continued training, not the
loss**, and the honest claim left is small: starlet and MAE (from p10) may add a little on M2, one seed, not established. The FRESH arms (which had a clean
control) are unaffected by this: they still beat the non-aug baseline on M1/M2.
**Published numbers it touches:** the 2026-09-25 "MAE helps M2 and M1 more than M0" paragraph and BLOG_PROGRESS.md section 1, both corrected. Also the res320 row:
scored at its own 320 px it is M0 -114.0 / +34.8 / +38.3 / -113.5 / +65.4 per cube (cubes 00 and 0025 fail), single seed; PSNR 37.36 is the lowest of any arm.
**RAM breakdown (first live reading).** `main` grows ~0.1 GB/epoch (2.4 to 8.6 GB across the two arms); workers' RSS stays flat (12.4 then 39.7 GB, shared pages).
The leak is in the main process. Cause still unknown. **Archived:** `results/05-unet-line-emission/v44_2026-09-25_failed_oom/`.
**Left:** a session to score `res480` and `native600` (fix in), and getting the two control checkpoints out of v44's Output (failed versions cannot be attached; download, re-upload as `.ckpt`).

---

## 2026-09-25 | run | 05 v43: last spectral-context arm done; Kaggle's push overwrote the 05 fixes again, restored by script

Ran the pre-fix cells (cell 0b pulled `ad600d2`), so nothing from the day's fixes was in it. One arm, `winner_k2` (two spectral neighbours),
seed 42, early stop at epoch 50: PSNR 42.81 (best of any arm), M0 +20.2 / M1 +65.9 / M2 +36.8 (5 cubes, one seed, clipped + signal-masked).
Same pattern as `winner_k1` (42.59, +33.3/+75.6/+40.5): spectral context is a pixel win and a moment loss, and the notebook's own line says
ranking by PSNR would have picked it. Two neighbours are no better than one. Archived: `results/05-unet-line-emission/v43_2026-09-25_6fb324d/`.
**Bug (RULES.md #2, third time).** Kaggle pushed v43 (`6fb324d`) over the notebook, reverting the per-arm scoring size, `STALE_MOMENT_ARMS`
and the two control arms. Caught by grepping the pulled file for `_arm_size`. Restored with `tools/reapply_05_fixes.py` (8 edits, idempotent).
**Published numbers it touches:** none new; the res320/480 rows in v42/v43 tables remain the invalid 256 px scores.
**Left:** `winner_hybrid_ft`, `winner_hybrid_p10_ft` and the res re-score, in one session, from a fresh Kaggle GitHub import.

---

## 2026-09-25 | bug | notebook 16's amplitude checks were meaningless on the SG v2 cube: clean and dirty are on different scales

Found by looking at the first SG v2 calibration plot (denoised against clean, pixel by pixel): slopes of ~300 and empty tiles. The
SG v2 cube's dirty is ~336x its clean at the systemic channel (~136x at the M0 peak), which this project already knew (PROGRESS.md
notes "dirty ~110x"; `kinematic_data_v2_amplitude_check.png`) but the harness did not account for. A model returns values in its INPUT's
units, so every amplitude comparison against an unscaled clean measured the data's units: PSNR (16.5 dB on that cube), M0 improvement,
channel and M0 error maps, calibration, invented structure, overshoot. Velocity quantities (M1, wiggle, M1 sharpness) are scale-free and
were never affected. The line-emission cubes are at 1.01 and are unaffected.

**Fixed.** `match_amplitude` puts clean in the input's units with the least-squares factor of dirty on clean, but only outside
0.8 to 1.25, so a cube whose halves already agree is left exactly as it was and the validated line-emission numbers (M0 +31.3 / M1 +77.2 /
M2 +84.6) cannot move. The factor is in every row (`amp_scale`) and on every amplitude figure's title. The correction is an assumption
(a single linear factor between the two cubes) and any SG v2 amplitude number should say it was rescaled.
**Published numbers it touches:** none. No published SG v2 result uses amplitude; they are all wiggle (velocity).

**What does and does not reproduce on SG v2.** The classic wiggle figure reproduces the earlier one to two decimals: residual RMS
clean 0.23 / dirty 0.21 / U-Net 0.22, and the smoothing is visible (the concentric ring and blotches in clean's residual are gone from
the models'). The wiggle CORRELATIONS are close but not identical to the published ones (dirty 0.873 against 0.862; `winner_aug_seed43`
0.790 against 0.760, and `sg_k3_fresh` and `winner_beam_seed42` swap order), so this is not a reproduction of that table. Notebook 16
continuum-subtracts every cube and starts the fit from the emission's shape; the older script did neither. The finding they support,
that every model loses to dirty on the wiggle here, holds: the new error ratio is 1.29 to 1.47 (all above 1).

---

## 2026-09-25 | bug | the line-emission wiggle geometry was degenerate on 2 of 5 cubes; the published nb08 wiggle numbers include them

**What was wrong.** `score_08_kinematic.py` (and most `experiments/` wiggle scripts) call `fit_keplerian(m1, mask, au_per_px)`
without `m0=`, so the fit starts from the MASK's shape, which `fit_keplerian`'s own docstring calls "the degenerate end of the
mass-inclination valley". **How it was caught.** Building the first real figures for notebook 16: the M0 radial profile was zero out
to 200 px and then rose, which meant the disk centre I had passed was wrong. The stored geometry had `cy = 600.0`, the image edge.
Refitting the same cube with `m0=` moved the centre to (299.9, 300.5), mass from 18.7 to 1.46 Msun (truth 1.0), residual rms from
9.4 to 4.4 km/s, and the fit cost to a third. Then the same refit on every case (about 45 s, no models needed):

| cube | as published (no m0) | with m0 |
|---|---|---|
| `run_0002_00560_rt_00` | centre (374, **600**), incl 73.0, mstar 18.7 | centre (300, 300), incl 48.9, mstar 1.46 |
| `run_0002_00560_rt_04` | centre (378, **600**), incl 74.2, mstar 0.80 | centre right, but incl 1.8, mstar **50.0 at the bound** (true incl 11.7: the documented mass-inclination degeneracy) |
| `rt_01`, `run_0025`, `run_0026`, and `sg_v2` | fine | identical |

So it is not a general failure: 3 of 5 line-emission cubes and the SG v2 cube were unaffected, and SG v2's dirty 0.87 (the anchor
of "doing nothing beats the model") stands. `nb08_kinematic_wiggle.json` stores exactly this: cubes `rt_00` and `rt_04` have `cy = 600.0`.

**Published numbers it touches.** The `nb08` kinematic wiggle table for those two cubes, and every mean over all five: kin_gamma0
**0.8155 against dirty 0.4284** (PROGRESS.md 2026-09-11 onward, the Phase J summary, and the line-emission points of
`headroom_scatter.png`, two of whose five points come from these fits, including the one at dirty 0.996). Not touched: SG v2, the SG
LOO folds and the SG holdout (inclination fixed at the true value, or fits identical either way).
**What still holds.** On the 3 cubes with valid free fits, kin_gamma0 (gamma 0) scores 0.7198 / 0.4993 / 0.8643 (mean **0.694**)
against dirty's 0.1182 / 0.2746 / 0.2062 (mean **0.200**): the win is intact and the gap is larger. What is no longer supported is the
pair of headline means, and the two high-dirty points used as evidence that models lose when dirty is already good.
**Read from stored geometry and refits, not from re-running any model**, so kin_gamma0's score on `rt_00` and `rt_04` under a valid
geometry is not yet known; notebook 16 will produce it.

**A second, separate limit found on the way.** Even with a valid geometry, `run_0002_00560_rt_00` (incl 63.7 deg) has dirty at 0.9966:
its residual against the flat Keplerian model is 4.4 km/s rms, dominated by the disk's flaring, and that shared error correlates
between clean and dirty whatever the denoiser does. The wiggle correlation is uninformative on that cube (no gain or loss can be read
from it). Notebook 16 now records `ref_resid_rms` so this is visible.
**Fixed with a metric that shared error cannot fool.** Notebook 16 also reports `resid_err_ratio` = rms(model residual - clean residual) /
rms(dirty residual - clean residual): subtracting clean's residual removes whatever the flat model leaves that clean and dirty share.
< 1 means the model brings the wiggle closer to clean's than dirty is. On that cube's saved maps, correlation gives 0.9907 / 0.9966 /
0.9967 for `sg_k3_fresh` / `winner_aug_seed43` / `winner_beam_seed42` (indistinguishable), while the error ratio gives 1.58 / 0.97 / 0.96:
`sg_k3_fresh` is worse than doing nothing there. (Computed from that preview's maps, which predate the geometry fix, so read it as a
demonstration that the metric discriminates, not as a result.)

**Fixed in notebook 16, not in the old scripts.** `checkpoint_eval.prepare` starts the fit from the emission's shape and, for the
line-emission cubes, holds the inclination at the true `.para` value (what `fit_keplerian` prescribes when it is known independently).
Under that protocol all 5 cubes fit sensibly (centre within 4.2 px, mass 1.2 to 1.75, none at a bound) and dirty's wiggle correlation
is 0.9966 / 0.1161 / **0.0435** / 0.2806 / 0.2165 (mean 0.331, against 0.428): `rt_04` is a heavily degraded cube, not the middling one
the degenerate fit made it. A `geom_ok` flag (fit converged, mass not pinned, centre within a tenth of the disk's extent of the M0
centroid) blanks the wiggle numbers when a fit fails instead of averaging them in (RULES.md #8). A related silent failure found and fixed
in the same pass: the `.para` lookup used `ho["folder"]`, which is a NAME, so the distance had fallen back to 140 pc for every cube and
the inclination was never fixed. The old scripts are left as the record; they are not edited.

---

## 2026-09-25 | added | notebook 16 now compares every checkpoint as images, not only numbers

The first version of 16 produced a results table, five aggregate figures and one stand-alone moment panel per checkpoint on
two cubes: to compare 40 checkpoints by eye you had to open 40 files, and the wiggle, the error and the channel views were
numbers or absent. Now scoring keeps each checkpoint's maps, blue/systemic/red channels, line profiles at three pixels and an
invented-structure map (small: a few MB per checkpoint per case), and `src/evaluation/checkpoint_figures.py` builds, for every
case, side-by-side sheets of ALL checkpoints next to clean and dirty on one colour scale taken from clean, never from a checkpoint:
M0, M1 and M2; error maps (denoised minus clean, scaled by dirty's own error, so "better than doing nothing" is visible); the
Keplerian residual of M1, the wiggle itself; the three channels and the systemic-channel error; an M1 gradient-magnitude
sharpness sheet; an invented-structure sheet; line profiles; the M0 radial profile and the M1 power spectrum against clean's;
and a large detail figure for the best checkpoints by M0. Across the run: the score tables as heat-coloured images, a scoreboard
of every metric (PSNR, SSIM, M0/M1/M2, wiggle, sharpness, invented blobs) with spread across cubes, the loss x source matrix
(Jason's asks 1 and 3), and a checkpoint x cube heatmap. About 20 figures per case, for all 7 cases by default.
Resume restores earlier sessions' maps; a scored row whose maps are missing is redone once so no checkpoint is a hole in a sheet.

Tested with `tests/test_checkpoint_figures.py` (all 19 per-case figure types plus the tables and matrices, from synthetic
artifacts, in seconds) and by building the real figures from three checkpoints on one full cube. Still not run on Kaggle.

---

## 2026-09-25 | bug | 05 scored the 320/480px arms at 256px: ask 2 (less downsampling) is still unanswered

**What was wrong.** Cell 18's `denoise_cube` did `F.interpolate(t, (TARGET_SIZE, TARGET_SIZE))` for every arm, so
`winner_aug_res320` and `winner_aug_res480`, trained at 320 and 480 px, were fed 256 px inputs when their moment maps
were computed. Their moment rows in v42 (M0 -12.3 / +9.7, M1 +50.0 / +66.1, M2 -4.4 / +54.5) are therefore not a test of
resolution. **How it was caught.** Reading `denoise_cube` while checking how the table treated resolution, before writing
up Jason's ask 2. The PSNR path was already right: `train_unet` and the resume path score on the arm's own view
(`VIEWS[view]()[1]`), so 37.36 and 40.18 dB stand, but PSNR does not compare across resolutions anyway.
**Published numbers it touches:** those two moment rows, in the v42 log and in anything quoting them. It does **not**
touch any 256px arm: their scoring size was correct. `winner_aug_native600` is gated off and has no row.

**Fixed.** `denoise_cube(..., size=)` with `_arm_size(name)` (320 / 480 / 600 for the resolution views, else 256), passed
from both callers. The res arms are added to `STALE_MOMENT_ARMS`, the mechanism built for the same class of bug when
`winner_beam` was scored without its beam vector, so their stored rows are dropped and re-scored at the right size on the next
run (checkpoints are in the recovered v38 dataset). Resolution changes and batch size changes together (16 at 256, 8 at 320, 6
at 480, forced by T4 memory, lr unchanged), so even a clean re-score will not isolate resolution; that confound is to be stated.
`checkpoint_eval.py` (notebook 16) takes each arm's size from its label, so it scores them correctly.

---

## 2026-09-25 | run | 05 v40, v41, v42: the loss sweep is done except one arm; MAE helps M2 and M1 more than M0

Three clean sessions, three new arms each (the session cap), archived under `results/05-unet-line-emission/v40_..`,
`v41_..`, `v42_..`. v42's moment table (5 holdout cubes, one seed per arm, clipped + signal-masked, mean across cubes)
gives Jason's asks 1 and 3 their first real answers. **Every number below is a mean over the 5 cubes, from one seed, against
a baseline whose spread is across 3 seeds (RULES.md #6); differences smaller than that spread are not established.**

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| `sweep_winner_aug` baseline (3 seeds) | 39.30 | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| `sweep_winner_p10` baseline (3 seeds) | 39.27 | 33.5 +/-9.6 | 70.7 +/-6.7 | 31.8 +/-11.1 |
| `sweep_winner` non-aug baseline (4 seeds) | 37.52 | 11.4 +/-27.4 | 55.6 +/-9.0 | 6.0 +/-35.1 |
| mae / wavelet / starlet / gradient, fine-tuned from aug | 39.78 / 40.13 / 40.32 / 40.14 | 26.3 / 38.8 / 42.6 / 32.1 | 77.4 / 76.1 / 80.5 / 78.7 | 70.5 / 70.0 / 81.1 / 77.3 |
| same four, fine-tuned from p10 | 40.34 / 40.41 / 40.20 / 40.12 | 43.5 / 41.4 / 44.5 / 41.7 | 79.6 / 78.1 / 77.1 / 80.3 | 83.8 / 73.1 / 79.6 / 70.5 |
| same four, fresh (no aug) | 39.96 / 40.03 / 39.90 / 39.73 | 36.0 / 20.4 / 38.1 / 33.9 | 75.2 / 66.1 / 77.8 / 75.6 | 69.1 / 60.9 / 75.6 / 56.4 |
| beam-sourced (mae / wavelet / starlet / gradient) | 39.99 / 40.12 / 39.62 / 40.22 | 36.4 / 29.3 / 31.5 / 31.2 | 76.9 / 71.6 / 72.4 / 74.1 | 66.4 / 70.9 / 66.2 / 42.9 |
| `winner_k1` (one spectral neighbour) | **42.59** | 33.3 | 75.6 | 40.5 |

**Ask 1, MAE.** It did not lift M0: fine-tuned from aug it is 26.3 against 29.2 (inside the seed spread), from p10 43.5
against 33.5 (about one spread). It lifts M1 (+3.4 and +8.9 pp) and M2 (+15.5 and +52 pp) more clearly. So the simple loss helped,
but on the velocity moments, not on M0.
**Ask 3, the other losses.** All four raise M2 above their source, and the fresh (from-scratch) arms, which have a clean
control in `sweep_winner` (same recipe, same view, only the loss differs), beat it on M1 by +10 to +22 pp (its spread is 9)
and on M2 by +50 to +70 pp. **No loss clearly beats the others**: the M0 range across the four (26 to 45) is inside single-seed noise.
Best M0 is `winner_starlet_p10_ft` (+44.5) and `winner_mae_p10_ft` (+43.5), indistinguishable.
**Unresolved confound.** Every `_ft` arm changes the loss AND continues a converged model for 30+ epochs at 0.1x lr, so part of
its gain may be the extra training. The fresh arms have a clean control; the `_ft` arms did not. Two control arms are now in 05
(`winner_hybrid_ft`, `winner_hybrid_p10_ft`: same source, same budget, original hybrid loss). Until they run, "the loss helped" is
established for the fresh arms and only suggested for the `_ft` ones.
**PSNR again does not rank.** `winner_k1` is far ahead on PSNR (42.59, +2.3 dB) and behind on M2 (40.5 against 55 to 84 for
the loss arms); the notebook's own line says ranking by PSNR would have picked it. That matches 2026-08-21: spectral context is a
pixel win, not a moment win. Beam-conditioned arms show no gain.

**Left in 05:** `winner_k2`, plus the two control arms, exactly one session at the cap of 3. `winner_aug_native600` stays gated.
The RAM leak was mild at 256px (27.7 to 24.7 GB free over 30 epochs, ~0.1 GB/epoch) and cost nothing; the per-epoch
`main/workers/cache` diagnostic is not live on Kaggle because that commit is unpushed.

---

## 2026-09-25 | added | 16-checkpoint-evaluation.ipynb: one protocol for every checkpoint, validated against v20

Until now every checkpoint family had its own scoring script and preprocessing (`score_08_kinematic.py`,
`wiggle_domain_split.py`, `score_sg_wiggle.py`, `m1_rendering_audit.py`), so no two checkpoints had ever been
put through the same checks on the same cubes, and none of the loss-sweep arms had been scored beyond PSNR (13's)
or beyond moments (05's). `src/evaluation/checkpoint_eval.py` gives every checkpoint one interface (single channel,
31-channel stack, 7-channel stack, beam-conditioned U-Net, conditional DDPM) and runs the same battery, composing only
functions that already have published numbers: pixel PSNR/SSIM, `moment_improvement` (M0/M1/M2), the quadratic-M1
Keplerian wiggle against one shared geometry per cube, gradient-energy and Laplacian sharpness of raw M1 as a ratio to
clean, `channel_artifacts` (invented blobs, overshoot, floor leak), the moment-map panel, and the (dirty resid_r, gain)
pair for the headroom scatter. Cross-domain transfer needs no separate code: a line-emission checkpoint on an SG case, and
the reverse, is the same call. Cases: the 5 line-emission holdouts, the SG v2 cube (channels 240-360), and the SG holdout
`run_9074_00025_rt_00` (inclination fixed at 20 deg). Resumable, row persisted the moment it exists.

**Validated before trusting it.** On `run_0002_00560_rt_00`, full 201 channels, `winner_aug_seed43` scores **M0 +31.3,
M1 +77.2, M2 +84.6**, identical to v20's published per-cube numbers for `sweep_winner_aug` seed 43 on that cube. (A first run
on a 60-channel slice gave M0 -60.8 / M1 -76.0: it cut the line. A slice is not a smoke test for moments.)

**Not yet run on Kaggle.** Local smoke test only: four families (aug43, kin_gamma0, sg_k3_fresh, winner_beam) on 60 channels of
one cube, plus the full-cube validation above (~3.5 min CPU for one checkpoint on one cube). Timing on a T4 is unmeasured.
Estimate, to be replaced by the first session's printed timings: inference is seconds per cube, but moment maps and artifact
counting are CPU-side, so roughly a minute or two per (checkpoint, cube) and several hours for the full ~40 checkpoints x 7
cases, across sessions.

**Limits, all stated in the notebook.** DDRM is not scored (it restores through the beam operator, not wired into this
protocol) and is listed with that reason; DDPM runs on one cube only (sampling is ~100x a U-Net forward), 25 steps, one draw.
Every case is continuum-subtracted, so the SG-trained family (trained without it) sees the closest available input on
line-emission cubes. PSNR here is per channel on the dirty-scale normalisation over the holdout cubes, not the 256px validation
set behind the PSNR in 05's tables, so the two are not comparable. SSIM is on every 2nd channel. `val_loss` is never compared
(RULES.md #4). Tables give spread across cubes, not seeds (RULES.md #6).

One thing already visible on the validation cube: dirty's own wiggle correlation is 0.996 there, so no model can gain on it,
the headroom effect again. `tests/test_checkpoint_eval.py` covers naming, discovery and architecture detection.

---

## 2026-09-25 | plan | PLAN.md rewritten around what Jason asked for, plus the ML4SCI org deadlines

The 2026-09-10 plan was built on our own reading of what mattered, and by 2026-09-24 its Block 2 had become a
simulated degradation sweep Jason never requested. Re-reading the 2026-09-12 transcript, the plan now starts from
his nine asks in his words (MAE first, less downsampling to 320/480, wavelet/starlet/NLL, focus on the best two or
three models, inference on real exoALMA fiducial 13CO `.image.fits`, MWC 758, DSHARP second, VLT optional, DS9, and
"a really good ALMA pipeline ... scientific insights" as the success measure), then the org email's dates (blog to
mentors Sep 25 17:00 US Central, PR to `ML4SCI/EXXA` and a 3-minute talk Sep 29, submission Nov 3), then a dated
timeline. Notebook 14 (native 600px), the `simobserve` sweep, NLL and VLT are listed as optional. Block 1 is kept
as history.

Status of his asks found while writing it: MAE done and did not lift M0 (+26.3% against the +29.2% baseline);
wavelet and starlet done and did (+38.8%, +42.6%); 320/480 **trained but not scored on moments**, and PSNR does
not compare across resolutions, so ask 2 has no answer yet; NLL not tried; exoALMA inference built, not run.

---

## 2026-09-24 | run + bug | 05 v38: first winner_aug res arms finished, RAM leak NOT fixed, 09-20 root cause refuted

Pulled `9856f17`, cells current (guard, `SEED_OVERRIDE`, `winner_aug_res*` all present, so
v36's stale-cell failure did not repeat). Archived `results/05-unet-line-emission/
v38_2026-09-24_crashed/`. Killed by host RAM at epoch 19 of the third new arm.

**Finished, the arms the mentor asked for:** `winner_aug_res320` seed 43 PSNR 37.3597 SSIM
0.9932; `winner_aug_res480` seed 43 PSNR 40.1815 SSIM 0.9969. From scratch, augmented,
winner_aug's hyperparameters. PSNR across 256/320/480 is on different pixel grids, so it does
not rank resolutions; only the 600px moments in section 6 do, and this run never got there.
Both are stranded in a failed version's Output (README has the by-hand recovery).

**Correction to 2026-09-20 ("RAM leak root cause found and fixed").** That entry said the
DataLoader fork-storm was the cause and `persistent_workers=True` the fix, and marked it
unverified. v38 is the verification, and it failed: with the fix live, 480px still declines
0.29 GB/epoch, the same as v33's 0.3 before it. Slope also scales with image size (0.11 GB/epoch
at 256px, 0.22 at 320, 0.29 at 480), and no memory is returned at arm boundaries (free RAM 17.1
then 16.4, and 2.3 then 2.3). The fork-storm theory is refuted; `persistent_workers` stays in
because it is harmless, but it is not the fix. The cause is still unknown.

*Published numbers this touches:* none of the PSNR values. But `context.md` and any draft that
says the leak is fixed is wrong, and 05/13/06/07's per-arm cap `MAX_NEW_ARMS_PER_SESSION` is
what keeps runs alive, not a repair. 13 finishing all 28 arms on v17 was the cap working over
many sessions, not evidence the leak is gone.

**Diagnostic added:** `_host_ram_note()` in `src/training/sweep.py` now appends `main / workers /
cache` (this process's RSS, its DataLoader children's summed RSS, kernel page cache) to every
epoch line. Hot-reloads through cell 0b, no cell edit. Whichever column grows with the epoch is
the leak. Not a fix; it makes the next run decisive instead of a third guess.

**Plan:** all remaining 05 arms except the gated `winner_aug_native600` are 256px (about 0.11
GB/epoch, 4 to 5 GB per arm), so three fit one fresh session. If the two res arms are not
recovered they retrain (about 2.2 h and 4.3 h) and must run with `MAX_NEW_ARMS_PER_SESSION = 1`.

---

## 2026-09-24 | run | 13 Version 17 -- all 28 loss-sweep arms complete, pushed as a merge not a clobber

Pushed the res-arm/lr/epoch fixes below, then `git push` rejected: Kaggle had already
pushed `aca35c5` (13 Version 17) to `midterm-prep` in the meantime. Fetched and checked
before merging, rather than force-pushing over it (RULES.md #2's usual failure runs the
other way: Kaggle's push overwriting committed fixes; this time the risk was symmetric,
my local fix overwriting Kaggle's finished training run if merged carelessly).

**v17 finished the whole notebook**: kin 8/8, sg 8/8, ddpm 8/8 (`gradient_ft`/`gradient_fresh`
were the last two, deferred since 09-20), ddrm 4/4 (`ddrm_l2_fresh` the final arm,
`best_val_loss` 24.35863, 50 epochs, no early stop). First time 13 has reached 28/28.

**v17's cells were the same stale-cell disease as 05 v36**, fingerprinted the same way:
`FINETUNE_LR_SCALE` still multiplying the diffusion fine-tune lr (dead code, since
`load_checkpoint` overwrites it either way -- see 2026-09-24 entry above), kin's
`max_epochs=45` not yet raised to 60. Neither affects v17's actual results: the lr fix is
comment-only (the real lr was always the source's saved value), and all 8 kin arms were
`SKIPPED, already done` this version, so `max_epochs` never executed.

**Resolved by merge, not overwrite.** `git checkout --theirs` took Kaggle's file (all 28
arms' outputs intact), then the lr-comment and kin-epoch fixes were reapplied by editing
only `source` fields at the same cell indices -- verified `cell 18`'s outputs (5985 chars)
survived byte-for-byte. Cell-order test still passes. No training data lost, no stale
code kept either.

**Numbers this touches:** none newly wrong. Per-arm PSNR/`best_val_loss` for the 27 arms
besides `ddrm_l2_fresh` are not yet in git -- resumed arms print `SKIPPED, already done`,
not their value, so pull `nb13_*.csv` from v17's Kaggle Output before quoting any of them
in RUNS.md, PROGRESS.md, or a blog draft.

---

## 2026-09-24 | bug | review of 05/13: three recipes were not what the log said they were

A full check of both notebooks against their own claims, before the Sep 25 meeting. Three
real defects, each found by reading what ran rather than what the code comments say.

**1. 05's resolution arms lost winner_aug's recipe on 09-16, and nobody noticed for five
runs.** The 09-15 entry below ("resolution arms rebuilt as winner_aug_seed43's recipe")
was true for one commit. `1832c54` (the host-RAM fix, 09-16) was built on a Kaggle copy of
05 downloaded before `090b5e8` landed, and silently reverted it: RULES.md #2's failure
mode, caused by the assistant, not by Kaggle's push. Caught by `git log -S SEED_OVERRIDE`:
the symbol appears in exactly two commits, the one that added it and the one that removed
it. Every res arm trained in v30 to v34 used the OLD recipe: no D4 augmentation, seed 42,
and `winner_res320`/`winner_res480` fine-tuned from `sweep_winner_aug` at 0.1x lr rather
than trained from scratch.

*Numbers this touches:* `winner_res320` 39.6676, `winner_res480` 40.1817,
`winner_res320_fresh` 38.9608 (09-20 entry, RUNS.md). They are valid measurements of the
old recipe, and must not be quoted as "winner_aug's recipe at 320/480". Fix: `090b5e8`
re-applied by hand (the patch no longer applies, context moved): `winner_aug_res320`/
`winner_aug_res480`/`winner_aug_native600`, WINNER hyperparameters, `augment=True` train
views, seed 43 via `SEED_OVERRIDE`, from scratch, default `min_epochs=50`. New arm names,
so resume trains them and the old rows stay as their own record. `winner_res480_fresh`
(never finished, RAM-killed in v33 and v34) is dropped, the new arms replace it.

**2. Diffusion fine-tune arms never used `FINETUNE_LR_SCALE`.**
`DenoisingDiffusion.load_checkpoint` restores the source checkpoint's Adam state, lr
included, overwriting the constructor's lr. Linear warmup cannot undo it, since `self.step`
comes back at 7656 and warmup only runs below one epoch's steps. Read directly from the
checkpoints (no torch needed, `.pth` is a zip): `ddpm_seed42` saved lr 2e-4, `ddrm_prior`
2e-5. So 13's `ddpm_*_ft` arms trained at 2e-4, not the 2e-5 the notebook claimed.

*Numbers this touches:* `ddpm_l1_ft` PSNR 37.2620 and every other finished `ddpm_*_ft`
arm: the lr is 2e-4, not 2e-5. The numbers themselves stand. Its first fine-tune epoch
shows no spike (val 1673 then 1123) and it beats `ddpm_l1_fresh` (35.7669), so this is
"continued training at the source's own lr under a new loss", a valid fine-tune, just not
the one described. Not retrained: every DDPM ft arm, finished or still to run, gets 2e-4 by
the same mechanism, so the family stays internally consistent, and the DDRM ft arms (not
yet run) get 2e-5 the same way. The fix below is behaviour-neutral on purpose. Fix:
13's dead `* FINETUNE_LR_SCALE` removed from both diffusion cells and the comments now say
what happens; `load_checkpoint` prints the effective lr so the next one is in the log.

**3. 13's kin fresh arms ran 45 epochs, below the 50-epoch minimum.** The kin cell passed
`min_epochs=50` with `max_epochs=45`. `train_unet` stops at max, and early stopping needs
`ep >= min_epochs`, so every kin fresh arm ran exactly 45 epochs with early stopping unable
to fire. *Numbers this touches:* `kin_gamma0_mae_fresh` 30.5207 and
`kin_gamma0_wavelet_fresh` 38.8636, plus the starlet/gradient fresh arms. The 30.52 in
particular may be budget-limited, and only a rerun would say. Fix: `max_epochs=60` (as the
sg cell already had). The finished kin rows are not retrained by this, since resume skips
them; retraining the four fresh arms is a separate, optional ~GPU cost.

---

## 2026-09-20 | bug | RAM leak root cause found and fixed: DataLoader worker fork-storm

05 Versions 33 and 34 (pulled `fe52be5`/Version 32, and `c7083fb`) both died on the exact
same arm, `winner_res480_fresh`, at RAM 0.9/31.3 GB free, no traceback -- host OOM-kill.
Archived at `results/05-unet-line-emission/v33_2026-09-20_crashed/` and
`v34_2026-09-20_crashed/`. Between them: all 12 loss-sweep + res-sweep arms reconfirmed
(V33's `winner_res480` PSNR 40.1817, `winner_res320_fresh` 38.9608; V34 retrained the same
set from scratch since a failed version's Output cannot be attached, same numbers), plus
`winner_gradient_ft` finally has a value (40.1396) after two sessions deferred.

**Root cause, found by inspection rather than another blind cap-and-retry:**
`train_unet`'s `DataLoader`s (`src/training/sweep.py`) never set `persistent_workers=True`.
With Kaggle's `num_workers=4` and PyTorch's default `persistent_workers=False`, BOTH loaders
tear down and respawn all 4 worker processes at the end of EVERY epoch's iteration -- a
fork-storm every epoch, for the whole run. This explains what three prior entries could only
describe: a steady ~0.3 GB/epoch decline, IDENTICAL regardless of arm or image resolution
(matching fixed per-process fork overhead, not data volume), that no `gc.collect()` or
`empty_cache()` call could touch because it isn't Python-heap or GPU memory.

**Fixed**, same pattern, everywhere a `DataLoader` sets `num_workers>0`: `sweep.py`
(`train_unet`, covers 05 and 13's kin/sg sections), notebook 13's DDPM/DDRM cells
(`_nw=4`), and notebooks 06 and 07 (`nw=4`/`num_workers=2`). Notebooks 10 and 11 use
`num_workers=0` (no workers spawned) -- not exposed, left alone.

Not yet re-verified against an actual Kaggle run. Next session's `RAM free` trend across
epochs is the test: flat (or only GPU-driver-noise-level drift) confirms the fix; still
declining means the leak has a second source.

## 2026-09-19 | run | both notebooks finish clean for the first time -- KeyError fix holds, RAM leak margin tightening

05 Version 30 (`22b01ae`) and 13 Version 5 (`6b3e6ea`), both pulled `d93803d`, both verified
against it (cell counts and every fix marker intact, nothing reverted), both archived at
`results/05-unet-line-emission/v30_2026-09-19_22b01ae/` and
`results/13-checkpoint-loss-sweep/v5_2026-09-19_6b3e6ea/`.

**05**: the crashed session's Output (2026-09-18, likely Version 29, never confirmed) was
never attached -- Kaggle can't attach a failed run and it wasn't manually recovered -- so
`winner_mae_ft`/`wavelet_ft`/`starlet_ft` retrained from scratch instead of resuming,
consuming the full `MAX_NEW_ARMS_PER_SESSION=3` cap on arms already done once. Numbers moved
slightly (expected, not a discrepancy) and all three still beat `sweep_winner_aug` (PSNR
39.30, M0 +29.2%, M1 +74.0%, M2 +55.0%) on every metric -- `winner_starlet_ft` (PSNR 40.32,
M0 +42.6%, M1 +80.5%, M2 +81.1%) is now the best arm in this notebook's history on every
metric. Section 6 completed this time; the `KeyError` fix (`d93803d`) held.

**13**: `kin_gamma0_mae_ft`/`_fresh` resumed correctly this time (CSV + checkpoint both
matched, `_import_prior_nb13` worked as designed). Trained `kin_gamma0_wavelet_ft` (PSNR
42.88, now the best kin_gamma0 arm) and `_fresh` (38.86), deferred starlet/gradient and all
of sg/ddpm/ddrm.

**RAM leak still not traced, and the margin is tightening.** 05: 28.2 -> 17.9 GB over 98
epochs (~105 MB/epoch). 13: 27.8 -> 3.3 GB over 86 epochs (~285 MB/epoch) -- closer to
exhaustion than Version 4's run (6.5 GB left at the same cap). The cap is sized in ARMS, not
epochs or GB, so it is not a fixed safety margin -- a future session with epoch-heavier arms
under the same cap could still hit zero. Worth an actual trace before it does.

## 2026-09-18 | bug | 05 crashed on a KeyError the session cap exposed; three loss arms beat winner_aug on every metric

Kaggle version unconfirmed (pulled `1832c54`), notebook downloaded and archived at
`results/05-unet-line-emission/v_pending_2026-09-17_1832c54/` pending the number (RULES.md
#10 -- rename once known). This is a real bug, not host RAM: `MAX_NEW_ARMS_PER_SESSION=3`
correctly trained 3 arms and deferred the rest (`winner_gradient_ft` included), then section
6's moment-map table crashed with `KeyError: 'M0'` trying to print `winner_gradient_ft`'s
row. `band_mom[name]` is built for every `CONFIGS` name unconditionally, valued `{}` when
nothing scored it yet; `if name not in band_mom: continue` never catches an empty-but-present
value, only a missing key. Section 6 onward, `collect_outputs` included, never ran.

Found the identical pattern in section 6d's headline figure (cell 24, `_arms = [n for n in
CONFIGS if n in band_mom]`) before it could cause a second crash on the next run -- fixed
both, same guard (`band_mom.get(name) and all(m in band_mom[name] for m in moments)`).

**Results that survived, before the crash -- all three beat `sweep_winner_aug` (PSNR 39.30,
M0 +29.2%, M1 +74.0%, M2 +55.0%) on every metric:**

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| `winner_mae_ft` | 39.93 | +40.1% | +77.1% | +70.0% |
| `winner_wavelet_ft` | 40.18 | +42.4% | +76.1% | +58.4% |
| `winner_starlet_ft` | 40.14 | +39.3% | +75.8% | +59.9% |

`moment_improvement`, clipped + signal-masked (RULES.md #6), 1 seed each -- not wiggle-scored
yet, this is the ranking metric, not the kinematic diagnostic that actually answers the
smoothing question. First time this project's loss sweep has moment-map evidence, not just
PSNR, and the direction agrees with PSNR for once: real, not just a pixel-metric artifact.

## 2026-09-18 | run | 13's first clean finish -- LR fix confirmed, RAM leak still present but contained

Kaggle Version 4 (exact number unconfirmed, author's next `Add Input` push will settle it),
`afb2bbd` pulled and verified live (`_arm_tag`/`no checkpoint -- will retrain` markers present
in the log). `MAX_NEW_ARMS_PER_SESSION=2` did its job: trained `kin_gamma0_mae_ft` and
`kin_gamma0_mae_fresh`, deferred the other 26 arms, finished clean, `collect_outputs` ran.

`kin_gamma0_mae_ft`: PSNR **42.1972**, SSIM 0.9954, 30 epochs, `lr` correctly scaled to 8.2e-5.
Up from **35.6845** at the broken full-lr run two sessions ago -- the `FINETUNE_LR_SCALE=0.1`
fix (`1832c54`) is confirmed doing real work, not a cosmetic change. `kin_gamma0_mae_fresh`:
PSNR 30.5207 (45 epochs), consistent with the prior fresh run's 30.3696. Neither wiggle-scored
yet; PSNR only, 31-channel val set, not comparable to 05's 1-channel PSNRs (RULES.md #4).

**RAM leak not located, but no longer fatal.** Free RAM 27.8 -> 6.5 GB over the two arms
(~2.3h, 75 epochs combined) -- roughly 280 MB/epoch, matching the slope from the crashed
runs. The session cap stopped training before it reached zero, so this is contained rather
than fixed. Worth a real trace once GPU-hour pressure allows it.

`sg_k3_fresh`/`ddpm_seed42`/`ddrm_prior` source checkpoints all located correctly from
`exxa-13-checkpoint-sources` -- those three sections are ready, just deferred by the cap.

## 2026-09-16 | bug | both notebooks killed by host RAM; fine-tune arms ran at from-scratch lr

**Runs:** 05 Version 28 died at 25367.8s, 13 Version 3 at 11907.4s. Both: Kaggle's "tried to
allocate more memory than is available", `DeadKernelError`, no Python traceback -- host RAM
(the ~30 GB CPU side), not GPU. So `a74022b`'s `gc.collect()`/`empty_cache()` after each arm
was aimed at the wrong memory. 05 v27 died at 25240s and v28 at 25368s despite different arms
and epoch budgets: a steady per-step leak, not one bad arm. 13 v3's `kin_gamma0_wavelet_finetune`
epoch times climbed 110s -> 165 -> 270 -> 334 -> 406s before the kill, the page-cache squeeze
of RAM filling; the two arms before it held ~100s for 75 epochs. **Leak not located.** Reading
`FITSChannelDataset` found no cache and no memmap views surviving `_to_native_float32`.

**Results that did survive (13 v3, persisted, attributable):** `kin_gamma0_mae_finetune` PSNR
35.6845 / SSIM 0.9917 (30 epochs, full lr -- see below, superseded recipe);
`kin_gamma0_mae_fresh` PSNR 30.3696 / SSIM 0.9724 (45 epochs). Pixel metrics only, not
wiggle-scored, 31-channel val set -- not comparable to 05's 1-channel PSNRs (RULES.md #4).

**Second bug, caught in the same log:** fine-tune arms used the arm's from-scratch lr
(8.2e-4). Both v3 fine-tune arms spiked at epoch 3 (mae val 0.0096 -> 0.0369; wavelet 0.0005 ->
0.0055) -- the pretrained weights knocked out, so "fine-tune" became a worse from-scratch run.
Likely also why `winner_mae` fell 39.81 -> 35.89 in v27. Notebook 10 already had
`FINETUNE_LR_SCALE = 0.1` for exactly this reason; 05 and 13 never adopted it. Also in 13:
the DDPM section ran FRESH arms at the fine-tune lr (2e-5), 10x below 06's from-scratch 2e-4.

**Fixed:** `FINETUNE_LR_SCALE = 0.1` in 05 and 13 (U-Net and diffusion); 13's DDPM base lr
2e-4. Fine-tune arms renamed (`_ft`) so v27/v28/v3 full-lr rows stay their own record and
retrain, rather than resume counting them as the current recipe. `MAX_NEW_ARMS_PER_SESSION`
(05: 3, 13: 2) stops a session cleanly before the leak can kill it; next session resumes.
`train_unet` now prints free host RAM every epoch (`/proc/meminfo`) and runs `gc.collect()`
per epoch, so the next log shows the leak's slope instead of just its end.

**Cost reality:** at 2-3 arms per session, 13's 28 arms need ~14 sessions and 05's ~18 new
arms ~6 -- well past one account's 30 GPU-h/week. Trimming scope is the author's call.

## 2026-09-15 | added | 05's resolution arms rebuilt as winner_aug_seed43's recipe, from scratch

Author direction: 320/480/600 should reproduce the best model's recipe exactly, trained from
scratch -- fine-tuning from 256px weights risks carrying that resolution's smoothing into the
new one. Checking the existing res arms against `winner_aug_seed43` found they were NOT the
same recipe: no D4 augmentation (the defining feature of `winner_aug`) and seed 42, not 43.

Replaced `winner_res320`/`_fresh`, `winner_res480`/`_fresh`, `winner_native600`/`_fresh` with
three arms: `winner_aug_res320`, `winner_aug_res480`, `winner_aug_native600`. WINNER
hyperparameters, `augment=True` on the 320/480/600 training views (val un-augmented), seed 43
via a new `SEED_OVERRIDE`, `min_epochs=50`, no `init_from`. One forced deviation: batch 8/6/4
instead of 16 for T4 memory, with lr kept identical rather than rescaled (noisier gradients --
a confound to name when reading the result). Seed 43 was winner_aug's best of three, inside
08's ~1 dB seed spread, so matching it keeps the recipe but won't carry that seed's luck.
`winner_aug_native600` still gated by `RUN_NATIVE600 = False`.

## 2026-09-15 | added | epoch budgets raised: fine-tune 6->30, fresh 20/15/10/6->50, across 05 and 13

`winner_mae`'s only completed fine-tune arm from the crashed run (see the bug entry just
below) stopped at epoch 10 on a 6-epoch floor -- author's call after seeing it: too thin to
trust the plateau, especially on a loss that changed scale entirely (MAE vs the original
hybrid MSE+SSIM). Raised uniformly: every fine-tune arm's `min_epochs` 6->30 (05's 14
loss/resolution arms; 13's kin/sg/ddpm/ddrm sections, replacing the earlier
per-checkpoint-matched floors of 15/10). Every fresh-init arm's `min_epochs` -> 50 (was 05's
shared default of 20; 13's ddpm/ddrm sections' 30).

**Cost, revised up substantially from the 20-35h estimate two entries below:** roughly 5x
the epoch floor on fine-tune arms and 1.7-8x on fresh arms (depending which arm's old
budget). Rough recompute for 05's new arms alone: ~9h floor on the 12 aug/p10/beam-source
fine-tune arms, ~3-4h on res320/res480 fine-tune, ~5h on the four fresh losses, ~4-7h on
res320/res480 fresh -- **21-25h just for 05's new arms' minimums**, before patience can push
any of them longer, and before 13's four sections (also raised) are counted at all. This is
a real, large increase, not a rounding change -- flagging it plainly rather than
understating it because the earlier estimate turned out low once already.

## 2026-09-15 | bug | 05 had no per-arm persistence, 7h run died and lost `winner_wavelet` entirely

`05-unet-line-emission.ipynb` NEVER had per-arm checkpoint persistence to `/kaggle/working`
-- section 9 was the only copy step, at the very end of the whole notebook, a pre-existing
gap that predates this session. Adding 20 loss/resolution-sweep arms onto it pushed one
session to ~7h (`DeadKernelError` at 25240.5s) before section 9 ever ran.

**Caught by:** reading the crash log the user pasted after the session died -- kernel death
traceback plus `papermill` writing `__notebook__.ipynb` (169905 bytes) as the final artifact,
confirming section 9 never executed.

**Consequence:** `winner_mae` seed 42 finished training (PSNR 35.8889) but its checkpoint
was never copied out of the ephemeral git clone -- the number survives only as printed text
in whatever Kaggle version the crash produced, not in a reloadable CSV or `.pth`. Nothing
downstream (moment maps, wiggle scoring) can be built on it without retraining.
`winner_wavelet` seed 42 was 5+ epochs in (val 0.0007, improving) when the kernel died;
`train_unet` only writes `ckpt_path` when the call **returns**, so nothing was ever written
for it at all -- total loss, retrains from scratch. Everything queued after it in `CONFIGS`
order never started. This is exactly notebook 06 v11's incident (RULES.md #1), in a
different notebook, because the fix from that incident was never backported here.

**Fixed:** `persist_ckpt()` defined in cell 6, called after every arm in both the
fresh-training and checkpoint-found-scoring branches of section 4's loop, copying the
checkpoint and `nb05_seed_repeats.csv` to `/kaggle/working` immediately. Cell-order test
passes. Not yet re-verified against an actual Kaggle run -- next session's crash-resistance
depends on this actually being pulled before running (RULES.md #2).

**Not yet done:** the crashed run itself needs archiving per RULES.md #10, failures included
-- Kaggle version number needed from the author, not visible from here.

No published numbers affected -- nothing from this run reached RUNS.md or the blog.

## 2026-09-15 | added | loss sweep expands to all 7 best_models checkpoints, plus a stale 06 restored

Continuation of the same day's earlier entry. Mentor direction (mentee call, 2026-09-12) was
to try MAE/wavelet/starlet/gradient losses and a 320/480px compromise; follow-up direction
this session was to run every arm BOTH fine-tuned from an existing best checkpoint AND fresh
(random init) -- not assumed to answer the same question -- and to cover all 7
`models/best_models/` checkpoints, not just `winner_aug_seed43`.

**`05-unet-line-emission.ipynb`**: the four losses now run from three fine-tune sources
(`winner_aug_seed43`, `winner_p10_seed44` -- highest untested PSNR/SSIM, `winner_beam_seed42`
-- needs the `beam` view) plus a `_fresh` random-init counterpart of each, so a source-name
comparison and a fine-tune-vs-fresh comparison are both possible without conflating them.
`winner_res320`/`winner_res480` get the same fine-tune/fresh split. 20 new arms total, on top
of the original 8. None need an external checkpoint upload -- all three fine-tune sources
train inside this same notebook, earlier in `CONFIGS`' dict order, so the checkpoint exists
by the time the loss arms need it.

**`13-checkpoint-loss-sweep.ipynb`** (new): covers the four checkpoints that don't fit 05's
1-channel line-emission U-Net shape -- `kin_gamma0` (31-ch spectral, kinematic_gamma pinned
at 0 to match its own training), `sg_k3_fresh` (7-ch spectral, self-gravitating domain, not
line-emission -- a scientific stretch, flagged as such rather than assumed to transfer),
`ddpm_seed42` and `ddrm_prior` (both diffusion). The diffusion pair needed a real design step,
not a mechanical loss swap: `noise_estimation_loss` (`src/training/diffusion.py`) gained
`loss_type` ("l1"/"l2" on the noise/v residual, the diffusion analogue of MAE) and
`aux_loss_name`/`aux_weight` (an optional wavelet/starlet/gradient term on the predicted-clean
estimate x0_hat, inverted from the noise/v prediction by the standard DDPM identity --
`WaveletLoss`/`StarletLoss`/`GradientLoss` have no meaning on a noise residual directly).
`AUX_WEIGHT` in the notebook is flagged explicitly as an order-of-magnitude guess (primary
loss sums squared error over ~65k pixels; the aux functions are per-pixel means), not tuned --
inspect the loss curve before trusting it.

Every arm in both notebooks: `finetune` (min_epochs=6, cheap) and `fresh` (min_epochs matching
that checkpoint's own original training budget). Fine-tune checkpoints for 13 staged at
`models/_kaggle-upload/13/` (gitignored, hardlinked, `.ckpt` extension per RULES.md #3) --
still needs manually creating the actual Kaggle Dataset and attaching it, which nothing here
can do without Kaggle credentials.

**Cost, not yet paid:** rough estimate from per-epoch timings elsewhere in this project puts
05's new arms at 7-15 GPU hours and 13's at 10-20+ GPU hours (the diffusion sections are the
uncertain part -- DDIM eval at K_AVG=4/25 steps is not cheap and nothing in this repo gives a
solid per-epoch number for it). 20-35 GPU hours combined exceeds one 12h Kaggle session for
either notebook alone; both resume correctly via their CSV skip-already-done logic, so this
is safe to spread across sessions but is not a one-run cost.

**Bug, separate from the above:** `06-ddpm-line-emission.ipynb`'s working-tree copy (present
before this session started, not produced by it) had reverted three committed fixes against
HEAD -- the `RESCALE_TO_DIRTY` pedestal fix, `RUN_SWEEP` flipped back to `True` (re-running a
sweep already marked "ANSWERED TWICE"), and the entire `## 13b. Diagnostics` section, dropping
cell count 47->45. Exactly the RULES.md #2 failure mode: Kaggle's own push sent a stale copy
back over committed work. Restored from HEAD (`git stash` first, the stale copy recoverable
if this diagnosis is wrong); cell-order test passes at 47 cells. Not yet re-verified against
Kaggle -- next session there needs the same check (rule 2: confirm what got pulled before
building on it).

## 2026-09-15 | added | 05 gains a loss-fn sweep and a resolution sweep, not yet run

Mentor direction (mentee call, 2026-09-12): the smoothing complaint on 256px output may be
loss-driven (try MAE), resolution-driven (600px trains huge, try a 320/480 compromise), or
both. `src/utils/losses.py` gained `MAELoss`, `WaveletLoss` (multi-level Haar DWT), `StarletLoss`
(a trous B3-spline, the transform the mentee already validated on real ALMA data), and
`GradientLoss` (Sobel edge L1, the pix2pix-style term without a discriminator). None need a
new dependency -- all fixed-kernel `conv2d`. `train_unet` (`src/training/sweep.py`) gained
`loss_name` (`LOSS_REGISTRY`), written into the checkpoint so `best_val_loss` stays
attributable to the objective that produced it (RULES.md #4/#6).

`05-unet-line-emission.ipynb` gained six exploratory (single-seed) arms: `winner_mae`,
`winner_wavelet`, `winner_starlet`, `winner_gradient` (all 256px, WINNER config, loss_name
swapped), `winner_res320`, `winner_res480` (WINNER config, new 320px/480px dataset views,
smaller batch). `winner_native600` stays gated off (`RUN_NATIVE600 = False`, RULES.md #1 --
has never once completed). None of the six are gated; they run by default alongside the
existing arms next session.

**Bug caught before running anything:** the "checkpoint found, scoring without retraining"
resume branch only special-cased `n_neighbors`/`use_beam` when picking the val set, so a
resumed `res320`/`res480`/`native600` run would score PSNR on the 256px `val_ds` while a
freshly-completed run of the same arm scores on its own view's val set -- same config name,
two metric bases, exactly what RULES.md #4 exists to prevent. Fixed: the resume branch now
uses `VIEWS[view]()[1]` for those three arms.

**Nothing measured yet.** Cell-order test passes; no Kaggle run has happened. Next: push
through Kaggle, verify a marker (`winner_gradient` in the loaded cell) per RULES.md #2, run.

## 2026-09-12 | added | PLAN.md and RUNS.md caught up to Phase J's close

Two doc gaps found while updating context.md yesterday, closed today. `PLAN.md`'s Block 2
still described the original single-cube ALMA verdict, which the headroom scatter now
predicts would likely just be an eighth instance of "doing nothing wins" -- real DSHARP dirty
data tends to sit in the low-degradation regime this project's own figure says models lose in.
Rewrote Block 2 as the degradation-axis redesign discussed 2026-09-11: inject the synthetic
signal into real DSHARP dirty at several noise/config levels, score recovery the same way
`headroom_scatter.png` does. Week 3 changes from a free overrun buffer to part of the sweep
itself, since this needs more `simobserve` runs than one. Exit criterion changed from "one
cube scored" to "at least 3 points on a real-noise degradation-vs-gain curve."

`results/RUNS.md` never had a section for notebook 12 (spectral context, the `k=3` result the
entire Phase J closing chain was triggered by) -- only `models/README.md` documented it.
Added, matching notebook 11's format, including the SG v2 non-transfer result and a link back
to the six-test chain. Flags an open item: notebook 12 has never been archived per RULES.md
#10 (no `results/12-sg-spectral-context/v<N>_.../` folder exists), unlike every other
notebook in the project. Deferred, not fixed today -- close before Block 3's cleanup pass.

---

## 2026-09-11 | run | headroom scatter: the SG thread's unifying figure, built from existing data only

Assembled from three already-logged JSON/PROGRESS sources, no new denoising: `sg_v2` (6
checkpoints, one cube, today's leaderboard + domain-split runs), `sg_loo_wiggle.json` (5 folds
x 3 trained arms, each fold's own dirty), `nb08_kinematic_wiggle.json` (5 cubes x
gamma=0/0.1, each cube's own dirty). Real per-point data, not per-regime means -- 31 points
total, each plotted at its OWN dirty resid_r rather than a pooled average, since the x-axis
IS the quantity the thesis is about and pooling would hide the within-regime spread that
makes it legible.

`experiments/headroom_scatter.py` -> `results/self-gravitating/headroom_scatter.png`.

Mean kinematic gain (model resid_r - dirty resid_r) by regime:

| regime | n | x range (dirty resid_r) | mean gain |
|---|---|---|---|
| line-emission holdouts (gamma=0/0.1) | 10 | 0.118 - 0.996 | **+0.391** |
| SG leave-one-out folds (frozen/finetune/fresh) | 15 | 0.468 - 0.996 | -0.089 |
| SG v2 cube (6 checkpoints) | 6 | 0.862 (fixed) | -0.163 |

**The scatter shows a monotonic decline in gain as dirty's own resid_r rises, not just three
separated clusters.** Every line-emission point sits above zero except one (`run_0002_00560_
rt_00`, dirty resid_r=0.996 -- the near-undetectable-wiggle cube flagged back in the original
kinematic-gamma sweep entry, sitting at gain~0 because there is no headroom there either,
consistent with the trend rather than an exception to it). LOO folds straddle zero, tilting
negative as their own dirty rises within the regime (compare the frozen arm at dirty=0.468,
gain=-0.01, against the same arm at dirty=0.723, gain=-0.08). SG v2's six points cluster at
the high-dirty end, all negative.

**This is the figure that reconciles the project's two headline results without either one
being wrong.** "Doing nothing beats every model" (SG v2, Phase H onward) and "kin_gamma0
crushes dirty" (notebook 08, today) are the same mechanism measured at opposite ends of one
axis, not a contradiction requiring a tie-breaker. Model value here is a property of how
degraded the input already is, not of architecture, training domain, or loss function --
every one of which was tested and shown NOT to explain the SG v2 losses in the preceding four
entries. This closes the SG investigation thread that opened with "why is the denoised image
smooth" three system-turns ago.

---

## 2026-09-11 | run | DDPM K_AVG=1 on SG v2: deliberately not run

The row was queued after v13 diagnosed the DDPM's moment collapse as posterior-mean averaging
over K draws, and K_AVG=1 is the natural counterfactual. Not run here because no outcome would
change a decision: DDRM, the diffusion family's measurement-consistent member, is the worst
row on this cube (0.568), so the family verdict on kinematics is already in from v13 and the
DDRM comparisons; and on this cube the headroom term (dirty at 0.862, headroom 0.109 to the
target ceiling) dominates any sampler-level effect, so a wiggle number here would confound the
pedestal mechanism with the input-quality regime rather than isolate it. Best realistic case
is DDPM moving from worst to mid-pack, still below dirty -- a seventh losing row bought at the
cost of porting the DDIM sampler (schedule reconstruction, EMA weights, normalisation
inversion), the exact class of work that has hidden a bug every time in this project (the
DotDict pickle break, the missing F import, the silent `beam=None`). The pedestal test belongs
where it was diagnosed, on the line-emission moment collapse, and remains open there. Not-run
with rationale is recorded instead of a number that cannot be interpreted.

---

## 2026-09-11 | run | leaderboard extends: winner_p10 and winner_beam also below dirty, beam OOD confirmed with real numbers

`experiments/wiggle_leaderboard_sg.py`, two more line-emission checkpoints on the SG v2 cube,
same protocol as every run in this thread. `winner_beam`'s beam vector taken directly from the
SG dirty header via `beam_features_of()`, confirmed real and non-zero before running
(`[0.983, 0.185, 0.160, 0.100]` -- BPA/BMAJ/BMIN are present in the header), not a degenerate
zero-vector test. `beam_dim=4` verified against the checkpoint's own stored value before
forcing it, and the state dict load asserted exact (no `strict=False` skip-on-mismatch).

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8620 |
| winner_aug_seed43 | 0.218 | 0.9821 | 0.7603 |
| winner_p10_seed44 | 0.219 | 0.9757 | 0.6912 |
| winner_beam_seed42 | 0.210 | 0.9794 | 0.7113 |

Both land below `winner_aug`, both below dirty. `winner_beam`'s residRMS (0.210) is actually
the lowest of the three trained models, closer to dirty's 0.213 than either of the others --
and it still loses on resid_r. Another instance of the RULES.md #6 lesson: RMS and the
wiggle's resid_r are not interchangeable, a method can look best on one and mid-pack on the
other because RMS doesn't see whether the residual PATTERN survives.

**Now six checkpoints tested on this cube** (winner_aug, winner_p10, winner_beam, sg_k3_fresh,
kin_gamma0, DDRM), spanning 3 architectures, 2 domains, 2 loss variants, a patience/early-stop
change, and a beam-conditioning arm run out of its training distribution. All six land in
0.568-0.760, all six lose to dirty's 0.862. This is now a strong pattern, not a handful of
coincidences: nothing tried has closed even half the gap to doing nothing on this specific
cube. The standing read from the last two entries holds and strengthens -- the SG v2 cube
cannot rank models, and the objective (or the input-quality regime, see the headroom table in
the domain-split entry) is where the remaining gap lives, not any one architecture or
checkpoint choice tried so far.

DDPM K_AVG=1 remains the one queued, unrun row (needs porting the DDIM sampler from
06-ddpm-line-emission.ipynb, not attempted inline).

---

## 2026-09-11 | run | rendering audit: the smoothness is in the M1 pixels, the renderer is exonerated

Before running any more checkpoints, settled whether the smooth look in the published M1
panels is in the arrays or in the mask/contour rendering. Logically the audit could only ever
change a presentation question -- mask, moment estimator and rendering are applied identically
to every panel, so they cancel in a between-method comparison and cannot manufacture the
dirty-0.862-vs-U-Net-0.760 gap, which is computed on raw arrays before any renderer runs.
`clean` is the control: if clean stays sharp through the same estimator, mask and rendering
while the U-Net goes smooth, the renderer is not the cause.

**First, a grep that shortcuts most of the question:** `wiggle_all_methods.py`,
`wiggle_patch_unet.py` and `wiggle_domain_split.py` contain **no `gaussian_filter` and no
`contour` call at all**. Every published panel is a raw `imshow` of the raw M1 with NaN
outside the mask. The contour-smoothing code in `moment_maps.py:435` is in a different
function that none of these scripts call. Contouring was never in the path.

`experiments/m1_rendering_audit.py`, plot-independent sharpness on the raw M1 arrays:

| cube | gradE (masked) | gradE (all px) | lapvar (frame) | hf>1/10 | hf>1/5 |
|---|---|---|---|---|---|
| clean | **4.994e-03** | 1.073e-03 | 7.569e-03 | 0.0308 | 0.0182 |
| dirty | **3.029e-03** | 1.447e-03 | 4.807e-03 | 0.0292 | 0.0172 |
| winner_aug (resize) | **1.928e-03** | 1.488e-03 | 5.324e-03 | 0.0289 | 0.0172 |

**Verdict: the masked gradient energy is decisive and the renderer is cleared.** Inside the
mask the U-Net's M1 carries 0.386 of clean's gradient energy against dirty's 0.607 -- a real,
large, monotonic difference in the RAW arrays with no mask or contour involved in the metric.
The ordering clean > dirty > U-Net matches both the visual impression and the resid_r ordering
(1.0 > 0.862 > 0.760) exactly.

**Honest note on the other three metrics: they are uninformative here, and should not be
quoted as if they corroborate.**
- `gradE (all px)` and `lapvar` are computed outside the mask as well, where M1 is fitted to
  noise and has no physical content. Both dirty and the U-Net exceed clean there simply
  because both carry noise where clean has none. The pre-registered mask-artifact trigger was
  "masked and all-pixel disagree for the U-Net ONLY"; they disagree for dirty (1.348) and the
  U-Net (1.386) almost equally, so the trigger does not fire and there is no mask artifact --
  but the column establishes that, not the main claim.
- The `hf` spectral fractions land within 0.0289-0.0308 for all three cubes. The mask-zeroing
  injects a hard edge whose broadband power dominates the FFT, and the mask is shared, so the
  ratio is set by the mask edge rather than by the disk. A null from a metric that turned out
  to be measuring the wrong thing, not evidence of similarity.

The 4x3 rendering matrix (`m1_rendering_audit.png`) shows the same ordering under imshow
no-mask, imshow masked (the published path), contourf-24 no-mask and contourf-24 masked. The
unmasked imshow column is the clearest: clean is visibly mottled with spiral texture, dirty
intermediate, the U-Net a smooth dipole with almost no texture. `_60lev.png` confirms level
quantization at 60 levels does not recover structure in the U-Net panel.

**Consequence: every published M1 figure stands as-is.** No reissue needed. The smoothness
readers see is a property of the model output, which is what those figures were drawn to show.

---

## 2026-09-11 | run | spectral context does NOT transfer to the SG v2 cube -- every trained checkpoint lands ~0.10 below dirty

Ran `kin_gamma0` (notebook 08's 31-channel stack, the only checkpoint with a large measured
win where there was headroom) on the SG v2 cube, through its own training preprocessing
(`subtract_continuum=True`, 31 clamped neighbours, min-max shared from the centre dirty
channel, `stack_target=True` so only the centre output channel is read, continuum added back
so the result lands in the same raw space as clean/dirty). 8.8 min on mps.
`experiments/wiggle_domain_split.py`, extended to carry all three checkpoints.

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| **dirty (do nothing)** | 0.213 | 0.9892 | **0.8620** |
| `winner_aug_seed43` (line-em, 1 channel) | 0.218 | 0.9821 | 0.7603 |
| `kin_gamma0` (line-em, 31-channel stack) | 0.217 | 0.9814 | 0.7581 |
| `sg_k3_fresh` (SG, 7-channel stack) | 0.242 | 0.9783 | 0.7053 |

**Spectral context does not transfer.** `kin_gamma0` scores 0.7581 here against
`winner_aug`'s 0.7603 -- indistinguishable, and slightly worse. The same checkpoint scores
0.8155 against its line-emission holdouts' dirty at 0.4284. The mechanism that wins there
does nothing here.

**The decisive pattern: three architecturally different models (1 / 7 / 31 input channels),
two training domains (line emission, self-gravitating), two loss variants (MSE, MSE +
kinematic term) all land in 0.705-0.760, and every one of them loses to dirty's 0.862.** The
model-to-model spread (0.055) is about half the gap from the best model to doing nothing
(0.104). When every model in a family that different clusters that tightly and all fall on the
same side of the baseline, the result is a property of the SETUP, not of any model's design.
The M1 panels show it directly: all three are visibly smoother than both clean and dirty and
nearly identical to each other.

**What is now tested and ruled out as the explanation**, each with a number:

| hypothesis | test | verdict |
|---|---|---|
| inference-time resize | patch inference | real but 0.025 |
| training resolution ceiling | clean through the 600->256->600 target path | 0.029, scores 0.9714 |
| training domain mismatch | SG-trained model on SG cube | refuted, it did WORSE |
| architecture / spectral context | `kin_gamma0` (k=15) on SG cube | refuted, matches the 1-channel model |

What remains is the objective and the benchmark's headroom, and those two are not separable
with what is on hand. Note the one genuinely non-MSE method tried on this cube, DDRM, is the
worst result in the entire comparison (0.568), so "swap the objective" is a hypothesis with a
discouraging first data point, not a demonstrated fix. State it that way in the writeup.

**Practical consequence: the SG v2 cube cannot rank models.** Every checkpoint scores
0.70-0.76 on it. It is a benchmark that says "do not denoise this cube", not one that says
which denoiser is better. Model selection has to happen on the line-emission holdouts and the
leave-one-out folds, where dirty sits at 0.428 and 0.664 and there is room to distinguish.

---

## 2026-09-11 | run | domain split: SG-trained loses to line-emission-trained on the SG cube -- domain is NOT the 0.186

Splits the 0.186 "loss/learning" term from today's resolution-vs-loss test into DOMAIN
(`winner_aug_seed43` trained on line emission, run on an SG cube) versus OBJECTIVE (MSE
posterior-mean averaging). `experiments/wiggle_domain_split.py`, same cube, same frac=0.05
mask, same shared geometry (mstar=0.643, matches every prior run of this comparison), each
checkpoint through its OWN training preprocessing read off the training code -- notebook 12's
`subtract_continuum=False`, 7 clamped neighbours, min-max shared from the centre dirty
channel, resize 256.

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8620 |
| `winner_aug_seed43` (line-emission trained) | 0.218 | 0.9821 | 0.7603 |
| `sg_k3_fresh` (SG trained, k=3) | 0.242 | 0.9783 | **0.7053** |

**The SG-trained model is WORSE here than the line-emission one, and both lose to dirty.**
Domain is not the explanation for the 0.186. `sg_k3_fresh`'s residual RMS (0.242) also
exceeds clean's own (0.233), so it is inflating residual amplitude rather than recovering
signal -- the same signature DDRM shows, and the M1 panel is visibly broader/smoother than
`winner_aug`'s.

**The cross-check is the real finding.** `sg_k3_fresh` BEAT dirty on its own notebook 12
holdout (0.681 vs that cube's dirty at 0.594). Here it LOSES to dirty (0.705 vs 0.862). Same
checkpoint, opposite verdict, different benchmark cube. What changed is how degraded the
input is:

| benchmark | dirty's own resid_r | headroom to the ~0.971 target ceiling |
|---|---|---|
| line-emission holdouts (notebook 08, n=5) | 0.4284 | ~0.54 |
| SG leave-one-out folds (notebook 11, n=5) | 0.6644 | ~0.31 |
| **SG v2 cube (this comparison)** | **0.8620** | **~0.109** |

**The SG v2 cube is by a wide margin the least-degraded benchmark in the project**, and it is
the one every "the model loses to dirty" headline comes from (Phase H onward). When dirty
already sits at 0.862, any model that smooths at all loses, because the smoothing is a
systematic error aligned with the signal while the noise it removes is not. When dirty sits
at 0.43, the same class of model wins by a wide margin (`kin_gamma0`: 0.8155).

**Consequence for the standing conclusions.** "Doing nothing beats both models" remains true
AS MEASURED, but its scope is now much narrower than it has been stated: it is a statement
about this one unusually clean cube, not about the models in general. The in-domain results
(notebook 08's `kin_gamma0`, notebook 12's k=3 on its own holdout) are not in conflict with
it and never were -- they are measured where there is headroom to win. Both belong in the
writeup, together, with the headroom column attached. Quoting either alone misrepresents the
result.

Objective/architecture remains the lever for the part that is real (spectral context, twice
confirmed). Domain: refuted. Resolution: 0.029. Inference resize: 0.025.

---

## 2026-09-11 | run | resolution-vs-loss split test: the objective is 6x the resolution term, native-res retrain is NOT the fix

Cheap decisive test, minutes of CPU, no training. Ran the CLEAN SG cube through the exact
training-target path (bilinear 600->256->600, per channel, what `FITSChannelDataset` does to
every clean target) and scored it against untouched clean with the usual shared geometry
(fit on clean, mstar=0.643, `at_bound=False`), frac=0.05, channels 240-360 step 1.

| signal | residRMS | resid r |
|---|---|---|
| clean (reference) | 0.233 | -- |
| clean@256 (the training-target path) | 0.228 | **0.9714** |
| dirty | 0.213 | 0.8620 |
| dirty@256 | 0.213 | 0.8617 |
| U-Net patch (native-res inference) | 0.211 | 0.7853 |
| U-Net resize (existing path) | 0.218 | 0.7603 |

**Decomposition of the model deficit:**

| term | cost in resid_r |
|---|---|
| resolution ceiling (clean -> clean@256) | 0.029 |
| loss / learning (clean@256 -> U-Net patch) | **0.186** |
| inference resize (patch -> resize) | 0.025 |

**The objective term is about 6x the resolution term.** A native-resolution retrain, the
GPU-weeks option that has been deferred twice, can recover at most 0.029 of resid_r on this
diagnostic. That is not where the deficit lives. The network is realizing only 0.785 of the
0.9714 its own 256px training targets already permitted, so the ceiling was never the data.

**Correction to this morning's entry ("resize round trip quantified: 89.6% of real structure
lost").** That measurement stands and is not withdrawn: a pure 600->256->600 round trip does
keep only 10.4% of the clean channel's Laplacian-variance sharpness. But that entry let a
pixel-scale SHARPNESS metric imply the resize was also the dominant cause of the WIGGLE
deficit, and it is not. `dirty@256` scores 0.8617 against `dirty`'s 0.8620 -- the resize is
worth 0.0003 on the wiggle. The GI wiggle evidently lives at a spatial scale coarser than the
~2.3px Nyquist cutoff a 2.34x downsample imposes, so it survives almost intact. Two different
questions with two different answers: **the resize costs visual sharpness, the objective costs
the wiggle.** Both matter, they need different fixes, and they should never again be quoted as
one problem.

**What this redirects.** Objective/architecture changes, not resolution, are where the GPU
time belongs -- which is exactly the direction notebook 12's spectral context (k=3, 0.681 vs
dirty's 0.594) and notebook 08's `kin_gamma0` (0.8155 mean vs dirty's 0.4284, in-domain) have
independently been pointing all along. Those two results and this test now agree.

Patch inference stays worth keeping (0.025, free, no retraining), it is just a small term, not
the lever.


## 2026-09-11 | run | `wiggle_patch_unet.py`: native-resolution patch inference, tested against the resize baseline

Direct test of the resize-vs-quality hypothesis from earlier today, on `winner_aug_seed43`
(no retraining, same checkpoint both ways). Same SG cube, same shared Keplerian geometry
(fit on clean, mstar=0.643, matches every prior run of this comparison), same frac=0.05 mask.
9 tiles/channel, Hann-window overlap-blended, ran on `mps`, resize-based 0.2 min, patch-based
1.9 min (the tile-count multiplier as expected).

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8620 |
| U-Net (resize, existing) | 0.218 | 0.9821 | 0.7603 |
| U-Net (patch, native res) | 0.211 | 0.9834 | **0.7853** |

**Real, measurable gain (resid_r +0.025, ~3% relative), but does not close the gap to dirty's
own 0.862, let alone approach clean.** This is the expected outcome given today's other
finding: `FITSChannelDataset` resizes to 256 on the TRAINING side too, for every checkpoint in
this project, so `winner_aug_seed43` has never been asked to represent detail finer than
256px. Patch inference removes the inference-side resize and recovers what the network
already learned, it cannot recover detail the network was never trained to produce. Confirms
the ceiling is primarily set during training, not at inference -- a native-resolution (or
patch-based) retrain is the only way past this, still deferred given the Nov 2 deadline and
Block 2 (ALMA) not yet started.

Figure: `results/self-gravitating/wiggle_patch_vs_resize.png`. Visually, the patch residual
panel carries more fine texture than the resize panel, consistent with the small quantitative
gain, still visibly coarser than dirty's own residual.

**Worth keeping regardless of the retrain decision**: patch inference is strictly better than
resize at zero extra cost (no retraining, ~10x runtime for this cube, U-Net only, not
DDRM-prohibitive). Candidate to become the default inference path for `winner_aug_seed43`,
not yet swapped into `wiggle_all_methods.py`/`score_08_kinematic.py`.

---

## 2026-09-11 | run | notebook 08 kinematic_gamma figures land, first run on mps

`figures_08_kinematic.py`, ran clean end to end on `device mps` (the Apple GPU fix landed
mid-way through the main sweep, this is the first script to actually pick it up). Denoised
one representative holdout cube (`run_0002_00560_rt_00`, the mstar=18.69 outlier fit flagged
earlier -- worth rerunning `--cube 1` or later for a less unusual example at some point) at
all 4 gammas, ~36 min total (9-11 min/gamma), faster than the CPU-only estimate but not the
dramatic speedup hoped for; mps has its own overhead on convolution-heavy nets.

`results/self-gravitating/nb08_kinematic_moments.png`: visually confirms the numbers.
gamma=0/0.1's M0 keeps the dirty cube's lobe structure while cleaning up the halo; M1 keeps
the smooth blue/red dipole. gamma=1's M0 nearly loses the lobes, M2 saturates solid yellow
across the whole field (same total-saturation failure mode already seen in DDPM's
sampler-collapse bug, worth noting as a recurring pattern in how these models fail, not
necessarily the same cause). gamma=10 is visibly noisy/grainy throughout, matches its high
per-cube variance in the numbers.

`results/self-gravitating/nb08_kinematic_vs_gamma.png`: one nuance the summary table alone
didn't show -- gamma=10 partially recovers above gamma=1 on 2 of 5 cubes (not a clean
monotonic collapse), though the mean stays well below gamma=0/0.1 either way. Confirms this
is a real per-cube effect, not a mean-hiding-a-clean-trend situation.

---

## 2026-09-11 | bug | resize round trip quantified: 89.6% of real structure lost, not the network's fault primarily

Follow-up to the 600->256->600 resize finding logged earlier today. That entry named the
resize as a second smoothing source alongside the known MSE regression-to-mean effect, but
did not separate how much each actually contributes. Isolated it directly: took the `clean`
channel (300) from `kinematic_data_v2`, noise-free, so any loss measured is genuine structure,
not noise being smoothed away, and ran it through a PURE bilinear 600->256->600 round trip
with NO network involved at all. Sharpness measured as Laplacian variance (`scipy.ndimage.
laplace`, standard no-reference high-frequency-content metric), cropped to the disk region
(the full 600x600 frame is mostly empty background and swamps the metric otherwise).

    clean, native 600x600:                 baseline
    clean, after resize round-trip only:   10.4% of native sharpness

**89.6% of the clean image's own real structure is destroyed by resize alone, zero network,
zero noise.** This is the dominant smoothing source, larger than the MSE regression-to-mean
effect flagged earlier, which only ever acts on an already-blurred 256px image -- the network
cannot recover detail that resize already discarded before inference began. Downsampling
600->256 is a 2.34x reduction, which by Nyquist removes everything finer than ~2.3px, and
that is exactly the scale spiral-wiggle substructure lives at in these disks.

**SCOPE CORRECTION (same day, see the split-test entry above):** this 89.6% figure is a
pixel-scale SHARPNESS measurement and stands as such, but it does NOT explain the wiggle
deficit. `dirty@256` scores resid_r 0.8617 against `dirty`'s 0.8620, so the resize is worth
0.0003 on the wiggle diagnostic. The wiggle lives at a coarser scale than the ~2.3px cutoff.
Read this entry as "resize costs visual sharpness", never as "resize costs the wiggle".

**Bigger implication: this is not only an inference-time artifact.** Every training pipeline
in this project (`FITSChannelDataset`, `target_size=256` default) resizes both dirty AND
clean to 256 for every training sample, on both line-emission and self-gravitating cubes,
both native 600x600. So no checkpoint currently in `models/best_models/` has ever been asked
to represent detail finer than 256px resolution, training-side. Patch-based inference
(`wiggle_patch_unet.py`, built earlier today, not yet run -- CPU still occupied by the
kinematic_gamma sweep and its figure generation) can only recover what a 256px-trained
network actually learned to reconstruct; it removes the inference-side resize but not the
training-side resolution ceiling underneath it. A full fix needs native-resolution (or
patch-based) TRAINING, not just inference, which is the "full retrain" option already
discussed and deferred given the Nov 2 deadline and Block 2 (ALMA) not yet started.

---

## 2026-09-11 | run | notebook 08's kinematic_gamma sweep, all 4 checkpoints scored on moments and the wiggle, 5 holdout cubes complete

`experiments/score_08_kinematic.py`, full run, 282 minutes on local CPU (mps fix landed
mid-sweep so this run itself stayed on the old CPU-only code path; the next run of this kind
gets the speedup). All 5 line-emission holdout cubes' Keplerian geometry fit free (no known
inclination for these, unlike the SG disks) and all 5 came back with `mstar_at_bound=False`
(RULES.md #8 checked, not assumed) -- one fit (`run_0002_00560_rt_00`, mstar=18.69) is a wide
outlier worth flagging but still clear of the 45 Msun (0.9x bound) cutoff.

Summary across the 5 holdouts (mean +/- std):

| gamma | M0 | M1 | M2 | wiggle resid_r |
|---|---|---|---|---|
| 0.0 (control) | -17.2+/-47.2 | +66.8+/-30.1 | +52.8+/-28.1 | **0.8155+/-0.2106** |
| 0.1 | +1.6+/-89.1 | +44.7+/-46.1 | +19.8+/-39.4 | **0.8242+/-0.1713** |
| 1.0 | -200.1+/-137.2 | -189.6+/-170.7 | -222.4+/-134.5 | 0.2323+/-0.2605 |
| 10.0 | -539.7+/-444.8 | -212.5+/-242.3 | -128.3+/-145.3 | 0.3728+/-0.4249 |
| dirty (no model) | -- | -- | -- | 0.4284+/-0.3556 |

**Headline: gamma=0 and gamma=0.1 both crush dirty on the wiggle (0.82 vs 0.43), but they are
statistically indistinguishable from each other.** gamma=0 is the architecture control --
`n_neighbors=15, out_channels=31`, the 31-channel spectral-context stack, with NO kinematic
loss term at all -- and it already gets 0.8155. Adding the kinematic loss at gamma=0.1 moves
that to 0.8242, well inside gamma=0's own std. **The real lever here is the wide spectral
context (31 neighbouring channels), not the kinematic loss term**, which mirrors notebook
12's finding on the self-gravitating side (spectral context wins, at a much smaller k=3).
This is a second, independent confirmation of the same mechanism on completely different data
(line emission vs self-gravitating) and a completely different architecture entry point
(fixed k=15 neighbour stack vs a k-sweep). The kinematic loss term itself is NOT shown to
help by this sweep -- it is not shown to hurt at gamma=0.1 either, that is the honest
reading, not "kinematic loss works."

gamma=1.0 and gamma=10.0 collapse hard, consistent with their exploding val_loss during
training (0.000721 -> 0.001721 -> 0.016901 -> 0.089597): the kinematic loss term dominates
the objective at high gamma and wrecks pixel-level reconstruction, which then wrecks moments
and the wiggle along with it. Not a subtle effect -- gamma=10 goes as low as -1260% M0 on one
cube.

One run-time anomaly: gamma=0.1 on `run_0002_00560_rt_01` took 69.5 minutes against every
other cube/gamma pair's 6-14 minutes. Not investigated further (the sweep as a whole still
finished and every number it produced is internally consistent with its neighbours), most
likely CPU contention from other work running on the same machine during that window rather
than anything wrong with that specific checkpoint or cube.

Saved to `results/self-gravitating/nb08_kinematic_wiggle.json`. Figures
(`figures_08_kinematic.py`) run immediately after, per the standing instruction not to run
them in parallel with the main sweep.

---

## 2026-09-11 | bug | U-Net/DDRM inference carries a 600->256->600 resize round trip, a second smoothing source beyond the loss function

Looking closely at `wiggle_all_methods.png` (regenerated by the notebook 09 v3 Kaggle rerun
below): the U-Net and DDRM M1 panels look visibly smoother than dirty and beam-only, more
than the known MSE regression-to-mean effect alone accounts for. Traced it to a second,
independent cause.

`self-gravitating cube and dirty cube/kinematic_data_v2/clean_sg.fits` is native
**600x600px**. `experiments/wiggle_all_methods.py`'s `unet_denoise()` and `ddrm_restore()`
both bilinear-resize the input down to `SIZE=256` before running the network, then resize
the output back up to 600x600 (`wiggle_all_methods.py:70,72,91,99`). That is a 2.34x
downsample followed by an upsample that cannot recover what was thrown away -- real spatial
detail below about 2.3px is gone before the network ever sees it, and the upsample just
interpolates smoothly between the coarse 256-grid points. `dirty` and `beam-only` never go
through this path; they stay at native 600x600 the whole time, only convolved with the beam
kernel. So this resize artifact hits exactly the two "denoised" columns and nothing else in
that comparison, which is consistent with what the figure shows.

Checked for other smoothing sources before settling on this explanation: the only
`gaussian_filter` call anywhere in `src/evaluation/` is `moment_maps.py:435`, and its own
comment says it draws a smoothed COPY for contour lines only, the displayed pixels are
untouched -- and this script does not even call that function, it builds its own `imshow`
panels directly from `rows[tag]["m1"]`. Ruled out, not the cause here.

**Two ways to fix it, neither of them a quick toggle:**
1. Retrain at native 600x600 (or patch-sized) resolution. Architecturally possible -- 600
   divides cleanly by 8, matching the current 3-downsample `MULTS=(1,2,4,8)` U-Net -- but
   every `TARGET_SIZE`/`SIZE` constant across notebooks 05/06/07/08/10/11/12 and
   `src/data/fits_cube_dataset.py`'s resize step would need to change, and every existing
   checkpoint would need retraining from scratch. Not something to take on lightly this
   close to the Nov 2 deadline with Block 2 (ALMA validation) not yet started.
2. Patch-based inference at native resolution using the checkpoints already trained:
   crop overlapping 256x256 patches directly out of the 600x600 cube (no resize needed, a
   patch already matches the net's trained input size), denoise each patch, stitch with
   overlap-blending. No retraining, only touches the inference/scoring scripts. Agreed as
   the next step to try, since it isolates whether resize is the dominant smoothing source
   without committing to a full retrain first.

Not yet implemented. Logging this now, before ALMA validation starts, since the same
inference pattern (global resize to a fixed square) is used everywhere in the scoring
scripts, not just this one comparison.

---

## 2026-09-10 | run | figure for notebook 11's leave-one-out wiggle (metrics-only, no re-denoise)

Notebook 11 had a JSON (`sg_loo_wiggle.json`, 5 folds) but no figure -- flagged as a gap
during a general audit of missing visual results across the SG-training notebooks. Built
`experiments/figures_sg_loo.py`, reading the JSON directly, no re-denoising: per-fold
`resid_r` bars (dirty/frozen/finetune/fresh) plus a mean +/- std summary panel, saved to
`results/self-gravitating/sg_loo_wiggle_vs_fold.png`. All 5 folds hold `mstar_at_bound=False`
(RULES.md #8 checked, not assumed).

Summary across the 5 leave-one-out folds:

| method | resid_r (mean +/- std) |
|---|---|
| dirty | 0.664 +/- 0.207 |
| frozen | 0.641 +/- 0.213 |
| finetune | 0.564 +/- 0.265 |
| fresh | 0.522 +/- 0.282 |

**None of the three trained arms beat the untouched dirty cube on this metric, and `fresh`
is the worst of the four, not the best.** This is consistent with the standing WITHDRAWAL
(`4e88cad`, notebook 10 V1's "fresh beats finetune" does not reproduce) rather than a new
finding: across genuine per-disk holdouts, SG training does not yet recover the wiggle
signature better than doing nothing. Per-fold spread is also large (std nearly half the
mean for finetune/fresh), and one fold (`run_9019_00019`) sits near 0.99 for every method
including dirty, meaning that disk's own wiggle is close to undetectable at this mask
regardless of denoising -- worth keeping in mind before averaging fold numbers together in
any future writeup.

---

## 2026-09-11 | run | frac sweep on `wiggle_all_methods.py`'s own comparison, confirms 0.05

Before trusting the frac=0.05 fix on this specific comparison (it had only been checked on
the SG holdout cube's mask sensitivity, not this line-emission-vs-SG cube one), swept it here
too, reusing the M1 maps cached in `experiments/wiggle_all_methods_step1.npz` (2026-08-28's
original-correction run) rather than re-denoising -- same trick as the earlier sweep, minutes
not hours.

| frac | mask% | dirty | beam-only | U-Net | DDRM | mstar |
|---|---|---|---|---|---|---|
| 0.02 (old) | 39.3% | 0.892 | 0.920 | 0.805 | 0.587 | 0.535 |
| 0.03 | 28.6% | 0.979 | 0.984 | 0.966 | 0.892 | **1.587** |
| **0.05 (new)** | 16.0% | 0.862 | 0.888 | 0.760 | 0.571 | 0.643 |
| 0.08 | 6.9% | 0.997 | 0.997 | 0.995 | 0.983 | 0.524 |
| 0.10 | 4.4% | 0.997 | 0.997 | 0.995 | 0.982 | 0.552 |
| 0.15 | 0.9% | 0.993 | 0.994 | 0.989 | 0.962 | **50.0 DEGEN** |
| 0.20 | 0.4% | 0.990 | 0.990 | 0.983 | 0.940 | **50.0 DEGEN** |

**Same three-zone shape as the SG mask-sensitivity sweep**, on independent data: loose (0.02)
pulls in a halo, 0.05 sits in the zone where the four methods actually separate, past ~0.08
the comparison flattens toward 1.0 (only the bright core survives, uninformative), past ~0.15
the fit degenerates.

**Caught one thing this sweep specifically exposed: `frac=0.03` is a bad fit, not a data
point.** `mstar` jumps to 1.587 there against 0.535 and 0.643 on either side, and every
correlation spikes with it -- the least-squares geometry fit landing on a different local
optimum at that mask size, not a real trend. Flagging it so it never gets read as a stronger
result than 0.05.

**Confirms 0.05 as the right choice for this comparison specifically**, not just inherited
from the SG holdout check. This is what the queued Kaggle rerun of `09-wiggle-scoring.ipynb`
will produce; the "third confirmation" table stays provisional until that actually lands.

---

## 2026-09-11 | bug | the "third confirmation" wiggle table was still on the flagged frac=0.02

The masking fix (Jason's flag, 0.02 -> 0.05, PROGRESS.md 2026-09-04) reached the SG training
scoring scripts but never reached `wiggle_all_methods.py`, the script actually behind the
"third confirmation" numbers (dirty 0.891 / beam-only 0.920 / U-Net 0.804 / DDRM 0.583,
reproduced 3x). All three of those reproductions, and `09-wiggle-scoring.ipynb`'s (its GPU
port), ran at the old threshold. Caught by checking every scoring script's actual `frac`
directly rather than assuming the fix had propagated.

Both fixed to `frac=0.05` now. `09-wiggle-scoring.ipynb`'s stale V1/V2 outputs cleared (same
reason as notebook 10's clearing on 2026-09-04: code and embedded output must not disagree in
a file about to be re-imported). Queued to rerun on Kaggle GPU. **The "third confirmation"
table is provisional until this reruns** -- the ordering held up to frac=0.10 in the earlier
mask-sensitivity sweep, but that was checked on the SG holdout cube, not this line-emission
comparison specifically, so it is not yet confirmed for this exact script.

---

## 2026-09-11 | run | scoring notebook 08's 4 checkpoints on moments AND the wiggle, started

`experiments/score_08_kinematic.py`, launched on local CPU. Sliding 31-channel window, centre-
channel readout (the standard read for a channel-stack model: one prediction per true target
position, not the whole predicted stack trusted at once). Normalisation matches training
exactly -- continuum-subtract first (mean of first/last 5 channels), then min-max both dirty
and clean using the CENTRE dirty channel's (lo, hi), shared across the neighbour stack, never
per-channel (`src/data/fits_cube_dataset.py`'s own documented reasoning for why: per-channel
normalisation would erase the relative amplitude along velocity that M1/M2 are computed from,
the same argument that already governs notebook 05's spectral-context arms).

Wiggle geometry is a FREE fit per cube (no stated ground truth for these disks, unlike the
self-gravitating ones), one fit on clean per cube, shared across dirty and all four gammas.
`mstar_at_bound` checked and printed per cube rather than assumed converged (RULES.md #8).

**Timed before committing to the full run:** 3.56 s/channel, so one cube x one gamma is ~12
min and the full 5-cube x 4-gamma sweep is ~4 hours on CPU -- the 31-channel model is far
heavier per forward pass than the SG scripts' 1-channel one. Not yet finished.

---

## 2026-09-11 | run | notebook 08's `gamma=10` rerun completes clean, the memory-clear fix held

Kaggle auto-push (`c1b2245`, "Version 5"), code `d9919bc`. `GAMMAS=[10.0]` only, per-arm
`gc.collect()` + `torch.cuda.empty_cache()` fix from the same day applied. No kernel death this
time: early stop at epoch 16 (best epoch 10, val_loss 0.0896), PSNR 24.333 / SSIM 0.9293 over
the full 31-channel stack (not comparable to single-channel PSNR elsewhere, the notebook's own
printed output says so).

All four arms of the sweep now complete:

| gamma | best epoch | val_loss |
|---|---|---|
| 0.0 | 27 | 0.000721 |
| 0.1 | 17 | 0.001721 |
| 1.0 | 28 | 0.016901 |
| 10.0 | 10 | 0.0896 |

`val_loss` is not comparable across gamma (RULES.md #4): the kinematic term's weight changes
the loss function's own scale, so a higher number at higher gamma does not mean worse pixel
performance on its own. Whether `kinematic_gamma>0` actually helps needs scoring on moments
and the wiggle, not read off this table.

**Not yet scored, and now lower priority than it was.** This run's original purpose (PLAN.md
Block 1) was a fast diagnostic on whether `KinematicLoss` was worth rebuilding for SG data
before spending days on it. Notebook 12's spectral-context result (two entries below) already
resolved that question positively without needing `KinematicLoss` at all, so this sweep is now
an informational data point on line emission rather than a blocking decision. Worth scoring
eventually, not urgent.

**`kin_gamma10.pth` downloaded and stored** in `models/08-kinematic/`, verified single-root,
strict-loadable, `epoch=10, val_loss=0.089597` matching this entry's table exactly. All four
gammas now in the local store, indexed in `models/README.md`.

**`kin_gamma10.pth` not yet pulled** (111.02 MB, Kaggle Output), the other three arms already
stored (`models/08-kinematic/`, entry earlier).

---

## 2026-09-11 | run | notebook 12, k=3: spectral context now exceeds doing nothing on the wiggle

Kaggle auto-push (`83c7ac0`, "Version 4"), single arm, `KS=[3]` only, reusing `k=0/1/2` from
the entry below rather than retraining them. Same holdout, same fixed-inclination geometry.

| k | in_channels | PSNR | M0 | M1 | M2 | wiggle resid_r |
|---|---|---|---|---|---|---|
| 0 | 1 | 30.053 | -26.5% | +4.2% | -44.4% | 0.366 |
| 1 | 3 | 32.828 | +29.2% | +31.0% | +62.5% | 0.590 |
| 2 | 5 | 33.576 | **+61.4%** | +31.4% | +67.6% | 0.506 |
| 3 | 7 | 35.114 | +53.0% | **+38.8%** | **+70.1%** | **0.681** |

**`k=3` exceeds dirty's own wiggle score (0.594) on this holdout**, not merely matches it, the
first SG-trained arm anywhere in this project to do so. Also the best PSNR, M1 and M2 in the
SG thread. `k=2`'s dip now reads as real non-monotonic structure rather than a fluke that
would wash out with one more point, since `k=3` recovered past it, though the mechanism for
why `k=2` specifically dips is not established.

**Block 1's decision gate (PLAN.md) is resolved: fast path, confirmed, not just directional.**
Spectral context is a working recipe that keeps the moment gains and improves the wiggle past
baseline. `k=3` is the strongest candidate on hand; `k=1` remains the cheapest (22 min vs
`k=3`'s 34) at a real but smaller cost (0.590 vs 0.681).

**Still n=1, one holdout disk, one seed, all `fresh`.** The magnitude and the above-baseline
result make this a genuine finding, but "`k=3` is the final recipe" isn't established past one
measurement, same caveat as `k=1`/`k=2` carried. A `k=4` point would say whether the trend
continues, plateaus, or was already past its peak at `k=3`; not run, lower priority now that
Block 1 has a working, positive result rather than an open question.

**Checkpoint not yet pulled** (`sg_k3_fresh.pth`, Kaggle Output).

---

## 2026-09-11 | run | notebook 12: spectral context fixes most of the wiggle loss, and improves the moments more than anything tried so far

Kaggle run (code `07f047f`, confirmed from cell 0b's log; no push commit, downloaded
manually). Same split as notebook 10 (train `{9015, 9019, 9032}`, val `9025`, holdout `9074`)
for direct comparability. All three arms `fresh` (random init), since no stored checkpoint
has `in_channels` matching `k>0` to fine-tune from. Holdout scored on moments AND the wiggle
directly (`fix_incl_deg=20`, this disk's `.para` truth), not moments alone.

| k | in_channels | PSNR | M0 | M1 | M2 | wiggle resid_r |
|---|---|---|---|---|---|---|
| 0 (control) | 1 | 30.053 | -26.5% | +4.2% | -44.4% | 0.366 |
| 1 | 3 | 32.828 | +29.2% | +31.0% | +62.5% | **0.590** |
| 2 | 5 | 33.576 | **+61.4%** | **+31.4%** | **+67.6%** | 0.506 |

**`k=1` recovers the wiggle to essentially the level of doing nothing.** dirty scored 0.594 on
this exact holdout (2026-09-10 entry); `k=1`'s 0.590 is a 0.004 difference, while every trained
arm from notebooks 10 and 11 sat well below that (`frozen` 0.487, `finetune` 0.465, `fresh`
0.278 on this same cube in notebook 11's fold 4). **And `k=1`/`k=2` post the largest moment
gains seen anywhere in the SG thread**, well past `k=0`'s own already-positive numbers.

**Mechanism, as predicted before this run (PLAN.md's Block 1 decision gate):** the `k=0` model
gets one dirty channel in, one clean channel out, with no information about neighbouring
velocity channels, so it cannot preserve a sub-channel velocity centroid it never sees.
Feeding `2k+1` channels while still predicting the centre channel gives it that information
directly, the same lever already proven on line emission (`winner_k1`/`k2`, 2026-08-20/21).

**Not monotonic, and this is reported rather than smoothed over: `k=2` has the best moments
and a LOWER wiggle than `k=1`** (0.506 vs 0.590), though still far above `k=0` and every prior
arm. Consistent with the mechanism, more context keeps helping pixel accuracy, but something
about `k=2`'s particular fit trades a little kinematic fidelity for it. On n=1 this cannot yet
be told apart from run-to-run noise the way `fresh`'s instability was shown to be real
(notebook 10 V1 vs V4). A third point (`k=3`) would say whether this is a real peak at `k=1`
or noise; not yet run.

**What this does NOT yet establish, same caveats as every SG result so far:** one holdout
disk, one seed, all `fresh` (no fine-tuned arm at `k>0` to compare against, since fine-tuning
from `winner_aug` needs a matching `in_channels=1` checkpoint that spectral context breaks by
construction). The magnitude here is large enough to be a real lever regardless, but "`k=1` is
the recipe" is not yet a claim past one measurement.

**Checkpoints not yet pulled** (`sg_k0_fresh.pth`, `sg_k1_fresh.pth`, `sg_k2_fresh.pth`,
Kaggle Output, `results/checkpoints/`), needed before storing per RULES.md #12.

---

## 2026-09-10 | code + added | notebook 12: spectral context on SG training, scored on the wiggle directly

Response to the finding above: the model gets one dirty channel in, one clean channel out,
zero information about neighbouring velocity channels, which is the documented mechanism
(2026-08-19 entry) for why per-channel denoising cannot preserve a sub-channel velocity
centroid. Spectral context (`n_neighbors=k`) already exists and was proven on line emission
(`winner_k1`/`k2`); this applies it to SG training for the first time, and scores the result
on the wiggle directly rather than only the moments.

Three arms, `k=0/1/2`, ALL `fresh` (random init): no stored checkpoint has `in_channels`
matching `k>0`, so fine-tuning from `winner_aug` is not available without first training a
matching-shape prior. `k0` is a same-day retrain of the fresh control, not a reused
checkpoint from notebook 10 or 11, so the comparison isn't confounded by which session
produced it. Same split as notebook 10 (train `{9015,9019,9032}`, val `9025`, holdout `9074`)
for direct comparability, inclination fixed at that disk's stated 20 deg throughout.

**Verified by executing the real cell sources** at reduced scale (64px, 1 epoch, 4
samples/cube, 10 scoring channels): shapes correct at each k (in_channels 1/3/5, out_channels
1 throughout), the neighbour-gathering denoise function (adapted from notebook 05's
`denoise_cube`, each neighbour normalised by the CENTRE channel's scale, ends clamped not
wrapped) runs without error at every k, wiggle scoring and the results table run. Numbers at
this scale are noise (10 channels sliced outside the line, `mstar` pins at its lower bound as
expected on flat signal). **Not yet run for real. No GPU numbers exist.**

---

## 2026-09-10 | run | sanity check: the wiggle finding is not a masking artifact

Before trusting the n=5 result below, checked whether fold 1's near-ceiling numbers (every
method within 0.006) were a `frac=0.05` masking effect that would dissolve at a tighter
threshold. `experiments/sanity_mask_frac_fold1.py`: denoised `run_9019` once, scored at
frac 0.05 / 0.10 / 0.15 without repeating the denoise per frac.

| frac | mask% | dirty | frozen | finetune | fresh |
|---|---|---|---|---|---|
| 0.05 | 24.4% | 0.9956 | 0.9956 | 0.9912 | 0.9901 |
| 0.10 | 5.1% | 0.9952 | 0.9950 | 0.9865 | 0.9827 |
| 0.15 | 1.4% | 0.9921 | 0.9917 | 0.9769 | 0.9687 |

**The gap widens as the mask tightens, it does not close.** No degenerate fit at any frac.
The ordering (dirty ~ frozen > finetune > fresh) is not a loose-mask artifact.

## 2026-09-10 | finding | across all 5 disks as genuine holdouts, SG training does not recover the wiggle, and `fresh` is worst

Extends the entry below from n=1 to n=5, using notebook 11's leave-one-out checkpoints
(`models/11-loo/`) so every disk is scored by a model that never trained on it.
`experiments/score_sg_wiggle_loo.py`, inclination held fixed at each disk's stated `.para`
truth throughout (20 deg for four disks, 30 for `run_9019`), one shared geometry per fold fit
on that fold's clean cube. 190 min total on CPU.

**resid r per fold (higher = wiggle preserved):**

| fold | holdout | dirty | frozen | finetune | fresh |
|---|---|---|---|---|---|
| 0 | run_9015 | 0.543 | 0.621 | 0.580 | 0.566 |
| 1 | run_9019 | 0.996 | 0.996 | 0.991 | 0.990 |
| 2 | run_9025 | 0.468 | 0.464 | 0.270 | 0.396 |
| 3 | run_9032 | 0.721 | 0.640 | 0.516 | 0.379 |
| 4 | run_9074 | 0.594 | 0.487 | 0.465 | 0.278 |
| **mean +/- std** | | **0.664 +/- 0.207** | **0.641 +/- 0.213** | **0.564 +/- 0.265** | **0.522 +/- 0.282** |

**Doing nothing wins on average, and training degrades the wiggle monotonically in 3 of 5
folds** (2, 3, 4: dirty > frozen > finetune > fresh, exactly the shape of the n=1 result
below). Folds 0 and 1 don't follow it: fold 0 has `frozen` slightly ahead of `dirty`, fold 1
has every method within 0.006 of ceiling (that disk's wiggle is essentially recoverable by
construction, mean resid RMS 1.45 km/s against clean's 1.448, barely denoised at all). Neither
exception reverses the aggregate: `dirty` and `frozen` sit within noise of each other, both
comfortably ahead of `finetune`, and `fresh` is worst on every single fold except fold 0.

**This is the opposite conclusion from notebook 10's headline, and it is not a contradiction,
it is a different metric.** M0/M1/M2 measure amplitude; the wiggle measures whether the
specific kinematic substructure survives. SG training makes the pixel-level moments better and
the kinematic diagnostic worse, on average, across every disk checked. The RETRACTION entry's
shape (2026-08-28: denoising improves moments, damages the diagnostic underneath them) now
holds for SG-trained models too, not just the original line-emission ones.

**`fresh` is confirmed as the least reliable arm on a second, independent measurement.** Its
std (0.282) is the largest of the four, consistent with the training-noise instability found
in notebook 10 V1 vs V4 (66pp swing on M0 between identical runs). Fold 4's fresh raw r is
0.0100, functionally uncorrelated with truth, worth flagging as a suspect value on its own
(RULES.md #8) even though `mstar_at_bound` was False in every fold's fit, so it is not the
degeneracy bug recurring.

**Caveats that still apply.** Three training disks per fold, one seed per fold, no seed
repeats, so cube variance and training variance remain confounded for `finetune` and `fresh`
exactly as flagged when this run was designed. `frozen`'s spread (0.213) is the one clean
cube-variance number in the table; it is comparable in size to `finetune`'s (0.265), meaning
most of what looks like "spread" here may be disk-to-disk kinematic difficulty rather than
training instability, except for `fresh`, whose spread exceeds even that baseline.

## 2026-09-10 | finding | SG training improves the moments and degrades the wiggle, on the one disk checked

Notebook 10's headline was scored on M0/M1/M2 amplitude. That is not the same question as
whether the kinematic signature (the Keplerian-subtracted residual) comes back, which is the
actual reason the SG thread exists. Scored it directly: `experiments/score_sg_wiggle.py`,
holdout `run_9074`, one shared geometry fit on clean with inclination FIXED at 20 deg (this
disk's stated `.para` truth), avoiding the mass-inclination degeneracy from the 2026-09-04
entry rather than re-triggering it.

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean (ref) | 2.341 | -- | -- |
| dirty | 2.746 | 0.267 | 0.594 |
| frozen | 3.461 | 0.209 | 0.487 |
| finetune | 3.569 | 0.183 | 0.459 |
| fresh | 3.627 | 0.143 | 0.424 |

**Doing nothing beats every trained model, and it degrades monotonically with how much the
model diverges from untouched: dirty > frozen > finetune > fresh.** The same shape as the
original line-emission-cube finding (RETRACTION entry, 2026-08-28): denoising improves pixel
moments and damages the kinematic diagnostic underneath them. Notebook 10's positive M1/M2
numbers do not mean the wiggle came back; on this disk it got worse the more the model was
trained.

**Not yet a claim, a data point.** n=1 holdout, and the baseline wiggle signal here is weak to
start: raw dirty-vs-clean r is only 0.267, against >0.9 on the disks used for the earlier line-
emission wiggle work. `run_9074` also has the smallest signal mask of the five disks (2.2% of
field, 2034 px), consistent with it being generally low-SNR. Could be a genuinely weak-signal
draw rather than a universal result. Needs scoring on at least one more disk with a stronger
baseline wiggle before this is reportable as more than "measured once, this is what it showed."

---

## 2026-09-10 | run | notebook 11 v2: leave-one-out, nothing is separable, but `frozen`'s spread isolates cube variance cleanly

5 folds, every cube holding out exactly once, seed fixed at 42. 128 min total. Version number
is author-reported (no push commit exists to verify it, unlike 10's V4); code `b7b140d`
confirmed from cell 0b's own log. Checkpoints NOT downloaded -- `loo*_*.pth` (10 files) still
only on Kaggle. Archived at `results/11-sg-loo/v2_2026-09-10_b7b140d/`.

| fold | holdout | frozen M0/M1/M2 | finetune M0/M1/M2 | fresh M0/M1/M2 |
|---|---|---|---|---|
| 0 | run_9015 | +23.6/-0.6/-22.3 | +35.0/+26.3/+9.3 | +37.4/+17.9/+7.4 |
| 1 | run_9019 | +23.8/+9.7/-76.3 | -24.1/-7.5/-44.0 | -6.3/-20.7/-139.7 |
| 2 | run_9025 | +4.6/+12.3/-9.5 | +13.2/+22.3/+7.1 | +14.2/+19.2/+7.7 |
| 3 | run_9032 | -111.3/-284.2/-112.8 | -68.2/-92.8/-36.9 | -288.7/-351.5/-246.5 |
| 4 | run_9074 | -10.3/-0.6/-43.6 | -15.1/+20.3/+6.2 | +10.7/+2.2/-48.4 |

**The notebook's own printed verdict: `fresh - finetune` is smaller than the fold-to-fold
spread on all three moments** (M0 -34.7+/-104.4pp, M1 -60.3+/-111.1pp, M2 -72.3+/-86.6pp).
Nothing separable at n=5, exactly what was expected going in.

**Fold 3 (`run_9032`) is catastrophic for every arm, `frozen` included.** Predicted in the
notebook before this run, from that cube's own diagnostics (rmsdiff 0.107 vs the others'
0.46-0.54, mask covering 98.9% of the field). Recorded in advance, not explained after.

**The one clean read: `frozen`'s fold-to-fold spread is pure cube variance**, since it never
trains (PSNR std 2.98, M0 std 56.3, M2 std 42.0). Against that baseline, `finetune` has LOWER
std on M0 and M2 and the best mean on all three moments -- "more stable and not worse," which
is a real, if modest, finding. "Finetune is better" is still not supported; the spreads remain
too large.

**Confirms the design caveat flagged before this ran.** Seed was fixed across folds to
isolate cube variance (RULES.md #6), but notebook 10 V1-vs-V4 showed `fresh` alone can move
66pp between IDENTICAL runs on ONE cube. So `fresh`'s per-fold numbers here mix cube variance
with that arm's own training noise, and cannot be read as a pure cube effect the way
`frozen`'s can. `finetune`'s spread is more trustworthy as a cube measurement, since notebook
10's logs showed its early-stopping point barely moved between V1 and V4 (epoch 32/24 both
times) where `fresh`'s did not (epoch 33/25 vs 23/15).

**Not yet done:** seed repeats per fold, which would actually separate cube variance from
training variance instead of confounding them for two of three arms. Checkpoints need pulling.

---

## 2026-09-04 | run | notebook 10 V4: V1's headline does not reproduce, `fresh` is high-variance

Re-ran 10 at identical settings (same seed, split, data; code `340e26c` vs V1's `be616fd`,
differing only by the `val_metrics` patch, which does not touch training) specifically to test
whether V1's `fresh` > `finetune` survived a second draw. Push `ee040ae`, Kaggle Version 4.
Archived at `results/10-sg-training/v4_2026-09-04_ee040ae/`.

| arm | PSNR V1 -> V4 | M0 | M1 | M2 |
|---|---|---|---|---|
| `frozen` | 29.032 -> 29.032 | -10.3 -> -10.3 | -0.6 -> -0.6 | -43.6 -> -43.6 |
| `finetune` | 30.859 -> 30.874 | -6.5 -> -7.0 | +36.5 -> +36.7 | +15.6 -> +15.3 |
| `fresh` | 30.024 -> 29.258 | **+5.0 -> -61.1** | +21.8 -> -27.3 | +26.0 -> -42.2 |

**WITHDRAWN: "`fresh` beats `finetune`" (2026-09-04, V1).** It was one draw of an arm that
moves 66 pp on M0 between identical runs. `fresh` in V4 is worse than the untrained baseline.

**Why this is trustworthy rather than just another noisy number:** `frozen` reproduced
*exactly* (it is inference only, so it must, and it did) and `finetune` reproduced to within
0.5 pp on every moment. The setup is not noisy in general; that arm is.

**Cause, and it is a known one.** v25 measured that early stopping on a noisy validation set
converts run-to-run nondeterminism into large performance swings. The logs show it directly:
`fresh` early-stopped at epoch 23 with best epoch 15 here, against epoch 33 / best 25 in V1 --
it stopped ten epochs sooner from a worse point. `finetune` stopped at 32 with best 24 in both
runs, because starting from trained weights it sits near a good solution and the stopping
decision is not delicate. Random init on three training disks is.

**What survives.** SG training closing the domain gap: `frozen` negative on all three moments
in both runs, `finetune` positive on M1 and M2 in both. That is the answer to Jason's
suggestion and it is now reproduced.

**Consequence for notebook 11.** Its LOO design fixes the seed across folds to isolate CUBE
variance. For `fresh` that isolates the wrong thing -- a fold-to-fold difference cannot be
read as a cube effect when the same cube moves 66 pp on its own. LOO stays valid for
`finetune`; for `fresh` it needs seed repeats per fold, or its per-fold numbers are a lower
bound on noise rather than a measurement. **This should be settled before 11 is run**, or its
`fresh` column will not mean what the notebook says it means.

**V4's checkpoints were not downloaded**, so `models/10-sg/` still holds V1's weights, which
is correct: the V1 archive's numbers were measured from exactly those (RULES.md #12).

## 2026-09-04 | code + added | notebook 11: leave-one-out, to test whether `fresh > finetune` survives

Notebook 10 V1's headline (`fresh` beating `finetune` on M0 and M2) rests on one holdout cube
and one seed, which is inside variance this project has already measured twice (V7/V9: M2
+18.4% -> +2.5% on the same cube with no config change; v25: 1.73 dB from an identical-seed
rerun). Five disks make leave-one-out available.

**5 folds, each 3 train / 1 val / 1 holdout, every cube holding out exactly once.** Training
seed fixed at 42 across folds, so the only thing varying is the fold: this measures spread
across CUBES, not seeds, and the notebook says so where it prints the number (RULES.md #6).
Folds are constructed explicitly rather than by reseeding `split_cubes`, so the assignment is
auditable instead of being a function of a seed, and each fold asserts no cube appears twice.

Reports `fresh - finetune` per moment against the fold-to-fold spread, compared against the
full standard deviation rather than the standard error, deliberately: n=5 supports no
significance claim and the wording must not imply one.

**Recorded before the run:** `run_9032` will produce a strange fold. Its synthesized pair came
out at rmsdiff 0.107 against the others' 0.46-0.54 and its signal mask covers 98.9% of the
field. Written down now so it is not explained away afterwards.

Results are written after every fold, so an interrupted session leaves usable partial data
rather than nothing. ~3 h expected (5 folds x 2 arms x ~18 min), 10 checkpoints at ~112 MB.

**Verified by executing the real cell sources** at reduced scale (64px, 1 epoch, 2 folds, 8
scoring channels): folds build and pass the leakage assertion, both arms train per fold, all
three arms score, per-fold persistence and the aggregate both run. **Not yet run for real.**
The reduced-scale artifacts were deleted rather than left in `results/` -- the same toy files
were briefly mistaken for results after notebook 10.

## 2026-09-04 | run | notebook 10 v1: SG training closes the domain gap, and `fresh` beats `finetune`

Kaggle Version 1 (push `cc194ef`), code `be616fd` confirmed from cell 0b. Dataset
`exxa-sg-synth-pairs`. Both arms converged and early-stopped (32 and 33 epochs, ~18 min each),
360 train items across 3 disks. Archived at `results/10-sg-training/v1_2026-09-04_cc194ef/`.

| arm | PSNR | SSIM | M0 | M1 | M2 |
|---|---|---|---|---|---|
| `frozen` | -- | -- | -10.3% | -0.6% | **-43.6%** |
| `finetune` | 30.859 | 0.98343 | -6.5% | **+36.5%** | +15.6% |
| `fresh` | 30.024 | 0.98055 | **+5.0%** | +21.8% | **+26.0%** |

**Training on SG data works.** `frozen` is negative on all three moments; both trained arms
are positive on M1 and M2. First measurement of that gap closing with a forward operator known
exactly rather than estimated.

**`fresh` beats `finetune` 2 of 3** (M0 by 11.5 pp, M2 by 10.4 pp; `finetune` takes M1 by
14.7 pp). **This contradicts the prediction written down before the run**, which was that
pretraining would help given only three disks. Recording that it was wrong rather than quietly
moving on.

**PSNR does not track the science, and with the baseline filled in this is the sharpest case
yet.** The run printed `nan` for `frozen` (it never goes through `train_unet`, so it has no
fixed-metric evaluation, and the results cell hardcoded nan -- a gap that put the hole exactly
where the baseline belongs). Scored afterwards with `val_metrics` itself on the same val split
(`experiments/frozen_val_metrics.py`; `finetune`/`fresh` reproduce to the digit, which is what
makes the baseline comparable):

| arm | PSNR | SSIM | MSE |
|---|---|---|---|
| `frozen` | 29.032 | 0.97815 | 0.001456 |
| `finetune` | **30.859** | **0.98343** | **0.000984** |
| `fresh` | 30.024 | 0.98055 | 0.001303 |

So `finetune` wins **every** pixel metric -- PSNR, SSIM, MSE and validation loss -- and still
loses M0 and M2 to `fresh`. Previously this pattern was PSNR being insensitive (05 v26's
spectral arms, the beam arm); here the training objective itself prefers the scientifically
worse model. And `frozen` is only 1.8 dB off the best arm while being ~60 pp worse on M2.
Notebook patched so future runs score all three arms instead of printing nan.

**What this does NOT establish.** One holdout cube, three training disks, one seed. V7/V9
measured M2 swinging +18.4% -> +2.5% on the same cube with no config change, and v25 measured
1.73 dB from an identical-seed rerun. A 10-15 pp gap on n=1 is inside that. `fresh > finetune`
is directional, not a finding. Also `frozen` scores M0 -10.3% here against -86.5% on Jason's
real pair, so the synthesized corruption is gentler than the real one and this understates the
true gap. And `run_9032`, one of the three training disks, came out at rmsdiff 0.107 against
the others' 0.46-0.54, so a third of the training data is easier than intended.

**Cleanup worth recording:** the local `results/checkpoints/sg_*.pth` and
`results/self-gravitating/sg_training_arms.json` left by the reduced-scale verification run
(1 epoch, 64px, 12 channels, M0 -111.6%) were deleted. They looked like results and were not,
which is the RULES.md #7 failure mode in miniature.

**Not yet done:** the real checkpoints are still only in the Kaggle Output and must be pulled
before the next run wipes them (RULES.md #1, #12), and these arms have not been scored on the
GI wiggle metric -- the kinematic question is what the SG data exists for, and `finetune`
winning M1 by 14.7 pp is the hint worth chasing.

## 2026-09-04 | code + added | notebook 10: frozen vs fine-tuned vs fresh on the synthesized SG pairs

Three arms against one baseline, on the pairs synthesized in the entry below:

| arm | init | trains on SG | the question it answers |
|---|---|---|---|
| `frozen` | `winner_aug` seed 43 | no | how large is the domain gap when the operator is known exactly |
| `finetune` | `winner_aug` seed 43 | yes | does SG training close it |
| `fresh` | random | yes | does line-emission pretraining help or hurt |

All three share `winner_aug`'s architecture and optimiser settings, so they differ only in
initialisation and whether they train. `finetune` runs at 0.1x the learning rate; at the full
rate the first steps overwrite the pretrained weights and the arm silently becomes an
expensive `fresh`.

**`train_unet` gained `init_state_dict=`** for this. It loads STRICTLY and raises on any
missing or unexpected key, because a silent partial load is the `winner_beam` failure again:
it trains, it reports numbers, and the numbers describe something other than the arm's name.

Scored on the holdout cube only, by moment improvement (RULES.md #4) rather than PSNR alone.

**Verified by executing the notebook's real cell sources** at reduced scale (64px, 1 epoch, 4
samples/cube, 12 scoring channels): the split is leakage-safe, `init_state_dict` loads its 256
tensors, both arms train and checkpoint, and the holdout scoring and results table run.
**Not yet run for real** -- no GPU numbers exist. Needs the `sg_synth` cubes (1.8 GB, of which
1.4 GB is the single 600x600x512 cube) plus the `winner_aug` checkpoint uploaded as a Kaggle
Dataset.

## 2026-09-04 | code | synthesized trainable SG pairs, since the shipped ones cannot train

Jason's suggestion (use SG data in training) is right in substance and blocked in practice:
the pairs he sent differ clean-to-dirty by 0.4-7% RMS against the training set's ~50%, entry
below. So the corruption gets applied here instead:
`dirty = beam (*) clean + beam (*) noise`, using the beam recovered from the v2 pair's
cross-spectrum. `experiments/synthesize_sg_pairs.py`, output under
`self-gravitating cube and dirty cube/sg_synth/` (1.8 GB, gitignored), manifest at
`results/self-gravitating/sg_synth_manifest.json`.

**Two bugs caught while building it, both in the same place: operator normalisation.**

1. The recovered beam **sums to 355**, not 1 -- the Jy/pixel-to-Jy/beam area factor, plus the
   129 px crop truncating the negative bowl. Convolving with it as-is multiplies the flux
   scale by 355, and the first run produced `rmsdiff/rms_clean` of ~347 on every cube: a scale
   factor wearing the costume of a corruption. That is the same signature the v2 pair shows
   (341), and it would have been actively harmful in training, because
   `fits_cube_dataset` normalises BOTH cubes by the DIRTY channel's range, so the target would
   have landed ~355x below the input. That is the Week-5 conditioning bug all over again.
   Fixed by normalising the beam to unit DC gain.
2. Convolving white noise with a unit-sum kernel suppresses its variance by a kernel-dependent
   factor (measured ~0.0384 here), so a sigma chosen before convolution silently tracks the
   beam normalisation rather than the requested level. Now calibrated after convolution:
   draw once, measure, rescale.

**Result, against the line-emission training set's 0.41-0.57 band:**

| run | source | channels | rmsdiff/rms_clean |
|---|---|---|---|
| `run_9019_00019_rt_00` | `run_sg_00019` | 512 | 0.4725 |
| `run_9015_00370_rt_00` | `run_sg_15` | 172 | 0.4908 |
| `run_9025_00370_rt_00` | `run_sg_25` | 172 | 0.5354 |
| `run_9032_00020_rt_00` | `run_sg_32` | 157 | **0.1065** |
| `run_9074_00025_rt_00` | `run_sg_74` | 201 | 0.4589 |

Four of five sit in the band. `run_9032` comes out much easier, consistent with it being an
odd cube generally: lowest in-signal RMS of the five by an order of magnitude, and the one
whose signal mask covered 98.9% of the field in the validation entry below, i.e. almost no
dynamic range in M0.

**Verified end to end through the real pipeline** (`experiments/check_sg_synth_pipeline.py`),
because being able to write FITS is not the same as being trainable: `split_cubes` discovers
all five with distinct RunIDs and splits them leakage-safely (3 train / 1 val / 1 holdout, no
cube in both), `FITSChannelDataset` returns matched finite (1, 256, 256) items, and the
**identity baseline scores 12.19 dB** -- against the U-Net's ~39 dB on line emission. There is
real signal to recover, which is precisely what the shipped pairs would have failed.

**What this is not.** Convolution plus correlated Gaussian noise is not an interferometer: no
uv sampling, no phase errors, no deconvolution residuals. It is a controlled approximation,
and it is exactly the forward model DDRM already assumes -- which is the upside, since `A` is
now known exactly rather than estimated at 0.80 held-out correlation. CASA `simobserve` on
these same clean cubes is the higher-fidelity version and remains the natural follow-up, and
would double as the first real work toward ALMA validation.

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
