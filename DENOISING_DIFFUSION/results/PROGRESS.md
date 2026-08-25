# Progress log

Chronological record of runs, arrivals and bugs, newest first. Written at the time, per
RULES.md #11. `RUNS.md` maps a number to the run that produced it; this file records what
state the project is in and how it got here.

Entry format: **date | trigger | notebook** then what happened, the evidence, the
consequence. Triggers are `run`, `added` (a notebook downloaded into the repo), `bug`.

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
