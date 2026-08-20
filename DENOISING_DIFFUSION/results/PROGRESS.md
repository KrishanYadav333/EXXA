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
