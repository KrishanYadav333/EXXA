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
