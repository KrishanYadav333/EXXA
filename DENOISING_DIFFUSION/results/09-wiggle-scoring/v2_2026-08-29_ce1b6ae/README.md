# 09 Kaggle Version 2 -- GPU scoring, third independent confirmation

No training. Same comparison as `experiments/wiggle_all_methods.py`, run on Kaggle T4 GPU
instead of a local CPU: dirty / U-Net / DDRM / beam-only against clean, scored with the
corrected `compare_wiggles()` (one shared Keplerian model, fit on the clean cube, reused for
every method).

Auto-pushed by Kaggle's GitHub integration as commit `ce1b6ae` ("Kaggle Notebook |
09-wiggle-scoring | Version 2"), which is the number this archive uses -- per RULES.md,
reconstructed from the push commit rather than from recollection. A slightly earlier
interactive run on the same kernel session, downloaded manually rather than committed, gave
essentially identical numbers (dirty 0.8907/beam-only 0.9198/U-Net 0.8040/DDRM 0.5835 vs this
run's 0.5833) but has no Kaggle version number of its own to attribute it to, so it is not
archived separately.

Pulled commit `7d73e2e` (confirmed from cell 0/0b's own log). Dataset resolved at
`/kaggle/input/datasets/krishanyadav333/kaggle-wiggle-scoring-dataset/`.

## Result

240-360, step 1, 121 channels, **7.2 minutes** (the same config took 140.9 min on local CPU,
about 19x):

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean | 0.182 | -- | -- |
| dirty | 0.170 | 0.9928 | 0.8907 |
| beam-only | 0.168 | 0.9947 | 0.9198 |
| U-Net | 0.169 | 0.9874 | 0.8040 |
| DDRM | 0.303 | 0.9527 | 0.5833 |

240-360, step 4, 31 channels, 1.8 minutes: all five methods bunch at 0.987-0.999, the known
`quadratic_moment1` coarse-sampling artifact, not a real finding (see
`RETRACTION_wiggle_methodology.md`'s note on this, carried into the script's own docstring).

## What this settles

Third independent reproduction of the retraction's corrected table, now across two machines
and both CPU and GPU:

| run | dirty | beam-only | U-Net | DDRM |
|---|---|---|---|---|
| original correction (2026-08-28) | 0.891 | 0.920 | 0.804 | 0.583 |
| local CPU rerun (2026-08-28) | 0.892 | 0.920 | 0.805 | 0.587 |
| this run, Kaggle GPU Version 2 (2026-08-29) | 0.891 | 0.920 | 0.804 | 0.583 |

No longer a one-off or provisional in any sense. The beam preserves the wiggle, the U-Net
degrades it, DDRM degrades it most.

## Gaps in this archive

- `results/self-gravitating/wiggle_all_methods.png` was regenerated on Kaggle (see the run
  log's "saved ->" line) but not downloaded from the Output tab, so it is not included here.
  The figure already committed at that path is from the local CPU rerun, not this run; the
  underlying numbers match closely enough that it is representative, but it is not this run's
  own output.
