# 09 Kaggle Version 1 -- GPU scoring, third independent confirmation

No training. Same comparison as `experiments/wiggle_all_methods.py`, run on Kaggle T4 GPU
instead of a local CPU: dirty / U-Net / DDRM / beam-only against clean, scored with the
corrected `compare_wiggles()` (one shared Keplerian model, fit on the clean cube, reused for
every method).

Pulled commit `9f59221` (confirmed from cell 0/0b's own log), the same commit the local CPU
rerun used. Dataset resolved at
`/kaggle/input/datasets/krishanyadav333/kaggle-wiggle-scoring-dataset/`.

## Result

240-360, step 1, 121 channels, **7.4 minutes** (the same config took 140.9 min on local CPU,
about 19x):

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean | 0.182 | -- | -- |
| dirty | 0.170 | 0.9928 | 0.8907 |
| beam-only | 0.168 | 0.9947 | 0.9198 |
| U-Net | 0.169 | 0.9874 | 0.8040 |
| DDRM | 0.302 | 0.9532 | 0.5835 |

240-360, step 4, 31 channels, 1.9 minutes: all five methods bunch at 0.987-0.999, the known
`quadratic_moment1` coarse-sampling artifact, not a real finding (see
`RETRACTION_wiggle_methodology.md`'s note on this, carried into the script's own docstring).

## What this settles

Third independent reproduction of the retraction's corrected table, now across two machines
and both CPU and GPU:

| run | dirty | beam-only | U-Net | DDRM |
|---|---|---|---|---|
| original correction (2026-08-28) | 0.891 | 0.920 | 0.804 | 0.583 |
| local CPU rerun (2026-08-28) | 0.892 | 0.920 | 0.805 | 0.587 |
| this run, Kaggle GPU (2026-08-29) | 0.891 | 0.920 | 0.804 | 0.584 |

No longer a one-off or provisional in any sense. The beam preserves the wiggle, the U-Net
degrades it, DDRM degrades it most.

## Gaps in this archive

- `results/self-gravitating/wiggle_all_methods.png` was regenerated on Kaggle (see the run
  log's "saved ->" line) but not downloaded from the Output tab, so it is not included here.
  The figure already committed at that path is from the local CPU rerun, not this run; the
  underlying numbers match closely enough that it is representative, but it is not this run's
  own output.
- Not yet added to `RUNS.md`.
