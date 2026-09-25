# 05-unet-line-emission, Version 47, completed, 2026-09-25

Kaggle push `773b0ce`. Cell 0b pulled `499030b`, so every 2026-09-25 fix was live (per-arm scoring size, size-scaled sub-batch,
`STALE_MOMENT_ARMS`, control arms) and the notebook Kaggle pushed still contains them. No traceback. **No training**: 38 checkpoints restored
(`[nb05 prior] restored 38`; the two control arms came from the recovered v44 dataset), every arm `checkpoint found, scoring without retraining`,
and 38 persisted. Versions 45 and 46 are not in git and not archived here; I do not know what they were.

## First valid resolution scores (Jason's ask 2)
Each arm scored at the size it was trained at (`winner_aug_native600` is gated off, `RUN_NATIVE600 = False`, and has no checkpoint). Means over the 5 holdout
cubes, one seed each, against the 256 px aug baseline (3 seeds):

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| aug at 256, baseline (3 seeds) | 39.30 | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| aug at 320 (seed 43) | 37.36 | -17.8 | 48.0 | -3.9 |
| aug at 480 (seed 43) | 40.18 | 39.7 | 73.5 | 67.3 |

Per cube, 320: M0 -114.0 / +34.8 / +38.3 / -113.5 / +65.4 (cubes `run_0002_00560_rt_00` and `run_0025_01000_rt_04` fail); 480: 33.6 / 55.4 / 60.0 / -12.9 / 62.3.
The control-arm rows are the same as in v44 (hybrid_ft +39.7/+80.9/+69.9, hybrid_p10_ft +40.4/+76.2/+68.4).
