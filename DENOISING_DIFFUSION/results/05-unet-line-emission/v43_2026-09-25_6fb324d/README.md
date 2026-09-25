# 05-unet-line-emission, Version 43, completed, 2026-09-25

Kaggle push `6fb324d`. Ran the same old cells as v42 (cell 0b pulled `ad600d2`, before the 2026-09-25 fixes), so it has NO
per-arm scoring size, NO hybrid control arms and NO RAM breakdown. No traceback. One new arm trained, `winner_k2` (two spectral
neighbours), seed 42, early stop at epoch 50, 88 s/epoch, RAM free 27.4 to 22.1 GB (~0.1 GB/epoch, harmless at 256 px).

| arm | seed | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|---|
| `winner_k2` | 42 | 42.8142 | +20.2 | +65.9 | +36.8 |

Same picture as `winner_k1` (42.59 / 33.3 / 75.6 / 40.5): best PSNR of any arm, worst M2 and M1 of the recent arms. The notebook's own
line says ranking by PSNR would have picked `winner_k2`; by M0 it picks `winner_starlet_p10_ft` (+44.5). One seed, 5 cubes.

The moment table is v42's plus this row; the `winner_aug_res320` / `res480` rows are still the invalid 256 px scores (see v42 README).
Kaggle's push replaced the fixed notebook with these old cells; the fixes were re-applied with `tools/reapply_05_fixes.py`.
