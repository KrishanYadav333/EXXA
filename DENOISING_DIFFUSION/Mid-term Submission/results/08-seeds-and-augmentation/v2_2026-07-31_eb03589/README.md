# 08-seeds-and-augmentation, code commit `eb03589` (2026-07-31)

The run that produced the four arms across seeds 42/43/44, and with them the twelve
checkpoints every later notebook reuses. Kaggle version numbers were never recorded in this
notebook's output, so `v2_` is this folder's ordering label rather than a verified version.

## Files

The notebook as executed, with its full stdout in the cell outputs, is in
[`../../../notebooks/08-seeds-and-augmentation.ipynb`](../../../notebooks/08-seeds-and-augmentation.ipynb)
rather than duplicated here.

| file | what |
|---|---|
| `figure_cell24.png` | the per-seed spread plot |
| `seed_repeats.csv` | per-run PSNR/SSIM/MSE for all twelve checkpoints, the record [`../../../MODELS.md`](../../../MODELS.md) tabulates |

## Why this run still matters

v2 and v4 are the **same twelve checkpoints**. v4 reused v2's weights and simply re-scored
them, so their PSNR/SSIM/MSE rows are byte-identical. Only the evaluation code differs:

```
v2 (eb03589)   raw:  no noise clip, no signal mask
v4 (1ca611f)   clip: 3-sigma noise clip added
```

That makes v2's moment scores the ones comparable to V12, because V12's published
+69.8 / +17.5 / +20.1 was measured on the same raw metric. v2 is the only seed-validated
result in the project that can sit beside the reference checkpoint without changing the
ruler. On that footing `winner_aug` scores **M0 +87.5% ± 4.2**, the highest M0 anywhere here
and at the lowest variance, over 3 seeds.

Its artifact diagnostics, though, are not usable. This run reports 0.0% invented structure
for all four arms, which is RULES.md #8 in action: the background mask selected zero pixels,
so the detector could not fire at all. The overshoot figures, 0.89 to 0.93 and every one
below 1.0, are the same artefact. v4's 22.3 to 39.0% are the numbers to use instead.

So the rule for this run is simple. Take its moments, take v4's artifact rates, and never mix
a moment number from one with the other.
