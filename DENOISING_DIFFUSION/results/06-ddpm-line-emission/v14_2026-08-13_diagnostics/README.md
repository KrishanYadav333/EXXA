# 06-ddpm-line-emission — diagnostics run (Kaggle v14)

Ran 2026-08-13 at code `5b02a5b`, on the checkpoint from v13 (`ddpm_seed42`, epoch 58,
v-prediction). **Nothing here trains** — every arm re-scores the same weights, restored from
the `exxa-ddpm-06-v13` dataset. Hit Kaggle's 12 h ceiling during the 6th arm.

## What ran

| arm | K_AVG | rescale | checkpoint | status |
|---|---|---|---|---|
| baseline | 4 | no | seed42 | complete (in the run log) |
| kavg1 | 1 | no | seed42 | complete |
| rescaled | 4 | yes | seed42 | complete |
| kavg1_rescaled | 1 | yes | seed42 | complete |
| patch_model | 4 | yes | sweep_patch | complete |
| tiled_native | 4 | yes | seed42 | **cut off** — 9 tiles/channel needs ~2.8 h per cube, ~14 h for the arm |

## Results, signal-masked metric, 5 held-out cubes

```
arm                              M0                M1                M2
baseline (K=4)          -56.9 ±149.9       13.7 ±90.0         5.2 ±83.6
improved (K=4)          -56.1 ±152.2       15.0 ±87.4        10.9 ±81.7
kavg1 (K=1)             -51.4 ±148.7       15.1 ±90.3         6.7 ±84.2
rescaled (K=4)         -264.5 ±158.7       38.3 ±24.6        42.4 ±24.4
kavg1+rescaled         -190.5 ±107.2       51.6 ±10.8        61.3 ±10.8
patch_model            -166.3 ±87.7       -68.1 ±29.0       -28.0 ±37.4
-----------------------------------------------------------------------
U-Net V12 (08 v4)         27.7 ±17.6       71.9 ±10.3       -10.5 ±39.7
```

## What the arms settled

**K_AVG is not the problem.** Dropping 4 draws to 1 moves M0 by +4.7 pp against a −56%
deficit. Posterior averaging is not what collapses the moments — and K=1 is 4× faster
(4.7 min/cube against 18.8), so the averaging was buying nothing here.

**The rescale splits the moments, and the reason is structural.** Matching each denoised
channel's mean and std to the DIRTY channel's:

- **destroys M0** (−56% → −264%), because M0 is a *sum*. Forcing the output's mean to
  dirty's re-imposes exactly the pedestal that denoising is supposed to remove. Verified on
  synthetic data: a **perfect** denoiser scores +100.0% on M0, and the same perfect output
  rescaled to dirty's statistics scores −0.0%. The rescale caps M0 at "no better than dirty"
  by construction.
- **transforms M1 and M2** (+15 → +52, +11 → +61) and collapses their spread by a factor of
  8 (std 90.3 → 10.8, 84.2 → 10.8). M1 and M2 are *ratios* normalised by the intensity sum,
  so they are insensitive to a constant offset and highly sensitive to the compressed
  dynamic range the sampler produces. Fixing the range is what they needed.

So the pedestal diagnosis was right about the symptom and wrong about the fix: the model's
output range is wrong, but correcting it against the dirty channel is the wrong reference for
an absolute quantity.

**The DDPM beats the U-Net on M2.** `kavg1_rescaled` scores **+61.3 ±10.8** against V12's
**−10.5 ±39.7** — a 72 pp gap on the moment every model in this project has struggled with,
and with a quarter of the variance. It still loses on M0 and M1.

**Patch training does not help.** `patch_model` is negative on all three moments (−166 / −68
/ −28). The 8400-patch training view was the strongest remaining lever for the small-data
hypothesis, and it made things worse rather than better.

## What to try next

Rescale the **standard deviation only**, leaving the mean alone — that is the part M1/M2
needed, without the part that caps M0. One re-score, no retraining.
