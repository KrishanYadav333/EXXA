# 05 v22 (Kaggle Version 22) — code `6f5c798` — sections 7-8 complete for the first time

Re-run purely to finish the artifact diagnostics. Section 7's CSV write had been failing on
a missing `n_background_px` column *after* the whole 300-channel analysis had run, so the
result was printed and then thrown away. Fixed in `6f5c798`; this run saved it.

**No training, and no new moment scores.** 3 checkpoints restored from v19's Output, 12 from
08, `winner_patch` scored from stored weights, 15 rows over 6 arms. The moment table is
identical to v20's. Everything new here is in sections 7-8.

## Artifact diagnostics — `sweep_winner_p10` seed 42, 300 validation channels

```
PEAK OVERSHOOT   denoised.max / clean.max
  mean 1.418 | median 0.848 | p90 1.054 | max 91.580
  channels overshooting by >10%: 8%

NEGATIVE FLOOR LEAK   denoised.min   (clean floor ~0)
  mean -0.13108 | most negative -0.21807

INVENTED STRUCTURE   (background above 20% of clean peak, blobs >= 20 px)
  channels with >=1 fake blob: 33% | blobs/channel 0.97
  mean invented background area: 6.6486% | worst channel: 99.9587%

  SNR split at median 3.9
    low-SNR  half: blobs/channel 1.63 | invented area 13.2590% | overshoot 1.992
    high-SNR half: blobs/channel 0.31 | invented area 0.0382% | overshoot 0.844
```

## Three readings, in order of confidence

**1. Invented structure is a threshold, not a gradient.** The scatter in `figure_cell24.png`
is bimodal: roughly 25 channels sit at ~100% invented background area and everything else at
~0%, with almost nothing between. Channels do not degrade smoothly as SNR falls, they flip.
Every channel in the failing cluster is below SNR ~0.5. The `6.65%` mean is therefore not a
typical channel, it is ~8% of channels failing almost completely.

**2. The floor leak is systematic, not occasional.** `denoised.min` clusters near -0.10 and
-0.155 and **no channel is near zero**, where the clean floor sits. Every channel is decoded
with a negative pedestal. This is the same class of failure as the DDPM's positive pedestal
in the blog's Section 7, on the other side of zero, and it is the mechanism behind the
overshoot statistic as well.

**3. The worst channels are where the metric stops meaning anything.** The three shown in
`figure_cell26.png` report `SNR 0.0`, `invented 99.1-99.96%`, on ~65,500 background pixels.
"Invented" is defined against 20% of the *clean peak*, and on a signal-free channel the clean
peak is a noise spike, so the bar is essentially zero and any pedestal clears it. These are
real failures, but the percentage attached to them is not a meaningful magnitude. This is
RULES.md #8 inverted: there the suspect number was a zero, here it is a 99.96%, and in both
cases the thing to check first is the denominator.

## Bearing on the blog

The low/high-SNR contrast is the strongest version of the faint-channel finding the project
has: **347x in invented area, 5.3x in blob rate, at the same split.** It is independent
support for the M2 wing-truncation hypothesis, since the model's inventions and the line
wings occupy the same faint regime. The blog currently quotes v18's `1.580 / 0.213` for this
split; these numbers are a different run of a different checkpoint and are not a correction
to it.
