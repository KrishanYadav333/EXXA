# 05, Kaggle Version 21, code `6f5c798`

The run that finally completed sections 7 and 8. Section 7's CSV write had been failing on a
missing `n_background_px` column *after* the whole 300-channel analysis had already run, so
the numbers were printed and then thrown away. That is fixed in `6f5c798`, and this run saved
them.

Nothing was trained and no new moment scores were produced. Three checkpoints came back from
v19's Output and twelve from notebook 08, `winner_patch` was scored from its stored weights,
giving 15 rows over 6 arms. The moment table is identical to the run before it. Everything
new is in sections 7 and 8.

## Artifact diagnostics: `sweep_winner_p10` seed 42, 300 validation channels

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

**Invented structure is a threshold, not a gradient.** The scatter in `figure_cell24.png` is
bimodal. Roughly 25 channels sit near 100% invented background area, everything else sits
near 0%, and almost nothing falls between. Channels do not degrade smoothly as SNR drops,
they flip. Every channel in the failing cluster is below SNR 0.5 or so. That makes the 6.65%
mean misleading as a typical channel; it is really about 8% of channels failing almost
completely.

**The floor leak is systematic rather than occasional.** `denoised.min` clusters near -0.10
and -0.155, and no channel comes near zero, which is where the clean floor sits. Every
channel is decoded with a negative pedestal. It is the same class of failure as the DDPM's
positive pedestal, just on the other side of zero, and it also drives the overshoot statistic.

**The worst channels are where the metric stops meaning anything.** The three in
`figure_cell26.png` report `SNR 0.0` and `invented 99.1-99.96%` across about 65,500
background pixels. "Invented" is measured against 20% of the clean peak, and on a signal-free
channel that peak is a noise spike, so the bar is effectively zero and any pedestal clears it.
These are real failures, but the percentage attached to them is not a meaningful magnitude.
It is RULES.md #8 inverted: there the suspect number was a zero, here it is 99.96%, and
either way the denominator is the thing to check first.

## Why the SNR split matters

The contrast between the two halves is the strongest version of the faint-channel finding
this project has: **347x in invented area and 5.3x in blob rate**, at the same split. It
supports the idea that M2 suffers from wing truncation, because the model's inventions and
the line wings occupy the same faint regime.

An earlier run (v18) reported 1.580 against 0.213 for the same split. That was a different
checkpoint on a different run, so it is a separate measurement rather than a correction to
this one.
