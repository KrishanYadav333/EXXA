# 05 — Kaggle Version 26 (2026-08-21)

The first run carrying all three post-v25 fixes, so the first whose summary table shows the
spectral arms and the first to measure out-of-band error. It answers a question and closes a
line of work.

## VIREO's band-limit form is refuted

```
run_0016_00550_rt_01   beam 2.47 px | k_cut 0.0980 | residual out-of-band 0.0180 | excess 0.9x
run_0036_00350_rt_03   beam 3.21 px | k_cut 0.0752 | residual out-of-band 0.0031 | excess 0.0x
median excess 0.5x | median 1.1% of residual power above the cutoff
```

The clean cubes are beam-convolved, so they are band-limited by construction, and a
prediction with power above the cutoff would be asserting structure the instrument could not
have measured. A loss could penalise that with one FFT per step. It would be pointed at
nothing: **excess is below 1**, meaning the model's error is if anything *smoother* than the
truth.

The invented structure is therefore beam-scale, not sharp. That is the worse of the two
possibilities scientifically, because a beam-scale blob is exactly what a real detection looks
like, and no band constraint can distinguish them. Together with Phase 0's `A = I`, this
closes the whole physics-informed line: DDRM, VIREO-lite's data-consistency term, and now the
band-limit variant.

## Spectral context: better pixels, not better science

Three runs of each arm at seed 42 now exist (this one, v25, and the unarchived first run).

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner_k1 (3 runs) | 41.26 ± 0.92 | 21.0 ± 9.3 | 63.4 ± 4.9 | 33.0 ± 16.3 |
| winner_k2 (3 runs) | 41.81 ± 0.55 | 16.6 ± 2.1 | 67.6 ± 6.1 | 23.8 ± 12.3 |
| winner_aug (3 seeds) | 39.30 ± 0.46 | 29.2 ± 7.2 | 74.0 ± 2.0 | 55.0 ± 13.9 |

Spectral context wins PSNR by about 2.4 dB and **loses M0 and M2 to augmentation**, across
three runs each rather than one. The gap is not seed noise this time.

So the mechanism does something real to pixel reconstruction and does not translate into
moment reliability. `winner_aug` remains the best arm on the metric that matters, as it has
since notebook 08.

## Provenance

Cell 0b pulled `df38336`, which is the Kaggle revert commit; it only touched the notebook, so
`src/` there already carried `out_of_band_power`. The cells came from Kaggle's own copy, which
had the fixes imported, which is why the check ran at all. Unlike v25's, **this version's push
preserved the committed cells** rather than reverting them.
