# 05 — Kaggle Version 25 (2026-08-21)

Code at `889cd44`, which the run's own cell 0b output confirms. That commit has the spectral
arms but **not** the three fixes that came after it, so this run:

- printed no `winner_k1` / `winner_k2` rows in section 6's summary table (`6df764d` fixed it);
  their per-cube moments are in `run_log.txt` and are recovered below
- did not run the out-of-band check that decides whether a band-limit loss is worth building
  (`84b6cb6` added it), so the VIREO question is still open
- would have diagnosed the wrong input in section 7 had a spectral arm been selected for it

## What it produced

Trained two arms, one seed each. Everything else resumed from stored checkpoints.

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner_k1 | 40.219 | 13.9 ± 21.1 | 58.5 ± 23.1 | 14.8 ± 27.9 |
| winner_k2 | 42.269 | 18.5 ± 62.2 | 69.6 ± 20.5 | 28.6 ± 34.5 |

Spreads are **across the 5 holdout cubes at one seed**, not across seeds.

## The reason this run matters more than its numbers

It is the second run of the same two arms at the same seed with the same code, and it does
not reproduce the first.

| arm, seed 42 | first run | this run | difference |
|---|---|---|---|
| k1 PSNR | 41.948 | 40.219 | 1.73 dB |
| k1 best epoch | 25 | 18 | |
| k1 M2 | +46.2% | +14.8% | 31.4 pp |
| k2 PSNR | 41.199 | 42.269 | 1.07 dB |

A third partial run of k1 reached 41.058 at epoch 22, so the three fixed-seed runs span
40.219 to 41.948, standard deviation 0.865 dB. `winner_aug`'s spread across three different
SEEDS is 0.455 dB. **Re-running one seed varies more than changing the seed**, which no error
bar in this project currently accounts for.

The cause is GPU nondeterminism: `torch.manual_seed` does not fix cuDNN's algorithm choice,
and the T4 x2 setup splits batches across devices. Nothing here is a bug in the notebook.

## Consequence

Any single-run difference below roughly 1.7 dB, or a few tens of points on a moment, is not
evidence. The k1 conclusion drawn from the first run alone -- that spectral context lifts M1
and M2 beyond the seed spread -- **does not survive this one** and is withdrawn. What holds is
the PSNR gain over the un-augmented baseline: 40.2 to 41.9 against 37.5, consistently several
dB across all three runs.
