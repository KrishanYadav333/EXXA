# results/

One folder per notebook. On this branch, one run per notebook, the run behind the matching
file in [`../notebooks/`](../notebooks/):

| folder | run | notebook |
|---|---|---|
| `05-unet-line-emission/v21_2026-08-17_6f5c798/` | Kaggle Version 21 | `05-unet-line-emission.ipynb` |
| `06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/` | Kaggle Version 13 | `06-ddpm-line-emission.ipynb` |
| `07-classical-baselines/v2_2026-07-31T1904_eb03589/` | v2 | `07-classical-baselines.ipynb` |
| `08-seeds-and-augmentation/v2_2026-07-31_eb03589/` | v2 | `08-seeds-and-augmentation.ipynb` |
| `09-architecture-comparison/v7_2026-08-02T1721_ee491fc/` | v7 | `09-architecture-comparison.ipynb` |

Each folder holds the run's figures and a README explaining what the run established and,
where relevant, what changed since the run before it. The notebook itself, with its full
stdout in the cell outputs, is in [`../notebooks/`](../notebooks/), not duplicated here.

**This is a pruned view.** `RUNS.md` and `PROGRESS.md` document the full run history back to
week 1, every sweep, every seed, every diagnostic pass, in text, but only the five runs
above have their figures archived on this branch. The rest, including their logs, are on
`midterm-prep`.

## Reading the numbers

Every moment-map number names its metric (raw / 3σ-clipped / clipped + signal-masked), 
numbers from different metrics are not comparable, and `RUNS.md` states which each run used.
`results/checkpoints/` is a local scratch path some notebooks write to; it is gitignored and
not part of this branch, see [`../MODELS.md`](../MODELS.md) for where checkpoints live.
