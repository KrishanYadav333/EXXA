# results/

One folder per notebook, holding the run behind the matching file in
[`../notebooks/`](../notebooks/):

| folder | run | notebook |
|---|---|---|
| `05-unet-line-emission/v21_2026-08-17_6f5c798/` | Kaggle Version 21 | `05-unet-line-emission.ipynb` |
| `06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/` | Kaggle Version 13 | `06-ddpm-line-emission.ipynb` |
| `07-classical-baselines/v2_2026-07-31T1904_eb03589/` | v2 | `07-classical-baselines.ipynb` |
| `08-seeds-and-augmentation/v2_2026-07-31_eb03589/` | v2 | `08-seeds-and-augmentation.ipynb` |
| `09-architecture-comparison/v7_2026-08-02T1721_ee491fc/` | v7 | `09-architecture-comparison.ipynb` |

Each folder carries that run's figures and a README covering what it established, and where
it matters, what changed since the run before. The notebook itself, with its full stdout
sitting in the cell outputs, is in [`../notebooks/`](../notebooks/) rather than copied here
a second time.

This is a pruned view. `RUNS.md` and `PROGRESS.md` still document the whole run history in
text, every sweep and seed and diagnostic pass, but only the five runs above keep their
figures on this branch. The rest, logs included, stay on `midterm-prep`.

## Reading the numbers

Every moment-map number names its metric, which is one of raw, 3σ-clipped, or clipped plus
signal-masked. Numbers from different metrics are not comparable, and `RUNS.md` states which
one each run used.

`results/checkpoints/` is a local scratch path some notebooks write to. It is gitignored and
not part of this branch. [`../MODELS.md`](../MODELS.md) says where the checkpoints actually
live.
