# 06 — Kaggle Version 13 Output

Drop the contents of notebook 06's **Version 13** Output here.

## Why this folder exists

Kaggle's *Add Input* pins a notebook input to that notebook's **latest** version, with no
version selector. The run after v13 crashed at cell 28 and became the latest, and its Output
is nearly empty — it died before section 15 ran. So v13's checkpoints can no longer be
reached by attaching the notebook to itself, and they have to come back through a Dataset.

## What to put here

From Kaggle: notebook 06 → Versions → **Version 13** → Output → download.

| file | size | needed for |
|---|---|---|
| `ddpm_seed42.pth` | 332 MB | **required** — the trained model every diagnostic arm re-scores |
| `ddpm_sweep_patch.pth` | 332 MB | the `patch_model` arm only |
| `ddpm_sweep_*.pth` (3 more) | 332 MB each | not needed; the sweep is settled |
| `ddpm_objective_sweep.csv` | small | section 7's table without re-running the sweep |
| `ddpm_seed_repeats.csv` | small | lets section 8 skip training |
| `moment_map_holdout_summary_ddpm.csv` | small | lets section 13 skip re-scoring |
| `ddpm_moment_maps.npz`, `*.png` | small | the figures |

Checkpoints are gitignored. The CSVs and PNGs are not — they are the run's record.

## Uploading them back to Kaggle

**Rename every checkpoint to `.ckpt` first.** A torch checkpoint is internally a zip, and
Kaggle unpacks archive-like files on Dataset upload, so a `.pth` arrives as a *directory* of
`data.pkl` / `byteorder` / `version` that `torch.load` rejects with `Is a directory`. This
destroyed two earlier uploads (RULES.md #3).

    ddpm_seed42.pth  ->  ddpm_seed42.ckpt

Cell 6b accepts `.pth`, `.pth.tar` and `.ckpt`, and maps all three to the `.pth.tar` name
section 8 loads. Notebook **Outputs** are mounted verbatim and need no renaming — this only
applies to the Dataset route.

## Then

Pull the notebook (`4d18986` or later), attach the dataset, Run All. Cell 6b should print:

    restored ddpm_seed42 -> ddpm_seed42.pth.tar
    restored BEST v-prediction checkpoint (epoch 58, best_val 56.6989)
    [resume] seeds already trained: [42]

If instead it says `seed 42 is in the CSV but its checkpoint is missing -- will retrain`,
the checkpoint did not arrive: check it is not a directory in the dataset's file browser.
