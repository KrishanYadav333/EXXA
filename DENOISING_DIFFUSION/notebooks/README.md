# Notebooks, as they ran

Each file here is the **executed** copy, real cell outputs, not a re-runnable script. Same
code that produced the midterm's numbers; this is the evidence it ran and what it produced.

| notebook | run | what it establishes |
|---|---|---|
| `05-unet-line-emission.ipynb` | Kaggle Version 21 | U-Net moment scores, seed bands, artifact diagnostics |
| `06-ddpm-line-emission.ipynb` | Kaggle Version 13 | conditional DDPM's PSNR/moment scores and the pedestal failure |
| `07-classical-baselines.ipynb` | v2, commit `eb03589` | classical filters against the U-Net, on moment maps |
| `08-seeds-and-augmentation.ipynb` | v2, commit `eb03589` | the 12-checkpoint seed bands the other notebooks reuse |
| `09-architecture-comparison.ipynb` | v7, commit `ee491fc` | U-Net vs autoencoder vs VAE, same protocol |

**On 05 v21 specifically:** its `winner_beam` arm was scored without feeding the trained beam
vector back in at inference (`UNet.forward` ignores a missing one silently), which understates
that arm's moment scores. The fix and the corrected number are on `midterm-prep`, not in this
notebook, see the note in `../MODELS.md` and `../results/RUNS.md`.

Full provenance, per-run READMEs, and every figure these produced are under
`../results/<notebook-name>/`. This folder is the five notebooks pulled into one place with
their outputs; that folder is the run-by-run detail behind each.
