# Notebooks, as they ran

Every file here is the executed copy, with its real cell outputs, rather than a clean script.
Same code that produced the midterm's numbers. This is the evidence it ran and what came out.

| notebook | run | what it establishes |
|---|---|---|
| `05-unet-line-emission.ipynb` | Kaggle Version 21 | U-Net moment scores, seed bands, artifact diagnostics |
| `06-ddpm-line-emission.ipynb` | Kaggle Version 13 | the conditional DDPM's PSNR and moment scores, and its pedestal failure |
| `07-classical-baselines.ipynb` | v2, commit `eb03589` | classical filters against the U-Net on moment maps |
| `08-seeds-and-augmentation.ipynb` | v2, commit `eb03589` | the 12-checkpoint seed bands the other notebooks reuse |
| `09-architecture-comparison.ipynb` | v7, commit `ee491fc` | U-Net against autoencoder and VAE, same protocol |

One caveat on 05 v21. Its `winner_beam` arm was scored without feeding the trained beam
vector back in at inference, and `UNet.forward` ignores a missing one silently, so the moment
scores this notebook prints for that arm are wrong: −95.7 / +14.1 / −27.3 where the real
figures are +9.6 / +63.2 / +20.9. The notebook keeps the old numbers because it is archived
as it ran. [`../results/RUNS.md`](../results/RUNS.md) carries the corrected row.

The figures each run produced, along with a README explaining it, are under
[`../results/`](../results/). This folder gathers the five notebooks in one place; that one
holds the detail behind each.
