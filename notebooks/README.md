# Notebooks, as they ran

Each file here is the **executed** copy — real cell outputs, not the re-runnable script
version at the repo root. Same code; this is the evidence it ran and what it produced.

| notebook | run | what it establishes |
|---|---|---|
| `05-unet-line-emission.ipynb` | Kaggle Version 24 | the U-Net moment scores, seed bands, artifact diagnostics, and the corrected beam-conditioning arm |
| `06-ddpm-line-emission.ipynb` | Kaggle Version 13 | the conditional DDPM's PSNR/moment scores and the pedestal failure |
| `07-classical-baselines.ipynb` | v2, commit `eb03589` | classical filters against the U-Net, on moment maps |
| `08-seeds-and-augmentation.ipynb` | v2, commit `eb03589` | the 12-checkpoint seed bands the other notebooks reuse |
| `09-architecture-comparison.ipynb` | v7, commit `ee491fc` | U-Net vs autoencoder vs VAE, same protocol |

For 05: Version 24 supersedes Version 22 (both diagnostics-complete) by fixing the beam arm's
scoring bug — see `../DENOISING_DIFFUSION/results/RUNS.md`. Version 22 and the crashed
Version 19 before it are also archived, without their own notebook copies, under
`../DENOISING_DIFFUSION/results/05-unet-line-emission/`.

Full provenance, per-run READMEs, and every figure these produced are under
`../DENOISING_DIFFUSION/results/<notebook-name>/`. This folder is the five headline
notebooks pulled into one place; that folder is the complete history.
