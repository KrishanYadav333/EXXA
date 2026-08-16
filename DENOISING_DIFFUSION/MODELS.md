# Model checkpoints

None of these are in this branch. Every checkpoint used in the midterm is at least 100 MB
(the DDPM's is 332 MB), `.pth`/`.ckpt`/`.pth.tar` are gitignored project-wide, and RULES.md
#3 exists specifically because a checkpoint uploaded the wrong way to Kaggle becomes
unusable. The real home for a checkpoint is the Kaggle Notebook Output (or Dataset) that
produced it, per notebook 05/06's own `_import_*` restore logic — not git.

This file is the substitute: which checkpoint, from which run, with the numbers that
identify it, so anyone can find the right one on Kaggle without guessing.

## U-Net (notebook 05 / 08)

12 core checkpoints trained once in notebook 08, reused everywhere since (05, the blog's
moment tables). Source: 08 Kaggle Version producing commit `1ca611f`
(`results/08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/`).

| config | seed | best epoch | val loss | PSNR | SSIM |
|---|---|---|---|---|---|
| v12 | 42 | 59 | 0.001492 | 38.659 | 0.99343 |
| v12 | 43 | 17 | 0.002018 | 36.671 | 0.99121 |
| v12 | 44 | 21 | 0.001848 | 37.466 | 0.99196 |
| winner | 42 | 21 | 0.001125 | 37.230 | 0.99228 |
| winner | 43 | 23 | 0.001060 | 37.724 | 0.99264 |
| winner | 44 | 23 | 0.001023 | 38.969 | 0.99292 |
| winner_aug | 42 | 26 | 0.001077 | 38.939 | 0.99240 |
| winner_aug | 43 | 46 | 0.000914 | 39.808 | 0.99346 |
| winner_aug | 44 | 29 | 0.001020 | 39.141 | 0.99277 |
| winner_p10 | 42 | 30 | 0.000920 | 38.954 | 0.99352 |
| winner_p10 | 43 | 35 | 0.000927 | 39.034 | 0.99355 |
| winner_p10 | 44 | 54 | 0.000809 | 39.830 | 0.99427 |

3 more, trained in notebook 05 directly (`sweep_winner` seed 49, `winner_beam`,
`winner_patch`, all seed 42). Source: `results/05-unet-line-emission/v21_2026-08-17_6f5c798/`.

| config | seed | best epoch | PSNR |
|---|---|---|---|
| sweep_winner | 49 | 11 | 36.162 |
| winner_beam | 42 | 24 | 38.710 |
| winner_patch | 42 | 21 | 33.960 |

**`winner_beam`'s checkpoint is fine; its moment-map score on this branch is not.** v21
scored it without feeding the trained beam vector back in at inference, which silently ran
the model with its conditioning branch dead (`UNet.forward` ignores `beam=None`). The fix
and the corrected score (M0 −95.7% → +9.6%, a 105-point swing) are on `midterm-prep`
(commit `a2af521` fixes it, run `v24` re-scores it) but excluded from this snapshot by
request — this branch is v21 as it stood at submission. See `results/RUNS.md` on
`midterm-prep` for the full correction if you need the accurate number.

**Architecture**: `base_channels=48, channel_multipliers=(1,2,4,8)` for winner/winner_aug/
winner_p10/winner_beam/winner_patch; `base_channels=32, (1,2,4)` for v12. `beam_dim=4` only
for winner_beam, `0` otherwise. Definition: `DENOISING_DIFFUSION/src/models/unet.py`.

## Conditional DDPM (notebook 06)

One production checkpoint, `ddpm_seed42.pth` (332 MB), from 06 Kaggle Version 13
(`results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/`): 60 epochs, v-prediction,
cosine schedule, min-SNR weighting, PSNR 38.180 / SSIM 0.9933.

Definition: `DENOISING_DIFFUSION/src/training/diffusion.py`.

## Getting a checkpoint

They live in each notebook's Kaggle Output, or in the Datasets built to move them across
Kaggle accounts (`exxa-nb08-checkpoints-v4`, `exxa-nb05-checkpoints-v19`) — see
`DENOISING_DIFFUSION/RULES.md` #1 and #3 on `midterm-prep` for why they are never in git,
and `notebooks/05-unet-line-emission.ipynb` (`_import_nb08`, `_import_prior_nb05`) for the
actual restore code.
