# Best models, finalized 2026-09-11

Hardlinks (RULES.md #12: nothing here is deleted from its original location, this folder
costs no extra disk). The three checkpoints this project would put forward right now, one
per role, chosen by the metric each is actually measured on, not by pixel metric alone.

| file | role | epoch | val_loss | PSNR | wiggle resid_r | source |
|---|---|---|---|---|---|---|
| `winner_aug_seed43.pth` | production line-emission U-Net | 46 | 0.000914 | 39.808 | 0.760 (frac=0.05) | `models/08-seeds/` |
| `winner_beam_seed42.pth` | beam-conditioned line-emission U-Net | 24 | -- | 38.710 | not scored | `models/05-unet/` |
| `sg_k3_fresh.pth` | self-gravitating, spectral context k=3 | 60 | 0.001366 | 35.114 | **0.681**, beats dirty's own 0.594 | `models/12-spectral/` |

`winner_aug_seed43` is the reference every wiggle comparison in this project is measured
against (`wiggle_all_methods.py`, `score_08_kinematic.py`, notebook 09). `winner_beam` is the
best beam-conditioned arm once the dead-conditioning-branch bug was fixed (v24, M0
-95.7%->+9.6%). `sg_k3_fresh` is the only self-gravitating-trained checkpoint in the project
to beat doing nothing on the wiggle across a genuine holdout.

## Deliberately not included, and why

- **`winner_p10_seed44.pth`** (`models/08-seeds/`) -- highest raw PSNR/SSIM of all 12 seed
  repeats (39.830 / 0.99427, edges out `winner_aug_seed43`), but never run through the
  wiggle comparison. Best on a metric that isn't the one this project cares about; genuinely
  untested on the one that is.
- **`ddpm_seed42.pth`** (`models/06-ddpm/`) -- best of the DDPM sweep (PSNR 38.180, SSIM
  0.9933), same gap: never scored on the wiggle.
- **`ddrm_prior.pth`** (`models/07-ddrm/`) -- worst of 4 methods on every wiggle
  confirmation to date (resid_r 0.568-0.587 against dirty's own 0.862-0.892). A real,
  reproduced negative result, not a candidate.
- **Notebook 10's `sg_fresh.pth`** -- WITHDRAWN. V1 looked like the best SG arm
  (+5.0/+21.8/+26.0 M0/M1/M2); V4, identical settings, same day, collapsed to
  -61.1/-27.3/-42.2. High-variance, does not reproduce.
- **`kin_gamma*.pth`** (`models/08-kinematic/`) -- kinematic-loss sweep, scoring in progress
  (4/5 holdout cubes as of this writing). No verdict yet; partial numbers show gamma=1/10
  degrading moments badly. Add here only if a gamma clears `winner_aug_seed43`'s 0.760 once
  the sweep finishes.

Revisit this list once `score_08_kinematic.py` finishes and once `wiggle_patch_unet.py`
(native-resolution patch inference, built 2026-09-11, not yet run) reports whether removing
the 600->256->600 resize changes which checkpoint or which inference method actually wins.
