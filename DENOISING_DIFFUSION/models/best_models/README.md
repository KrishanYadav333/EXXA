# Best models, finalized 2026-09-11

Hardlinks (RULES.md #12: nothing here is deleted from its original location, this folder
costs no extra disk).

## Confirmed on wiggle evidence, not just pixel metric

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

## `ddrm_prior.pth` -- kept here, but it is a negative result, not a candidate

`models/07-ddrm/`. The diffusion-restoration method, worst of 4 in every wiggle confirmation
to date (resid_r 0.568-0.587 against dirty's own 0.862-0.892, frac=0.05 corrected numbers).
Not untested, actually tested and it lost, repeatedly, on GPU and CPU both. Kept in this
folder at the user's request since it is still the best (only) checkpoint of its kind, but it
should not be read as an endorsement -- see `PROGRESS.md` for the reproduction history.

## `untested/` -- best on PSNR/SSIM, never run through the wiggle comparison

| file | role | PSNR | SSIM |
|---|---|---|---|
| `winner_p10_seed44.pth` | line-emission U-Net, patch-10 augmentation | **39.830** (highest of all 12 seed repeats) | **0.99427** (highest) |
| `ddpm_seed42.pth` | production DDPM | 38.180 | 0.9933 |

Both beat `winner_aug_seed43` on raw pixel metric. Neither has ever been scored on the wiggle.
This project has repeatedly shown pixel metric does not track wiggle recovery (RULES.md #4;
DDRM's good RMS but worst wiggle score; notebook 10 `fresh`'s good V1 moments with an
already-bad wiggle; notebook 11's better moments correlating with worse wiggle across the
board) -- so "best PSNR" earns a place here to keep it on record, not a promotion to the
confirmed table above until someone actually runs it through `compare_wiggles()`.

## Not included at all, and why

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
