# 09 Kaggle Version 3 -- GPU scoring, frac=0.05 corrected numbers, first confirmed run

No training. Same comparison as `experiments/wiggle_all_methods.py`: dirty / U-Net / DDRM /
beam-only against clean, scored with the corrected `compare_wiggles()` (one shared Keplerian
model, fit on the clean cube, reused for every method). This is the rerun that settles the
frac fix from `613aa0e` (2026-09-10): v2's "third confirmation" table was computed at
`frac=0.02`, the mask width Jason flagged as too loose, not the `frac=0.05` used everywhere
else. This run is the first GPU confirmation at the corrected frac.

GitHub-imported fresh so the notebook picked up the fix, then Run All. Not auto-pushed
through Kaggle's GitHub integration (no matching push commit in the git log), downloaded
manually instead -- same situation as v2, `code` is the commit confirmed pulled from the
run's own cell 0/0b log rather than a push commit.

Pulled commit `5c51ad7`. Dataset resolved at
`/kaggle/input/datasets/krishanyadav333/kaggle-wiggle-scoring-dataset/`.

## Result

240-360, step 1, 121 channels, 7.4 minutes:

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8619 |
| beam-only | 0.211 | 0.9912 | 0.8878 |
| U-Net | 0.219 | 0.9821 | 0.7601 |
| DDRM | 0.392 | 0.9471 | 0.5681 |

Shared geometry (fit on clean): mstar=0.644 Msun, incl=27.4deg, pa=0.4deg, vsys=0.111.
Well clear of the 50 Msun bound, not degenerate (RULES.md #8).

240-360, step 4, 31 channels, 1.9 minutes: dirty 0.873 / beam-only 0.882 / U-Net 0.764 /
DDRM 0.630 -- same known coarse-sampling compression toward 1.0 as every previous run at this
config, not a separate finding (see the script's own docstring).

## What this settles

Cross-validates against the local CPU frac=0.05 sweep run the same day (`5c51ad7`, on
`wiggle_all_methods_step1.npz`'s cached M1 maps), independent code path (GPU vs CPU) and
independent cube read (fresh FITS load vs cached maps):

| run | dirty | beam-only | U-Net | DDRM | mstar |
|---|---|---|---|---|---|
| local CPU frac sweep, frac=0.05 (2026-09-10/11) | 0.862 | 0.888 | 0.760 | 0.571 | 0.643 |
| this run, Kaggle GPU Version 3, frac=0.05 (2026-09-11) | 0.862 | 0.888 | 0.760 | 0.568 | 0.644 |

Matches to three decimal places on three of four methods, DDRM within 0.003. The frac=0.05
"third confirmation" table is no longer provisional -- it is now the frac the mask-fix
required, not the flagged frac=0.02 it was accidentally still running at.

## New finding surfaced while reviewing this run's own figure

`wiggle_all_methods.png` (included in this archive) shows the U-Net and DDRM M1 panels
visibly smoother than dirty and beam-only, more than the known MSE regression-to-mean effect
alone explains. Traced to `unet_denoise()`/`ddrm_restore()` resizing the native 600x600 cube
down to `SIZE=256` for inference and back up afterward -- a lossy round trip that dirty and
beam-only never go through. Logged in full in `PROGRESS.md` (2026-09-11, "U-Net/DDRM
inference carries a 600->256->600 resize round trip"). Not yet fixed; patch-based inference
at native resolution agreed as the next step, no retraining required since it reuses the
existing checkpoints.
