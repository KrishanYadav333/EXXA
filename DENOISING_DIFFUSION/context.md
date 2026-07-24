# EXXA Line-Emission Denoising — Context

Local-only tracking doc (gitignored). Last updated: **2026-07-24** (week 7 of 22; midterm evaluation Aug 10–14).

## Project

GSoC 2026, ML4Sci EXXA — denoising synthetic ALMA observations of protoplanetary disks
(hydro sims + radiative transfer). Mentors: Sergei Gleyzer, Jason Terry. Student: Krishan Yadav.
Track: **line-emission velocity cubes** (mentor pivot 2026-06-18, away from patch-based continuum images).

## Working setup

- Branch: `line-emission`. Work scoped to `DENOISING_DIFFUSION/` + root `05-unet-line-emission.ipynb`.
- Root `05` notebook is the **only** Kaggle-linked notebook — all training runs happen there
  (bootstrap cell clones the repo on Kaggle; local machine has **no FITS data and no checkpoints**).
- Kaggle: T4×2, DataParallel. Results pulled back as CSVs/pngs into `results/`; frozen notebook
  snapshots archived in `notebooks/kaggle_versions/`.
- Rule: never claim a result "done" unless it actually ran with real data and produced artifacts.

## Data

14 cubes, 11 RunIDs, `(201, 600, 600)` FITS pairs (dirty/clean). Cube-level split with
`n_holdout=3` RunID groups (0002/0025/0026 = 5 cubes held out, never in train/val).
Channels sampled per cube (Gaussian sampler, center 100), downsampled to 256×256,
**continuum-subtracted** (mean of first/last CONTINUUM_N line-free channels), per-channel
min-max normalized using the **dirty** channel's (min,max) for both dirty and clean
(invertibility at inference; clean can exceed 1 → linear output head).

## Reference baseline — V12 (U-Net)

`unet_line_emission_continuum_best.pth`, epoch 28, val_loss 0.0034 (HybridLoss 0.8·MSE + 0.2·(1−SSIM)):
- Validation (100 ch): **PSNR 32.95 dB | SSIM 0.9857 | MSE 0.000681**
- 5-cube holdout moments: **M0 +69.8%±15.2% | M1 +17.5%±7.8% | M2 +20.1%±14.3%**
- Known issues: ~15% peak overshoot (1.151× clean max at ch100), slight negative floor leak
  (−0.0017 vs clean ~0), M2 highest cube-to-cube variance.

V7: first continuum A/B — SSIM 0.9868, single-holdout M0 +84.9/M1 +20.9/M2 +18.4 (one cube, no error bar).
V9: continuum_n ablation — **n=1 beats n=5** on every pixel metric; never folded back into the
main pipeline (still trains n=5) — open decision with Jason.

## DDPM (side comparison — parked)

Conditional DDPM (`src/training/diffusion.py`, DDIM sampling). First Kaggle run was garbage
(PSNR 14.2, M0 −1545%) — root cause: EMA `mu=0.999` frozen near random init on a ~1300-step run,
sampler drew from it. Fixed (bias-corrected EMA warmup) + posterior-mean sampling (`n_avg`),
grad-clip 1.0, LR warmup, retune (N_SAMPLES 150, 60 epochs, ema 0.99, K_AVG 4). **Retrain never
ran** — Jason's 2026-07-20 call: *stick with the U-Net*, DDPM is a comparison baseline only.
Structural truth: single-draw diffusion loses to a regressor on PSNR/moment metrics by construction.

## Mentor direction (2026-07-20 meeting)

1. **Beam metadata** (brief; drop if no help): 4-vector `[sin(2·BPA), cos(2·BPA), BMAJ·3600, BMIN·3600]`
   (BPA deg→rad) as extra model input reaching the upsampling path.
2. **Hyperparameter sweeps** (the real focus): random first, then Bayesian seeded with the random
   runs; sweep width/depth/lr/scheduler/loss-weights; score on FIXED metric (PSNR/MS-SSIM);
   early stopping min ~20 / max ~100 / patience 3–5. W&B optional.
3. **Self-gravitating dataset**: Jason will send — moment-map TEST only (kinematic substructure
   recovery), not training. Blocked on him.

## Current state (2026-07-24, pre-meeting)

Implemented + pushed (`0bc1c82`, docs `122568c`): beam features in dataset (`beam_features_of`,
`return_beam`), U-Net `beam_dim` conditioning via time-embedding add (checkpoint back-compat),
sweep harness `src/training/sweep.py` (train_unet with early stopping, run_sweep random search,
crash-safe CSV, OOM halving), root 05 rebuilt: beam A/B **with no-beam control** (isolates beam
from the early-stopping schedule change), 5-holdout moment maps, 12-run sweep + correlation
analysis. Tests green locally (`tests/test_beam_sweep.py`, `test_ema_warmup.py`, `test_ddpm_sampling.py`).
Design doc: `BEAM_AND_SWEEP.md`.

**Not yet run on Kaggle** — no beam/sweep numbers exist yet. Beam-variance tripwire in §3 of the
notebook: if beam vectors identical across cubes, beam experiment is null by construction.
