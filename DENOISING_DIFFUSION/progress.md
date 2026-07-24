# Progress Log — Line-Emission Denoising

Local-only tracking doc (gitignored). Newest first. Only entries whose work actually ran
with real data claim results; implementation-only entries say so.

## 2026-07-24 — Beam conditioning + sweep harness implemented (NOT yet run)

- Implemented both action items from the 2026-07-20 mentor meeting; pushed `0bc1c82`:
  - `beam_features_of(header)` → `[sin(2·BPA), cos(2·BPA), BMAJ·3600, BMIN·3600]`
    (deg→rad; verified against Jason's example header → `[0.5573, 0.8303, 0.1535, 0.1185]`);
    `FITSChannelDataset(return_beam=True)`; warns if beam identical across cubes.
  - `UNet(beam_dim=4)`: beam MLP added to the time embedding → conditions every block incl.
    upsampling path; `beam_dim=0` keeps V12 checkpoint loading unchanged.
  - `src/training/sweep.py`: `train_unet` (early stop min 20/max 100/patience 5, best-epoch
    restore) + `run_sweep` (random; lr log-uniform 1e-4–3e-3, width {16,32,48,64}, depth,
    α∈[0.5,0.95] with β=1−α, sched patience, use_beam; fixed PSNR/SSIM/MSE scoring;
    crash-safe CSV per run; OOM batch-halving).
  - Root 05 notebook rebuilt (23 cells): beam A/B **with control** (same harness, only
    `use_beam` differs — avoids conflating beam with the schedule change vs V12's fixed
    30 epochs), 5-holdout moment maps for beam model, 12-run sweep + correlations, persist cell.
  - Tests green locally: `test_beam_sweep.py` (4 checks). Docs: `BEAM_AND_SWEEP.md` (`122568c`).
- **Pending**: Kaggle run of the new notebook (beam A/B ~1 h, sweep ~3–5 h). No numbers yet.
- Meeting with Jason today (he may only make the second half — flying).

## 2026-07-20 — Mentor meeting (showed V7/V9/V12 + DDPM run)

- Jason's calls: **stick with U-Net**; brief beam-metadata investigation (drop if no help);
  then hyperparameter sweeps (random → Bayesian seeded with random runs; fixed scoring metric;
  early stopping min ~20/max ~100/patience 3–5); self-gravitating cubes coming from him as a
  moment-map TEST only (not training) — still not received.
- Beam spec he gave: `sin(2·BPA), cos(2·BPA), bmaj·3600, bmin·3600`, BPA in degrees in header.
  Example header: BPA=16.9333°, BMAJ=4.2629e-5°, BMIN=3.2923e-5°.

## 2026-07-20 — DDPM diagnosis, fix, retune (retrain never ran; parked per Jason)

- First DDPM Kaggle run (30 ep, ~1320 steps): **PSNR 14.25 | SSIM 0.55 | M0 −1545.6%** — broken.
- Root cause: EMA `mu=0.999` (~1000-step time constant) left the EMA shadow near random init;
  `evaluate()/sample()` use EMA weights → sampler ran on near-random weights. Fixed with
  bias-corrected (Adam-style) EMA warmup + `num_updates` in checkpoint state (`bbfa26c`).
- Also shipped (`dc1a19b`): posterior-mean sampling `n_avg` (single diffusion draw carries full
  posterior variance — averages of K draws are what PSNR/SSIM reward), grad-clip 1.0, linear LR
  warmup; notebook retune N_SAMPLES 150 / 60 ep / ema 0.99 / K_AVG 4.
- Honest ceiling: even fixed, single-model DDPM likely lands below V12 on pixel metrics —
  regression beats generation in this regime; frame as comparison baseline, not the workhorse.
- Notebook housekeeping: standalone `06-ddpm-line-emission.ipynb` + generator deleted
  (`f12a20c`) — Part 2 lived only in root 05; then root 05 rebuilt DDPM-only (`6c9a200`);
  now rebuilt again as beam+sweep (see 2026-07-24). DDPM cells remain in git history.

## Earlier (from repo history / kaggle_versions/README)

- **V12** (`82abd23`): consolidation — no-continuum path dropped, 5-cube holdout eval added
  (first error bars), `continuum_of` import fix. Ep 28, val_loss 0.0034; PSNR 32.95 / SSIM
  0.9857 / MSE 0.000681; M0 +69.8±15.2 / M1 +17.5±7.8 / M2 +20.1±14.3. **Reference checkpoint.**
- **V9**: continuum_n ablation — n=1 ≥ n=5 on all pixel metrics (0.9867 vs 0.9793 SSIM);
  never folded back (main pipeline still n=5) — open decision.
- **V7**: first continuum-subtraction side-by-side — SSIM 0.9868, single-holdout
  M0 +84.9 / M1 +20.9 / M2 +18.4.
- Pre-V7: shared-dirty-scale normalization fix (clean-by-own-scale destroyed M0, −6400%);
  2026-06-18 pivot to line emission, cube-level split, full-image 256².

## Open items

1. Run beam A/B + sweep on Kaggle → paste numbers back.
2. V12 not yet archived in `notebooks/kaggle_versions/` (V7/V9 are).
3. n=1 vs n=5 continuum decision (V9 finding) — ask Jason.
4. Self-gravitating cubes from Jason — not received.
5. Bayesian sweep seeded from random runs — after random stats exist.
6. DDPM retrain with all fixes — only if time/interest permits (parked).
7. Midterm evaluation Aug 10–14 (week 7/22 now).
