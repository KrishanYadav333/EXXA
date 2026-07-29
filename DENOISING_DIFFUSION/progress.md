# Progress Log — Line-Emission Denoising

Local-only tracking doc, tracked in git as of `b6cdd79`. Newest first. Only entries whose
work actually ran with real data claim results; implementation-only entries say so.

## 2026-07-26 — Beam A/B + 12-run sweep RAN on Kaggle (Version 15); DDPM notebook split off

- **Beam A/B + sweep (05, from the 2026-07-24 implementation below) ran on Kaggle, pulled and
  verified from real notebook output — supersedes the "not yet run" status of that entry:**
  - Control (no beam): PSNR 33.70 | SSIM 0.9865 | MSE 0.000597 (early-stopped ep 48, best ep 43).
  - Beam: **+1.242 dB PSNR, +0.00169 SSIM** over control → pixel-metric verdict "beam helps."
  - Beam 5-cube moment-map holdout is mixed: M0 +59.8%±33.0% (V12: +69.8%±15.2% — mean
    *dropped*, variance more than doubled; one cube fell to +3.0% M0), M1 +19.2%±8.4%
    (slightly better), M2 +23.2%±20.2% (slightly better, more variance). Beam trades M0
    reliability for a small pixel-metric gain — not an unambiguous win.
  - 12-run random sweep: **best config base=48, mult 1×2×4×8, lr=0.00082, alpha=0.888,
    use_beam=False → PSNR 37.11 | SSIM 0.9902 | MSE 0.000290** — beats V12 by +4.16 dB.
    Alpha correlates strongest with PSNR (r=+0.62); **beam correlates negatively (r=-0.33)**,
    contradicting the A/B's own verdict — all top-5 configs have `use_beam=False`.
  - **Not done yet: the PSNR-37.11 config has never been through the all-5-holdout moment-map
    eval** — only the beam model got that. Real but unverified on the metric that matters.
- **DDPM split into its own standalone notebook** (`06-ddpm-line-emission.ipynb`, root,
  self-contained git-clone bootstrap — not a separate git branch; that plan changed once a
  standalone-notebook approach proved workable without depending on Kaggle's native
  GitHub-linked-notebook sync). Never touches U-Net/V12 files.
  - The 2026-07-20 retune (N_SAMPLES 150, 60 ep, ema 0.99, K_AVG=4 posterior-mean, DDIM 25
    steps) **did run**, correcting this log's prior "retrain never ran" status: checkpoint
    saved, best val loss 18.3447 @ epoch 43.
  - Validation (300 ch): PSNR 18.75 | SSIM 0.4652 | MSE 0.0284. PSNR up vs the original broken
    run (14.25 → 18.75) but **SSIM down** (0.55 → 0.4652) — "better than last time" only half
    true. Still far below V12.
  - All-5-holdout moment-map eval crashed before scoring any cube: missing
    `torch.nn.functional` import (`F.interpolate` called, never imported). Fixed and pushed
    (`aedf0f9`, moved to root in `09bfcb9`) — **not yet re-run to completion**, no DDPM
    moment-map number exists.
- **Correction, not a recurrence:** `context.md`/`progress.md` (this file) were never "lost" —
  both have been tracked in git since `b6cdd79`. Today's context.md rewrite folded in the
  above beam/sweep/DDPM results that a stale draft had omitted; this entry does the same here.

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

1. Run the PSNR-37.11 sweep winner through the all-5-holdout moment-map eval — the real
   open question now, ahead of everything below.
2. Re-run `06-ddpm-line-emission.ipynb` Section 11 (import fix already committed) for the
   first real DDPM-vs-V12 moment-map comparison.
3. Resolve the beam M0-variance tradeoff — decide whether beam ships given the sweep's
   negative correlation with PSNR.
4. V12 not yet archived in `notebooks/kaggle_versions/` (V7/V9 are).
5. n=1 vs n=5 continuum decision (V9 finding) — ask Jason.
6. Self-gravitating cubes from Jason — not received.
7. Bayesian sweep seeded from the 12 random runs now that alpha stands out (r=+0.62).
8. Midterm evaluation Aug 10–14 (week 7-8/22 now).
