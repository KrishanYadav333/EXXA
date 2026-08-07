# Progress Log — Line-Emission Denoising

Local-only tracking doc, tracked in git as of `b6cdd79`. Newest first. Only entries whose
work actually ran with real data claim results; implementation-only entries say so.

## 2026-07-31 — First classical baseline on line emission (07 v2); tuning bug found

**The project's central claim now has evidence.** No classical filter had ever been scored on
line-emission data, so "the learned denoiser beats traditional processing" was unsupported on
this dataset. Notebook 07 ran (CPU-only, zero GPU quota) and produced it.

Channel-level, 300 validation channels at 256px:

| method | param | PSNR | SSIM | MSE |
|---|---|---|---|---|
| dirty (unfiltered) | – | 20.43 | 0.3095 | 0.010952 |
| **gaussian** | σ=4.0 | **24.96** | 0.7005 | 0.004022 |
| median | 9 | 22.80 | 0.5300 | 0.006658 |
| wiener | 9 | 22.83 | 0.5383 | 0.006596 |
| **U-Net V12** | – | **32.95** | 0.9857 | 0.000681 |

5-cube holdout moment maps (mean ± std):

| method | M0 | M1 | M2 |
|---|---|---|---|
| gaussian σ=4 (native 600) | +11.7 ± 2.1 | +1.1 ± 0.9 | +0.5 ± 0.4 |
| median 9 | +3.9 ± 0.9 | +0.3 ± 0.3 | +0.2 ± 0.1 |
| wiener 9 | +5.0 ± 1.0 | +0.3 ± 0.5 | +0.2 ± 0.2 |
| **gaussian @256 round trip** | **+36.5 ± 3.5** | +4.3 ± 3.7 | +2.1 ± 1.8 |
| **U-Net V12** | **+69.8 ± 15.2** | +17.5 ± 7.8 | +20.1 ± 14.3 |

**A methodology bug the run itself exposed.** Section 6 pushes the best filter through the
network's 256 round trip and labelled the difference a "penalty", assuming a round trip can only
hurt. It came out **+24.9 pp POSITIVE** on M0. A filter tripling its score by being downsampled
first is a symptom, not a result.

Cause: `tune_on_validation` ran on 256×256 channels while `denoise_cube` applied the chosen
parameter at native 600×600. Filter parameters are in **pixels**, so σ=4 tuned at 256 smooths
2.3× too little at 600 — and the round trip scored better precisely because it restored the
resolution the filter was tuned for. Compounding it, the optimum σ=4.0 sat on the **grid edge**
(old grid topped out at 4.0), so the filter was never given its best. Both handicapped the
classical side, the opposite of the notebook's stated "generous to classical" framing.

Fixed in `be7d79d`: tuning now runs at native resolution, grids widened to bracket the optimum
(gaussian to 16.0, windows to 21), an explicit warning fires on grid-edge hits, and the headline
reports whichever classical *configuration* scores best rather than the native row.

**What to quote until 07 is re-run:** the defensible U-Net advantage is **+33.3 pp on M0**
(69.8 vs the round-trip figure 36.5), not the +58.1 pp the notebook printed against the
mis-tuned native row. Still decisive. A re-run (~10 min, CPU) gives the final number.

Run outputs are preserved in git at `7dbd17b`; the current notebook is the fixed version and
carries no outputs. The result CSVs were persisted to `/kaggle/working` but never downloaded —
re-run and commit those instead, since these numbers are superseded.

## 2026-07-30 — V16: sweep winner FAILED to reproduce; two artifact "known issues" corrected

Kaggle Version 16 (`227d2fc`, on `line-emission`) retrained the sweep winner and ran the
artifact diagnostics. Three findings, all from real notebook output.

**1. The +4.16 dB headline was not real.**
- Sweep run 7 recorded **37.109 dB**. Retrained under the same config/split/patience:
  **30.278 dB**. Reproduction delta **−6.831 dB**, and **−2.672 dB vs V12** — i.e. worse
  than the reference it was supposed to beat.
- Mechanism: the sweep run reached epoch 38 (best 33); the retrain early-stopped at
  **epoch 21** (best 16). Only the seed differed (sweep used 49, retrain 42). The val curve
  plateaued sooner by luck, `patience=5` fired, and the model undertrained.
- **Early stopping amplifies seed variance into ~7 dB swings** on this small, noisy
  300-channel val set. That is the real finding, and it is a methodological one.
- Moment maps agree the winner is not an improvement: M0 **+64.4%±14.6** (V12 +69.8±15.2,
  −5.4 pts), M1 +13.8±6.9 (−3.7 pts), M2 +23.1±13.0 (+3.0 pts). 5/5 cubes positive.
  Notebook verdict correctly refused promotion. **V12 remains the reference checkpoint.**

**2. The sweep's stated conclusion is confounded** (`src/evaluation/sweep_analysis.py`,
re-analysis of the committed CSV — no GPU). `epochs_run` correlates **+0.650** with PSNR,
more than any hyperparameter, and is itself correlated **+0.535** with alpha. Partialling
out training duration reorders everything:

| predictor | raw r | partial r | shift |
|---|---|---|---|
| base_channels | +0.430 | **+0.821** | +0.391 (suppressed) |
| epochs_run | +0.650 | +0.480 | −0.170 |
| alpha | +0.622 | **+0.428** | −0.195 (inflated) |
| use_beam | −0.327 | −0.390 | −0.063 |
| log10 lr | −0.116 | +0.183 | +0.299 |

Wider models early-stop sooner, masking their benefit. **Model width, not alpha, looks like
the driver** — the opposite of what the sweep concluded and of what would have seeded the
Bayesian follow-up. Caveat carried with every citation: n=12, partial correlations on 12
points are noisy, not significance tests.

**3. CORRECTION 2026-07-31 — the artifact diagnostics were measuring nothing.** The SNR
column came back `NaN` for every channel, and the cause invalidates the whole panel:
`channel_artifacts` defined background as `clean < 10% of peak`, which assumes the clean
background sits near zero. Under shared dirty-scale normalisation it does not — continuum
subtraction makes DIRTY's minimum negative, so clean's zero background maps to
`-lo/(hi-lo)`, about **+0.32** on a representative channel. The mask therefore selected
**zero pixels**, which means:
- **"0 of 100 channels contain invented structure" is vacuous**, not evidence — `invented`
  is `background & (...)`, and an empty background is empty by construction.
- **"floor leak +0.145, no negative leak" restates the offset**, since clean's floor is not
  ~0 in this normalisation.
- **the low-SNR split never ran**, so "is hallucination low-SNR specific?" — the question
  flagged to Jason — remains completely open.
- **overshoot 0.929 is offset-compressed**: a shared positive offset drags `max/max` toward
  1, so the "model undershoots" reading understates the true deviation.

Fixed: all four quantities are now measured relative to each channel's own floor and span,
making them invariant to any affine rescaling, and `summarise` reports mask health so an
empty background can never again pass as a clean result. Verified that an invented source the
old code scored as zero blobs is now detected. **Nothing about hallucination is currently
established; the panel must be re-run.**

The original point does still stand — a single channel cannot characterise an artifact — but
the replacement numbers below were not evidence either, and are struck.

~~**3. Two documented artifacts do not survive n=100**~~ (superseded by the correction above) (artifact diagnostics over 100 val
channels, on the *winner* checkpoint — not V12, so these do not retroactively disprove V12's
figures):
- **Peak overshoot**: documented as "~15% overshoot (1.151×)" from channel 100 alone. Actual
  distribution **mean 0.929, median 0.907** — the model *undershoots* on average. Only 19% of
  channels overshoot by >10% (p90 1.161, max 1.424).
- **Negative floor leak**: documented as −0.0017. Actual **mean min +0.145, most negative
  +0.086** — strictly positive, no negative leak. New hypothesis worth testing: a +0.145
  positive pedestal summed over 201 channels would inflate M0, a plausible cause of the M0
  regression above.
- **Invented structure ("hallucination")**: **0% of 100 channels** contained a fake blob
  (background >20% of clean peak, ≥20 px). Report as *not detected under this definition*,
  not solved — may be config-specific or the threshold may be lenient.

Same lesson as V7/V9 in a new guise: the artifact list was built from single-channel
anecdotes and does not survive a distribution.

**Infrastructure built this session** (branch `midterm-prep`, `line-emission` untouched):
- `src/evaluation/classical.py` + `07-classical-baselines.ipynb` — tuned Gaussian/median/
  Wiener on the 5-cube moment protocol. **No classical filter had ever been run on
  line-emission data**, so "how much better than a Gaussian filter?" had no answer. CPU-only,
  zero GPU quota. Deliberately generous to classical: native 600×600 (no 256 round-trip
  penalty the U-Net pays) and per-filter tuning on validation.
- `FITSChannelDataset(augment=True)` — D4 augmentation (8 lossless orientations, applied
  identically to dirty and clean). Pipeline had none, on 14 cubes. Off by default.
- `repeat_config` + `08-seeds-and-augmentation.ipynb` — 3 configs × 3 seeds with a verdict
  that refuses gaps inside the combined spread. Built before V16 landed; V16 is exactly the
  failure mode it exists to catch.
- Test suite runnable locally on **Python 3.12.13** (matches Kaggle exactly): 11 script
  modules + 60 pytest cases green. Previously the suite could not run at all on this machine,
  which is why the DDPM missing-import bug cost a whole GPU session.
- `requirements.txt` permitted an environment where moment-map evaluation **crashes**:
  bettermoments 1.10 needs `np.trapezoid` (numpy ≥2.0) while astropy <6.1 pins numpy <2.
  Empirically confirmed — astropy 6.0.1 + numpy 2 breaks FITS loading itself
  (`numpy._core.umath has no attribute '_ljust'`). Pinned numpy≥2.0.0, astropy≥6.1.0.
- Fixed a latent bug: `normalize_image(method="percentile")` returned 1.0000001 on float32.

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
6. ~~Self-gravitating cubes from Jason — not received.~~ Received 2026-08-07 (see below).
7. Bayesian sweep seeded from the 12 random runs now that alpha stands out (r=+0.62).
8. Midterm evaluation Aug 10–14 (week 7-8/22 now).

## 2026-08-07 — Jason clarifies midterm deliverable: a blog post, not a report

Emailed Jason 2026-08-03 asking what to prepare for Aug 10–14: format, whether current
U-Net result is sufficient, whether ALMA validation is needed by then. Reply came 2026-08-07:

- **Deliverable is a WordPress blog post** (ML4Sci's standard midterm ask), not a formal
  written report or a notebook walkthrough. Should cover background, methods, initial
  results. Explicitly **does not need to be complete or polished** ("we don't expect it to
  be perfect because you're not done yet").
- **U-Net is fine to present as the focus** — Jason confirmed directly, no further tuning
  demanded before midterm.
- **ALMA real-data validation is deferred to the FINAL blog**, not midterm. "For now, just
  focus on the synthetic data, we can get to ALMA data for the final blog."
- Include a link to a public notebook alongside the post.
- **"Don't worry about passing your evaluation though. You're fine there."** — direct
  reassurance from the mentor; the bar for Aug 10–14 is lower than the rigor this session has
  been pursuing (seed bands, corrected metrics, cross-notebook comparability). That rigor is
  still worth finishing for the *final* evaluation and for the blog's own credibility, but it
  is not gating the midterm.
- **New data received**: self-gravitating cube + a matching dirty cube, shared via Google
  Drive (link in the email thread, not the repo). Jason: "I'll give you more later." Not yet
  pulled into the repo or looked at — open item.

Replied same day with the planned blog structure: classical baselines → architecture
comparison (autoencoder, VAE, U-Net, DDPM) → U-Net results on synthetic line-emission data →
link to public notebook. ALMA left for the final blog per Jason's steer.

**Consequence for priority ordering**: the blog needs the U-Net story to be presentable, not
every notebook to be re-run under the corrected metric. 06 (DDPM) having a real, if weak,
completed result (PSNR ~18–19 dB) is enough to write the "why U-Net over DDPM" comparison
section — it does not need the memory/seed fixes chased earlier today to be *good*, just to
exist and be honestly reported.
