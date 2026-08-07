# EXXA — Denoising Astronomical Observations of Protoplanetary Disks
## Project Context File for Agentic IDE

Local-only tracking doc, tracked in git as of `b6cdd79` (syncing across machines). Last
updated: 2026-07-26 (week 7-8 of 22; midterm evaluation Aug 10–14).

---

## 1. WHO / WHAT / WHY

**Contributor:** Krishan Yadav (GitHub: KrishanYadav333)
**Program:** Google Summer of Code 2026, ML4Sci organization
**Project:** EXXA — Denoising Astronomical Observations of Protoplanetary Disks
**Mentors:** Sergei Gleyzer (University of Alabama, ml4-sci@cern.ch), Jason Terry
(Oxford University, jpterry@uga.edu — preferred contact, NOT Mattermost)
**Duration:** 22 weeks, May 25, 2026 – November 3, 2026 (350-hour project)
**Midterm Evaluation:** Aug 10–14, 2026
**Final Submission:** Oct 26 – Nov 3, 2026

---

## 2. CURRENT COMPUTE ENVIRONMENT

- **Primary:** Kaggle Notebooks, T4 ×2 GPUs
- Repository: KrishanYadav333/EXXA, branch `line-emission`
- Notebooks are the source of truth for Kaggle-run work — two root-level notebooks, each
  self-contained (own git-clone bootstrap, doesn't depend on Kaggle's native GitHub-linked-
  notebook sync):
  - `05-unet-line-emission.ipynb` — all U-Net work (V12, beam conditioning, sweep)
  - `06-ddpm-line-emission.ipynb` — DDPM only, standalone, never modifies U-Net/V12 files
- Local machine has **no FITS data and no checkpoints** — all training happens on Kaggle;
  results pulled back as CSVs/pngs into `results/`.
- For current repo structure, run `find`/`git ls-files` directly rather than trusting a
  hand-maintained tree here — this doc has gone stale on file paths more than once.

---

## 3. PROJECT HISTORY — MAJOR PHASES

### Phase A: Continuum Data (Weeks 2-4) — Background/Deprioritized
Worked with `clean.npy`/`dirty.npy` (975 samples, 600x600, continuum images from PHANTOM+MCFOST
synthetic ALMA observations). Trained AE, VAE, U-Net, DDPM using 64x64 PATCHES.

Results (continuum, patch-based):
| Model | PSNR | SSIM | MSE |
|---|---|---|---|
| AE HybridLoss | 19.92 | 0.7609 | 0.0139 |
| VAE | 20.00 | 0.7059 | 0.0133 |
| U-Net | 20.63 | 0.7044 | 0.0117 |
| DDPM (Kaggle, 17.2M, patches) | 14.59 | 0.2207 | 0.2869 |

DDPM significantly underperformed even classical Gaussian filtering (SSIM 0.42) — root cause:
model config never actually scaled up as planned (agreed ch=128 6-level, ran ch=64 4-level
instead — Rule 2 violation caught and documented), plus patches fundamentally limiting.

### Phase B: THE PIVOT (Week 4, mentor meeting 2026-06-18)
Mentor (Jason Terry) directed a full pivot after reviewing weak patch-based/DDPM results:
1. **Patches deprecated project-wide.** Full images only (downsample if needed for memory).
2. **Move to line emission data** (continuum treated as "sidetrack").
3. **Go back to last week's architecture (U-Net)**, not DDPM, on full images.
4. New dataset: FITS cubes, ~201 velocity channels each, clean/dirty pairs per RunID folder.
5. Channel sampling: Gaussian centered near index 100, ~75%+ mass in [50,150] (avoid
   high-velocity/continuum-dominated extreme channels).
6. **Cube-level train/val/holdout split** — hold out ENTIRE cubes (not channels) for
   inference-only end-to-end testing (denoise full cube -> moment maps).
7. New evaluation tool: `bettermoments` package — generates Moment 0/1/2 maps, the actual
   scientific deliverable astronomers use.

### Phase C: Line Emission Pipeline Built (Week 4-5)
- `src/data/channel_sampler.py` — Gaussian channel sampling, calibrated (mean 73.8% across
  20 seeds lands in [50,150], satisfies mentor's "at least 75%" as a floor)
- `src/data/cube_split.py` — RunID-grouped split, leakage-safe (RT variants of same RunID
  never split across train/val/holdout). **Dataset: 14 cubes total, 11 RunIDs** (not ~20 as
  originally estimated from mentor's verbal description). n_holdout=3 RunIDs (0002, 0025,
  0026) = 5 cubes held out. 7 train cubes, 2 val cubes.
- `src/data/fits_cube_dataset.py` — FITSChannelDataset, per-channel memmap load, downsample
  to 256x256, full images (no patches)
- `src/evaluation/moment_maps.py` — bettermoments wrapper, generate_moment_maps(fits_path)

### CRITICAL BUG FOUND AND FIXED (Week 5)
**Symptom:** First end-to-end moment-map test (denoise full held-out cube -> compare M0/M1/M2
vs clean) showed catastrophic failure: M0 -6402%, M1 -0.6%, M2 -9.7%.

**Root cause (confirmed via channel-100 diagnostic):** Training normalized clean target by
clean's OWN per-channel (min,max) — independent scale from dirty. Model learned to output in
clean-normalized space, but inference decoded using DIRTY's (min,max) (correctly — clean
min/max isn't available at real inference), creating a scale mismatch. Clean background floor
~0; dirty min is negative -> every background pixel decoded to dirty's negative floor -> M0
went strongly negative.

**Fix (commit 5ed8fc6):**
1. `fits_cube_dataset.py`: normalize BOTH dirty and clean by the DIRTY channel's (min,max) —
   shared scale, invertible at inference. Clean is NOT clipped (peak may exceed 1).
2. Dropped sigmoid from model output (linear head) — sigmoid can't represent shared-scale
   clean values that exceed 1.
3. Verified: denoised min flipped from -0.0047 (matched dirty, wrong) to +0.0001 (near
   clean's ~0, correct).

**Result after fix (single test cube):** M0 -6402% -> -1376%, M1 +13.6%, M2 +18.6-25.2%.

### Phase D: Continuum Subtraction (Week 5-6, mentor suggestion)
Mentor (email): noted continuum contamination in some "clean" images. Suggested averaging
first/last few channels into a 2D template, subtracting from all channels to isolate pure line
emission. Implemented + tested (n=1 vs n=5 window — tied). Notebook:
`notebooks/06-unet-line-emission-continuum.ipynb` (lives in the `notebooks/` subfolder, not
root — root `06` is now the DDPM notebook, see §2).

| Metric | Baseline (no subtraction) | Continuum-subtracted |
|---|---|---|
| PSNR | 31.02 dB | 33.18 dB (V7) / 32.95 dB (V12) |
| SSIM | 0.9832 | 0.9868 (V7) / 0.9857 (V12) |
| MSE | 0.000933 | 0.000602 (V7) / 0.000681 (V12) |

Moment maps (single test cube, V7): M0 -1672% -> +84.9%, M1 +13.6% -> +20.9%, M2 +18.6% ->
+18.4%. Confirmed continuum contamination was actively degrading M0; removing it first fixed
this.

### V7/V9 Variance Discovery (Methodology Finding)
V7 and V9, same config retrained twice, evaluated on the SAME single holdout cube: M2 swung
+18.4% -> +2.5% with zero config change — pure run-to-run/cube-selection variance. Revealed
single-cube evaluation is unreliable; multi-cube averaging required (V7 also showed a "1.4x
peak overshoot" at channel 100, a persistent bias across runs).

### V12 — REFERENCE CHECKPOINT (confirmed, fully verified)
First run evaluated across ALL 5 holdout cubes.

**Config:** DenoisingUNet 3.4M params, HybridLoss(alpha=0.8, beta=0.2), linear output head,
Adam lr=1e-3, ReduceLROnPlateau(factor=0.5, patience=5), batch=8, fixed 30 epochs (best @
epoch 28, val_loss 0.0034), 256x256 full images, continuum subtraction n=5, shared dirty-scale
normalization, seed=42, n_holdout=3 RunIDs (=5 cubes), 7 train / 2 val cubes.

**Validation (100 channels):** PSNR 32.95 dB | SSIM 0.9857 | MSE 0.000681

**5-cube holdout moment maps:**
| Cube | M0 | M1 | M2 |
|---|---|---|---|
| run_0002_00560_rt_00 | 82.9% | 18.4% | 18.1% |
| run_0002_00560_rt_01 | 55.1% | 17.5% | 8.4% |
| run_0002_00560_rt_04 | 79.8% | 15.3% | 10.2% |
| run_0025_01000_rt_04 | 79.8% | 29.1% | 44.1% |
| run_0026_00005_rt_04 | 51.6% | 7.4% | 19.6% |

**Summary: M0 +69.8%±15.2%, M1 +17.5%±7.8%, M2 +20.1%±14.3%.** All 5 cubes positive on all 3
moments — strongest, most honestly-evaluated result to date.

### Known Open Issues — REVISED 2026-07-30 after measuring distributions (V16)

Items 1, 2 and 4 below were each recorded from a **single channel (ch 100)**. V16 ran the
diagnostics over **100 validation channels** and two of them do not survive. Measured on the
*sweep-winner* checkpoint, not V12, so V12's own published figures are not retroactively
disproved — but the anecdotal framing is not trustworthy for either model.

1. **Peak overshoot — CORRECTED.** Documented as "~15% overshoot (1.151×)". Measured
   distribution: **mean 0.929, median 0.907** — the model *undershoots* on average. 19% of
   channels overshoot by >10%; p90 1.161, max 1.424. So overshoot is a tail behaviour, not
   the central tendency.
2. **Negative floor leak — NOT PRESENT.** Documented as −0.0017. Measured: **mean min +0.145,
   most negative +0.086** — strictly positive. Open hypothesis in the other direction: a
   +0.145 pedestal summed over 201 channels would *inflate* M0, a plausible contributor to
   the winner's M0 regression (+64.4% vs V12's +69.8%).
3. **M2 highest cube-to-cube variance** (std 14.3% on mean 20.1%) — still open; likely a
   small-dataset effect (14 cubes, no more available).
4. **Hallucination — NOT DETECTED at n=100.** Zero of 100 channels contained an invented
   blob (background pixels >20% of clean peak, connected region ≥20 px). Treat as *not
   reproduced under this definition*, NOT as solved: it may be config-specific, or the
   threshold may be too lenient to catch what was seen by eye.
5. **Early stopping is a major variance source — NEW, and the most consequential.**
   `patience=5` on the noisy 300-channel validation set turns seed differences into large
   performance swings: the sweep winner scored 37.11 dB at epoch 38 in the sweep and
   30.28 dB stopping at epoch 21 on retrain. `epochs_run` correlates +0.650 with PSNR, more
   than any hyperparameter. Any config comparison that does not control for it is unreliable.

### Phase E — Beam-Metadata Conditioning + Hyperparameter Sweep (Week 7, RUN, verified 2026-07-26)
Per the 2026-07-20 mentor meeting: brief beam-metadata investigation first (drop if no help),
then hyperparameter sweeps as "the real focus" (random search first, Bayesian seeded with
random runs once justified; fixed PSNR/SSIM/MSE scoring; early stopping min 20/max 60/
patience 5). Implemented in `05-unet-line-emission.ipynb`, pushed `0bc1c82`/`7b0d335`/
`122568c`. **Ran on Kaggle 2026-07-26 (Version 15), results pulled and verified from real
notebook output — this is the project's most current result.**

**Beam A/B** (identical harness, only `use_beam` differs, isolates beam from the early-stopping
schedule change vs V12's fixed 30 epochs):
- Control (no beam): PSNR 33.70 | SSIM 0.9865 | MSE 0.000597 (early-stopped ep 48, best ep 43)
- Beam-conditioned: **+1.242 dB PSNR, +0.00169 SSIM** over control -> pixel-metric verdict
  "beam helps"
- **But moment-map holdout (5 cubes) is mixed, not a clean win:** M0 +59.8%±33.0% (V12:
  +69.8%±15.2% — mean DROPPED, variance more than doubled), M1 +19.2%±8.4% (slightly better),
  M2 +23.2%±20.2% (slightly better, more variance). One cube
  (`run_0025_01000_rt_04`) nearly failed on M0: only +3.0%. **Beam trades M0 reliability for a
  small pixel-metric gain — not an unambiguous improvement.**

**Random sweep (12 runs, search space: base_channels [16,32,48,64], channel_multipliers
{1-2-4, 1-2-2-4, 1-2-4-8}, lr [1e-4,3e-3], alpha [0.5,0.95], sched_patience {3,5,8},
use_beam {T,F}):**
- **Best config: base=48, mult 1×2×4×8, lr=0.00082, alpha=0.888, use_beam=False -> PSNR
  37.11 dB | SSIM 0.9902 | MSE 0.000290** — beats V12 by +4.16 dB, beats the beam run too.
- Alpha correlates strongest with PSNR (r=+0.62, higher alpha/more-MSE-weight -> better).
- **Beam correlates NEGATIVELY with PSNR across the sweep (r=-0.33)** — contradicts the A/B's
  own "beam helps" verdict. All top-5 sweep configs have `use_beam=False`.
- **RESOLVED 2026-07-30 by V16, negatively: the 37.11 dB did not reproduce.** Retrained under
  the same config and split it scored **30.28 dB** (−6.83 dB reproduction gap, −2.67 dB vs
  V12) because early stopping fired at epoch 21 instead of 38 under a different seed. Moment
  maps confirm it is not an improvement: M0 +64.4%±14.6 vs V12's +69.8%±15.2. **V12 stays the
  reference.** See progress.md 2026-07-30.
- **The "alpha matters most" reading is confounded.** `epochs_run` correlates +0.650 with PSNR
  and +0.535 with alpha. Controlling for it, `base_channels` rises +0.430 → **+0.821** while
  alpha falls +0.622 → +0.428. A Bayesian sweep seeded on alpha would chase the wrong
  parameter; **width is the better target**. Reproduce with
  `python src/evaluation/sweep_analysis.py --csv results/sweep_results.csv`. Caveat: n=12,
  noisy, not a significance test.

### Diffusion Model — Standalone Notebook, Partially Run (Recent, still not comparable to V12)
`06-ddpm-line-emission.ipynb` (root, standalone, self-contained git-clone bootstrap — no
dependency on Kaggle's native GitHub-linked-notebook sync, so it works regardless of that
limitation). Retuned config: N_SAMPLES 150, 60 epochs, ema 0.99, K_AVG=4 posterior-mean
averaging, DDIM 25-step sampling. **This retrain (which progress.md previously flagged as
"never ran") did run:**
- Training completed, real checkpoint saved (best val loss 18.3447 @ epoch 43).
- Validation (300 channels, DDIM+posterior-mean): PSNR 18.75 dB | SSIM 0.4652 | MSE 0.0284 —
  PSNR up vs the earlier broken run (14.25 -> 18.75) but **SSIM down** (0.55 -> 0.4652), so
  "better than last time" is only half true. Still far below V12 (32.95/0.9857).
- **All-5-holdout moment-map eval (the number actually comparable to V12's headline) crashed**
  on a missing `torch.nn.functional` import before scoring a single cube — fixed in the
  committed notebook (`aedf0f9`), **not yet re-run to completion**. No DDPM moment-map number
  exists yet.
- V12/U-Net files untouched by this notebook, as required.

---

## 4. KEY REFERENCE PAPERS

- Terry, Hall, Abreau, Gleyzer (2022) — ApJ 941:192 — planet detection via ML on continuum-
  synthetic ALMA channel maps, PHANTOM+MCFOST pipeline.
- Terry, Hall, Abreau, Gleyzer (2023) — ApJ 947:60 — HD 142666 planet kink discovery, DSHARP
  catalog validation target.
- Tanmay's GSoC 2025 blog (EXXA predecessor) — origin of the HybridLoss (MSE+SSIM) idea;
  `DENOISING_DIFFUSION/GSOC_2025_EXXA_Main.ipynb` is the actual predecessor notebook
  (Colab, patch-based, ermongroup/ddim + bahjat-kawar/ddrm lineage) — superseded, not a
  source of new material for current work.

---

## 5. MENTOR FEEDBACK LOG (chronological, authoritative)

**Week 2 (meeting):** Feed channels one at a time (post-midterm, for cubes). Try hybrid
loss (Tanmay's blog). Try VAE. Use Kaggle/Colab. Gradient accumulation for small batches.
Email jpterry@uga.edu, not Mattermost.

**Week 4 (meeting, 2026-06-18) — THE PIVOT:** see Phase B above.

**Week 5-6 (email, post-results):** Continuum contamination note + subtraction suggestion
(see Phase D) — confirmed to help significantly, especially M0. Called results "really
good... for a first week on the data."

**2026-07-20 (meeting):** Beam-metadata brief (drop if no help), then hyperparameter sweeps
as "the real focus" (see Phase E). Self-gravitating dataset coming as a moment-map TEST only,
not training — still not received.

**Most recent status — UNRESOLVED, needs a real date:** a follow-up meeting is logged as
"held Monday, after a prior missed call." The only confirmed Monday meeting on record is
2026-07-20 (above). If this is a distinct, later meeting, it needs its own date before this
line is trustworthy — don't cite it as a separate event until that's pinned down.

---

## 6. CONVENTIONS AND RULES

No separate `AGENT_RULES.md` file exists in this repo (checked repeatedly — gitignored by
`**/AGENT_RULES.md` but never created). Rules live here:

- Cast `dirty` arrays to float32 immediately after loading
- seed=42 everywhere (data splits, torch/numpy RNG)
- Notebooks are the source of truth for Kaggle-run work — avoid fragmenting logic into
  standalone .py files for anything iteration-heavy; keep `src/` for genuinely reusable,
  stable code (models, losses, established data utilities)
- **Rule 1:** never claim something is "done" without real execution + real artifacts on disk
- **Rule 2:** never silently deviate from an agreed spec — flag immediately, explain, ask
  before proceeding on higher-risk deviations
- Cube-level (not channel-level) splitting is mandatory for all line-emission work — held-out
  cubes are inference-only, never touched by train/val
- Moment-map evaluation must average across ALL available holdout cubes (currently 5), never
  rely on a single cube's numbers as representative (lesson from V7/V9)
- No co-author trailers in git commits (explicit user rule)

---

## 7. UPCOMING MILESTONES

**Midterm deliverable, confirmed by Jason 2026-08-07 (email): a WordPress blog post**, not a
formal report or notebook walkthrough — background + methods + initial results, explicitly
allowed to be unfinished/imperfect, with a link to a public notebook. **ALMA real-data
validation is deferred to the final blog**, not required for midterm. Jason's own words:
"Don't worry about passing your evaluation though. You're fine there." This resets the bar
for the items below — they remain worth finishing, but none of them gate Aug 10–14.

Blog structure as sent to Jason: classical baselines → architecture comparison (autoencoder,
VAE, U-Net, DDPM) → U-Net results on synthetic line-emission data → public notebook link.

- Run the winning sweep config (PSNR 37.11) through the all-5-holdout moment-map eval — see
  the 2026-08-07 correction below: this was already done (V16/V17), and the "failure" was a
  seed mismatch (sweep row used seed 49, retrain used SEED=42), not a real regression.
- Re-run `06-ddpm-line-emission.ipynb` Section 11 — **done 2026-08-07**, first completed run:
  PSNR ~18–19 dB vs U-Net's 37–39 dB. Real, if weak, result — sufficient for the blog's
  "why U-Net over DDPM" comparison section without further tuning.
- Resolve the beam M0-variance tradeoff — decide whether beam ships at all given the sweep's
  negative correlation
- Loss reweighting / non-negativity-penalty refinement for overshoot + floor leak — lower
  priority than the sweep-winner validation above
- Resolve or characterize the low-SNR hallucination issue — quantified 2026-08-06/07 via
  notebook 08's artifact diagnostics (floor-relative fix): invented structure in 22–39% of
  channels depending on config, ~7x worse below median SNR than above it
- Real ALMA DSHARP validation — **confirmed deferred to the FINAL blog**, not midterm
- Two new cubes (self-gravitating + matching dirty) received from Jason via Google Drive
  2026-08-07, not yet pulled into the repo or examined
- Repo cleanup + comprehensive README
- Midterm blog post (Aug 3-9 prep window, evaluation Aug 10-14) — write-up not yet started


---

## Correction (2026-08-07): the sweep winner's "7 dB reproduction failure"

Recorded above as the sweep winner (PSNR 37.11) failing to reproduce, scoring 30.28 in V16
and 29.92 in V17. That framing is wrong.

`run_sweep` trains run *i* at `seed + i`. The winning row is run index 7, so it was trained
at **seed 49**. Notebook 05's retrain calls `train_unet(..., seed=SEED)` with **SEED = 42**.
Identical hyperparameters, different seed. Dataset (350/100), split, early-stopping schedule
and PSNR calculation were all verified identical.

Every sweep row therefore used a different seed (42-53), so the search confounds
configuration with seed and its maximum is partly an order statistic. Notebook 08 measured
this configuration over seeds 42/43/44 at **37.97 +/- 0.90 dB**, which contains 37.109, and
V12's arm at 37.60 +/- 1.00 -- "INDISTINGUISHABLE from seed noise" in 08's own words.

The sweep results are not invalid, but no single row is reproducible from its
hyperparameters alone. `run_sweep` now records `seed` per row.
