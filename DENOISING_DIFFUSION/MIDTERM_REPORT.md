# GSoC 2026 — EXXA Midterm Report

**Denoising Astronomical Observations of Protoplanetary Disks**

**Contributor:** Krishan Yadav ([@KrishanYadav333](https://github.com/KrishanYadav333))
**Organisation:** ML4Sci — EXXA
**Mentors:** Sergei Gleyzer (University of Alabama), Jason Terry (Oxford University)
**Period covered:** Community bonding through week 8 (2026-05-25 → 2026-07-26)
**Branch:** [`line-emission`](https://github.com/KrishanYadav333/EXXA/tree/line-emission)

> Status markers used throughout: **[verified]** = ran on real data, artifacts on disk;
> **[pending]** = implemented but not yet executed. No result is claimed without artifacts.

---

## 1. Summary

The project set out to denoise synthetic ALMA observations of protoplanetary disks. Following
mentor direction on 2026-06-18, the focus moved from patch-based continuum images to
**full-image line-emission velocity cubes**, evaluated with the moment maps astronomers
actually use (`bettermoments` M0/M1/M2) rather than pixel metrics alone.

**Headline result [verified]:** a U-Net denoiser (reference checkpoint **V12**) improves all
three moment maps on **every one of 5 held-out cubes**, with no cube made worse:

| Metric | Result |
|---|---|
| Validation PSNR | **32.95 dB** |
| Validation SSIM | **0.9857** |
| Validation MSE | **0.000681** |
| Moment 0 (intensity) | **+69.8% ± 15.2%** |
| Moment 1 (velocity) | **+17.5% ± 7.8%** |
| Moment 2 (dispersion) | **+20.1% ± 14.3%** |

Improvement is measured against the dirty cube: `100 × (1 − |denoised−clean| / |dirty−clean|)`,
mean absolute difference over finite pixels, averaged across 5 holdout cubes with a standard
deviation. Holdout cubes are inference-only and never seen in training or validation.

A 12-run hyperparameter sweep has since found a configuration reaching **37.11 dB** (+4.16 dB
over V12) **[verified]**, whose moment-map validation is the current run in flight
**[pending]** — see §6.

---

## 2. Data and problem setup

**Source data.** 14 FITS cube pairs (dirty/clean) spanning 11 distinct RunIDs, each
`(201, 600, 600)` — 201 velocity channels of a 600×600 sky image. Synthetic observations from
hydrodynamic simulation + radiative transfer (PHANTOM + MCFOST lineage). Total ≈7.6 GB; hosted
as a Kaggle Dataset, never committed to git.

**Why this is not generic image denoising.** The "dirty" image is an interferometric
reconstruction from sparse *uv*-coverage, so its noise is spatially correlated PSF/sidelobe
structure, not independent per-pixel noise. The clean/dirty pair therefore differs by a
structured, beam-dependent artifact field.

**Splitting [verified].** Cube-level, grouped by RunID (`src/data/cube_split.py`). Radiative
transfer variants of the same simulation (`run_0002_00560_rt_00/_rt_01/_rt_04`) are near-
duplicates, so splitting them across train and test would leak. Grouping by RunID prevents
this: 3 RunID groups (`0002`, `0025`, `0026`) = **5 cubes** held out entirely, 7 train, 2
validation.

**Channel sampling [verified].** Channels are sampled per cube from a Gaussian centred on
channel 100 (`src/data/channel_sampler.py`), avoiding the line-free high-velocity extremes.
Calibrated at 73.8% of mass inside [50,150] averaged over 20 seeds, against the mentor's ~75%
guidance.

**Preprocessing [verified].** Continuum subtraction (§4), downsample to 256×256, per-channel
min–max normalisation using the **dirty** channel's (min, max) for both dirty and clean — see
§3 for why this specific choice is load-bearing.

---

## 3. The normalisation bug — and why it mattered

The first end-to-end test (denoise a full holdout cube → moment maps) produced **M0 −6402%**:
denoising made the intensity map dramatically *worse* than the dirty input, while per-channel
SSIM looked healthy at 0.81. That contradiction is the most instructive result of the project
so far.

**Root cause.** Training normalised the clean target by *clean's own* per-channel (min, max) —
a scale independent of dirty's. The model therefore learned to emit values in clean-normalised
space. At inference, clean's min/max is unavailable by definition, so decoding used dirty's
(min, max). Clean's background floor sits at ~0 while dirty's minimum is negative, so every
background pixel decoded to dirty's negative floor instead of ~0, imposing a negative DC offset
on each channel. Summed over 201 channels, M0 went sharply negative.

**Fix** (commit [`5ed8fc6`](https://github.com/KrishanYadav333/EXXA/commit/5ed8fc6)):

1. Normalise **both** dirty and clean by the **dirty** channel's (min, max) — a shared scale
   using only statistics available at inference time. Clean is deliberately *not* clipped, as
   its peak can exceed 1 on this scale.
2. Replace the sigmoid output with a **linear head**, since sigmoid cannot represent
   shared-scale clean values above 1.
3. Verified by channel diagnostic: denoised minimum moved from −0.0047 (tracking dirty, wrong)
   to +0.0001 (tracking clean's ~0, correct).

M0 recovered from −6402% to −1376% immediately, and to strongly positive after continuum
subtraction (§4).

**Transferable lesson:** any normalisation must be invertible from information available at
inference. A per-channel pixel metric can look fine while a cube-integrated scientific
quantity is catastrophically wrong — which is why moment maps, not SSIM, are the deliverable.

---

## 4. Continuum subtraction

The mentor observed residual continuum emission (the static dust disk) contaminating the
"clean" line-emission images, and suggested estimating it from the line-free channels and
subtracting.

**Implementation [verified]** (`continuum_of` in `src/data/fits_cube_dataset.py`): average the
first and last *n* channels into a 2D template, subtract from every channel, before
normalisation. Applied identically in training and evaluation.

**Result [verified]** — large gains, most dramatically on M0:

| Metric | No subtraction | Continuum-subtracted |
|---|---|---|
| PSNR | 31.02 dB | 33.18 dB |
| SSIM | 0.9832 | 0.9868 |
| MSE | 0.000933 | 0.000602 |
| M0 (single cube) | **−1672%** | **+84.9%** |
| M1 (single cube) | +13.6% | +20.9% |
| M2 (single cube) | +18.6% | +18.4% |

Continuum contamination was actively destroying M0 after denoising; removing it first was the
difference between failure and a usable result. Window size made little difference (n=1 vs n=5
tied on pixel metrics; n=5 is the pipeline default, n=1 marginally better in a later ablation —
an open decision, §7).

---

## 5. Evaluation methodology — and a correction to it

**The V7/V9 finding [verified].** Two runs of the *same configuration*, evaluated on the *same
single* holdout cube, produced M2 improvements of **+18.4%** and **+2.5%**. Zero configuration
change; the entire difference was run-to-run and cube-selection variance.

This invalidated the project's evaluation practice up to that point. Adopted as a standing rule:
**moment-map results must be averaged over all available holdout cubes and reported with a
standard deviation.** Single-cube numbers are not quoted as representative.

V12 was the first run evaluated across all 5 holdout cubes, which is why it — and not the
higher-SSIM V7 — is the reference checkpoint.

### V12 reference — per-cube detail [verified]

| Holdout cube | M0 | M1 | M2 |
|---|---|---|---|
| `run_0002_00560_rt_00` | +82.9% | +18.4% | +18.1% |
| `run_0002_00560_rt_01` | +55.1% | +17.5% | +8.4% |
| `run_0002_00560_rt_04` | +79.8% | +15.3% | +10.2% |
| `run_0025_01000_rt_04` | +79.8% | +29.1% | +44.1% |
| `run_0026_00005_rt_04` | +51.6% | +7.4% | +19.6% |
| **mean ± std** | **+69.8% ± 15.2%** | **+17.5% ± 7.8%** | **+20.1% ± 14.3%** |

**Configuration:** `DenoisingUNet` 3.4M parameters, `HybridLoss(α=0.8·MSE + 0.2·(1−SSIM))`,
linear output head, Adam lr=1e-3, `ReduceLROnPlateau(factor=0.5, patience=5)`, batch 8, 30
epochs (best epoch 28, val loss 0.0034), 256×256, continuum subtraction n=5, shared dirty-scale
normalisation, seed 42.

---

## 6. Beam conditioning and hyperparameter sweep

Both items came from the 2026-07-20 mentor meeting: a brief beam-metadata investigation
("drop it if it doesn't help"), then hyperparameter sweeps as the main effort.

### 6.1 Beam conditioning [verified] — a useful negative result

The observation beam is exposed as a 4-vector `[sin(2·BPA), cos(2·BPA), BMAJ″, BMIN″]` and
injected via the time-embedding path so it reaches every block including the decoder
(`UNet(beam_dim=4)`). Tested as an A/B against a no-beam control on an identical training
harness, so the comparison isolates beam from the schedule change.

| | PSNR | SSIM | M0 | M1 | M2 |
|---|---|---|---|---|---|
| Control (no beam) | 33.70 dB | 0.9865 | — | — | — |
| Beam-conditioned | **+1.24 dB** | +0.0017 | +59.8% ± 33.0% | +19.2% ± 8.4% | +23.2% ± 20.2% |
| V12 reference | 32.95 dB | 0.9857 | +69.8% ± 15.2% | +17.5% ± 7.8% | +20.1% ± 14.3% |

**Beam improved pixel metrics while degrading the scientific deliverable.** M0 fell 10 points
below V12 and its variance more than doubled; one cube (`run_0025_01000_rt_04`) collapsed to
+3.0% M0. M1 and M2 improved slightly, also with more variance.

This is the report's second methodological point: **pixel-metric gains do not imply
scientific gains, and a project measuring only PSNR/SSIM would have shipped this as an
improvement.**

### 6.2 Random hyperparameter sweep [verified]

12 random configurations over base width {16,32,48,64}, channel multipliers
{1-2-4, 1-2-2-4, 1-2-4-8}, lr (log-uniform 1e-4–3e-3), HybridLoss α (0.5–0.95), scheduler
patience {3,5,8}, beam on/off. Early stopping min 20 / max 60 / patience 5. Every run scored on
**fixed** PSNR/SSIM/MSE — never on the swept loss, whose weights differ between runs. Raw data:
[`results/sweep_results.csv`](results/sweep_results.csv).

Top 5 by PSNR:

| base | multipliers | lr | α | beam | PSNR | SSIM | MSE |
|---|---|---|---|---|---|---|---|
| 48 | 1×2×4×8 | 8.20e-4 | 0.888 | ✗ | **37.11** | 0.9902 | 0.000290 |
| 64 | 1×2×2×4 | 2.57e-4 | 0.597 | ✗ | 36.46 | 0.9914 | 0.000299 |
| 16 | 1×2×4 | 9.48e-4 | 0.848 | ✗ | 36.01 | 0.9900 | 0.000377 |
| 32 | 1×2×2×4 | 7.42e-4 | 0.864 | ✗ | 35.56 | 0.9886 | 0.000428 |
| 48 | 1×2×2×4 | 2.04e-4 | 0.620 | ✗ | 35.34 | 0.9904 | 0.000388 |

Correlation with PSNR: **α r = +0.62** (strongest), base width r = +0.43, log(lr) r = −0.12,
**beam r = −0.33**.

Two findings worth noting. First, α — the MSE/SSIM loss balance — matters more than width or
learning rate, and the sweep prefers *more* MSE weight than V12's 0.8. Second, **beam
correlates negatively with PSNR across the sweep and appears in none of the top 5**, which
contradicts the A/B's own verdict and reinforces §6.1: beam is not a reliable win.

### 6.3 Sweep-winner moment-map validation [pending]

Per §6.1's lesson, the 37.11 dB configuration is not treated as an improvement until it is
scored on the same 5-cube moment-map protocol. That run is in flight; the notebook prints an
explicit verdict that refuses promotion on PSNR alone. **This report will be updated with the
result.**

---

## 7. Conditional DDPM comparison [partially verified]

Diffusion is named in the project description, so a conditional DDPM is maintained as a
comparison baseline (`06-ddpm-line-emission.ipynb`). It predicts the noise added to the clean
channel conditioned on `[dirty, x_t]`, sampled with DDIM.

**A prior run failed instructively.** The first attempt produced PSNR 14.25 / M0 −1546%. Root
cause: EMA decay of 0.999 has a ~1000-step time constant, but the run was only ~1320 steps, so
the EMA shadow weights were still near random initialisation — and the sampler draws from the
EMA weights. Fixed with bias-corrected (Adam-style) EMA warmup
([`bbfa26c`](https://github.com/KrishanYadav333/EXXA/commit/bbfa26c)), plus posterior-mean
sampling, gradient clipping, and LR warmup
([`dc1a19b`](https://github.com/KrishanYadav333/EXXA/commit/dc1a19b)).

**Retuned run [verified]:** 60 epochs, 20.7M parameters, EMA 0.99, DDIM 25 steps, K=4
posterior-mean averaging. Best validation loss 18.34 at epoch 43. Validation over 300 channels:
**PSNR 18.75 dB | SSIM 0.4652 | MSE 0.0284**. PSNR improved substantially over the broken run
(14.25 → 18.75) but SSIM regressed (0.55 → 0.4652), and both remain far below V12.

**Moment-map comparison [pending]** — the run crashed before scoring any cube; fixed and
awaiting re-run.

**Expected outcome, stated in advance.** A diffusion model samples from `p(clean | dirty)`;
a single draw carries the full posterior variance, whereas PSNR and SSIM reward the posterior
*mean*. A regression model targets that mean directly. Posterior-mean averaging over K draws
(implemented) narrows the gap but does not close it. The honest expectation is that the DDPM
underperforms the U-Net on these metrics **by construction**, and its value here is as a
documented comparison rather than a candidate for the final pipeline — consistent with the
mentor's direction to stay with the U-Net.

---

## 8. Known artifacts and open issues

Three artifacts persist across V7/V9/V12, so far documented only from a single channel. A
per-channel diagnostic (`src/evaluation/artifacts.py`, 8 tests) now measures all of them across
every validation channel; results **[pending]** the current run.

1. **Peak overshoot** — denoised peak ≈1.151× clean peak at channel 100 (≈15% too bright).
2. **Negative floor leak** — denoised minimum −0.0017 against clean's ~0.
3. **Invented structure ("hallucination")** — in at least one faint, low-SNR channel the model
   produced a second point source that does not exist in the ground truth, on a raised
   background. **This is the most scientifically serious issue in the project:** in a disk map,
   invented structure reads as a false detection. Suspected cause is that neither MSE nor SSIM
   penalises asserting plausible structure. The new diagnostic counts invented regions and
   splits by SNR to establish whether it is a low-SNR-specific failure.
4. **M2 variance** — the highest cube-to-cube spread (±14.3% on +20.1%), most likely a
   small-dataset effect: 14 cubes is the entire available corpus.

Open decisions: continuum window n=1 vs n=5 (a later ablation favoured n=1 on pixel metrics;
the pipeline still uses n=5); whether beam conditioning ships at all given §6.

---

## 9. Deliverables to date

| Deliverable | Status | Location |
|---|---|---|
| FITS cube data pipeline | [verified] | `src/data/` — `cube_split`, `channel_sampler`, `fits_cube_dataset` |
| U-Net denoiser + hybrid loss | [verified] | `src/models/unet.py`, `src/utils/losses.py` |
| Conditional DDPM + DDIM sampling | [verified] | `src/models/diffusion_unet.py`, `src/training/diffusion.py` |
| Moment-map evaluation | [verified] | `src/evaluation/moment_maps.py` |
| Artifact diagnostics | [verified] (code) | `src/evaluation/artifacts.py` + tests |
| Early-stopping trainer + sweep harness | [verified] | `src/training/sweep.py` |
| U-Net training notebook | [verified] | `05-unet-line-emission.ipynb` |
| DDPM notebook | [verified] | `06-ddpm-line-emission.ipynb` |
| Reference checkpoint (V12) | [verified] | `results/checkpoints/` |
| Result artifacts (CSVs, figures) | [verified] | `results/` |
| Test suite | [verified] | `tests/` — 9 modules |

Compute: Kaggle Notebooks, dual Tesla T4 with `DataParallel`. Both notebooks bootstrap
themselves by cloning the repository, so a run is reproducible from a blank kernel plus the
Kaggle Dataset.

---

## 10. Plan for the second half

1. **Complete the moment-map validations in flight** — sweep winner and DDPM. Whichever U-Net
   configuration wins on moment maps becomes the reference.
2. **Bayesian sweep** seeded with the 12 random runs, now that α is identified as the dominant
   parameter (mentor's stated sequence).
3. **Address the hallucination artifact** once characterised — candidate directions include a
   non-negativity penalty, a loss term penalising asserted structure in low-signal regions, and
   evaluating whether posterior averaging suppresses it.
4. **Self-gravitating cubes** from the mentor as a moment-map *test* set (kinematic
   substructure recovery), not training data. Awaiting delivery.
5. **Real ALMA validation** against DSHARP — qualitative first, the official task list's
   endpoint and the largest remaining gap.
6. **Documentation and reproducibility** — the public curated dataset and preprocessing
   pipeline deliverable.

---

## 11. Reproducing this work

```bash
git clone -b line-emission https://github.com/KrishanYadav333/EXXA.git
cd EXXA/DENOISING_DIFFUSION
pip install -r ../requirements.txt
python -m pytest tests/ -q          # or: python tests/test_artifacts.py
```

Training requires the line-emission FITS cubes (Kaggle Dataset, ~7.6 GB — not in git). On
Kaggle: attach the dataset as an Input, enable GPU and Internet, and run
`05-unet-line-emission.ipynb` top to bottom.

**Conventions:** seed 42 throughout; cube-level splitting is mandatory; holdout cubes are
inference-only; moment-map results are always averaged across all 5 holdout cubes with a
standard deviation; no result is recorded as complete without real execution and artifacts on
disk.
