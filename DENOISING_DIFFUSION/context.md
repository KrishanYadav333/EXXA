# EXXA — Denoising Astronomical Observations of Protoplanetary Disks
## Project Context File for Agentic IDE

---

## 1. WHO / WHAT / WHY

**Contributor:** Krishan Yadav (GitHub: KrishanYadav333)
**Program:** Google Summer of Code 2026, ML4Sci organization
**Project:** EXXA — Denoising Astronomical Observations of Protoplanetary Disks
**Mentors:** Jason Terry (Oxford University, jpterry@uga.edu — preferred contact, do NOT use Mattermost)
**Org head:** Katia Matcheva
**Duration:** 22 weeks, May 25, 2026 – November 3, 2026 (350-hour project)
**Midterm Evaluation:** Aug 10–14, 2026
**Final Submission:** Oct 26 – Nov 3, 2026

**The problem:** Protoplanetary disks (rotating gas/dust disks around young stars where planets form) are observed by ALMA and VLT telescopes. These observations contain significant noise from atmospheric interference, instrumental limitations, and observational constraints. This noise obscures disk structures (rings, gaps, spiral arms) that signal forming planets. Prior work (Terry et al. 2022, 2023 — see Section 5) showed ML can detect planets via kinematic "kinks" in velocity channel maps, but these signatures become nearly invisible in noisy data.

**The goal:** Build an ML pipeline that denoises these observations — faster and more effectively than classical filtering — while preserving scientifically meaningful structures. This sits upstream of planet-detection pipelines: better denoising → better planet detection.

**Official Expected Results (from GSoC project description):**
1. ML denoising model tailored to astronomical data characteristics
2. Detailed performance analysis vs. traditional methods AND real ALMA/VLT data
3. Publicly available curated dataset + preprocessing/augmentation pipeline
4. Documentation for architecture, training, and reproducibility

**Tech stack required:** Python, PyTorch, C/Fortran (Fortran/C only relevant if touching PHANTOM/MCFOST simulation code directly — not expected for the ML pipeline itself)

---

## 2. CURRENT COMPUTE ENVIRONMENT

- **Primary (Week 1-3):** Local machine, NVIDIA GeForce RTX 2050, 4GB VRAM
- **Future (Week 4+):** Kaggle Notebooks, T4 ×2 GPUs (~16GB VRAM each, NOT pooled — DataParallel splits batches, doesn't combine memory)
- Patch-based training (64×64) to fit in 4GB VRAM constraint
- Gradient accumulation ×4 to simulate larger batch sizes

---

## 3. DATASET — GROUND TRUTH FACTS (verified, do not re-derive)

### 3A. LINE EMISSION CUBES — **ACTIVE DATASET (as of 2026-06-18 pivot)**

Location: `data/Line Emission Data/<run_id>/` — one subfolder per cube.

Per subfolder (verified by inspecting `run_0002_00560_rt_00_dirty.fits`):
- `*_clean.fits` — ground-truth cube
- `*_dirty.fits` — noisy cube (the input to denoise)
- `*.para` — simulation parameter file

| Property | Value (verified) |
|---|---|
| FITS data shape | **(201, 600, 600)** = (velocity channels, Dec, RA) |
| dtype | float32 (`>f4` big-endian — cast on load) |
| Axes | CTYPE1=RA---TAN, CTYPE2=DEC--TAN, **CTYPE3=VELO-LSR** (velocity), CDELT3=0.1 |
| Units | JY/BEAM |
| Value range (example dirty) | ~[-0.0066, 0.0404], mean ~5e-4 — **NOT pre-normalized to [0,1]** (unlike the continuum .npy); per-cube/per-channel normalization needed |
| NaNs | 0 (in inspected sample) |
| Cubes present | 14 (28 FITS files), ~7.6 GB; mentor still uploading toward ~20 |

- **Naming convention:** `run_<RunID>_<StepID>_rt_<PP>` (e.g. `run_0002_00560_rt_00`). `rt_00/01/04` are different radiative-transfer **parameter variants** (inclination etc.) of the same run+step — you do NOT need every variant; prefer breadth across distinct RunIDs.
- These are synthetic PPV (position-position-velocity) cubes from PHANTOM + MCFOST, same lineage as Terry et al. 2022/2023.
- Paired supervised: each dirty cube has an exact clean counterpart.
- **Channel sampling (mentor-specified):** do NOT use every channel. Sample ~50 channels/cube via a Gaussian centered at channel **index 100**, with **~75% of samples in [50, 150]**. Avoid extreme high-velocity channels (very low / very high indices) — mostly continuum, little signal, "will confuse the model."
- **Cube-level holdout (mentor-specified):** set aside **3–4 entire cubes** for inference-only — NOT used for train OR val. Split is at the CUBE level, not the channel level. Held-out cubes are denoised channel-by-channel, then fed through `bettermoments` to compare moment maps (the real scientific test).
- **Full-image input** (mentor-specified): feed whole channel maps (600×600, downsample to 256×256/300×300 if memory-constrained), NOT 64×64 patches. See Section 6 (2026-06-18).

### 3B. CONTINUUM DATA — **BACKGROUND/FOUNDATIONAL (de-prioritized 2026-06-18)**

File: `data/clean.npy`, `data/dirty.npy` — still present, prior work valid, but NO LONGER the active thread.

| Property | clean.npy | dirty.npy |
|---|---|---|
| Shape | (975, 600, 600) | (975, 600, 600) |
| dtype | float32 | **float64 — must cast to float32 before use** |
| Range | [0, 1] | [0, 1] |
| Memory | ~1.4 GB | ~2.8 GB |

- 975 paired continuum images, already normalized to [0,1]. All Week 2–4 results (AE/VAE/U-Net/DDPM) are on this data and remain valid documented progress — just not the current focus.
- Real ALMA validation data source (later): DSHARP catalog (https://almascience.eso.org/almadata/lp/DSHARP/) — HD 97048, HD 163296, HD 142666.

---

## 4. REPOSITORY STRUCTURE

```
DENOISING_DIFFUSION/
├── data/
│   ├── clean.npy
│   └── dirty.npy
├── src/
│   ├── data/
│   │   ├── dataset.py          # AstroDataset + dataloader pipeline
│   │   ├── fits_loader.py      # FITS format loading utilities
│   │   ├── preprocessing.py    # normalization, cropping
│   │   └── augmentation.py     # rotations, flips, noise variation
│   ├── models/
│   │   ├── autoencoder.py      # DenoisingAutoencoder (plain conv AE)
│   │   ├── vae.py              # DenoisingVAE (encoder->mu/logvar->reparam->decoder)
│   │   ├── unet.py             # DenoisingUNet() [3.4M, 4GB-safe] and create_model() [51M, full DDPM]
│   │   ├── noise_scheduler.py  # DDPM linear/cosine beta schedules
│   │   └── ddpm.py             # diffusion training/sampling logic (Week 4)
│   ├── training/
│   │   └── trainer.py          # DDPM training loop, logging, checkpoints
│   ├── inference/              # inference pipeline scripts
│   ├── utils/
│   │   └── losses.py           # HybridLoss (MSE+SSIM), VAELoss (MSE+SSIM+KL)
│   └── baselines.py            # classical: Gaussian, Median, Wiener + metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset stats, baselines
│   ├── 02_autoencoder_model.ipynb     # AE MSE + Hybrid, complete
│   ├── 03_vae_model.ipynb             # VAE, complete
│   └── 04_unet_model.ipynb            # U-Net, complete
├── tests/
│   ├── test_noise_scheduler.py
│   ├── test_blocks.py
│   ├── test_trainer.py
│   ├── test_inferencer.py
│   └── test_metrics.py
├── results/
│   ├── checkpoints/           # model checkpoints
│   ├── stats/                 # dataset visualizations
│   ├── autoencoder_hybrid_loss.png
│   ├── vae_loss.png
│   ├── unet_loss.png
│   ├── noise_progression.png
│   ├── metrics_week3_final.csv
│   └── *.png                  # loss curves
└── experiments/
    ├── hybrid_vs_mse_comparison.png
    ├── vae_vs_ae_comparison.png
    ├── unet_vs_ae_vae_comparison.png
    └── unet_vs_all_comparison.png
```

---

## 5. KEY REFERENCE PAPERS (read, summarized — do not need to re-read PDFs)

**Terry, Hall, Abreau, Gleyzer (2022) — "Locating Hidden Exoplanets in ALMA Data Using Machine Learning"** (ApJ 941:192)
- Trained EfficientNetV2 and RegNet classifiers on synthetic velocity channel maps (600×600, C=47/61/75 channels) generated via 1000 PHANTOM SPH simulations + MCFOST radiative transfer
- Detects presence/location of planets via non-Keplerian "kinks" in channel maps
- Applied successfully to real ALMA data: HD 97048, HD 163296
- Noise: rms 5–20%, added per-pixel independently (Table 1 has full simulation parameter ranges)
- **Relevance:** This is the downstream task our denoising pipeline should improve. Better denoised channels → better kink visibility → better planet detection by their classifiers.

**Terry, Hall, Abreau, Gleyzer (2023) — "Kinematic Evidence of an Embedded Protoplanet in HD 142666"** (ApJ 947:60)
- Applied the above models to HD 142666 (DSHARP catalog), found a previously-overlooked kink, confirmed via SPH simulation with 5 MJ planet at 75 au
- **Relevance:** HD 142666 is a good real-data validation target later in our project.

**Tanmay's GSoC 2025 blog (EXXA predecessor project — "Foundation Models for Exoplanet Characterization")**
- Used MAE (Masked Autoencoder) with custom radial/elliptical feature engineering
- **Critical contribution we adopted:** Hybrid loss = α·MSE + β·(1−SSIM). He achieved MSE 0.000154, SSIM 0.967611 with this on his task.
- Also suggested (current mentor meeting, Week 2): try VAE instead of plain AE for denoising; Kaggle works well for training.

**Ho, Jain, Abbeel (2020) — DDPM paper** — theoretical basis for diffusion model work, arxiv 2006.11239
**Ronneberger et al. (2015) — U-Net paper** — arxiv 1505.04597

---

## 6. MENTOR FEEDBACK LOG (chronological, authoritative — follow these literally)

**Week 2 meeting (Jason Terry + Tanmay):**
- "Having all the channels in the model at once was an absolute massive pain... maybe we could feed them in one by one." → For line emission cubes: denoise channel-by-channel, NOT whole cube at once.
- Jason will provide dirty synthetic line emission cubes post-midterm.
- Suggested: compute moment maps (clean vs dirty vs denoised) — richer scientific comparison than raw pixel metrics, what astronomers actually use to infer disk processes/planet masses.
- "Play with your loss function... I think it's in Tanmay's blog... robust loss function" → implement MSE+SSIM hybrid loss (✅ done Week 3).
- Try Google Colab/Kaggle for GPU (plan for Week 4+).
- If using small batches: look into gradient accumulation (✅ implemented, accumulation_steps=4).
- Tanmay: try VAE instead of plain conv autoencoder, "specialized in denoising tasks" (✅ done Week 3).
- **Communication preference: EMAIL, not Mattermost.** Jason explicitly said he won't reliably see Mattermost messages.

**Week 4 meeting (2026-06-18, Jason Terry) — MAJOR PIVOT (follow literally):**
- **Abandon patches.** "These results make me really question whether... the patch approach is something we should invest a lot of time in"; "these patches just are not going to work." Deprecated project-wide, not just for DDPM.
- **Full image in at once.** "Really try to focus on seeing if you can get the full image in at once, instead of these patches." Downsampling to 300×300 or 256×256 explicitly approved if 600×600 won't fit — "just like a parameter," unscale later with more memory. Rationale: global context — disk symmetries and substructures span large azimuthal distances that patches destroy.
- **Pivot to line emission NOW.** "Let's go ahead and move to the line emission data... this continuum data... is just going to be a sidetrack." Continuum = sidetrack, not the priority.
- **Architecture: go back to last week's U-Net**, full images. NOT the DDPM (underperformed), NOT patches. "We can maybe find more fancy architectures later." Goal: get *any* line-emission denoising working — "no one's done that yet."
- **Line emission format:** FITS cubes, ~201 velocity channels, subfolders with clean+dirty pairs, naming `RunID_StepID_RT#`. Each cube ~200× a continuum image.
- **Channel sampling:** ~50 channels/cube, Gaussian centered at index ~100, ~75% in [50,150]; avoid extreme high-velocity channels (mostly continuum, will confuse the model).
- **Cube-level holdout:** hold out 3–4 ENTIRE cubes for inference only (not train, not val). Channel-level holdout is not enough.
- **Breadth over depth:** prefer channels across many distinct runs over many channels from few runs; if downloading few, take from different RunIDs.
- **`bettermoments` package:** generate moment maps (Moment 0/1/2 + "quadratic") from cubes. "An individual line map means very little — it's putting them together in an entire cube that... gives you the power." Practical focus: just learn the two functions to make the plots. This is the ultimate scientific evaluation: dirty cube vs denoised cube moment maps.
- Train on individual channels → at inference denoise an entire held-out cube → moment maps → compare. That comparison is the real test of scientific value.

---

## 7. RESULTS SO FAR (Week 3 completion — June 12, 2026)

Unified leaderboard, 100 validation samples, ranked by SSIM (the metric that matters — PSNR/MSE favor blurry classical outputs that destroy disk structure):

| Rank | Method | PSNR (dB) | SSIM | MSE |
|---|---|---|---|---|
| 1 | **AE HybridLoss** (30ep) | 19.92 | **0.7609** ★ | 0.013920 |
| 2 | VAE MSE+SSIM+KL (30ep) | 20.00 | 0.7059 | 0.013336 |
| 3 | U-Net HybridLoss (30ep) | 20.63 | 0.7044 | **0.011653** ★ |
| 4 | AE MSE-only (30ep) | 20.25 | 0.6158 | 0.012804 |
| 5 | Gaussian σ=2 | 22.78 | 0.4230 | 0.006380 |
| 6 | Median 3×3 | **22.88** ★ | 0.3591 | 0.006317 |
| 7 | Wiener | 22.57 | 0.3398 | 0.006702 |
| 8 | Noisy input | 21.57 | 0.1924 | 0.008391 |

**Key insights:**
- Neural models beat best classical method by **+80% on SSIM** (0.76 vs 0.42)
- AE HybridLoss is current leader for structure preservation (SSIM 0.76)
- U-Net has best pixel accuracy (MSE 0.0117) but slightly lower SSIM due to supervised training mode
- Classical filters win PSNR/MSE by blurring — they destroy the exact structures (rings, gaps) we need
- **Always report and prioritize SSIM for this task**

**Loss function used:** `HybridLoss(alpha=0.8, beta=0.2)` = 0.8·MSE + 0.2·(1−SSIM). VAE additionally adds `kl_weight=0.001 · KL_divergence`.

**Training config:** 64×64 random patches, batch size 16, Adam lr=1e-3, 30 epochs, gradient accumulation steps=4 (effective batch 64), local RTX 2050.

**Model parameters:**
- AE / VAE: 1,734,305 parameters
- U-Net: 3,424,065 parameters

**Checkpoints saved:**
- `results/checkpoints/autoencoder_hybrid_best.pth` (epoch 27)
- `results/checkpoints/vae_best.pth` (epoch 23)
- `results/checkpoints/unet_best.pth` (epoch 23)

---

## 8. CURRENT WEEK STATUS

### >>> ACTIVE FOCUS (as of 2026-06-18 mentor meeting): LINE EMISSION PIVOT <<<

**Major pivot — supersedes the old Week 5–22 plan where relevant.** Decided in the 2026-06-18
meeting with Jason Terry (see Section 6 entry of same date):

1. **Patch-based training is DEPRECATED project-wide.** The 64×64 patch approach underperformed
   (DDPM SSIM ~0.22; weak continuum patch results). No future training on random patches.
2. **Full-image input** going forward — whole 600×600 channel maps, downsampled to 256×256 or
   300×300 if needed for memory ("just a parameter"; unscale later on a bigger machine).
3. **Pivot from continuum → line emission data.** Continuum work is now background/foundational.
4. **Go back to last week's U-Net** (Week 3 `src/models/unet.py`), NOT the DDPM (underperformed),
   on full images — get *any* line-emission denoising working first; fancier architectures later.
5. **New eval target:** denoise whole held-out cubes channel-by-channel → `bettermoments` moment
   maps (Moment 0/1/2 + quadratic) → compare dirty vs denoised. This is the scientific test.

**Immediate action items (in order):**
1. [DONE] Locate line emission FITS data — present at `data/Line Emission Data/` (14 cubes, 7.6 GB).
2. [DONE] Inspect cube structure — (201, 600, 600), VELO-LSR axis, Jy/beam, not normalized.
3. [TODO] FITS cube loader + Gaussian channel sampler (center 100, ~75% in [50,150]).
4. [TODO] Cube-level train/val/test split; hold out 3–4 cubes inference-only.
5. [TODO] Adapt Week-3 U-Net to full images (256×256/300×300), full-image dataset (no patches).
6. [TODO] Install + test `bettermoments` on one clean/dirty pair.

---

### Prior status (continuum thread, now background)

**Week 3 (Jun 8–12): COMPLETE ✅**

**Completed deliverables:**
1. ✅ Hybrid Loss implementation (MSE + SSIM) → SSIM improved 0.62 to 0.76
2. ✅ VAE architecture and training → SSIM 0.71
3. ✅ U-Net architecture and training → SSIM 0.70, MSE 0.0117
4. ✅ Gradient accumulation (×4 steps)
5. ✅ 4 visual comparisons generated:
   - `experiments/hybrid_vs_mse_comparison.png`
   - `experiments/vae_vs_ae_comparison.png`
   - `experiments/unet_vs_ae_vae_comparison.png`
   - `experiments/unet_vs_all_comparison.png`
6. ✅ Unified 8-method leaderboard (`results/metrics_week3_final.csv`)
7. ✅ Training curves for all models saved
8. ✅ All notebooks executed and outputs baked in
9. ✅ Week 3 presentation and speaker notes created

**Week 4 (Jun 15–18): DDPM built (6-level ch=128, 128px, multi-GPU), then SUPERSEDED by the
2026-06-18 line-emission pivot above.** DDPM code/notebooks (`05`, `06`, master §Scaled DDPM)
remain in the repo as foundational work; DDPM is PAUSED, not the active thread.

---

## 9. CONVENTIONS AND RULES FOR THIS CODEBASE

- Always cast `dirty` arrays to `float32` immediately after loading (dtype mismatch with `clean`)
- Use `random_state=42` for all train/val splits — consistency across notebooks
- Default train/val split: 80/20 (780 train, 195 val)
- Default loss: `HybridLoss(alpha=0.8, beta=0.2)` unless explicitly testing alternatives
- Visualization colormap: `inferno` for all disk images
- Checkpoint naming: `results/checkpoints/{model_name}_best.pth`, saved on best val loss
- All metrics tables ranked by SSIM (not PSNR/MSE) as primary metric, per Section 7 rationale
- For multi-GPU (future): wrap model in `nn.DataParallel(model)`, verify with `torch.cuda.device_count()`
- Never claim a deliverable is "done" until it has been actually executed with real data and produced real artifacts (loss curves, sample images, metrics) — code existing and passing smoke tests is NOT the same as a completed deliverable

---

## 10. UPCOMING MILESTONES (REVISED after 2026-06-18 line-emission pivot)

The line-emission work moved up from Week 8–9 to NOW. Revised near-term plan (supersedes old
Week 5–22 ordering where it conflicts):

- **Now (pivot week):** FITS cube loader + Gaussian channel sampler; cube-level split with 3–4
  inference-only holdout cubes; full-image U-Net (256×256/300×300) on line emission; `bettermoments`
  install + smoke test. Deliverable: first line-emission denoising + moment-map comparison.
- **Next:** iterate U-Net on full images; per-cube/per-channel normalization tuning; expand to more cubes as mentor uploads them.
- **Later (still on the roadmap):** augmentation pipeline; revisit larger/fancier architectures
  (incl. revisiting DDPM on full images) once a baseline works; real ALMA DSHARP validation
  (HD 97048, HD 163296, HD 142666); public dataset release.
- **Midterm (Aug 10–14):** working line-emission denoiser + moment-map evaluation on held-out cubes.

Background (continuum, de-prioritized): DDPM full benchmark — paused, not abandoned.

---

**Last updated:** June 18, 2026 (line-emission pivot — Week 4 meeting with Jason Terry)
