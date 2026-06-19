# Progress Tracking -- DENOISING_DIFFUSION

**Project:** ML4Sci — Denoising Astronomical Observations of Protoplanetary Disks  
**Author:** Krishan Yadav  
**Branch:** `week-3`

---

## Task Log

## Week 1 Progress -- Data Pipeline, Baselines, and Autoencoder Setup

### [2026-06-03] [DONE] Environment Verification
- CUDA available: `True` — GPU: `NVIDIA GeForce RTX 2050` (4 GB VRAM)
- PyTorch 2.7.1+cu118, Python 3.13, Windows (num_workers=0 required)

---

### [2026-06-03] [DONE] Data Inspection — `src/dataset_stats.py`

| File | Shape | dtype | Mean | Std |
|---|---|---|---|---|
| `clean.npy` | (975, 600, 600) | float32 | 0.1525 | 0.2508 |
| `dirty.npy` | (975, 600, 600) | float64 | 0.1777 | 0.1941 |

- **Fix applied:** cast `dirty` to `float32` on load — halves memory (2.8 GB → 1.4 GB)
- **Visualisations:** `results/stats/sample_grid.png`, `pixel_distribution.png`, `mean_images.png`, `noise_difference.png`

---

### [2026-06-03] [DONE] Classical Baselines — `src/baselines.py`

Averaged over 50 samples (600×600 full images):

| Method | PSNR (dB) | SSIM | MSE |
|---|---|---|---|
| Noisy input | 21.66 | 0.2358 | 0.009048 |
| Gaussian σ=1 | 22.86 | 0.4771 | 0.006917 |
| **Gaussian σ=2** | 22.92 | **0.5214** | **0.006806** |
| Median 3×3 | 22.93 | 0.4358 | 0.006921 |
| Wiener | 22.69 | 0.4151 | 0.007186 |

- **ML target:** PSNR > 22.93, SSIM > 0.52, MSE < 0.006806
- Notebook: `notebooks/01_data_exploration.ipynb`

---

### [2026-06-03] [DONE] Denoising Autoencoder — `src/models/autoencoder.py`

- 3-level encoder (1→32→64→128ch) + bottleneck (256ch) + 3-level decoder
- ConvBlock: Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU
- Parameters: **1,734,305**
- Input/output: `(B, 1, 64, 64)` → `(B, 1, 64, 64)` with Sigmoid

---

### [2026-06-05] [DONE] Autoencoder Training (MSE-only, 30 epochs)

- Script: `src/train_autoencoder.py`
- Checkpoint: `results/checkpoints/autoencoder_best.pth`
- Loss plot: `results/autoencoder_loss.png`
- Setup: 780/195 train/val split, batch=16, Adam lr=1e-3, ReduceLROnPlateau

| Epoch | Train MSE | Val MSE | LR |
|---|---|---|---|
| 1 | 0.045146 | 0.026865 | 1e-3 |
| 8 | 0.012505 | 0.008527 | 1e-3 |
| 19 | 0.011269 | 0.007249 | 5e-4 |
| **26** | 0.010042 | **0.007239** | 2.5e-4 |

- Best: epoch 26, val MSE = **0.007239**

---

## Week 2 Progress -- Hybrid Loss, VAE, Unified Evaluation, and U-Net Preparation

### [2026-06-08] [DONE] Hybrid Loss — `src/utils/losses.py` + `HybridLoss`

- `HybridLoss(alpha=0.8, beta=0.2)` = 0.8×MSE + 0.2×(1−SSIM)
- Full 30-epoch training: best val hybrid loss = **0.034365** (epoch 27)
- MSE sub-component: **0.007181** — beats MSE-only model (0.007239)
- Checkpoint: `results/checkpoints/autoencoder_hybrid_best.pth`
- Loss plot: `results/autoencoder_hybrid_loss.png`
- `train_autoencoder.py` updated with `--loss mse|hybrid` CLI flag

---

### [2026-06-09] [DONE] Test Suite Fixes — `tests/test_unet.py`

- Rewrote `test_unet.py` as proper `pytest` functions with fixtures
- Added graceful OOM handling for scale tests on 4 GB GPU
- All **64/64 tests pass** (`python -m pytest`)

---

### [2026-06-11] [DONE] VAE Architecture — `src/models/vae.py`

- `DenoisingVAE(latent_dim=128)` — probabilistic extension of the autoencoder
- Encoder → μ/log_var maps (B,128,8,8) → reparameterize → decoder
- Reparameterization: z = μ + ε·σ, ε~N(0,I) (differentiable sampling)
- At inference: z = μ (no sampling noise)
- Parameters: **~3.2M**
- Input/output: `(B, 1, 64, 64)` → `(B, 1, 64, 64)` + μ + log_var

---

### [2026-06-11] [DONE] VAE Loss — `src/utils/losses.py` + `VAELoss`

- `VAELoss(alpha=0.8, beta=0.2, gamma=0.001)` = 0.8×MSE + 0.2×(1−SSIM) + 0.001×KL
- KL = −0.5×mean(1 + log_var − μ² − exp(log_var))
- Both `HybridLoss` and `VAELoss` exported from `src/utils/__init__.py`

---

### [2026-06-11] [DONE] VAE Training Notebook — `notebooks/03_vae_model.ipynb`

8 sections, fully executed with baked-in outputs:

1. Setup and imports
2. VAE architecture — encoder/reparameterize/decoder, shape walkthrough
3. VAE loss — MSE + SSIM + KL, sanity check
4. Data pipeline — same PatchDataset as autoencoder
5. Training — 30 epochs, gradient accumulation steps=4, per-epoch table
6. Save checkpoint → `results/checkpoints/vae_best.pth`
7. Visual comparison — VAE vs Hybrid AE vs Clean GT vs Dirty (inferno cmap)
8. Summary table

**Training results (best @ epoch 23/30):**

| Epoch | tr_total | tr_mse | tr_ssim | tr_kl | val_total | LR |
|---|---|---|---|---|---|---|
| 1 | 0.2545 | 0.1183 | 0.7972 | 0.4218 | 0.2687 | 1e-3 |
| 14 | 0.0627 | 0.0131 | 0.2578 | 0.6049 | 0.0460 | 1e-3 |
| **23** | 0.0445 | 0.0104 | 0.1783 | 0.5466 | **0.0392** | 5e-4 |
| 30 | 0.0472 | 0.0120 | 0.1854 | 0.5421 | 0.0562 | 2.5e-4 |

- Best val loss: **0.039160** at epoch 23
- Checkpoint: `results/checkpoints/vae_best.pth`

---

### [2026-06-11] [DONE] Unified Evaluation — `src/evaluate_all.py` + `results/metrics_final.csv`

All 7 methods evaluated on the **same 100 val samples** via sliding-window patch inference.

| Rank | Method | PSNR (dB) | SSIM | MSE |
|---|---|---|---|---|
| 1 | **AE HybridLoss (30ep)** | 19.92 | **0.7609** | 0.013920 |
| 2 | **VAE MSE+SSIM+KL (30ep)** | 20.00 | **0.7059** | 0.013336 |
| 3 | AE MSE-only (30ep) | 20.25 | 0.6158 | 0.012804 |
| 4 | Gaussian σ=2 | 22.78 | 0.4230 | 0.006380 |
| 5 | Median 3×3 | **22.88** | 0.3591 | **0.006317** |
| 6 | Wiener | 22.57 | 0.3398 | 0.006702 |
| 7 | Noisy input | 21.57 | 0.1924 | 0.008391 |

**Key insight:** SSIM is the correct metric — neural models preserve disk structure
**+80% better** than the best classical filter. PSNR/MSE favour blurring at the cost of
smearing disk gaps (the opposite of what the task requires).

- Final evaluation cell added to `notebooks/03_vae_model.ipynb` (Cell 20)
- CSV saved: `results/metrics_final.csv`
- Evaluation script: `src/evaluate_all.py`

---

### [2026-06-11] [DONE] U-Net Inspection and Fix — `src/models/unet.py`

**Architecture (UNet class):**
- `SinusoidalPositionalEmbedding` — t (scalar) → (B, time_emb_dim) sinusoidal vector
- `ResidualBlock` — GroupNorm→SiLU→Conv3×3→time_proj→GroupNorm→SiLU→Conv3×3 + shortcut
- `Downsample` — strided Conv2d (×0.5 spatial)
- `Upsample` — ConvTranspose2d (×2 spatial)
- Encoder: `enc_blocks` + `enc_downs` (num_res_blocks per level)
- Bottleneck: num_res_blocks at lowest resolution
- Decoder: `dec_blocks` + `dec_ups` (skip concat + num_res_blocks+1 per level)
- Output: GroupNorm → SiLU → Conv3×3 (unbounded, no sigmoid — correct for noise prediction)

**forward(x, timesteps) → tensor** same spatial size as input

**Two presets added:**

| Factory | Input | Output | Params | Use |
|---|---|---|---|---|
| `DenoisingUNet()` | `(B,1,64,64)` | `(B,1,64,64)` | **3.4M** | Patch denoising on 4 GB GPU |
| `create_model()` | `(B,2,H,W)` | `(B,1,H,W)` | 51M | DDPM with large GPU |

- Forward pass verified for both presets ✅
- `DenoisingUNet` exported from `src/models/__init__.py`

---

## Week 3 Progress -- U-Net Training, Visuals, and Final Metrics

### [2026-06-12] [DONE] Last-Week Feedback Follow-up

| Feedback / Task | Status |
|---|---|
| Use SSIM in loss function | **DONE** |
| Look at Tanmay's blog for robust loss | **DONE** |
| Hybrid MSE + SSIM loss implemented | **DONE** |
| Try VAE | **DONE** |
| Gradient accumulation | **DONE** |
| Keep doing what you're doing | **DONE** |

---

### [2026-06-12] [DONE] Week 3 Planning Completion

| Planned Task | Status |
|---|---|
| Complete autoencoder training on the dataset | **DONE** |
| Evaluate autoencoder: PSNR/SSIM vs baselines | **DONE** |
| Start U-Net training using existing `unet.py` | **DONE** |
| Visual comparisons: noisy -> autoencoder output -> U-Net output | **DONE** |

---

### [2026-06-12] [DONE] U-Net Training Notebook -- `notebooks/04_unet_model.ipynb`

Final Week 3 U-Net notebook created and executed:

1. Setup and imports -- `DenoisingUNet`, `HybridLoss`, `clean.npy`, `dirty.npy`, 80/20 split (`random_state=42`)
2. Model summary -- **3,424,065 parameters**, forward pass `(2,1,64,64) -> (2,1,64,64)`
3. Training loop -- 30 epochs, `HybridLoss(alpha=0.8, beta=0.2)`, Adam lr=1e-3, batch=16, grad accumulation=4
4. Loss curves -- saved to `results/unet_loss.png`
5. Visual comparison -- 3 random validation samples, 5 columns, SSIM below outputs
6. Unified metrics table -- all 8 methods on 100 validation samples
7. Final project progress summary cell added

**Training results:**

| Setting | Value |
|---|---|
| Best epoch | **23 / 30** |
| Best validation hybrid loss | **0.036899** |
| Checkpoint | `results/checkpoints/unet_best.pth` |
| Loss plot | `results/unet_loss.png` |
| Visual comparison | `experiments/unet_vs_all_comparison.png` |

Redundant development notebook `notebooks/04_unet_training.ipynb` was removed to keep only the final Week 3 notebook.

---

### [2026-06-12] [DONE] Week 3 Unified Metrics -- `results/metrics_week3_final.csv`

All 8 methods evaluated on the same 100 validation samples and ranked by SSIM.

| Rank | Method | PSNR | SSIM | MSE |
|---|---|---|---|---|
| 1 | **AE HybridLoss** | 19.9152 | **0.7609** | 0.013920 |
| 2 | VAE (MSE+SSIM+KL) | 19.9951 | 0.7059 | 0.013336 |
| 3 | U-Net HybridLoss | 20.6345 | 0.7044 | 0.011653 |
| 4 | AE MSE-only | 20.2513 | 0.6158 | 0.012804 |
| 5 | Gaussian s=2 | 22.7803 | 0.4230 | 0.006380 |
| 6 | Median 3x3 | **22.8835** | 0.3591 | **0.006317** |
| 7 | Wiener | 22.5687 | 0.3398 | 0.006702 |
| 8 | Noisy Input | 21.5703 | 0.1924 | 0.008391 |

**Best model so far by SSIM:** `AE HybridLoss`, SSIM = **0.7609**.

**Week 3 completion checklist:**

| Task | Status |
|---|---|
| U-Net implementation | **DONE** |
| U-Net training 30 epochs | **DONE** |
| Visual comparison | **DONE** |
| Unified metrics table | **DONE** |

---

## Files Created / Modified (Week 3)

| File | Status | Description |
|---|---|---|
| `src/models/vae.py` | **NEW** | DenoisingVAE architecture |
| `src/utils/losses.py` | **UPDATED** | Added VAELoss, fixed docstring |
| `src/utils/__init__.py` | **UPDATED** | Export VAELoss |
| `src/models/__init__.py` | **UPDATED** | Export DenoisingVAE, DenoisingUNet |
| `src/models/unet.py` | **UPDATED** | DenoisingUNet preset, full docstring, __main__ |
| `src/train_autoencoder.py` | **UPDATED** | --loss mse|hybrid CLI, sub-loss logging |
| `src/evaluate_all.py` | **NEW** | Unified evaluation script (all 7 methods) |
| `notebooks/02_autoencoder_model.ipynb` | **UPDATED** | HybridLoss 30-ep training, executed |
| `notebooks/03_vae_model.ipynb` | **NEW** | VAE notebook, fully executed |
| `notebooks/04_unet_model.ipynb` | **NEW** | Final U-Net notebook, fully executed, includes summary |
| `notebooks/04_unet_training.ipynb` | **REMOVED** | Redundant older U-Net development notebook |
| `results/checkpoints/autoencoder_hybrid_best.pth` | **NEW** | Best hybrid AE (epoch 27) |
| `results/checkpoints/vae_best.pth` | **NEW** | Best VAE (epoch 23) |
| `results/checkpoints/unet_best.pth` | **NEW** | Best U-Net (epoch 23) |
| `results/autoencoder_hybrid_loss.png` | **NEW** | Hybrid training curves |
| `results/vae_loss.png` | **NEW** | VAE training curves |
| `results/unet_loss.png` | **NEW** | U-Net training curves |
| `results/metrics_final.csv` | **NEW** | All-methods PSNR/SSIM/MSE table |
| `results/metrics_week3_final.csv` | **NEW** | Week 3 all-methods table including U-Net |
| `experiments/hybrid_vs_mse_comparison.png` | **NEW** | AE visual comparison |
| `experiments/vae_vs_ae_comparison.png` | **NEW** | VAE vs AE visual comparison |
| `experiments/unet_vs_all_comparison.png` | **NEW** | Clean/Noisy/AE Hybrid/VAE/U-Net comparison with SSIM |
| `tests/test_unet.py` | **UPDATED** | Proper pytest fixtures, OOM handling |

---

## Week 4 Progress -- Conditional DDPM (Diffusion Model)

### [2026-06-18] [DONE] U-Net training/evaluation verified intact

- Supervised `DenoisingUNet` (3,424,065 params) forward pass OK, `results/checkpoints/unet_best.pth`
  loads cleanly, Week-3 metrics table preserved in `results/metrics_week3_final.csv`.

---

### [2026-06-18] [DONE] Adapted diffusion model from notebook -- `src/models/diffusion_unet.py`

Ported the config-driven `DiffusionUNet` from `GSOC_2025_EXXA_Main.ipynb` (ermongroup/ddim +
bahjat-kawar/ddrm lineage) into the repo:

- `get_timestep_embedding`, `nonlinearity` (swish), `Normalize` (GroupNorm), `Upsample`,
  `Downsample`, `ResnetBlock`, `AttnBlock`, `DiffusionUNet`.
- Conditional: input is `cat([x_cond, x_t], dim=1)` (2-ch), output predicted noise (1-ch).
- `DotDict` config + `default_diffusion_config()` factory; exported via `src/models/__init__.py`.

**Reduced model size for RTX 2050 (4 GB):**

| Param | Notebook | This repo |
|---|---|---|
| `ch` | 128 | **64** |
| `ch_mult` | [1,1,2,2,4,4] (6 levels) | **[1,2,2,4] (4 levels)** |
| `attn_resolutions` | [16] | [16] |
| `num_res_blocks` | 2 | 2 |
| Params | ~110M | **17,216,193** |

- Forward pass `(2,2,64,64) -> (2,1,64,64)` verified on GPU.
- Train-step peak VRAM at batch 16: **2.07 GB / 4.29 GB** -- comfortable headroom.

---

### [2026-06-18] [DONE] Diffusion algorithm -- `src/training/diffusion.py`

Adapted the notebook's "Diffusion Algorithm" cell for single-GPU, single-channel:

- `EMAHelper`, `get_beta_schedule`, `data_transform`/`inverse_data_transform`,
  conditional `noise_estimation_loss`, DDIM `generalized_steps` sampling.
- `DenoisingDiffusion` runner: `train` (antithetic timestep sampling + EMA),
  per-epoch loss tracking, best-checkpoint saving, DDIM `sample`, and `evaluate`
  (PSNR/SSIM/MSE in [0,1]).
- Loads this repo's `AstroDataset` patch batches `(B, n, 2, H, W) -> (B*n, 2, H, W)`,
  channel layout `[dirty, clean]`. Both arrays already in [0,1] (verified).

---

### [2026-06-18] [DONE] Training entry point -- `src/train_diffusion.py`

- 80/20 split (`random_state=42`), patch loaders (size 64, 4 patches/image),
  CLI for epochs/batch/lr/timesteps, loss-curve + best-checkpoint saving, DDIM eval.
- Smoke test (32 imgs, 1 epoch) passed end-to-end: train -> checkpoint -> load -> DDIM sample -> metrics.

---

### [2026-06-18] [DONE] Diffusion training run (local RTX 2050, 40 epochs)

- `python -m src.train_diffusion --epochs 40 --batch-size 4 --patch-n 4`
- 780/195 train/val images, effective batch 16 patches, GPU 2.7/4.1 GB @ 99%
  (epoch time ~90-170 s, with some spikes to 800-1100 s from background GPU contention).
- **Best val noise-loss: 11.9737 @ epoch 33** (step 6435). Loss is noisy epoch-to-epoch
  (random-timestep + antithetic sampling), trending down overall:

| Epoch | Train | Val | |
|---|---|---|---|
| 1 | 579.88 | 193.76 | *best |
| 2 | 164.29 | 128.50 | *best |
| 3 | 106.98 | 98.14 | *best |
| 32 | 22.91 | 27.11 | |
| **33** | 25.20 | **11.97** | *best |
| 34 | 26.14 | 18.55 | |
| 40 | 21.37 | 27.84 | |

- Artifacts: `results/checkpoints/diffusion_best.pth.tar`, `results/diffusion_loss.png`,
  `results/logs/diffusion_train.log`.

**DDIM evaluation (25 steps, 128 val patches), vs Week-3 baselines (ranked by SSIM):**

| Method | PSNR (dB) | SSIM | MSE |
|---|---|---|---|
| AE HybridLoss (Wk3) | 19.92 | **0.7609** | 0.013920 |
| VAE (Wk3) | 20.00 | 0.7059 | 0.013336 |
| U-Net HybridLoss (Wk3) | 20.63 | 0.7044 | 0.011653 |
| Gaussian σ=2 | 22.78 | 0.4230 | 0.006380 |
| **DDPM (this run, 40 ep)** | **14.75** | **0.2348** | **0.305799** |

**Interpretation:** the noise objective trained fine (per-pixel noise MSE ≈ 0.003), but the
*sampled* reconstructions are weak — SSIM 0.235 sits below all Week-3 regression models and
even the noisy input. This is the expected signature of an **under-trained DDPM**: ~6.4k steps
is tiny next to the source notebook's ~2M-iteration target. Motivates moving training to a
faster GPU (Kaggle) for many more steps.

---

### [2026-06-18] [DONE] Kaggle migration -- `notebooks/05_diffusion_kaggle.ipynb`

Self-running Kaggle notebook (21 cells) so the DDPM can train on a faster T4/P100 (16 GB):

- Clones the fork (`KrishanYadav333/EXXA`, `week-4` branch), `pip install pytorch-msssim`,
  imports `DiffusionUNet` / `DenoisingDiffusion` from `src/` (single source of truth).
- Auto-discovers `dirty.npy` / `clean.npy` under `/kaggle/input/` (data uploaded separately
  as a Kaggle Dataset -- too large for git: 2.7 GB + 1.4 GB).
- Larger effective batch (32 patches), `EPOCHS=300`, DDIM eval + sample visualisation,
  outputs saved to `/kaggle/working/`. Optional scale-up knob (ch=128 / more levels) documented.

**All notebooks made Kaggle-runnable.** Prepended an idempotent "Kaggle bootstrap" cell to
`01_data_exploration`, `02_autoencoder_model`, `03_vae_model`, `04_unet_model` that (on Kaggle)
clones the fork, `pip install`s `pytorch-msssim` + `torchinfo`, symlinks the uploaded
`dirty.npy`/`clean.npy` Dataset as `./data`, and `chdir`s into `notebooks/` so the existing
`../data` and `../results` relative paths resolve. The cell is a **no-op off Kaggle**, so local
runs are unaffected. Also fixed two `__file__`-based path lines in `02` (undefined in any kernel)
to `os.path.abspath('..')`. All five notebooks validate under `nbformat`.

---

### [2026-06-19] [DONE] Multi-GPU support (Kaggle T4 x2)

- `DenoisingDiffusion` now wraps the model in `nn.DataParallel` to split each batch across
  all visible GPUs. Auto-enabled when `torch.cuda.device_count() > 1`; `data_parallel=False`
  forces single-GPU. EMA + checkpoints operate on the **unwrapped** model, so the saved
  `state_dict` has no `module.` prefix and stays portable across single-/multi-GPU.
- `train_diffusion.py`: added `--data-parallel` / `--no-data-parallel` (auto by default).
- `05_diffusion_kaggle.ipynb`: prints GPU count, notes DataParallel, bumps `BATCH_IMAGES` to 16
  (64 patches, ~32/GPU) to keep both T4s busy.
- **Bug fix:** `DotDict` now converts nested dicts recursively, so nested config writes persist
  (`cfg.model.ch = 128`, `cfg.diffusion.num_diffusion_timesteps = ...`). Previously these were
  silently dropped (a throwaway copy was mutated), so the notebook's scale-up hints were no-ops.

---

### [2026-06-19] [DONE] Unified research notebook -- `notebooks/00_master_research_notebook.ipynb`

Comprehensive faithful merge of notebooks 01->05 into one Kaggle-runnable research notebook
(85 cells). Replaces the earlier slim orchestrator (`00_master_week1_to_week4.ipynb`, removed).
Preserves the entire journey rather than summarising it:

- **Phase 1-2 audit** (markdown): notebook inventory, cell-action ledger, dependency graph, dedup notes.
- **Environment**: one Kaggle bootstrap, single global `CONFIG` (+ `QUICK_TEST`), `seed_everything`,
  auto GPU/multi-GPU detection, Kaggle-resolved `CKPT_DIR`/`OUT_DIR`.
- **Data**: load once, EDA, quality checks, preprocessing notes, ONE canonical `PatchDataset`,
  ONE split + loaders, ONE sliding-window evaluator (merged from the 4x / 2x duplicates).
- **Architectures shown via `inspect.getsource`** of the real `src/` code (AE, VAE, UNet,
  NoiseScheduler, beta schedule, noise loss, DDIM steps, DiffusionUNet) -- no drift, fully preserved.
- **Per-model sections** with Evolution callouts (Previous/New/Why/Benefit): AE (MSE + Hybrid both
  trained), VAE (+latent-space analysis), U-Net, conditional DDPM (multi-GPU, resume support).
- **Two evaluation protocols**: full-image sliding-window leaderboard (reproduces Week-3 numbers)
  and a shared-patch unified table incl. DDPM (fair, single protocol).
- **Final pipeline** (`denoise_image(img, method)` for ae/vae/unet/ddpm) + Conclusion + Phase-6
  validation report (deps, bugs, conflicts, dedup, runtime, memory, GPU recs).

New/risky code paths smoke-tested locally (getsource, all trainers + `_MSEWrap`, latent math,
`ssim_torch` unified metrics, `denoise_image` incl. DDPM branch, baselines) before commit.

---

### [2026-06-19] [DONE] Scaled DDPM subsection added to master notebook

Added a "Week-4 Scaled Upgrade" subsection (16 cells -> notebook now 101 cells) after the
64px/17M baseline diffusion section, per the updated Week-4 plan (Kaggle T4x2):

- **Bigger backbone:** `DiffusionUNet` ch=128, ch_mult=[1,1,2,2,4] (5 levels, attn@16) =
  **71.4M params** (middle ground between the old 17M and the 110M 6-level / 51M create_model
  versions). Baseline 64px/17M run is **kept** for the scaling-story record.
- **Higher resolution:** 128x128 patches (4x the area of 64px).
- **Forward diffusion viz:** q(x_t|x_0) over T=1000 shown on a real patch (t=0..999).
- **Schedules:** linear vs cosine beta/alpha-cumprod plots (NoiseScheduler from PR #22),
  `SCALE_BETA_SCHEDULE` switch.
- **Auto OOM-fallback builder:** probes one fwd+bwd step; on CUDA OOM halves batch, then drops to
  64px -- since T4 VRAM couldn't be verified offline.
- **Deliverables:** training loss curve + first DDIM sample outputs (dirty->DDPM->clean) + PSNR/SSIM/MSE.

Why not literal "6-level / 51M / 600x600": 6-level needs input divisible by 32 (600 isn't) and 110M
@600^2 overflows 16 GB; documented in the subsection. All new paths smoke-tested locally -- the 71.4M
model trained at 128px even on the 4 GB RTX 2050 (batch_img=2), so it fits a T4 with headroom.

---

### [2026-06-19] [DONE] Week-4 plan-fidelity pass

Brought the scaled DDPM in line with the official Week-4 day-by-day plan:

- **Model = literal 6-level** `ch=128, ch_mult=[1,1,2,2,4,4]`, attn@16 (**109.7M params**, the plan's
  "51M-style" config) at 128x128. Note 6-level *does* build at 128px (128/32=4) -- the earlier
  divisible-by-32 objection only applied to full 600px. Verified build + forward (1,2,128,128)->(1,1,128,128).
- **DDIM sampling steps 25 -> 50** (`CONFIG['DDIM_STEPS']`), plan Task 4.1.
- **Forward-diffusion viz t-values -> [0,250,500,750,999]**, plan Task 1.2.
- **Per-10-step loss logging** added to `DenoisingDiffusion.train(log_every_step=...)` and used on both
  the baseline and scaled training calls (Day-3 deliverable "loss logged every 10 steps").
- OOM-fallback builder updated to retry the 6-level config. All verified locally.

Remaining (user-side): run the notebook on Kaggle (upload data, Run All) to produce the actual
Day-5 artifacts -- loss curve, sample PNGs, preliminary metrics.

---

## Next Steps (Week 4 cont.)

- [ ] Push Week-4 `src/` modules + Kaggle notebook to fork `week-4` branch (code only, no data).
- [ ] Upload `dirty.npy` / `clean.npy` as a Kaggle Dataset; run `05_diffusion_kaggle.ipynb`.
- [ ] Train DDPM for many more steps on Kaggle; re-evaluate SSIM vs the baselines above.
- [ ] If diffusion still trails, weigh longer training / bigger model vs. the regression U-Net.
