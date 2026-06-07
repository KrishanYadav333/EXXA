# Progress Tracking -- DENOISING_DIFFUSION

This document tracks all tasks, actions, and progress made during the GSoC 2026 session.

---

## Task Log

### [2026-06-03] [DONE] Verification of CUDA
- **Task**: Verify CUDA availability in Python / PyTorch.
- **Status**: Completed successfully.
- **Output**: 
  - CUDA available: `True`
  - GPU Device: `NVIDIA GeForce RTX 2050`

---

### [2026-06-03] [DONE] Repository Structure Mapped
- **Task**: Map and document the directory structure of `DENOISING_DIFFUSION`.
- **Status**: Completed successfully.
- **Details**:
  - `data/`: `clean.npy` (1.4GB), `dirty.npy` (2.8GB)
  - `src/data/`: `dataset.py`, `fits_loader.py`, `preprocessing.py`, `augmentation.py`
  - `src/models/`: `unet.py`, `noise_scheduler.py`
  - `src/training/`: `trainer.py`
  - `src/inference/`: (empty -- to be implemented)
  - `tests/`: `test_data_pipeline.py`, `test_noise_scheduler.py`

---

### [2026-06-03] [DONE] Data Inspection
- **Task**: Inspect shapes, bounds, and dtypes of `clean.npy` and `dirty.npy`.
- **Status**: Completed successfully.
- **Output**:

| File | Shape | Min | Max | dtype |
|------|-------|-----|-----|-------|
| `clean.npy` | `(975, 600, 600)` | 0.0 | 1.0 | `float32` |
| `dirty.npy` | `(975, 600, 600)` | 0.0 | 1.0 | `float64` |

- **Notes**:
  - 975 samples, each 600x600 pixels
  - Both already normalized to [0, 1]
  - **dtype mismatch** -- `dirty` is `float64`, `clean` is `float32` -> need to cast `dirty` to `float32` before training

---

### [2026-06-03] [DONE] Exploration Notebook Created
- **Task**: Create a proper Jupyter notebook for data exploration and pipeline validation.
- **Status**: Completed successfully.
- **File**: `notebooks/01_data_exploration.ipynb`
- **Covers**:
  1. Environment & GPU verification
  2. Dataset loading & inspection (shape, dtype, min/max)
  3. Visual exploration -- clean/dirty pairs + pixel histograms
  4. Data pipeline smoke test (`AstroDataset` + `create_dataloaders`)
  5. DDPM noise schedule check (alpha bar progression)
  6. UNet sanity check (forward pass with dummy input)

### [2026-06-03] [DONE] Classical Baselines -- `src/baselines.py`
- **Task**: Run Gaussian, Median, Wiener denoising on 50 samples and measure PSNR/SSIM/MSE.
- **Status**: Completed successfully.
- **Results** (averaged over 50 samples, 600x600 images):

| Method | PSNR (up) | SSIM (up) | MSE (down) |
|--------|--------|--------|-------|
| Noisy (input) | 21.6579 | 0.2358 | 0.009048 |
| Gaussian sigma=1 | 22.8625 | 0.4771 | 0.006917 |
| Gaussian sigma=2 | 22.9181 | **0.5214** | **0.006806** |
| Median size=3 | **22.9336** | 0.4358 | 0.006921 |
| Wiener | 22.6930 | 0.4151 | 0.007186 |

- **Key findings**:
  - Gaussian sigma=2 wins on SSIM (0.52) and MSE -- best overall classical baseline.
  - Median wins marginally on PSNR but lower SSIM than Gaussian.
  - All classical methods improve over the noisy input.
  - **DDPM target**: must beat PSNR > 22.93, SSIM > 0.52.

### [2026-06-03] [DONE] Baseline Visualization -- `src/visualize.py`
- **Task**: Create visualization script for classical baselines.
- **Status**: Completed successfully.
- **File**: `src/visualize.py`
- **Output**: Generates comparison plots of Clean (GT), Noisy Input, Gaussian, Median, and Wiener filtering.

### [2026-06-03] [DONE] Denoising Autoencoder -- `src/models/autoencoder.py`
- **Task**: Build first ML model -- encoder-bottleneck-decoder architecture.
- **Status**: Completed successfully. Forward pass verified on GPU.
- **File**: `src/models/autoencoder.py`
- **Details**:
  - Architecture: 3-level encoder (1->32->64->128), bottleneck (256), 3-level decoder
  - ConvBlocks: Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU
  - Total parameters: **1,734,305**
  - Input/Output: `(B, 1, H, W)` -> `(B, 1, H, W)` with sigmoid activation
  - Forward pass test: [DONE] `(2, 1, 64, 64)` -> `(2, 1, 64, 64)`

---

### [2026-06-05] [DONE] Autoencoder Training -- `src/train_autoencoder.py`
- **Task**: Train DenoisingAutoencoder for 30 epochs on full dataset.
- **Status**: Completed successfully.
- **Files**:
  - Script: `src/train_autoencoder.py`
  - Checkpoint: `results/checkpoints/autoencoder_best.pth`
  - Loss plot: `results/autoencoder_loss.png`
  - Notebook: `notebooks/02_autoencoder_model.ipynb`
- **Training Setup**:
  - Data: 975 images (600x600) -> 1 random 64x64 patch per image per epoch
  - Split: 780 train / 195 val (80/20, seed=42)
  - Optimizer: Adam (lr=1e-3), ReduceLROnPlateau (factor=0.5, patience=5)
  - Loss: MSELoss, Batch size: 16, Epochs: 30, ~2s/epoch on RTX 2050
- **Training Results (first 5 epochs)**:

```
 Epoch     Train MSE       Val MSE          LR     Time
---------------------------------------------------------
     1      0.045146      0.026865    1.00e-03    2.4s <<
     2      0.019390      0.013693    1.00e-03    1.9s <<
     3      0.017383      0.023118    1.00e-03    1.9s
     4      0.017881      0.010610    1.00e-03    1.9s <<
     5      0.014661      0.013101    1.00e-03    1.9s
```

- **Key epochs**:

| Epoch | Train MSE | Val MSE | LR |
|-------|-----------|---------|----|
| 1 | 0.045146 | 0.026865 | 1e-3 |
| 8 | 0.012505 | 0.008527 | 1e-3 |
| 19 | 0.011269 | 0.007249 | 5e-4 |
| **30** | 0.009467 | **0.007082** | 2.5e-4 |

- **Best Checkpoint**: Epoch 30 -- Val MSE = **0.007082**
- **LR Schedule**: 1e-3 -> 5e-4 (ep 14) -> 2.5e-4 (ep 25)
- **Comparison vs Classical Baselines** (MSE on noisy input = 0.009048):
  - Gaussian sigma=2: MSE = 0.006806
  - Autoencoder (30 ep): MSE = **0.007082** -- -21.7% vs noisy, close to Gaussian sigma=2

---

## Next Steps

- [x] Add baseline visualization to `notebooks/GSOC_2025_EXXA_Main.ipynb`
- [x] Build denoising autoencoder (`src/models/autoencoder.py`)
- [x] Train autoencoder for 30 epochs -- best val MSE: 0.007082 (epoch 30)
- [x] Save best checkpoint to `results/checkpoints/autoencoder_best.pth`
- [x] Generate loss curves -> `results/autoencoder_loss.png`
- [ ] Evaluate autoencoder PSNR/SSIM vs classical baselines
- [ ] Visual denoising comparison on real test samples
- [ ] Run tests: `pytest tests/`
- [ ] Begin DDPM training loop validation
