# GSoC 2026 -- EXXA Denoising Diffusion
## Work Log: Week 1 and Week 2

**Author:** Krishan Yadav
**Project:** ML4Sci -- Denoising Astronomical Observations of Protoplanetary Disks
**Period:** May 27 -- June 7, 2026

---

## Repository Layout

```
DENOISING_DIFFUSION/
|
|-- data/
|   |-- clean.npy               # (975, 600, 600) float32 -- ground truth images
|   +-- dirty.npy               # (975, 600, 600) float64 -- noisy observations
|
|-- src/
|   |-- __init__.py
|   |-- baselines.py            # classical denoising filters + metric computation
|   |-- dataset_stats.py        # statistical analysis + visualization of dataset
|   |-- visualize.py            # single-image visualization helpers
|   |-- visualize_baselines.py  # comparison plots for all classical methods
|   |-- train_autoencoder.py    # full training script for the autoencoder
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- autoencoder.py      # DenoisingAutoencoder (1.73M params)
|   |   |-- unet.py             # U-Net skeleton (not yet trained)
|   |   +-- noise_scheduler.py  # DDPM noise schedule (forward process)
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- dataset.py          # AstroDataset + create_dataloaders()
|   |   |-- fits_loader.py      # FITS file loading utilities
|   |   |-- preprocessing.py   # normalization + resize helpers
|   |   +-- augmentation.py    # random rotation, flip, crop augmentation
|   |
|   +-- training/
|       |-- __init__.py
|       +-- trainer.py          # generic Trainer class (for DDPM later)
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb   # data loading, stats, baselines
|   +-- 02_autoencoder_model.ipynb  # autoencoder architecture + training results
|
|-- results/
|   |-- autoencoder_loss.png        # train vs val loss curves
|   |-- checkpoints/
|   |   +-- autoencoder_best.pth    # best saved model (epoch 30)
|   +-- stats/
|       |-- sample_grid.png
|       |-- pixel_distribution.png
|       |-- mean_images.png
|       |-- noise_difference.png
|       |-- baseline_visual.png
|       +-- baseline_metrics_chart.png
|
|-- tests/
|   |-- test_data_pipeline.py
|   |-- test_noise_scheduler.py
|   +-- test_unet.py
|
|-- presentation.html           # Reveal.js presentation for Week 2 meeting
|-- progress.md                 # running progress log
|-- readme.md                   # project readme
+-- ARCHITECTURE.md             # full system design + phase plan
```

---

## Week 1 -- Foundation and Baselines (June 3, 2026)

### 1.1 Environment Setup

Verified that the local machine is correctly configured for GPU training:
- Python 3.13
- PyTorch 2.7.1+cu118
- CUDA 11.8 -- NVIDIA GeForce RTX 2050 (4 GB VRAM)

Key constraint discovered: `num_workers=0` is mandatory on Windows because Python's spawn-based multiprocessing breaks PyTorch DataLoaders. This is hardcoded throughout the training scripts.

Console encoding is `cp1252` (Windows default), which does not support Unicode. All print statements use ASCII-safe strings with `>>` instead of Unicode arrows, and no emoji.

---

### 1.2 Dataset Inspection

**Files involved:** `src/dataset_stats.py`

The dataset consists of 975 paired protoplanetary disk images generated from hydrodynamic simulations (PHANTOM SPH + MCFOST radiative transfer), following the same pipeline as Terry et al. (2022).

| Property | clean.npy | dirty.npy |
|----------|-----------|-----------|
| Shape | (975, 600, 600) | (975, 600, 600) |
| dtype | float32 | float64 |
| Min | 0.0 | 0.0 |
| Max | 1.0 | 1.0 |
| Mean | 0.1525 | 0.1777 |
| Std | 0.2508 | 0.1941 |
| Memory | ~1.4 GB | ~2.8 GB |

**Key issue found:** `dirty.npy` is saved as `float64` while PyTorch and clean data use `float32`. Feeding `float64` tensors into CUDA would cause type-mismatch errors and consume unnecessary VRAM.

**Fix:** Cast dirty to `float32` immediately on load:
```python
dirty = np.load(dirty_path).astype(np.float32)
```

This halved the memory footprint from 2.8 GB to 1.4 GB and eliminated all downstream dtype errors.

**Visualizations generated:**
- `results/stats/sample_grid.png` -- 3 random clean/dirty pairs side by side (inferno colormap)
- `results/stats/pixel_distribution.png` -- overlapping histograms of pixel values from 100 random samples
- `results/stats/mean_images.png` -- mean over all 975 images for clean and dirty separately
- `results/stats/noise_difference.png` -- mean absolute difference (dirty - clean) as a heatmap

How `dataset_stats.py` works:
- Loads both arrays using `np.load()`
- Casts dirty to float32
- Uses `np.mean(arr, axis=0)` to collapse all 975 images into one mean image
- Randomly samples 100 image indices with `np.random.choice()`, flattens them with `.flatten()`, and plots overlapping histograms with `plt.hist(density=True, alpha=0.5)`
- Computes `np.abs(dirty - clean)` then takes the mean across axis 0 for the noise heatmap

---

### 1.3 Classical Denoising Baselines

**Files involved:** `src/baselines.py`, `src/visualize_baselines.py`

Before training any neural network, we established what classical image processing filters can achieve on this dataset. This gives us a concrete minimum target for our ML models to beat.

**Three filter families tested:**
1. **Gaussian Blur** (`scipy.ndimage.gaussian_filter`) -- applies a Gaussian smoothing kernel to reduce high-frequency noise. Tested with sigma=1 and sigma=2.
2. **Median Filter** (`scipy.ndimage.median_filter`) -- replaces each pixel with the median of its local neighborhood. Tested with a 3x3 kernel. Good at removing salt-and-pepper noise.
3. **Wiener Filter** (`scipy.signal.wiener`) -- an adaptive linear filter that minimizes mean squared error. Adjusts locally based on signal and noise variance.

**Metrics used** (`skimage.metrics`):
- `peak_signal_noise_ratio(data_range=1.0)` -- measures ratio of maximum possible signal power to noise power, expressed in dB. Higher is better.
- `structural_similarity(data_range=1.0)` -- measures perceptual similarity including luminance, contrast, and structure. Scale of 0 to 1. Higher is better. This metric is critical because it penalizes blurring of the disk gaps we care about.
- MSE -- computed directly as `np.mean((clean - denoised)**2)`. Lower is better.

**Results averaged over 50 samples:**

| Method | PSNR (dB) | SSIM | MSE |
|--------|-----------|------|-----|
| Noisy input | 21.66 | 0.2358 | 0.009048 |
| Gaussian sigma=1 | 22.86 | 0.4771 | 0.006917 |
| Gaussian sigma=2 | 22.92 | 0.5214 | 0.006806 |
| Median 3x3 | 22.93 | 0.4358 | 0.006921 |
| Wiener | 22.69 | 0.4151 | 0.007186 |

**Findings:**
- Gaussian sigma=2 is the overall best classical method: highest SSIM (0.5214) and lowest MSE (0.006806)
- Median wins marginally on PSNR but lags badly on SSIM, meaning it smears structural features
- Classical methods gain only ~1.3 dB PSNR over noisy input
- **ML target: PSNR > 22.93, SSIM > 0.52, MSE < 0.006806**

How `baselines.py` works:
- `gaussian_denoise(img, sigma)` wraps `gaussian_filter()` directly
- `median_denoise(img, size)` wraps `median_filter()` directly
- `wiener_denoise(img)` wraps `scipy.signal.wiener()`
- `compute_metrics(clean, denoised)` calls PSNR, SSIM, MSE and returns a dict
- `run_baselines(clean, dirty, n_samples)` iterates over the first n_samples images, runs all four methods on each, appends metrics to lists, then prints averaged results

How `visualize_baselines.py` works:
- Takes sample index 42 and applies all filters to it
- Generates a 2x3 matplotlib grid (Clean, Noisy, Gaussian, Median, Wiener, Difference map)
- Calls `run_baselines()` on 50 samples to get averaged metrics
- Generates a dual-axis bar chart with PSNR on the left y-axis and SSIM on the right y-axis using `ax.twinx()`

---

## Week 2 -- Autoencoder Architecture and Training (June 5, 2026)

### 2.1 Convolutional Autoencoder Architecture

**File:** `src/models/autoencoder.py`

The first ML model is a pure Encoder-Bottleneck-Decoder convolutional autoencoder. It takes a noisy 64x64 patch and reconstructs the corresponding clean patch. No skip connections (unlike a U-Net), which forces the bottleneck to learn a compact representation.

**Building block -- `ConvBlock`:**
```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 3x3 conv, preserve spatial dims
            nn.BatchNorm2d(out_ch),                   # normalize batch activations
            nn.ReLU(inplace=True),                    # non-linearity, in-place saves memory
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # second 3x3 conv
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
```
Two 3x3 convolutions per block, each followed by BatchNorm and ReLU. Padding=1 on a 3x3 kernel preserves spatial dimensions (same-padding).

`BatchNorm2d` normalizes the output of each channel to zero mean and unit variance across the batch. This stabilizes training and allows higher learning rates.

**Full architecture -- tensor shapes at each stage:**

```
Input:         (B,   1,  64, 64)
enc1 (32ch):   (B,  32,  64, 64)  -- ConvBlock(1, 32)
pool:          (B,  32,  32, 32)  -- MaxPool2d(2), halves spatial dims
enc2 (64ch):   (B,  64,  32, 32)  -- ConvBlock(32, 64)
pool:          (B,  64,  16, 16)
enc3 (128ch):  (B, 128,  16, 16)  -- ConvBlock(64, 128)
pool:          (B, 128,   8,  8)
bottleneck:    (B, 256,   8,  8)  -- ConvBlock(128, 256) -- compressed representation
up3:           (B, 128,  16, 16)  -- ConvTranspose2d(256, 128, 2, stride=2)
dec3:          (B, 128,  16, 16)  -- ConvBlock(128, 128)
up2:           (B,  64,  32, 32)  -- ConvTranspose2d(128, 64, 2, stride=2)
dec2:          (B,  64,  32, 32)  -- ConvBlock(64, 64)
up1:           (B,  32,  64, 64)  -- ConvTranspose2d(64, 32, 2, stride=2)
dec1:          (B,  32,  64, 64)  -- ConvBlock(32, 32)
out:           (B,   1,  64, 64)  -- Conv2d(32, 1, 1) -- 1x1 conv to collapse channels
sigmoid:       (B,   1,  64, 64)  -- clamp output to [0, 1]
```

`ConvTranspose2d(in, out, kernel=2, stride=2)` doubles the spatial resolution -- it is the decoder's upsampling operation, learnable (unlike bilinear upsampling).

Total parameters: **1,734,305** (about 6.6 MB in float32)

---

### 2.2 Training Data Pipeline -- `PatchDataset`

**File:** `src/train_autoencoder.py`

Why patches instead of full images: A single 600x600 float32 image occupies 1.37 MB. At batch size 16, a full-image batch would be 22 MB just for inputs -- not accounting for activations and gradients. This exceeds 4 GB VRAM quickly. Instead, we extract one random 64x64 patch per image per epoch.

```python
class PatchDataset(Dataset):
    def __getitem__(self, idx):
        ps = self.patch_size  # 64

        # random top-left corner for this epoch
        r = np.random.randint(0, self._h - ps + 1)   # 0 to 536
        c = np.random.randint(0, self._w - ps + 1)   # 0 to 536

        # slice the patch
        dirty_patch = self.dirty[idx, r:r+ps, c:c+ps]
        clean_patch = self.clean[idx, r:r+ps, c:c+ps]

        # per-patch min-max normalization based on the dirty patch range
        lo, hi = dirty_patch.min(), dirty_patch.max()
        if hi > lo:
            dirty_patch = (dirty_patch - lo) / (hi - lo)
            clean_patch = np.clip((clean_patch - lo) / (hi - lo), 0.0, 1.0)

        # add channel dimension: (H, W) -> (1, H, W)
        return (
            torch.from_numpy(dirty_patch[np.newaxis]),
            torch.from_numpy(clean_patch[np.newaxis]),
        )
```

The random patch location changes every epoch for every image, effectively giving the model 975 * ~536 * ~536 = ~280 million possible patches to learn from. This is a massive form of data augmentation that prevents overfitting.

The per-patch normalization uses the dirty patch's own min/max so the model always receives values in [0, 1] regardless of where in the image the patch came from (some regions are much brighter than others).

**Data split:**
```python
train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42)
# -> 780 train images, 195 val images
```

**DataLoaders:**
```python
DataLoader(train_ds, batch_size=16, shuffle=True,
           num_workers=0, pin_memory=True)
```
- `shuffle=True` on train to randomize batch order each epoch
- `num_workers=0` because Windows spawn-based multiprocessing breaks DataLoader
- `pin_memory=True` pins CPU memory so the `.to(device)` transfer is faster (uses DMA instead of page-by-page copy)

Batches per epoch: 780 / 16 = 48 train batches, 195 / 16 = 12 val batches

---

### 2.3 Training Loop

**File:** `src/train_autoencoder.py`

**Hyperparameters:**
```python
PATCH_SIZE  = 64
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.20
SEED        = 42
NUM_WORKERS = 0   # Windows requirement
```

**Optimizer:** `torch.optim.Adam(model.parameters(), lr=1e-3)`

Adam (Adaptive Moment Estimation) maintains per-parameter learning rates adjusted based on first and second moment estimates of gradients. Substantially faster to converge than plain SGD on this type of image reconstruction task.

**Loss:** `nn.MSELoss()` -- computes the mean squared pixel error between the model's output and the clean ground truth patch.

**Scheduler:** `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)`

If the validation loss does not improve for 5 consecutive epochs, the learning rate is halved. This prevents the model from oscillating around a minimum without converging into it.

The LR dropped twice during training:
- After epoch 13: 1e-3 -> 5e-4
- After epoch 24: 5e-4 -> 2.5e-4

**The `run_epoch` helper:**
```python
def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for dirty, clean in loader:
            dirty = dirty.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(dirty)
            loss = criterion(pred, clean)
            if train:
                optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * dirty.size(0)

    return total_loss / len(loader.dataset)
```

- `model.eval()` disables Dropout and BatchNorm's running stats update during validation
- `torch.no_grad()` disables gradient tracking entirely during validation -- saves VRAM and speeds up evaluation
- `set_to_none=True` on `zero_grad` sets gradients to `None` rather than zero tensors, which is slightly faster

**Checkpoint saving:** Every time val loss improves, the checkpoint is saved with:
```python
torch.save({
    'epoch':                epoch,
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss':             val_loss,
    'train_loss':           train_loss,
}, ckpt_path)
```
Saving the optimizer state allows training to be resumed exactly from this point.

---

### 2.4 Training Results

Full 30 epoch run on NVIDIA RTX 2050, approximately 2 seconds per epoch.

**Key milestone epochs:**

| Epoch | Train MSE | Val MSE | LR |
|-------|-----------|---------|-----|
| 1 | 0.045146 | 0.026865 | 1e-3 |
| 4 | 0.017881 | 0.010610 | 1e-3 |
| 8 | 0.012505 | 0.008527 | 1e-3 |
| 19 | 0.011269 | 0.007249 | 5e-4 |
| 26 | 0.009941 | 0.007243 | 2.5e-4 |
| **30** | **0.009467** | **0.007082** | 2.5e-4 |

**Best checkpoint:** Epoch 30 -- Val MSE = 0.007082
Saved to: `results/checkpoints/autoencoder_best.pth`
Loss curve: `results/autoencoder_loss.png`

**Comparison vs classical baselines (MSE):**

| Method | MSE | vs Noisy Input |
|--------|-----|----------------|
| Noisy Input | 0.009048 | baseline |
| Autoencoder (30 ep) | 0.007082 | -21.7% |
| Gaussian sigma=2 (best classical) | 0.006806 | -24.8% |

The autoencoder already achieved -21.7% MSE reduction over noisy input after just 30 short epochs. It is close to the best classical filter and expected to surpass it with more training or augmentation.

---

### 2.5 Visualization Scripts

**`src/dataset_stats.py`**
Run with: `python src/dataset_stats.py`
Outputs 4 statistical visualizations to `results/stats/`

**`src/visualize_baselines.py`**
Run with: `python src/visualize_baselines.py`
Outputs 2 baseline comparison visualizations to `results/stats/`

**`src/visualize.py`**
Helper module with utility functions for rendering single images and clean/dirty comparison grids.

---

### 2.6 Notebooks

**`notebooks/01_data_exploration.ipynb`**
- Environment and GPU check
- Load and inspect data arrays
- dtype mismatch detection and fix
- Run classical baselines on 50 samples
- Bar chart and image comparison visualizations
- Results summary table

**`notebooks/02_autoencoder_model.ipynb`**
- Architecture description and tensor shape walkthrough
- Model instantiation and parameter count
- PatchDataset and DataLoader setup
- Training loop (or skip to load the saved checkpoint)
- Loss curve visualization
- Visual comparison: dirty vs clean vs autoencoder output on real samples

---

## What Is NOT Done Yet (Pending for Week 3+)

| Task | File | Status |
|------|------|--------|
| Evaluate autoencoder PSNR/SSIM vs baselines | new script needed | pending |
| Visual comparison on full 600x600 images | new script needed | pending |
| Run existing test suite | `pytest tests/` | pending |
| U-Net with skip connections | `src/models/unet.py` | skeleton exists, not trained |
| DDPM noise scheduler integration | `src/models/noise_scheduler.py` | exists, not connected |
| Full DDPM training loop | `src/training/trainer.py` | skeleton exists, not complete |
| Dataset class integration | `src/data/dataset.py` | exists, import errors pending |

---

## How Everything Connects

```
data/clean.npy  \
                 +-- PatchDataset (__getitem__ extracts random 64x64 patches)
data/dirty.npy  /         |
                           |
                     DataLoader (batch_size=16, shuffle, pin_memory)
                           |
                     DenoisingAutoencoder.forward(dirty_batch)
                           |
                     nn.MSELoss(prediction, clean_batch)
                           |
                     loss.backward()  ->  Adam.step()
                           |
                     ReduceLROnPlateau.step(val_loss)
                           |
                     autoencoder_best.pth  (saved when val_loss improves)
                           |
                     autoencoder_loss.png  (train vs val MSE curves)
```

The baselines pipeline runs separately and feeds into the presentation charts. The autoencoder results will be added to the same chart once PSNR/SSIM are computed on full-size images.
