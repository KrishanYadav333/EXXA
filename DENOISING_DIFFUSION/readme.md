# EXXA Denoising Diffusion

Denoising pipeline for protoplanetary disk observations using classical baselines, convolutional autoencoders, and diffusion models.

**GSoC 2026 -- EXXA Project**

---

## Project Structure

```
DENOISING_DIFFUSION/
├── data/                          # clean.npy, dirty.npy (975 x 600x600)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data inspection, pipeline validation
│   └── 02_autoencoder_model.ipynb # Autoencoder architecture + training
├── experiments/                   # Generated comparison plots
├── results/
│   ├── autoencoder_loss.png       # Training loss curves
│   └── checkpoints/               # Saved model weights
├── screenshots/                   # GSoC proposal figures
├── scripts/
│   └── generate_screenshots.py    # Proposal screenshot generator
│
├── src/
│   ├── __init__.py
│   ├── baselines.py               # Classical denoising (Gaussian, Median, Wiener)
│   ├── visualize.py               # Baseline visualization helpers
│   ├── train_autoencoder.py       # Autoencoder training script
│   ├── data/
│   │   ├── dataset.py             # AstroDataset, DataLoaders
│   │   ├── fits_loader.py         # FITS file loading
│   │   ├── preprocessing.py       # Normalization, patching
│   │   └── augmentation.py        # Data augmentation transforms
│   ├── models/
│   │   ├── autoencoder.py         # DenoisingAutoencoder (1.7M params)
│   │   ├── unet.py                # UNet for DDPM
│   │   └── noise_scheduler.py     # DDPM noise schedule
│   └── training/
│       └── trainer.py             # DDPM training loop
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_noise_scheduler.py
│   └── test_unet.py
│
├── progress.md                    # Task log and results tracking
├── ARCHITECTURE.md                # Detailed architecture documentation
├── MIDTERM_PLAN.md                # GSoC midterm milestone plan
└── PRE_GSOC_STRATEGY.md           # Pre-GSoC preparation strategy
```

## Setup

```bash
# Dependencies
pip install torch numpy matplotlib scikit-learn scikit-image scipy torchinfo

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Classical Baselines
```bash
python src/baselines.py
```

### Train Autoencoder
```bash
python src/train_autoencoder.py
```

### Run Tests
```bash
pytest tests/
```

## Results

### Classical Baselines (50 samples, 600x600)

| Method         | PSNR   | SSIM   | MSE     |
|----------------|--------|--------|---------|
| Noisy input    | 21.66  | 0.2358 | 0.009048|
| Gaussian s=2   | 22.92  | 0.5214 | 0.006806|
| Median (3x3)   | 22.93  | 0.4358 | 0.006921|

### Autoencoder (30 epochs, RTX 2050)

| Metric        | Value   |
|---------------|---------|
| Best val MSE  | 0.007082|
| Best epoch    | 30      |
| Parameters    | 1,734,305 |
| Time/epoch    | ~2s     |

## Hardware

- GPU: NVIDIA GeForce RTX 2050
- CUDA: Available
