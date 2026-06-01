# GSoC 2026 Test Task - Denoising Astronomical Observations

**Applicant**: Krishan Yadav  
**GitHub**: @KrishanYadav333  
**Project**: Denoising Astronomical Observations of Protoplanetary Disks  
**Date**: March 12, 2026

---

## Task Overview

This test demonstrates understanding of:
1. Image denoising techniques
2. Deep learning for astronomical data
3. Code quality and documentation
4. Experimental methodology

---

## Approach

### Part 1: Simple Baseline Denoiser (Autoencoder)

Implement a basic convolutional autoencoder as baseline before tackling diffusion models.

**Architecture**:
```
Encoder: Conv2D → ReLU → MaxPool (repeat 3x)
Bottleneck: Latent representation
Decoder: ConvTranspose2D → ReLU → Upsample (repeat 3x)
Output: Same size as input
```

**Implementation**: `test_task/simple_denoiser.py`

### Part 2: Data Handling

Use the existing data pipeline I implemented:
- Load dirty.npy and clean.npy
- Apply proper preprocessing
- Train/validation split

### Part 3: Training & Evaluation

- Loss: MSE + perceptual loss
- Metrics: PSNR, SSIM
- Visualizations: Before/after comparisons

### Part 4: Analysis

Compare against:
- Median filter (baseline)
- Gaussian filter
- My autoencoder

---

## Implementation Plan

```
test_task/
├── simple_denoiser.py       # Autoencoder model
├── train_baseline.py        # Training script
├── evaluate.py              # Evaluation & metrics
├── visualize.py             # Result visualization
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## Expected Results

**Quantitative**:
- Baseline PSNR: ~25 dB
- Autoencoder PSNR: ~28-30 dB
- Training time: <1 hour on CPU

**Qualitative**:
- Preserved disk structures
- Reduced noise significantly
- No artifacts introduced

---

## Next: Implementation

Shall I implement this test task solution now?
