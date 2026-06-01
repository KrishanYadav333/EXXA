# GSoC 2026 Midterm Plan - EXXA Denoising
**Krishan Yadav** | Midterm Evaluation: July 6-10, 2026 | **40 Days to Deliver**

---

## Executive Summary
- **Goal**: Working end-to-end DDPM denoising pipeline for protoplanetary disk observations
- **Availability**: 8-10 hours/day
- **GPU**: Available locally
- **Current Blockers**: Missing `dataset.py`, untracked code, stuck PRs
- **Midterm Demo**: Trained model + inference notebook showing before/after denoising

---

## Phase 1: Foundation (May 27 - June 2) - 5 Working Days

### Goal: Get codebase to passing tests

#### Day 1-2: Fix Code Structure & Missing Files
**Task**: Create `src/data/dataset.py` 
- Implement `AstroDataset` class (wraps numpy/FITS data)
- Implement `create_dataloaders()` factory
- Use existing preprocessing/augmentation
- **Expected**: Tests pass, ~30 mins

**Task**: Organize untracked files into feature branch
```bash
# Create branch for Phase 1 fixes
git checkout -b feat/phase1-foundation-fix
# Move untracked code into proper locations
# Commit incrementally
```

#### Day 3: Run Tests & Fix Failures
```bash
cd DENOISING_DIFFUSION
pytest tests/test_data_pipeline.py -v
# Fix any remaining import/logic errors
```

#### Day 4-5: Review Existing Code
- Audit `fits_loader.py` (FITS file loading - ready?)
- Audit `preprocessing.py` (normalization - ready?)
- Audit `augmentation.py` (data augmentation - ready?)
- Audit `noise_scheduler.py` (q_sample - ready?)
- **Output**: Code review document noting readiness

---

## Phase 2: Build Training Pipeline (June 2 - June 20) - 13 Working Days

### Goal: Trainable DDPM model end-to-end

#### Sprint 2.1: Model Architecture (June 2-6)
**Create**: `src/models/unet.py`
- Implement U-Net backbone for denoising
- Include time embedding for diffusion timestep
- Include skip connections from encoder to decoder
- Target: 3-4M parameters (reasonable for protoplanetary disks)
- **Tests**: Verify forward pass on (B, C, 64, 64) tensors

```python
# Model checklist:
- [ ] Input: (x_t, t) -> noise prediction
- [ ] Skip connections from encoder to decoder
- [ ] Time embedding injected at multiple levels
- [ ] Residual blocks in bottleneck
- [ ] Output: predicted noise same shape as input
```

#### Sprint 2.2: Training Loop (June 6-15)
**Create**: `src/training/trainer.py`
- Loss function: MSE between predicted & sampled noise
- Optimizer: AdamW
- LR scheduler: Cosine annealing with warmup
- Checkpointing every N epochs
- Logging: epoch, loss, val_loss
- **Tests**: 2-3 epochs on small synthetic data

```python
# Training checklist:
- [ ] Data loader integration
- [ ] Forward pass: x_t = q_sample(x0, t)
- [ ] Model predicts noise
- [ ] Loss = MSE(predicted_noise, true_noise)
- [ ] Backward pass + optimizer step
- [ ] Checkpoint saving
```

#### Sprint 2.3: Config & Reproducibility (June 15-20)
**Create**: `configs/training.yaml`
```yaml
model:
  type: "unet"
  channels: 64
  depths: [2, 2, 2, 2]
  
training:
  epochs: 50
  batch_size: 16
  lr: 1e-4
  
diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
```

**Create**: `src/training/config.py` for loading/validation

---

## Phase 3: Evaluation & Inference (June 20 - July 3) - 10 Working Days

#### Sprint 3.1: Inference Pipeline (June 20-25)
**Implement** in `noise_scheduler.py`:
- `p_sample()` - single reverse diffusion step
- `p_sample_loop()` - full denoising from noise to clean image
- Use trained model to denoise real noisy observations

#### Sprint 3.2: Evaluation Metrics (June 25-30)
**Create**: `src/evaluation/metrics.py`
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- MSE (Mean Squared Error)
- Visual comparison plots

#### Sprint 3.3: Midterm Demo (June 30 - July 3)
**Create**: `notebooks/midterm_demo.ipynb`
- Load trained checkpoint
- Apply to test images
- Show before/after denoising
- Compute metrics
- Plot disk structures (rings/gaps preserved?)

---

## Checkpoint: Midterm Evaluation (July 6-10)

### Expected Deliverables:
1. ✅ Working data pipeline (FITS → PyTorch DataLoader)
2. ✅ Trained DDPM model (trained on synthetic data)
3. ✅ Inference pipeline (denoise function)
4. ✅ Evaluation metrics (PSNR, SSIM)
5. ✅ Demonstration notebook
6. ✅ Clean documentation

### Metrics to Report:
- Training loss convergence
- Validation PSNR/SSIM on test set
- Sample denoised images
- Time to denoise 1 image

---

## Git & PR Strategy

### Consolidate Scattered Branches
You currently have these feature branches (some old):
- feat/config-system
- feat/config-system-v2
- feat/unet-blocks
- feat/training-loop
- feat/model-skeleton
- feat/noise-scheduler
- feature/denoising-config-foundation

**Action**: Create single unified branch for Phase 1-3
```bash
git checkout -b feat/ddpm-implementation
# Build everything here, commit incrementally
# Each commit = 1 logical feature (e.g., "feat: add dataset.py", "feat: add U-Net model")
```

### PR Timeline for Midterm
- **June 7** (End of Phase 1): PR #1 → Data Pipeline (dataset.py + passing tests)
- **June 20** (End of Phase 2): PR #2 → Training (U-Net + trainer.py)
- **June 30** (End of Phase 3): PR #3 → Evaluation + Inference

### Keep PRs Focused
Each PR = 1 logical unit (~300-500 lines)
- Easy to review
- Easy to debug if issues
- Shows consistent progress to mentors

---

## Daily Schedule (8-10 hrs/day)

**Suggested Breakdown**:
- **2 hrs**: Code implementation
- **1 hr**: Testing & debugging
- **30 min**: Documentation
- **1 hr**: Code review (your own + others')
- **Flexible**: Adjust based on complexity

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Dataset not available | Use synthetic numpy arrays (dirty.npy + clean.npy) |
| Training too slow on CPU | Use GPU (you have it), reduce batch size if needed |
| Model not converging | Start with small model, simple data |
| Mentors request changes | Small PRs = faster feedback loops |
| Not enough time for diffusion models | Fall back to Autoencoder denoising |

---

## Resources

### Code to Reuse
- ✅ `src/data/fits_loader.py` (mostly done?)
- ✅ `src/data/preprocessing.py` (mostly done?)
- ✅ `src/data/augmentation.py` (mostly done?)
- ✅ `src/models/noise_scheduler.py` (q_sample done)

### To Create
- ❌ `src/data/dataset.py` (URGENT - fix import errors)
- ❌ `src/models/unet.py` (U-Net architecture)
- ❌ `src/training/trainer.py` (training loop)
- ❌ `src/evaluation/metrics.py` (PSNR, SSIM, MSE)
- ❌ `notebooks/midterm_demo.ipynb` (showcase notebook)

---

## Success Criteria for Midterm

✅ **Code works end-to-end**: Data → Model → Inference  
✅ **Tests pass**: All data pipeline + model tests green  
✅ **Model trained**: At least 1 checkpoint saved  
✅ **Metrics computed**: PSNR/SSIM on test set  
✅ **Documented**: Code comments + README updates  
✅ **PRs clean**: No merge conflicts, focused changes  

---

## Post-Midterm Plan (July 11 - Aug 24)

After midterm evaluation, you'll have 9+ weeks to:
1. Explore diffusion-specific improvements (DDIM, classifier-free guidance)
2. Extend to real ALMA/VLT observations
3. Compare with classical denoising (Gaussian, Wiener filters)
4. Optimize inference speed
5. Build visualization tools
6. Write final paper/report

---

## Questions to Answer First

Before you start Phase 1, clarify with mentors:
1. Do you have access to the EXXA dataset (FITS files)?
2. Or should you use the synthetic `dirty.npy` / `clean.npy` for training?
3. Any preferences on model architecture (from mentors)?
4. Target inference time per image?
