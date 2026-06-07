# Denoising Diffusion System Architecture

## Project Overview

A complete end-to-end system for denoising astronomical observations of protoplanetary disks using Denoising Diffusion Probabilistic Models (DDPM).

**Goal**: Replace traditional CLEAN algorithm with faster, more accurate ML-based denoising.

**Input**: Noisy ALMA/VLT observations (FITS files)  
**Output**: Clean, high-fidelity images preserving scientific features  
**Method**: DDPM trained on synthetic observations from hydrodynamic simulations

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Raw Data     │───>│ Preprocessing │───>│ Augmentation │          │
│  │ - FITS files │    │ - Normalize  │    │ - Rotations  │          │
│  │ - .npy files │    │ - Pad/Resize │    │ - Flips      │          │
│  │ - Train/Val  │    │ - Transform  │    │ - Noise      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                    │                    │                  │
│         └────────────────────┴────────────────────┘                  │
│                              │                                       │
│                     ┌────────▼────────┐                             │
│                     │  PyTorch        │                             │
│                     │  DataLoader     │                             │
│                     └────────┬────────┘                             │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               │ Batches: (B, C, H, W)
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         DDPM MODEL                                   │
│  ┌────────────────────────────────────────────────────────┐         │
│  │              Forward Diffusion Process                 │         │
│  │  Clean Image ─> + Noise (t=0->T) ─> Pure Noise         │         │
│  │  x₀          β₁   βₜ              βₜ    xₜ            │         │
│  └────────────────────────────────────────────────────────┘         │
│                              │                                       │
│  ┌────────────────────────────────────────────────────────┐         │
│  │                    U-Net Backbone                       │         │
│  │                                                         │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │         │
│  │  │ Encoder  │  │ Bottleneck│  │ Decoder  │            │         │
│  │  │ - Conv   │─>│ - ResBlocks│─>│ - UpConv │            │         │
│  │  │ - Down   │  │ - Attention│  │ - Skip   │            │         │
│  │  │ - Skip   │  │            │  │   Conn   │            │         │
│  │  └──────────┘  └──────────┘  └──────────┘            │         │
│  │       ▲                                                 │         │
│  │       │                                                 │         │
│  │  ┌────┴──────────┐                                     │         │
│  │  │ Time Embedding│                                     │         │
│  │  │ Sinusoidal PE │                                     │         │
│  │  └───────────────┘                                     │         │
│  │                                                         │         │
│  │  Output: Predicted Noise epstheta(xₜ, t)                    │         │
│  └────────────────────────────────────────────────────────┘         │
│                              │                                       │
│  ┌────────────────────────────────────────────────────────┐         │
│  │           Reverse Diffusion (Inference)                │         │
│  │  Pure Noise ─> - Predicted Noise (t=T->0) ─> Clean     │         │
│  │  xₜ           epstheta                          x₀          │         │
│  └────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ Predictions
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      TRAINING PIPELINE                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Loss         │    │ Optimizer    │    │ Scheduler    │          │
│  │ - MSE Loss   │───>│ - AdamW      │───>│ - Cosine LR  │          │
│  │ - L1 Loss    │    │ - β=(0.9,..) │    │ - Warmup     │          │
│  │ - Perceptual │    │ - Weight     │    │              │          │
│  └──────────────┘    │   Decay      │    └──────────────┘          │
│                      └──────────────┘                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Logging      │    │ Checkpointing│    │ Validation   │          │
│  │ - Wandb      │    │ - Best Model │    │ - PSNR/SSIM  │          │
│  │ - TensorBoard│    │ - Last Model │    │ - Visual     │          │
│  │ - Metrics    │    │ - Resume     │    │   Inspection │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ Trained Model
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    INFERENCE PIPELINE                                │
│  ┌────────────────────────────────────────────────────────┐         │
│  │ 1. Load noisy observation (FITS)                       │         │
│  │ 2. Preprocess (normalize, pad)                         │         │
│  │ 3. Run reverse diffusion (T->0 steps)                   │         │
│  │ 4. Postprocess (unpad, denormalize)                    │         │
│  │ 5. Save clean image (FITS)                             │         │
│  └────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ Clean Images
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    EVALUATION & VALIDATION                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Metrics      │    │ Comparison   │    │ Sci. Valid.  │          │
│  │ - PSNR       │    │ - vs CLEAN   │    │ - Feature    │          │
│  │ - SSIM       │    │   algorithm  │    │   Preserv.   │          │
│  │ - MS-SSIM    │    │ - vs Median  │    │ - Flux       │          │
│  │ - LPIPS      │    │   Filter     │    │   Conservation│          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Data Pipeline ([DONE] IMPLEMENTED)

**Location**: `src/data/`

**Components**:
- `dataset.py`: PyTorch Dataset for paired noisy/clean images
- `fits_loader.py`: Load ALMA/VLT FITS observations
- `preprocessing.py`: Normalization, padding, transforms
- `augmentation.py`: Physics-aware data augmentation

**Status**: [DONE] Complete (24/24 tests passing)

**Input**: 
- Training: `dirty.npy` (noisy), `clean.npy` (ground truth)
- Inference: `.fits` files from telescopes

**Output**: PyTorch tensors ready for model training

---

### 2. DDPM Model ([*] TO IMPLEMENT)

**Location**: `src/models/`

#### 2.1 U-Net Architecture (`unet.py`)

```python
class UNet(nn.Module):
    """
    U-Net backbone for diffusion model.
    
    Architecture:
    - Encoder: 4 downsampling blocks (64->128->256->512 channels)
    - Bottleneck: ResNet blocks with self-attention
    - Decoder: 4 upsampling blocks with skip connections
    - Time conditioning: Sinusoidal positional embeddings
    
    Input: (B, C, H, W) image + time step t
    Output: (B, C, H, W) predicted noise
    """
    
    def __init__(
        self, 
        in_channels=1,      # Grayscale astronomical images
        out_channels=1,     # Predicted noise
        base_channels=64,   # Base feature channels
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.1
    ):
        # Implementation details below...
```

**Key Components**:
1. **ResidualBlock**: Conv + GroupNorm + SiLU activation
2. **AttentionBlock**: Self-attention for capturing global context
3. **Downsample/Upsample**: Spatial resolution changes
4. **TimeEmbedding**: Sinusoidal positional encoding for timestep t

#### 2.2 DDPM Core (`ddpm.py`)

```python
class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    Implements:
    - Forward diffusion: q(x_t | x_0) 
    - Reverse diffusion: p_theta(x_{t-1} | x_t)
    - Noise schedule: β_t (linear or cosine)
    """
    
    def __init__(
        self,
        model,              # U-Net backbone
        timesteps=1000,     # Diffusion steps
        beta_schedule='linear',
        loss_type='mse'
    ):
        # Precompute diffusion process constants
        self.betas = self._get_beta_schedule(...)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # ... other constants
        
    def forward_diffusion(self, x0, t, noise=None):
        """Add noise to clean image at timestep t."""
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * eps
        
    def reverse_diffusion_step(self, xt, t):
        """Predict one denoising step."""
        # x_{t-1} = mu_theta(x_t, t) + sigma_t * z
        
    def sample(self, shape):
        """Generate clean image from pure noise."""
        # Start from x_T ~ N(0, I)
        # Iteratively denoise: x_{T-1}, x_{T-2}, ..., x_0
```

**Key Methods**:
1. `q_sample()`: Forward diffusion (add noise)
2. `p_sample()`: Reverse diffusion (remove noise)
3. `p_losses()`: Training loss computation
4. `sample()`: Full sampling from noise to image

#### 2.3 Conditional DDPM (`conditional_ddpm.py`)

```python
class ConditionalDDPM(DDPM):
    """
    Conditional diffusion model.
    
    Conditions on:
    - Noisy observation (guide denoising)
    - Observation parameters (telescope, wavelength)
    - Noise level estimate
    """
    
    def __init__(self, model, condition_type='concatenation'):
        super().__init__(model)
        self.condition_type = condition_type
        
    def forward(self, x, t, condition):
        """Forward pass with conditioning."""
        if self.condition_type == 'concatenation':
            # Concatenate noisy obs with input
            x = torch.cat([x, condition], dim=1)
        elif self.condition_type == 'cross_attention':
            # Use cross-attention mechanism
            pass
```

---

### 3. Training Pipeline ([*] TO IMPLEMENT)

**Location**: `src/training/`

#### 3.1 Trainer (`trainer.py`)

```python
class DDPMTrainer:
    """
    Training orchestration for DDPM.
    
    Handles:
    - Training loop (epochs, batches)
    - Validation monitoring
    - Checkpointing
    - Logging (wandb, tensorboard)
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config
    ):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize logging, checkpointing, etc.
        
    def train_epoch(self):
        """Train for one epoch."""
        for batch in self.train_loader:
            images, _ = batch
            
            # Sample random timesteps
            t = torch.randint(0, self.timesteps, (batch_size,))
            
            # Forward diffusion (add noise)
            noise = torch.randn_like(images)
            noisy_images = self.model.q_sample(images, t, noise)
            
            # Predict noise
            predicted_noise = self.model(noisy_images, t)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
    def validate(self):
        """Validation with sample generation."""
        # Compute metrics on validation set
        # Generate sample images
        # Log to wandb
```

#### 3.2 Loss Functions (`losses.py`)

```python
class DiffusionLoss:
    """
    Loss functions for diffusion models.
    """
    
    @staticmethod
    def simple_loss(predicted, target):
        """MSE between predicted and true noise."""
        return F.mse_loss(predicted, target)
    
    @staticmethod
    def vlb_loss(model, x0, t):
        """Variational lower bound loss."""
        # Full ELBO computation
        
    @staticmethod
    def hybrid_loss(predicted, target, x0, model):
        """Combination of simple + VLB losses."""
        return simple_loss + λ * vlb_loss
```

#### 3.3 Configuration (`config.py`)

```python
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Model
    image_size: int = 64
    in_channels: int = 1
    base_channels: int = 64
    
    # Diffusion
    timesteps: int = 1000
    beta_schedule: str = 'linear'
    
    # Training
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    
    # Data
    train_data_path: str = "data/dirty.npy"
    val_split: float = 0.1
    num_workers: int = 4
    
    # Logging
    log_every: int = 100
    sample_every: int = 1000
    save_every: int = 5000
    
    # Hardware
    device: str = 'cuda'
    mixed_precision: bool = True
```

---

### 4. Inference Pipeline ([*] TO IMPLEMENT)

**Location**: `src/inference/`

#### 4.1 Inference Engine (`inference.py`)

```python
class DDPMInference:
    """
    Inference pipeline for denoising new observations.
    """
    
    def __init__(self, model_path, config):
        self.model = self._load_model(model_path)
        self.config = config
        
    def denoise_observation(self, fits_path, output_path):
        """
        Denoise a single FITS observation.
        
        Steps:
        1. Load FITS file
        2. Preprocess (normalize, pad)
        3. Run reverse diffusion
        4. Postprocess (unpad, denormalize)
        5. Save as FITS
        """
        # Load observation
        image, header = load_fits(fits_path)
        
        # Preprocess
        image_norm, stats = preprocess(image)
        image_padded, padding = pad_to_multiple(image_norm, 16)
        
        # Convert to tensor
        x = torch.from_numpy(image_padded).unsqueeze(0).unsqueeze(0)
        
        # Denoise with DDPM
        with torch.no_grad():
            # Start from noisy image + additional noise
            x_T = x + noise_factor * torch.randn_like(x)
            
            # Reverse diffusion
            for t in reversed(range(self.timesteps)):
                x_T = self.model.p_sample(x_T, t)
                
            x_clean = x_T
            
        # Postprocess
        x_clean = unpad_image(x_clean, padding)
        x_clean = denormalize(x_clean, stats)
        
        # Save FITS
        save_fits(x_clean, output_path, header)
        
        return x_clean
        
    def batch_denoise(self, input_dir, output_dir):
        """Denoise all FITS files in directory."""
        # Process multiple observations
```

#### 4.2 Fast Sampling (`sampling.py`)

```python
class DDIMSampler:
    """
    DDIM: Denoising Diffusion Implicit Models.
    
    Faster sampling (50 steps instead of 1000).
    """
    
    def __init__(self, model, ddim_steps=50):
        self.model = model
        self.ddim_steps = ddim_steps
        
    def sample(self, x_T):
        """Fast sampling with DDIM."""
        # Skip timesteps strategically
        # 10-20x faster than DDPM
```

---

### 5. Evaluation & Validation ([*] TO IMPLEMENT)

**Location**: `src/evaluation/`

#### 5.1 Metrics (`metrics.py`)

```python
class DenoisingMetrics:
    """
    Evaluation metrics for image denoising.
    """
    
    @staticmethod
    def psnr(clean, denoised):
        """Peak Signal-to-Noise Ratio."""
        mse = torch.mean((clean - denoised) ** 2)
        return 10 * torch.log10(1.0 / mse)
        
    @staticmethod
    def ssim(clean, denoised):
        """Structural Similarity Index."""
        # Use pytorch-msssim library
        
    @staticmethod
    def lpips(clean, denoised):
        """Perceptual similarity."""
        # Use learned perceptual metric
        
    @staticmethod
    def feature_preservation_score(clean, denoised):
        """
        Astronomy-specific metric.
        
        Checks if important features preserved:
        - Disk structures
        - Spiral arms
        - Gaps
        - Peak intensities
        """
```

#### 5.2 Comparison (`comparison.py`)

```python
class BaselineComparison:
    """
    Compare DDPM against traditional methods.
    """
    
    def __init__(self):
        self.methods = {
            'median_filter': self.median_filter,
            'gaussian_filter': self.gaussian_filter,
            'clean_algorithm': self.clean_algorithm,  # Radio astronomy standard
            'ddpm': self.ddpm
        }
        
    def compare_all(self, noisy_obs, clean_truth):
        """Run all methods and compute metrics."""
        results = {}
        
        for name, method in self.methods.items():
            denoised = method(noisy_obs)
            results[name] = {
                'psnr': psnr(clean_truth, denoised),
                'ssim': ssim(clean_truth, denoised),
                'time': execution_time
            }
            
        return results
```

---

### 6. Utilities ([*] TO IMPLEMENT)

**Location**: `src/utils/`

#### 6.1 Visualization (`visualization.py`)

```python
def plot_diffusion_process(images, timesteps):
    """Visualize forward diffusion."""
    # Show x_0, x_250, x_500, x_750, x_1000
    
def plot_denoising_progress(model, noisy_image):
    """Visualize reverse diffusion."""
    # Show denoising step-by-step
    
def compare_results(noisy, clean_truth, denoised):
    """Side-by-side comparison."""
    # 3-panel figure
```

#### 6.2 Experiment Tracking (`tracking.py`)

```python
class ExperimentTracker:
    """
    Integration with experiment tracking tools.
    """
    
    def __init__(self, project_name='exxa-denoising'):
        self.wandb_run = wandb.init(project=project_name)
        
    def log_metrics(self, metrics, step):
        """Log training metrics."""
        wandb.log(metrics, step=step)
        
    def log_images(self, images, caption):
        """Log sample generations."""
        wandb.log({"samples": [wandb.Image(img) for img in images]})
```

---

## File Structure (Complete)

```
DENOISING_DIFFUSION/
├── src/
│   ├── __init__.py
│   │
│   ├── data/                          # [DONE] COMPLETE
│   │   ├── __init__.py
│   │   ├── dataset.py                 # AstroDataset class
│   │   ├── fits_loader.py             # FITS file operations
│   │   ├── preprocessing.py           # Normalization, padding
│   │   └── augmentation.py            # Physics-aware transforms
│   │
│   ├── models/                        # [*] TO BUILD
│   │   ├── __init__.py
│   │   ├── unet.py                    # U-Net architecture
│   │   ├── blocks.py                  # ResBlock, AttentionBlock
│   │   ├── ddpm.py                    # DDPM core logic
│   │   ├── conditional_ddpm.py        # Conditional variant
│   │   └── embeddings.py              # Time embeddings
│   │
│   ├── training/                      # [*] TO BUILD
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop
│   │   ├── losses.py                  # Loss functions
│   │   ├── config.py                  # Configuration dataclass
│   │   └── callbacks.py               # Training callbacks
│   │
│   ├── inference/                     # [*] TO BUILD
│   │   ├── __init__.py
│   │   ├── inference.py               # Main inference engine
│   │   ├── sampling.py                # DDPM/DDIM samplers
│   │   └── batch_processing.py        # Batch inference
│   │
│   ├── evaluation/                    # [*] TO BUILD
│   │   ├── __init__.py
│   │   ├── metrics.py                 # PSNR, SSIM, etc.
│   │   ├── comparison.py              # Baseline comparisons
│   │   └── scientific_validation.py   # Astronomy-specific checks
│   │
│   └── utils/                         # [*] TO BUILD
│       ├── __init__.py
│       ├── visualization.py           # Plotting utilities
│       ├── tracking.py                # Experiment logging
│       └── checkpoint.py              # Model saving/loading
│
├── tests/                             # [DONE] Data pipeline tested
│   ├── test_data_pipeline.py          # 24 tests (passing)
│   ├── test_unet.py                   # [*] TO ADD
│   ├── test_ddpm.py                   # [*] TO ADD
│   └── test_training.py               # [*] TO ADD
│
├── configs/                           # [*] TO CREATE
│   ├── base_config.yaml               # Base hyperparameters
│   ├── small_model.yaml               # Fast training config
│   └── production.yaml                # Full-scale config
│
├── scripts/                           # [*] TO CREATE
│   ├── train.py                       # Training entry point
│   ├── inference.py                   # Inference entry point
│   ├── evaluate.py                    # Evaluation script
│   └── download_data.py               # Data download helper
│
├── notebooks/                         # [*] TO CREATE
│   ├── 01_data_exploration.ipynb      # Explore training data
│   ├── 02_model_architecture.ipynb    # Visualize U-Net
│   ├── 03_diffusion_process.ipynb     # Understand diffusion
│   └── 04_results_analysis.ipynb      # Analyze trained model
│
├── docs/                              # [DONE] Started
│   ├── DATA_PIPELINE_IMPLEMENTATION.md
│   └── ARCHITECTURE.md                # This file
│
├── experiments/                       # [*] WILL BE AUTO-GENERATED
│   ├── exp001/                        # First experiment
│   │   ├── config.yaml
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── samples/
│   └── exp002/
│
├── trained_models/                    # [*] WILL BE CREATED
│   ├── ddpm_base.pth
│   └── ddpm_best.pth
│
├── data/                              # [*] USER DOWNLOADS
│   ├── dirty.npy                      # Noisy synthetic obs
│   ├── clean.npy                      # Ground truth
│   └── real_observations/             # Real ALMA/VLT data
│       ├── obs001.fits
│       └── obs002.fits
│
├── ARCHITECTURE.md                    # This file
├── PRE_GSOC_STRATEGY.md              # [DONE] Complete
├── requirements.txt                   # [*] TO UPDATE
└── README.md                          # [*] TO UPDATE
```

---

## Technology Stack

### Core Framework
- **PyTorch 2.0+**: Deep learning framework
- **PyTorch Lightning**: Training orchestration (optional)
- **Torchvision**: Transform utilities

### Scientific Computing
- **NumPy**: Array operations
- **SciPy**: Scientific algorithms
- **Astropy**: FITS file handling, astronomy utilities

### Experiment Tracking
- **Weights & Biases (wandb)**: Experiment logging
- **TensorBoard**: Alternative logging
- **Matplotlib**: Visualization

### Evaluation
- **pytorch-msssim**: SSIM metrics
- **lpips**: Perceptual similarity
- **scikit-image**: Image metrics

### Development
- **pytest**: Unit testing
- **black**: Code formatting
- **mypy**: Type checking
- **pre-commit**: Git hooks

---

## Data Flow

### Training Phase

```
Raw Data (dirty.npy, clean.npy)
    │
    ├─> AstroDataset
    │   ├─ Load paired images
    │   ├─ Extract random patches
    │   └─ Apply augmentation
    │
    ├─> DataLoader
    │   └─ Batch, shuffle, parallel loading
    │
    ├─> Forward Diffusion
    │   ├─ Sample timestep t ~ Uniform(0, T)
    │   ├─ Sample noise eps ~ N(0, I)
    │   └─ Compute x_t = sqrt(alpha_t)*x_0 + sqrt(1-alpha_t)*eps
    │
    ├─> U-Net Prediction
    │   ├─ Input: (x_t, t)
    │   └─ Output: eps_theta(x_t, t)
    │
    ├─> Loss Computation
    │   └─ L = ||eps - eps_theta||^2
    │
    ├─> Backpropagation
    │   ├─ Compute gradients
    │   ├─ Update weights
    │   └─ Log metrics
    │
    └─> Validation
        ├─ Generate samples
        ├─ Compute PSNR/SSIM
        └─ Save checkpoint if best
```

### Inference Phase

```
Noisy FITS Observation
    │
    ├─> Load & Preprocess
    │   ├─ Read FITS file
    │   ├─ Normalize intensity
    │   └─ Pad to model size
    │
    ├─> Initialize Noise
    │   └─ x_T = noisy_obs + small_noise
    │
    ├─> Reverse Diffusion Loop (T->0)
    │   For t = T, T-1, ..., 1:
    │   ├─ Predict noise: eps_theta(x_t, t)
    │   ├─ Compute mean: mu_theta(x_t, t)
    │   ├─ Sample: x_{t-1} ~ N(mu_theta, sigma_t^2I)
    │   └─ (optionally log progress)
    │
    ├─> Extract Clean Image
    │   └─ x_0 = final denoised image
    │
    └─> Postprocess & Save
        ├─ Remove padding
        ├─ Denormalize intensity
        ├─ Write FITS file
        └─ Preserve header metadata
```

---

## Implementation Phases

### Phase 1: Foundation ([DONE] COMPLETE)
**Status**: Done  
**Duration**: 3 days

- [x] Data pipeline (loading, preprocessing, augmentation)
- [x] Test suite (24 unit tests)
- [x] Documentation (architecture, strategy)
- [x] Repository structure

### Phase 2: Model Architecture ([*] NEXT - Days 1-3)
**Priority**: HIGH  
**Estimated**: 3-4 days

- [ ] U-Net backbone implementation
  - [ ] ResidualBlock
  - [ ] AttentionBlock
  - [ ] Encoder/Decoder
  - [ ] Time embedding
- [ ] DDPM core logic
  - [ ] Noise scheduler
  - [ ] Forward diffusion (q_sample)
  - [ ] Reverse diffusion (p_sample)
  - [ ] Loss computation
- [ ] Unit tests for model components
- [ ] Model architecture notebook

**Deliverable**: Working U-Net that can predict noise

### Phase 3: Training Pipeline ([*] Days 4-6)
**Priority**: HIGH  
**Estimated**: 3 days

- [ ] Training loop implementation
- [ ] Loss functions
- [ ] Optimizer & scheduler setup
- [ ] Logging integration (wandb)
- [ ] Checkpointing system
- [ ] Configuration management
- [ ] Training script

**Deliverable**: Can train model end-to-end

### Phase 4: Initial Training ([*] Days 7-10)
**Priority**: MEDIUM  
**Estimated**: 3-4 days

- [ ] Download training data (dirty.npy, clean.npy)
- [ ] Train small model (quick validation)
- [ ] Monitor training metrics
- [ ] Generate first samples
- [ ] Debug issues
- [ ] Tune hyperparameters

**Deliverable**: First trained model checkpoint

### Phase 5: Inference & Evaluation ([*] Days 11-14)
**Priority**: MEDIUM  
**Estimated**: 4 days

- [ ] Inference pipeline
- [ ] Fast sampling (DDIM)
- [ ] Metrics computation (PSNR, SSIM)
- [ ] Baseline comparison
- [ ] Visualization tools
- [ ] Inference script

**Deliverable**: Can denoise new observations

### Phase 6: Optimization & Scaling ([*] Days 15-21)
**Priority**: MEDIUM  
**Estimated**: 7 days

- [ ] Train full-scale model
- [ ] Hyperparameter optimization
- [ ] Mixed precision training
- [ ] Multi-GPU support
- [ ] Batch inference optimization
- [ ] Model compression

**Deliverable**: Production-ready model

### Phase 7: Validation on Real Data ([*] Days 22-28)
**Priority**: HIGH (for GSoC success)  
**Estimated**: 7 days

- [ ] Obtain real ALMA/VLT observations
- [ ] Test on real data
- [ ] Compare with CLEAN algorithm
- [ ] Scientific validation
- [ ] Performance benchmarking
- [ ] Results documentation

**Deliverable**: Proof of scientific utility

### Phase 8: Documentation & Paper ([*] Days 29-35)
**Priority**: HIGH (for GSoC deliverables)  
**Estimated**: 7 days

- [ ] Complete API documentation
- [ ] User guide (how to use)
- [ ] Developer guide (how to extend)
- [ ] Tutorial notebooks
- [ ] Results analysis
- [ ] Technical report/paper draft

**Deliverable**: Publication-ready documentation

---

## Key Design Decisions

### 1. Why U-Net Architecture?

**Reasons**:
- Skip connections preserve spatial information
- Proven for image-to-image tasks
- Handles multiple resolutions
- Standard in diffusion literature

### 2. Why DDPM over Other Denoising Methods?

**Advantages**:
- State-of-the-art image generation quality
- Flexible: can adapt to various noise types
- Probabilistic: captures uncertainty
- Recent success in scientific imaging

**Compared to**:
- Traditional filters: Less adaptive
- VAE/GAN: Mode collapse, training instability
- Other denoisers: Less flexible

### 3. Image Size: 64x64 vs 256x256?

**Start with 64x64**:
- Faster iteration during development
- Lower memory requirements
- Sufficient for proof-of-concept

**Scale to 256x256**:
- Higher fidelity for science
- Requires more compute
- Can use progressive training

### 4. Conditional vs Unconditional?

**Start Unconditional**:
- Simpler to implement
- Learn pure noise->clean mapping

**Add Conditioning Later**:
- Use noisy observation as condition
- Better guided denoising
- Higher performance

---

## Performance Requirements

### Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, V100)
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ for datasets and checkpoints
- **Time**: 1-2 days for full training (1000 epochs)

### Inference
- **GPU**: Can run on CPU (slower) or same GPU
- **Speed**: ~10-30 seconds per image (1000 steps)
  - DDIM: ~1-3 seconds (50 steps)
- **Memory**: 2GB+ RAM

---

## Success Metrics

### Technical Metrics
- **PSNR**: >30 dB (good), >35 dB (excellent)
- **SSIM**: >0.90 (good), >0.95 (excellent)
- **Training Loss**: Converges to <0.01
- **Inference Speed**: <30s per 256x256 image

### Scientific Metrics
- **Feature Preservation**: Spiral arms, gaps, rings intact
- **Flux Conservation**: Total flux preserved within 5%
- **Noise Reduction**: SNR improvement >10dB
- **vs CLEAN**: Competitive or better quality, 10x+ faster

### GSoC Success Criteria
- [ ] Working end-to-end pipeline
- [ ] Trained model on synthetic data
- [ ] Validation on real observations
- [ ] Performance comparison published
- [ ] Code documented and tested
- [ ] Reproducible results

---

## References

### Key Papers
1. **DDPM**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
2. **DDIM**: "Denoising Diffusion Implicit Models" (Song et al., 2020)
3. **U-Net**: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)

### Astronomical Context
4. **CLEAN**: "An introduction to CLEAN" (Hogbom, 1974)
5. **ALMA**: Atacama Large Millimeter/submillimeter Array documentation
6. **Protoplanetary Disks**: Recent observational reviews

### Code References
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
- Annotated DDPM: https://huggingface.co/blog/annotated-diffusion
- PyTorch Lightning: https://lightning.ai/

---

## Next Steps for Implementation

### Immediate (This Week):
1. **Create model architecture** (Phase 2)
   - Start with `unet.py`
   - Implement ResidualBlock, AttentionBlock
   - Add time embedding
   - Write unit tests

2. **Build trainer** (Phase 3)
   - Create `trainer.py` skeleton
   - Implement basic training loop
   - Add wandb logging

3. **Download data**
   - Get dirty.npy and clean.npy
   - Verify with existing data pipeline
   - Create train/val split

### This Month:
1. Train first model (Phase 4)
2. Implement inference (Phase 5)
3. Generate samples and compute metrics

### For GSoC Application:
1. Submit PR with data pipeline (already done!)
2. Complete Phase 2 (model architecture)
3. Show preliminary training results
4. Document progress

---

**Questions? Ready to start implementing?**

Let me know which phase you want to tackle first, and I'll help you implement it step by step!
