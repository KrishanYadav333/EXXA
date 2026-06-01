# EXXA - Local Setup Guide

This guide will help you set up the EXXA project locally on your machine.

## Prerequisites

- Python 3.10+ (Currently using Python 3.13.5 )
- Git
- 10+ GB free disk space (for datasets and models)
- CUDA-capable GPU (recommended for training)

## Quick Setup

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/ML4SCI/EXXA.git
cd EXXA
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Core Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install core requirements
pip install -r requirements.txt
```

### 4. Install Subproject-Specific Dependencies

Choose the subproject(s) you want to work on:

#### Anomaly Detection
```bash
cd ANOMALY_DETECTION/EXXA_Final_JTerry
pip install -r requirements.txt
cd ../..
```

#### Atmosphere Characterization
```bash
cd ATMOSPHERE_CHARACTERIZATION
pip install jupyter numpy pandas matplotlib scikit-learn tensorflow torch
cd ..
```

#### Foundation Models
```bash
cd FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION
pip install torch torchvision numpy scipy astropy pillow scikit-learn
cd ..
```

#### Equivariant Networks
```bash
cd EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/final_submission
pip install -r requirements.txt
cd ../..
```

#### Denoising Diffusion (New Project)
```bash
cd DENOISING_DIFFUSION
pip install torch torchvision numpy scipy matplotlib jupyter diffusers accelerate
cd ..
```

## Project Structure

```
EXXA/
 ANOMALY_DETECTION/          # Protoplanetary disk anomaly detection
 ATMOSPHERE_CHARACTERIZATION/ # Exoplanet atmosphere analysis
 FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION/
 EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/
 DENOISING_DIFFUSION/        # Denoising astronomical observations
 DUST_CONTINUUM_APPROCH/     # Planet detection from dust continuum
 KINEMATIC_APPROACH/         # Kinematic analysis
 QUANTUM_MACHINE_LEARNING_FOR_EXOPLANET_CHARACTERIZATION/
 TIME_SERIES_APPROACH/       # Kepler data analysis
 NEURAL_NETWORK_CLASSIFIER/  # TESS exoplanet classifier
```

## Verification

Test your installation:

```python
# test_setup.py
import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy version: {np.__version__}")
print(" Setup successful!")
```

Run: `python test_setup.py`

## Common Issues

### Issue: CUDA not available
**Solution:** Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Import errors
**Solution:** Make sure virtual environment is activated:
```bash
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

### Issue: Memory errors during training
**Solution:** Reduce batch size or use gradient accumulation

## GPU Setup (Optional but Recommended)

Check GPU availability:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

## Next Steps

1. **Explore a subproject:** Navigate to any subproject folder
2. **Read the README:** Each subproject has specific instructions
3. **Run examples:** Try running existing notebooks or scripts
4. **Start contributing:** Pick an issue or feature to work on

## Data Setup

Most projects require datasets. Check individual project READMEs for:
- Data download links
- Expected data formats
- Preprocessing requirements

## Getting Help

- **Main repository:** https://github.com/ML4SCI/EXXA
- **Issues:** Report problems on GitHub Issues
- **Contact:** ml4-sci@cern.ch
- **Documentation:** Check individual project READMEs

## Contributing

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes
3. Commit: `git commit -m "Description"`
4. Push: `git push origin feature-name`
5. Create a Pull Request

## IDE Setup (Optional)

### VS Code
Install recommended extensions:
- Python
- Jupyter
- Pylance
- GitLens

### PyCharm
Configure interpreter to use the virtual environment at `./venv`

---

**Last Updated:** March 2026
**Python Version:** 3.10+
**Primary Framework:** PyTorch 2.0+

