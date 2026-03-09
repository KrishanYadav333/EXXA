#  EXXA Quick Reference Guide

##  Your Setup is Complete!

All core packages are installed:
-  PyTorch 2.10.0
-  NumPy 2.4.3
-  SciPy 1.17.1
-  PyTorch Lightning 2.6.1
-  And more...

**Note:** Running on CPU mode. For GPU training, install CUDA toolkit.

##  Next Steps

### 1. Choose Your Project

```bash
# Explore a specific project
cd ANOMALY_DETECTION/EXXA_Final_JTerry
# or
cd DENOISING_DIFFUSION
```

### 2. Activate Environment (Always!)

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# You should see (venv) in your prompt
```

### 3. Run Example Code

**Test PyTorch:**
```python
import torch
x = torch.rand(5, 3)
print(x)
print(f"GPU Available: {torch.cuda.is_available()}")
```

**Start Jupyter:**
```bash
jupyter notebook
# Or
jupyter lab
```

##  Common Commands

### Virtual Environment
```bash
# Activate
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate

# Check installed packages
pip list

# Install additional package
pip install <package-name>
```

### Running Scripts
```bash
# Run Python script
python script_name.py

# Run with specific arguments
python train.py --epochs 10 --batch-size 32

# Check script help
python script_name.py --help
```

### Jupyter Notebooks
```bash
# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab (Modern UI)
jupyter lab

# Convert notebook to Python
jupyter nbconvert --to script notebook.ipynb
```

##  Project Structure Guide

```
EXXA/
 venv/                    # Your virtual environment
 requirements.txt         # Core dependencies
 test_setup.py           # Installation test
 SETUP.md                # Detailed setup guide

 ANOMALY_DETECTION/      # Transformer-based anomaly detection
    EXXA_Final_JTerry/
       models/         # Model architectures
       utils/          # Utility functions
       train_*.py      # Training scripts
    EXXA_Midterm_JTerry/

 DENOISING_DIFFUSION/    #  NEW PROJECT - Needs contributors!
    readme.md
    GSOC_2025_EXXA_Main.ipynb

 [Other projects...]
```

##  Working with Specific Projects

### Anomaly Detection
```bash
cd ANOMALY_DETECTION/EXXA_Final_JTerry
pip install -r requirements.txt
python train_multiloader_transformer.py --help
```

### Foundation Models (MAE)
```bash
cd FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION
# Open notebooks in Jupyter
jupyter notebook Notebooks/combined_notebook.ipynb
```

### Equivariant Networks
```bash
cd EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/final_submission
pip install -r requirements.txt
# Explore notebooks/
```

##  Troubleshooting

### Package Import Errors
```bash
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall package
pip install --force-reinstall <package-name>
```

### CUDA Not Available
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Errors
- Reduce batch size in training scripts
- Close other applications
- Use gradient accumulation

##  Useful Development Tools

### Code Formatting
```bash
pip install black flake8
black script.py          # Format code
flake8 script.py         # Check style
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature-name

# Stage changes
git add file.py

# Commit
git commit -m "Description of changes"

# Push to your fork
git push origin feature-name
```

### Experiment Tracking
```bash
# Weights & Biases (already installed)
wandb login
# Then use in your training scripts
```

##  Tips for GSoC Contributors

### 1. Start Small
- Run existing code first
- Understand the workflow
- Make incremental improvements

### 2. Document Your Work
- Add comments to code
- Update README files
- Write clear commit messages

### 3. Ask for Help
- Email: ml4-sci@cern.ch
- GitHub Issues
- Project-specific blogs

### 4. Test Your Changes
```bash
pytest                   # Run tests
python test_setup.py    # Verify setup
```

##  For Denoising Diffusion Contributors

This is a **NEW** project - great opportunity!

**What's Needed:**
1. Research diffusion models for image denoising
2. Implement training pipeline
3. Create synthetic training data
4. Test on real astronomical observations
5. Document everything

**Getting Started:**
```bash
cd DENOISING_DIFFUSION
pip install diffusers accelerate
jupyter notebook GSOC_2025_EXXA_Main.ipynb
```

**Key Papers to Read:**
- DDPM (Ho et al., 2020)
- Applications in scientific imaging
- Astronomical image processing

##  Cheat Sheet

```bash
# Always start with
.\venv\Scripts\Activate.ps1

# Check environment
python test_setup.py

# Start Jupyter
jupyter lab

# Install new package
pip install package-name

# Save requirements
pip freeze > my_requirements.txt

# Get help
python script.py --help
```

##  Success Checklist

- [ ] Virtual environment activated
- [ ] All packages installed (test_setup.py passes)
- [ ] Explored at least one subproject
- [ ] Ran example code successfully
- [ ] Jupyter notebooks working
- [ ] Git repository set up
- [ ] Ready to contribute!

---

**Need Help?** Check SETUP.md or reach out to ml4-sci@cern.ch

**Happy Coding! **

