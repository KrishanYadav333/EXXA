# Pre-GSoC Contribution Guide for EXXA

## Welcome!

This guide will help you make meaningful contributions to EXXA before GSoC 2026.

## Your Current Setup Status

- [x] Local environment configured
- [x] Virtual environment created with core dependencies
- [x] Git configured with your fork
- [x] Ready to contribute!

## Contribution Areas

### 1. HIGH PRIORITY: Denoising Diffusion Project (NEW)

**Why this is perfect for pre-GSoC:**
- Brand new project with minimal code
- Ground-floor opportunity to make significant impact
- Directly related to GSoC 2026 project description
- High visibility for your contributions

**Current State:**
- Basic notebook exists with data loading
- Needs: Complete implementation from scratch

**Specific Tasks You Can Do:**

#### Phase 1: Research & Planning (Week 1-2)
```
[ ] Read key papers on diffusion models
    - DDPM (Denoising Diffusion Probabilistic Models)
    - DDIM (Denoising Diffusion Implicit Models)
    - Applications in scientific imaging
    
[ ] Document findings in DENOISING_DIFFUSION/research_notes.md

[ ] Create architecture design document
    - Model architecture choices
    - Training strategy
    - Data pipeline design
```

#### Phase 2: Data Pipeline (Week 3-4)
```
[ ] Create data loading utilities
    - FITS file readers for ALMA/VLT data
    - Data augmentation functions
    - Train/val/test splits
    
[ ] Write preprocessing scripts
    - Noise characterization
    - Normalization strategies
    - Data validation
    
[ ] Add unit tests for data pipeline
```

#### Phase 3: Model Implementation (Week 5-6)
```
[ ] Implement DDPM architecture
    - U-Net backbone
    - Time embedding
    - Noise scheduler
    
[ ] Create training script
    - Loss functions
    - Optimizer configuration
    - Logging with wandb
    
[ ] Add model checkpointing
```

#### Phase 4: Validation (Week 7-8)
```
[ ] Compare against traditional methods
    - CLEAN algorithm
    - Median filtering
    
[ ] Create evaluation metrics
    - PSNR, SSIM
    - Scientific metrics (feature preservation)
    
[ ] Generate result visualizations
```

**Quick Start for Denoising Diffusion:**

```bash
cd DENOISING_DIFFUSION

# Create project structure
mkdir -p src/models src/data src/utils tests docs

# Start with data exploration
jupyter notebook GSOC_2025_EXXA_Main.ipynb
```

### 2. MEDIUM PRIORITY: Documentation & Testing

**Why important:**
- Shows attention to detail
- Benefits entire community
- Easy to start, immediate impact

**Specific Tasks:**

```
[ ] Add docstrings to existing code
    - ANOMALY_DETECTION/EXXA_Final_JTerry/models/
    - FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION/Scripts/
    
[ ] Write unit tests
    - Data loading functions
    - Model utilities
    - Preprocessing scripts
    
[ ] Create tutorials
    - "How to train a model from scratch"
    - "How to use pre-trained models"
    - "Data preparation guide"
    
[ ] Improve README files for subprojects
```

### 3. Code Quality Improvements

```
[ ] Add type hints to Python functions
[ ] Create consistent error handling
[ ] Add logging throughout codebase
[ ] Refactor duplicate code
[ ] Add configuration file support (YAML/JSON)
```

### 4. Experiments & Analysis

```
[ ] Reproduce existing results
[ ] Try different hyperparameters
[ ] Compare model architectures
[ ] Benchmark performance
[ ] Create comparison charts
```

## Recommended Contribution Path

### Week 1-2: Foundation
1. Read DENOISING_DIFFUSION notebook thoroughly
2. Research diffusion models (papers, tutorials)
3. Create research notes document
4. Set up project structure

### Week 3-4: First PR
1. Implement data loading utilities
2. Add proper documentation
3. Write tests
4. Submit PR with clear description

### Week 5-6: Core Implementation
1. Implement basic DDPM model
2. Create training script
3. Add experiment tracking
4. Document progress

### Week 7-8: Validation & Results
1. Train on sample data
2. Compare results
3. Create visualizations
4. Write blog post about your work

## How to Submit Contributions

### 1. Create a Feature Branch

```bash
# For denoising diffusion work
git checkout -b feature/denoising-data-pipeline

# For documentation
git checkout -b docs/improve-readme

# For bug fixes
git checkout -b fix/model-loading-bug
```

### 2. Make Your Changes

```bash
# Add files
git add src/data/dataloader.py tests/test_dataloader.py

# Commit with clear message
git commit -m "Add FITS file dataloader with unit tests

- Implement FITSDataset class for ALMA observations
- Add data augmentation pipeline
- Include comprehensive unit tests
- Add documentation and examples"
```

### 3. Push to Your Fork

```bash
git push origin feature/denoising-data-pipeline
```

### 4. Create Pull Request

1. Go to https://github.com/KrishanYadav333/EXXA
2. Click "Compare & pull request"
3. Write clear description:
   - What problem does it solve?
   - How did you implement it?
   - How to test it?
   - Screenshots/results if applicable

### 5. Engage with Review

- Respond to feedback promptly
- Make requested changes
- Ask questions if unclear
- Be open to suggestions

## Contribution Quality Guidelines

### Code Style
- Follow PEP 8 for Python
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused (single responsibility)

### Documentation
- Write clear docstrings
- Include usage examples
- Document assumptions
- Explain "why" not just "what"

### Testing
- Write tests for new functions
- Aim for >80% code coverage
- Test edge cases
- Include integration tests

### Commits
- One logical change per commit
- Clear, descriptive messages
- Reference issues if applicable

## Communication

### Before Starting Major Work
1. Check existing issues/PRs
2. Create an issue describing your proposal
3. Wait for mentor feedback (24-48 hours)
4. Start implementation after approval

### Asking Questions
- Email: ml4-sci@cern.ch
- Include: What you've tried, specific question, context
- Be patient - mentors are busy!

### Getting Help
- Check QUICKSTART.md for setup issues
- Review existing code for patterns
- Google error messages
- Ask for help when stuck >2 hours

## Success Metrics

Good pre-GSoC contributions show:

1. **Technical Skills**
   - Clean, working code
   - Good software engineering practices
   - Understanding of ML/astronomy concepts

2. **Communication**
   - Clear PR descriptions
   - Responsive to feedback
   - Asks thoughtful questions

3. **Initiative**
   - Self-directed learning
   - Identifies problems
   - Proposes solutions

4. **Commitment**
   - Regular contributions
   - Follows through on PRs
   - Helps reviewers

## Example First Contribution

**Easy Win: Add Data Loading Utility**

```python
# DENOISING_DIFFUSION/src/data/fits_loader.py

from astropy.io import fits
import numpy as np
import torch
from torch.utils.data import Dataset

class AstroImageDataset(Dataset):
    """
    Dataset for loading astronomical FITS images.
    
    Args:
        file_list: List of paths to FITS files
        transform: Optional transform to apply
        normalize: Whether to normalize images
    
    Example:
        >>> dataset = AstroImageDataset(['image1.fits', 'image2.fits'])
        >>> image = dataset[0]
        >>> print(image.shape)  # (1, H, W)
    """
    
    def __init__(self, file_list, transform=None, normalize=True):
        self.file_list = file_list
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        # Load FITS file
        with fits.open(self.file_list[idx]) as hdul:
            image = hdul[0].data
            
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Normalize if requested
        if self.normalize:
            image = (image - image.mean()) / image.std()
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image
```

Then add tests, documentation, and create a PR!

## Resources

### Papers to Read
1. DDPM: https://arxiv.org/abs/2006.11239
2. DDIM: https://arxiv.org/abs/2010.02502
3. Diffusers Library: https://huggingface.co/docs/diffusers

### Tutorials
1. PyTorch DDPM Tutorial: https://huggingface.co/blog/annotated-diffusion
2. Diffusion Models from Scratch: https://www.youtube.com/watch?v=HoKDTa5jHvg

### Tools
- wandb: Experiment tracking
- tensorboard: Visualization
- pytest: Testing framework
- black: Code formatting

## Timeline for GSoC Application

- **Now - February 2026**: Make contributions
- **February 2026**: GSoC projects announced
- **March 18 - April 2**: GSoC application period
- **April 2026**: Application review
- **May 2026**: GSoC begins

**Pro Tip:** Start contributing NOW! Strong pre-GSoC contributions significantly increase acceptance chances.

## Next Steps

1. [ ] Read this guide completely
2. [ ] Pick one task from "Phase 1" above
3. [ ] Create a feature branch
4. [ ] Start working!
5. [ ] Ask questions when stuck
6. [ ] Submit your first PR

## Questions?

Email: ml4-sci@cern.ch
Subject: "[EXXA] Pre-GSoC Contribution Question"

Good luck with your contributions! We're excited to work with you.
