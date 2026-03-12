# Pre-GSoC Contribution Strategy for Denoising Project

**Project**: Denoising Astronomical Observations of Protoplanetary Disks
**Mentors**: Sergei Gleyzer (U. Alabama), Jason Terry (Oxford)
**Duration**: 175/350 hours
**Application Deadline**: April 2, 2026
**Current Date**: March 10, 2026

## URGENT Timeline (Applications open in 8 days!)

### Week 1: March 10-16 (BEFORE applications open)
**Goal**: Create solid foundation and demonstrable contributions

#### Days 1-2 (Mar 10-11): Project Setup
- [x] Set up development environment
- [ ] Create proper project structure
- [ ] Research diffusion models (DDPM, DDIM)
- [ ] Document understanding of project requirements
- [ ] Analyze existing data (dirty.npy, clean.npy)

#### Days 3-5 (Mar 12-14): Core Implementation
- [ ] Implement data loading pipeline for FITS files
- [ ] Create data augmentation utilities
- [ ] Write comprehensive unit tests
- [ ] Begin DDPM model architecture
- [ ] Set up experiment tracking (wandb)

#### Days 6-7 (Mar 15-16): Documentation & PR
- [ ] Document all implementations
- [ ] Create example notebooks
- [ ] Submit pull requests to your fork
- [ ] Prepare contribution summary for application

### Week 2: March 17-23 (Application period starts Mar 18)
**Goal**: Complete application while continuing contributions

#### Mar 17: Application Preparation
- [ ] Draft GSoC proposal (use template)
- [ ] List all your contributions
- [ ] Create timeline for 175h and 350h options
- [ ] Prepare CV highlighting relevant skills

#### Mar 18-19: Submit Application
- [ ] Complete test task (check link in description)
- [ ] Finalize proposal document
- [ ] Submit via Google form
- [ ] Email confirmation to ml4-sci@cern.ch

#### Mar 20-23: Continue Contributing
- [ ] Complete DDPM implementation
- [ ] Run initial training experiments
- [ ] Compare results with baseline
- [ ] Create result visualizations

### Week 3-4: March 24 - April 2 (Before deadline)
**Goal**: Strengthen application with additional contributions

- [ ] Implement data preprocessing pipeline
- [ ] Add support for different data formats
- [ ] Create comprehensive documentation
- [ ] Run performance benchmarks
- [ ] Help other contributors (show collaboration skills)

## What Mentors Look For

Based on the project description, demonstrate:

1. **Technical Skills**
   - PyTorch proficiency
   - Understanding of diffusion models
   - Data pipeline development
   - Scientific computing (numpy, scipy, astropy)

2. **Domain Understanding**
   - Knowledge of astronomical observations
   - Understanding of ALMA/VLT data
   - Awareness of traditional methods (CLEAN algorithm)
   - Appreciation for scientific rigor

3. **Software Engineering**
   - Clean, well-documented code
   - Unit testing
   - Version control (Git)
   - Reproducible research practices

4. **Communication**
   - Clear documentation
   - Responsive to feedback
   - Thoughtful questions
   - Collaborative attitude

## Strategic Contributions to Make NOW

### 1. Data Pipeline (HIGHEST PRIORITY)
**Why**: Foundation for everything else, shows you understand astronomical data

**Implementation Tasks**:
```python
# Create these files:
DENOISING_DIFFUSION/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fits_loader.py      # Load FITS files from ALMA/VLT
│   │   ├── synthetic_data.py   # Handle simulation outputs
│   │   ├── augmentation.py     # Data augmentation techniques
│   │   └── preprocessing.py    # Normalization, cropping, etc.
│   └── ...
└── tests/
    └── test_data_pipeline.py    # Unit tests
```

**Key Features to Implement**:
- FITS file reader with proper header handling
- Data normalization strategies
- Augmentation (rotation, flipping - respecting physics!)
- Train/val/test split utilities
- Batch loading with PyTorch DataLoader

### 2. DDPM Model Architecture (HIGH PRIORITY)
**Why**: Core of the project, shows ML expertise

**Implementation Tasks**:
```python
# Create these files:
src/
├── models/
│   ├── __init__.py
│   ├── unet.py            # U-Net backbone with attention
│   ├── ddpm.py            # DDPM implementation
│   ├── noise_scheduler.py # Forward/reverse diffusion process
│   └── time_embedding.py  # Sinusoidal time embeddings
```

**Key Components**:
- U-Net with skip connections
- Self-attention layers
- Time conditioning
- Noise scheduling (linear, cosine)
- Forward and reverse diffusion

### 3. Training Pipeline (MEDIUM PRIORITY)
**Why**: Demonstrates end-to-end understanding

**Implementation Tasks**:
```python
# Create these files:
src/
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Training loop with PyTorch Lightning
│   ├── losses.py          # Loss functions (L1, L2, perceptual)
│   └── metrics.py         # PSNR, SSIM, MS-SSIM
```

### 4. Documentation (CRITICAL)
**Why**: Shows communication skills and understanding

**Create**:
- `PROJECT_OVERVIEW.md` - Your understanding of the problem
- `ARCHITECTURE.md` - Model design decisions
- `DATA_FORMAT.md` - Description of input/output data
- Example notebooks with clear explanations

## Specific Contributions for Application

When applying, reference these concrete contributions:

### Contribution #1: Data Pipeline Implementation
```
Title: "Implement FITS data loading pipeline for astronomical observations"

Description:
- Created FITSDataset class for loading ALMA/VLT observations
- Implemented physics-aware data augmentation (rotation, flipping)
- Added comprehensive unit tests with >90% coverage
- Documented data format specifications

Impact: Provides foundation for training pipeline, ensures proper 
handling of astronomical data formats
```

### Contribution #2: DDPM Architecture
```
Title: "Implement DDPM model architecture for image denoising"

Description:
- Built U-Net backbone with self-attention layers
- Implemented noise scheduling algorithms (linear, cosine)
- Added time conditioning with sinusoidal embeddings
- Created modular, extensible design

Impact: Core model architecture ready for training experiments
```

### Contribution #3: Training Infrastructure
```
Title: "Set up training pipeline with experiment tracking"

Description:
- Created PyTorch Lightning training module
- Integrated wandb for experiment tracking
- Implemented multiple loss functions and metrics
- Added model checkpointing and logging

Impact: Enables systematic experimentation and result comparison
```

## Research to Complete

### Papers to Read (Priority Order):
1. **DDPM** (Ho et al., 2020) - https://arxiv.org/abs/2006.11239
   - Original denoising diffusion paper
   - Read: Sections 1-3, Algorithm 1

2. **DDIM** (Song et al., 2020) - https://arxiv.org/abs/2010.02502
   - Faster sampling method
   - Read: Introduction, Section 2

3. **Score-Based Generative Models** (Song & Ermon, 2019)
   - Theoretical foundation
   - Read: Section 3

4. **Diffusion Models in Medical Imaging** (Recent review)
   - Similar application domain
   - Learn from scientific imaging context

5. **ALMA Data Processing** (CASA documentation)
   - Traditional methods
   - Understand what you're improving

### Technical Concepts to Master:
- [ ] Forward diffusion process (adding noise)
- [ ] Reverse diffusion process (denoising)
- [ ] Noise scheduling strategies
- [ ] U-Net architecture for images
- [ ] Self-attention mechanisms
- [ ] FITS file format and headers
- [ ] Astronomical coordinate systems
- [ ] Radio interferometry basics

## Application Components

### 1. Test Task (REQUIRED)
Check link in project description. Complete thoroughly:
- Show ML skills
- Demonstrate code quality
- Add detailed documentation
- Submit via Google form

### 2. Project Proposal (REQUIRED)
Structure:
```
1. Personal Background
   - Your education and experience
   - Why you're interested in this project
   - Relevant coursework/projects

2. Project Understanding
   - Problem statement in your words
   - Current limitations of traditional methods
   - How diffusion models can help

3. Technical Approach
   - Model architecture choice (DDPM vs DDIM vs hybrid)
   - Training strategy
   - Data augmentation techniques
   - Evaluation metrics

4. Implementation Plan
   - Detailed timeline (week-by-week)
   - Milestones and deliverables
   - Risk mitigation strategies

5. Expected Outcomes
   - What you'll deliver
   - How it advances the field
   - Future extensions

6. Timeline Comparison
   - 175 hour version (minimum viable product)
   - 350 hour version (full implementation + extensions)
```

### 3. CV (REQUIRED)
Highlight:
- Python/PyTorch experience
- ML projects (especially computer vision)
- Scientific computing background
- Open source contributions
- Academic achievements
- Relevant coursework

## Communication Strategy

### Before Application (Now):
- DO: Email ml4-sci@cern.ch with specific questions
- DON'T: Contact mentors directly
- DO: Contribute to codebase
- DON'T: Spam with trivial questions

### Good Email Example:
```
Subject: [EXXA Denoising] Question about Data Format

Dear ML4SCI Team,

I'm preparing my GSoC application for the "Denoising Astronomical 
Observations" project and have a question about the training data.

I've examined the existing notebook (GSOC_2025_EXXA_Main.ipynb) 
which uses .npy files with paired dirty/clean observations (64x64x1).

Questions:
1. Will the actual project use FITS files from simulations?
2. What dimensions should the model support (64x64, 128x128, 256x256)?
3. Are there any specific data augmentation techniques to avoid 
   (that might violate physical constraints)?

I'm currently implementing a FITS data loader and want to ensure 
it matches project requirements.

Thank you,
Krishan Yadav
GitHub: KrishanYadav333
```

### After Submission:
- Be responsive to questions
- Continue contributing
- Help other applicants (collaboration++)
- Stay engaged in community

## Success Metrics

Your application will be strong if you:

1. **Technical Demonstration**
   - [ ] 2-3 substantial PRs merged
   - [ ] Working code that runs without errors
   - [ ] Test coverage >80%
   - [ ] Clear documentation

2. **Understanding**
   - [ ] Can explain diffusion models clearly
   - [ ] Understand astronomical data challenges
   - [ ] Know traditional methods (CLEAN)
   - [ ] Aware of project scope

3. **Planning**
   - [ ] Realistic timeline
   - [ ] Well-defined milestones
   - [ ] Risk awareness
   - [ ] Clear deliverables

4. **Communication**
   - [ ] Clear proposal writing
   - [ ] Responsive to feedback
   - [ ] Thoughtful questions
   - [ ] Collaborative attitude

## Next Steps (RIGHT NOW)

### Step 1: Complete Test Task (TODAY)
- Read test instructions
- Implement solution
- Document thoroughly
- Submit ASAP

### Step 2: Set Up Project Structure (TODAY)
```bash
cd DENOISING_DIFFUSION
mkdir -p src/data src/models src/training src/utils tests notebooks docs
```

### Step 3: Implement Data Pipeline (Mar 11-13)
- Start with FITS loader
- Add augmentation
- Write tests
- Create PR

### Step 4: Research & Document (Mar 11-14)
- Read DDPM paper
- Create research notes
- Document understanding

### Step 5: Start DDPM Implementation (Mar 14-16)
- U-Net backbone
- Basic forward/reverse process
- Initial training script

### Step 6: Prepare Application (Mar 17)
- Draft proposal
- Update CV
- Prepare test submission

### Step 7: Submit Application (Mar 18-19)
- Final proposal review
- Submit via form
- Confirmation email

## Resources

### Code Examples:
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
- Annotated DDPM: https://huggingface.co/blog/annotated-diffusion
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/

### Papers:
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Diffusion Models Beat GANs: https://arxiv.org/abs/2105.05233

### Astronomy:
- ALMA CASA: https://casa.nrao.edu/
- FITS format: https://fits.gsfc.nasa.gov/
- AstroPy: https://www.astropy.org/

### Contact:
- Email: ml4-sci@cern.ch
- Subject prefix: "[EXXA Denoising]"
- Application form: (provided in description)

## Immediate Action Items

**Today (March 10)**:
- [ ] Read complete test task instructions
- [ ] Start test task implementation
- [ ] Set up project structure
- [ ] Begin DDPM paper reading

**Tomorrow (March 11)**:
- [ ] Complete test task
- [ ] Implement FITS data loader
- [ ] Write unit tests
- [ ] Start research notes document

**March 12-14**:
- [ ] Submit test task
- [ ] Create PRs for contributions
- [ ] Continue DDPM implementation
- [ ] Draft proposal outline

**March 15-17**:
- [ ] Finalize initial contributions
- [ ] Complete proposal draft
- [ ] Update CV
- [ ] Prepare all materials

**March 18** (APPLICATION OPENS):
- [ ] SUBMIT APPLICATION

Remember: Quality over quantity. 2-3 excellent, well-documented contributions 
are better than 10 rushed ones.

Good luck!
