# EXXA - Exoplanet eXploration with AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**EXXA** is a comprehensive machine learning project under [ML4SCI](https://ml4sci.org/) focused on applying state-of-the-art AI techniques to exoplanet research and protoplanetary disk analysis.

##  Overview

This repository contains multiple Google Summer of Code (GSoC) contributions exploring various machine learning approaches for:
- **Detecting exoplanets** in protoplanetary disks
- **Characterizing atmospheres** of exoplanets
- **Denoising astronomical observations**
- **Analyzing kinematic data** from telescopes
- **Applying quantum ML** to exoplanet science

##  Quick Start

### Prerequisites
- Python 3.10+ (Tested with Python 3.13)
- 10+ GB free disk space
- CUDA-capable GPU (recommended, but CPU works)

### Installation

```bash
# Clone the repository
git clone https://github.com/ML4SCI/EXXA.git
cd EXXA

# Run setup script (Windows)
.\setup.ps1

# Or install manually
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_setup.py
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

##  Projects

### 1. Anomaly Detection in Protoplanetary Disks
**Goal:** Detect non-Keplerian features in disk observations using unsupervised learning

**Tech:** Transformers, Autoencoders, Domain Adaptation
-  [`ANOMALY_DETECTION/`](ANOMALY_DETECTION/)
-  [Blog Post](https://exxaanomalydetection.wordpress.com/2023/09/23/gsoc-2023-final-report/)

### 2. Atmosphere Characterization
**Goal:** Identify chemical species from exoplanet transmission spectra

**Tech:** CNN, LSTM, GRU, POSEIDON
-  [`ATMOSPHERE_CHARACTERIZATION/`](ATMOSPHERE_CHARACTERIZATION/)
-  [Blog 1](https://medium.com/@shuklag554/exoplanet-atmosphere-characterization-gsoc24-ml4sci-5f78f85faa13) | [Blog 2](https://medium.com/@shuklag554/exoplanet-atmosphere-characterization-gsoc24-ml4sci-part-2-96392e3ba190)

### 3. Foundation Models (MAE)
**Goal:** Self-supervised learning on protoplanetary disk images

**Tech:** Masked Autoencoders, Vision Transformers
-  [`FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION/`](FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION/)

### 4. Equivariant Networks
**Goal:** Leverage rotational symmetries in astronomical images

**Tech:** e2cnn, Steerable CNNs, Equivariant VGG16
-  [`EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/`](EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/)
-  Best Result: 96% accuracy

### 5. Denoising Diffusion (New!)
**Goal:** Denoise ALMA/VLT observations using diffusion models

**Tech:** DDPM, DDIM, Diffusion Networks
-  [`DENOISING_DIFFUSION/`](DENOISING_DIFFUSION/)
-  **In Progress - Contributions Welcome!**

### 6. Dust Continuum Approach
**Goal:** Detect planets in dust continuum images

**Tech:** FARGO3D, RADMC3D, CNNs
-  [`DUST_CONTINUUM_APPROCH/`](DUST_CONTINUUM_APPROCH/)
-  [Blog Post](https://medium.com/@mihirtripathi97/exxa-detecting-planets-in-dusty-disks-bd5a7db30cc8)

### 7. Kinematic Approach
**Goal:** Find exoplanets using kinematic analysis

**Tech:** RegNet, EfficientNetV2, PHANTOM+MCFOST
-  [`KINEMATIC_APPROACH/`](KINEMATIC_APPROACH/)
-  [Blog Post](https://medium.com/@jason.terry47/finding-exoplanets-with-deep-learning-1d271c73e588)

### 8. Quantum Machine Learning
**Goal:** Apply QML to exoplanet characterization

**Tech:** Quantum Algorithms, POSEIDON
-  [`QUANTUM_MACHINE_LEARNING_FOR_EXOPLANET_CHARACTERIZATION/`](QUANTUM_MACHINE_LEARNING_FOR_EXOPLANET_CHARACTERIZATION/)
-  [Blog Post](https://medium.com/@sourishphate/quantum-machine-learning-for-exoplanet-characterization-gsoc-25-ml4sci-c6c6cb4590b9)

### 9. Time Series Approach
**Goal:** Analyze Kepler light curves

**Tech:** Time series analysis, Deep learning
-  [`TIME_SERIES_APPROACH/`](TIME_SERIES_APPROACH/)

### 10. Neural Network Classifier
**Goal:** Binary classification of TESS exoplanet candidates

**Tech:** CNNs, TESS data
-  [`NEURAL_NETWORK_CLASSIFIER/`](NEURAL_NETWORK_CLASSIFIER/)

##  Contributing

We welcome contributions! Here's how to get started:

1. **Pick a project** that interests you
2. **Read the subproject README** for specific requirements
3. **Fork the repository** and create a feature branch
4. **Make your changes** following Python best practices
5. **Submit a pull request** with a clear description

### Contribution Ideas
-  Improve documentation and tutorials
-  Fix bugs or add tests
-  Implement new features
-  Experiment with new architectures
-  Add visualization tools
-  Optimize performance

See individual project READMEs for specific contribution opportunities.

##  Resources

### Papers & Documentation
- Check individual project folders for relevant papers
- See `references/` folder for literature

### Datasets
- Most projects use simulated data from FARGO3D, PHANTOM, MCFOST
- Real observations from ALMA, VLT, Kepler, TESS
- Contact mentors for dataset access

##  GSoC Information

**Organization:** [ML4SCI](https://ml4sci.org/)  
**Mentors:** Available via ml4-sci@cern.ch  
**Apply:** [Google Form](https://forms.gle/...)

### Participating Organizations
- University of Alabama
- Oxford University
- CERN

##  Contact

- **Questions:** ml4-sci@cern.ch
- **Issues:** [GitHub Issues](https://github.com/ML4SCI/EXXA/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ML4SCI/EXXA/discussions)

##  License

See individual project folders for licensing information.

##  Acknowledgments

This project is part of Google Summer of Code and is supported by ML4SCI, CERN, and participating universities.

---

**Last Updated:** March 2026  
**Active Projects:** 10  
**Contributors:** Multiple GSoC participants

