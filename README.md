# EXXA

ML4Sci EXXA — machine learning for exoplanet and protoplanetary-disk science.

**GSoC 2026 work in this fork:** *Denoising Astronomical Observations of Protoplanetary Disks*
— Krishan Yadav, mentored by Sergei Gleyzer and Jason Terry.

Active development is on the
[`line-emission`](https://github.com/KrishanYadav333/EXXA/tree/line-emission) branch, in
[`DENOISING_DIFFUSION/`](DENOISING_DIFFUSION/).

| | |
|---|---|
| **Project README** | [DENOISING_DIFFUSION/readme.md](DENOISING_DIFFUSION/readme.md) |
| **Results write-up** | [DENOISING_DIFFUSION/MIDTERM_REPORT.md](DENOISING_DIFFUSION/MIDTERM_REPORT.md) |
| **Architecture** | [DENOISING_DIFFUSION/ARCHITECTURE.md](DENOISING_DIFFUSION/ARCHITECTURE.md) |

**Current result:** a U-Net denoiser improves all three moment maps (M0 +69.8% ± 15.2%,
M1 +17.5% ± 7.8%, M2 +20.1% ± 14.3%) across all 5 inference-only holdout cubes of synthetic
ALMA line-emission data, with no cube made worse.

Training notebooks (Kaggle-linked entry points):

- [`05-unet-line-emission.ipynb`](05-unet-line-emission.ipynb) — U-Net denoiser
- [`06-ddpm-line-emission.ipynb`](06-ddpm-line-emission.ipynb) — conditional DDPM comparison

Other top-level directories in this repository are upstream ML4Sci EXXA project tracks
(anomaly detection, atmosphere characterisation, kinematics, and others), unmodified by this
project.
