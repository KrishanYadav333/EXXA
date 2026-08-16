# EXXA — GSoC 2026 midterm snapshot

*Denoising Astronomical Observations of Protoplanetary Disks* — Krishan Yadav, mentored by
Jason Terry and Gaurav S., ML4Sci EXXA.

**This branch, `midterm_completed`, is a frozen submission snapshot.** It is not where
development happens — that is `midterm-prep` — and it is not kept up to date after
submission. It exists to be exactly what the midterm delivered:

- the five notebooks at the repo root, and their **executed copies with real outputs** in
  [`notebooks/`](notebooks/) — 05 (U-Net), 06 (DDPM), 07 (classical baselines), 08 (seed
  sweep), 09 (architecture comparison)
- [`MODELS.md`](MODELS.md) — every checkpoint referenced, with its numbers and where to get
  it (checkpoints themselves are never in git, see below)
- [`DENOISING_DIFFUSION/src/`](DENOISING_DIFFUSION/src/) — the library the notebooks import
- [`DENOISING_DIFFUSION/results/`](DENOISING_DIFFUSION/results/) — every run's log, figures
  and README, `RUNS.md` mapping numbers to runs, `PROGRESS.md` the chronological record
- the midterm write-up: [`DENOISING_DIFFUSION/BLOG_MIDTERM.md`](DENOISING_DIFFUSION/BLOG_MIDTERM.md)
  and its rendered HTML

Other ML4Sci EXXA project tracks that share this repository upstream (anomaly detection,
atmosphere characterisation, kinematics, and others) are **not included on this branch** —
they were never part of this deliverable. They are on `main` and `line-emission`.

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
