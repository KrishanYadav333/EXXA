# EXXA: Denoising Astronomical Observations of Protoplanetary Disks

A machine-learning pipeline for denoising synthetic ALMA observations of protoplanetary
disks. It is evaluated on the moment maps astronomers actually use, not on pixel metrics
alone.

**GSoC 2026 · ML4Sci · EXXA.** Krishan Yadav, mentored by Jason Terry and Gaurav S.

> **This branch is a frozen submission snapshot.** It holds the five notebooks that produced
> the midterm's numbers with their outputs intact, plus the library and results behind them.
> Development happens on `midterm-prep`; nothing here is updated after submission. Everything
> for this deliverable sits under this one directory, including this file.

---

## Result

A U-Net denoiser (reference checkpoint **V12**) improves all three moment maps on every one
of the 5 held-out cubes, with no cube made worse:

| | PSNR | SSIM | MSE | M0 (intensity) | M1 (velocity) | M2 (dispersion) |
|---|---|---|---|---|---|---|
| **V12** | 32.95 dB | 0.9857 | 0.000681 | **+69.8% ± 15.2%** | **+17.5% ± 7.8%** | **+20.1% ± 14.3%** |

Moment improvement is `100 × (1 − |denoised−clean| / |dirty−clean|)`, a mean absolute
difference over finite pixels, averaged across the 5 inference-only holdout cubes with the
standard deviation alongside.

A later 3-seed search reached **39.30 dB** using D4 augmentation. That result, the classical
baselines, the architecture comparison and the conditional DDPM all live in `notebooks/`,
with every number traced in `results/RUNS.md`.

---

## What makes this problem specific

**The noise is structured.** A "dirty" interferometric image carries correlated PSF and
sidelobe artifacts from sparse *uv*-coverage. It is not independent per-pixel noise, which is
what most denoising literature assumes.

**The data is 3D.** Line-emission cubes are `(201, 600, 600)`: 201 velocity channels. The
scientific product is the moment maps computed over the whole cube, not any single channel.

**Pixel metrics can mislead.** More than one result here improved PSNR while degrading the
moment maps. Beam conditioning was one. The conditional DDPM was another: it lands within
about 1 dB of the U-Net on PSNR while its M0 deficit is two orders of magnitude larger. That
is why evaluation is moment-map-first.

**Normalisation has to be invertible at inference.** Normalising the clean target by its own
statistics, which are not available at real inference, once produced M0 of −6402% while the
per-channel SSIM still looked healthy.

---

## Layout

```
DENOISING_DIFFUSION/
├── README.md                      # this file
├── MODELS.md                      # every checkpoint referenced, and where it lives
├── requirements.txt
│
├── notebooks/                     # the five notebooks as they ran, outputs intact
│   ├── 05-unet-line-emission.ipynb        (v21)
│   ├── 06-ddpm-line-emission.ipynb        (v13)
│   ├── 07-classical-baselines.ipynb       (v2)
│   ├── 08-seeds-and-augmentation.ipynb    (v2)
│   └── 09-architecture-comparison.ipynb   (v7)
│
├── src/
│   ├── data/
│   │   ├── cube_split.py          # RunID-grouped cube-level split (leakage-safe)
│   │   ├── channel_sampler.py     # Gaussian channel sampling (centred on ch 100)
│   │   ├── fits_cube_dataset.py   # FITSChannelDataset, continuum_of, beam_features_of
│   │   ├── stacked_pair.py        # (dirty, clean) -> (2,H,W) adapter for the DDPM
│   │   ├── dataset.py             # continuum-era loaders
│   │   ├── fits_loader.py, preprocessing.py, augmentation.py
│   ├── models/
│   │   ├── unet.py                # DenoisingUNet (optional beam conditioning)
│   │   ├── diffusion_unet.py      # conditional DDPM backbone + config
│   │   ├── noise_scheduler.py     # linear / cosine beta schedules
│   │   ├── autoencoder.py, vae.py # continuum-era baselines
│   ├── training/
│   │   ├── sweep.py               # early-stopping trainer + random sweep harness
│   │   ├── diffusion.py           # DDPM training, EMA, DDIM sampling
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── moment_maps.py         # bettermoments wrapper (M0/M1/M2)
│   │   └── artifacts.py           # per-channel overshoot / floor-leak / invented-structure
│   ├── utils/losses.py            # HybridLoss(alpha*MSE + beta*(1-SSIM)), VAELoss
│   └── baselines.py               # classical filters (Gaussian, median, Wiener)
│
├── tests/                         # run standalone or under pytest
└── results/
    ├── RUNS.md                    # every number, mapped to the run that produced it
    ├── PROGRESS.md                # chronological log of runs, arrivals and bugs
    └── <notebook>/<version>/      # figures and a README for the run behind each notebook
```

`results/` here holds only the run behind each notebook, one folder apiece. `RUNS.md` still
records the full history in text, back to the first line-emission runs in late June. The
intermediate runs' logs and figures, and the write-up prose, stay on `midterm-prep` rather
than being duplicated into a submission snapshot.

---

## Setup

```bash
git clone -b midterm_completed https://github.com/KrishanYadav333/EXXA.git
cd EXXA/DENOISING_DIFFUSION
pip install -r requirements.txt
```

Beyond the usual stack this needs `astropy` for FITS, `bettermoments` for the moment maps,
and `pytorch-msssim` for the SSIM loss term.

### Tests

```bash
python -m pytest tests/ -q
python tests/test_artifacts.py     # the modules also run standalone
```

They use synthetic arrays, so nothing has to be downloaded first.

### Reproducing a run

Neither the line-emission FITS cubes (14 cubes across 11 RunIDs, about 7.6 GB) nor any
trained checkpoint is in git. [MODELS.md](MODELS.md) says what each checkpoint is and where
to get it. Each notebook bootstraps itself by cloning the repository, so a run starts from a
blank Kaggle kernel:

1. New Kaggle notebook, GPU on, Internet on.
2. `Add Input` → the line-emission Dataset. The bootstrap finds it under `/kaggle/input/`.
3. Paste in a notebook from `notebooks/` and run top to bottom.

Everything here was computed on Kaggle's dual Tesla T4 with `DataParallel`.

---

## Conventions

These are project rules rather than preferences. Each one exists because breaking it cost a
result. The full list, with the incident behind every rule, is `RULES.md` on `midterm-prep`.
The ones worth knowing to read this branch:

- **Cube-level splitting is mandatory.** Hold out whole cubes grouped by RunID. Radiative
  transfer variants of one simulation are near-duplicates and leak if they straddle a split.
- **Holdout cubes are inference-only.** Training and validation never touch them.
- **Moment-map results are averaged over all 5 holdout cubes**, always with a spread.
- **Normalise using only statistics available at inference time.**
- **Every number traces to the run that produced it.** See `results/RUNS.md` and the README
  in each run folder.
- `seed=42` throughout, and `dirty` arrays are cast to `float32` on load.

---

## Status and history

`results/RUNS.md` maps every number on this branch to its run, and `results/PROGRESS.md` is
the chronological log. The short version:

| Phase | Outcome |
|---|---|
| Weeks 2–3 | Continuum baselines: classical filters, autoencoder, VAE, patch U-Net (best SSIM 0.76) |
| Week 4 | Patch DDPM underperformed classical filtering, prompting the mentor pivot to full-image line emission |
| Week 5 | Line-emission U-Net; found and fixed the normalisation bug (M0 −6402%, then recovered) |
| Weeks 5–6 | Continuum subtraction, suggested by the mentors, took M0 from −1672% to **+84.9%** |
| Weeks 6–7 | V7/V9 variance discovery led to the 5-cube evaluation protocol, and **V12** became the reference |
| Weeks 7–8 | Beam conditioning, a 12-run sweep reaching 37.11 dB (later traced to seed luck), DDPM retune |
| Weeks 8–9 | Seed-validated bands, classical baselines on line emission, architecture comparison, the DDPM rebuilt and diagnosed, and the midterm write-up |
