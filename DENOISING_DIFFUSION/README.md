# EXXA — Denoising Astronomical Observations of Protoplanetary Disks

Machine-learning pipeline for denoising synthetic ALMA observations of protoplanetary disks,
evaluated on the moment maps astronomers actually use rather than pixel metrics alone.

**GSoC 2026 · ML4Sci · EXXA** — Krishan Yadav, mentored by Jason Terry and Gaurav S.

> **This branch, `midterm_completed`, is a frozen submission snapshot** — the five notebooks
> that produced the midterm's numbers, real outputs intact, plus the library, tests and
> documentation behind them. It is not where development happens (that is `midterm-prep`)
> and it is not kept up to date after submission. Everything for this deliverable lives
> under this one directory, including this file.

---

## Result

A U-Net denoiser (reference checkpoint **V12**) improves all three moment maps on **every one of
5 held-out cubes**, with no cube made worse:

| | PSNR | SSIM | MSE | M0 (intensity) | M1 (velocity) | M2 (dispersion) |
|---|---|---|---|---|---|---|
| **V12** | 32.95 dB | 0.9857 | 0.000681 | **+69.8% ± 15.2%** | **+17.5% ± 7.8%** | **+20.1% ± 14.3%** |

Moment improvement is `100 × (1 − |denoised−clean| / |dirty−clean|)` (mean absolute difference
over finite pixels), averaged over 5 inference-only holdout cubes ± standard deviation.

A 3-seed hyperparameter search since reached **39.30 dB** with D4 augmentation; its full
moment-map validation, the classical-baseline comparison, the architecture comparison and the
conditional-DDPM result are all in the midterm write-up:
**[BLOG_MIDTERM.md](BLOG_MIDTERM.md)** (also rendered as [`blog_midterm.html`](blog_midterm.html)
and [`blog_midterm_blogger.html`](blog_midterm_blogger.html)), with the fuller
**[MIDTERM_REPORT.md](MIDTERM_REPORT.md)** covering the same ground in more depth.

---

## What makes this problem specific

- **The noise is structured.** A "dirty" interferometric image carries correlated PSF/sidelobe
  artifacts from sparse *uv*-coverage, not independent per-pixel noise.
- **The data is 3D.** Line-emission cubes are `(201, 600, 600)` — 201 velocity channels — and
  the scientific product is the moment maps computed over the *whole* cube, not any single
  channel.
- **Pixel metrics can mislead.** Several results in this project improved PSNR while degrading
  the moment maps — a beam-conditioning A/B, and the conditional DDPM's ~1 dB PSNR gap
  against an M0 deficit two orders of magnitude larger. Evaluation is moment-map-first.
- **Normalisation must be invertible at inference.** Normalising the clean target by its own
  statistics — unavailable at real inference — once produced M0 −6402% while per-channel SSIM
  looked healthy.

---

## Layout

```
DENOISING_DIFFUSION/
├── README.md                      # this file
├── MODELS.md                      # every checkpoint referenced, and where it lives
├── BLOG_MIDTERM.md                # the midterm write-up (+ rendered, self-contained HTML)
├── MIDTERM_REPORT.md              # the fuller results write-up
├── ARCHITECTURE.md                # architecture documentation
├── BEAM_AND_SWEEP.md              # beam conditioning + sweep design notes
├── context.md, progress.md        # project state and the week-by-week run log
│
├── notebooks/                     # the five notebooks AS THEY RAN, outputs intact
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
├── tests/                         # test modules, run standalone or via pytest
└── results/
    ├── RUNS.md                    # every number, mapped to the run that produced it
    ├── PROGRESS.md                # chronological log: runs, arrivals, bugs
    └── <notebook>/<version>/      # the run behind each notebooks/ entry above:
                                    # log, figures, and a README explaining the run
```

`results/` on this branch holds only the run behind each notebook in `notebooks/` — one
folder per notebook. `RUNS.md` and `progress.md` still record the full history in text, back
to week 1; the intermediate runs' logs and figures are on `midterm-prep`, not duplicated here.

---

## Setup

```bash
git clone -b midterm_completed https://github.com/KrishanYadav333/EXXA.git
cd EXXA/DENOISING_DIFFUSION
pip install -r requirements.txt
```

Key dependencies beyond the usual stack: `astropy` (FITS), `bettermoments` (moment maps),
`pytorch-msssim` (SSIM loss term).

### Tests

```bash
python -m pytest tests/ -q
python tests/test_artifacts.py     # individual modules also run standalone
```

Tests use synthetic arrays and need no data download.

### Reproducing a run

The line-emission FITS cubes (14 cubes, 11 RunIDs, ~7.6 GB) and every trained checkpoint are
**not in git** — see [MODELS.md](MODELS.md) for what each checkpoint is and where it lives.
Each notebook bootstraps itself by cloning the repository, so a run is reproducible from a
blank Kaggle kernel:

1. New Kaggle notebook, GPU on, Internet on.
2. `Add Input` → the line-emission Dataset (the bootstrap locates it under `/kaggle/input/`).
3. Paste in a notebook from `notebooks/` and run top to bottom.

Compute used throughout: Kaggle dual Tesla T4 with `DataParallel`.

---

## Conventions

Project rules, not preferences — each exists because breaking it cost a result. The full list
with the incident behind each is `RULES.md` on `midterm-prep` (not on this snapshot branch).
The ones that matter for reading this branch:

- **Cube-level splitting is mandatory.** Hold out whole cubes grouped by RunID; RT variants of
  one simulation are near-duplicates and leak if split.
- **Holdout cubes are inference-only** — never touched by training or validation.
- **Moment-map results are averaged over all 5 holdout cubes with a standard deviation.**
- **Normalise using only statistics available at inference time.**
- **Numbers are traced to the run that produced them** — see `results/RUNS.md`, and the
  attribution note at the top of every notebook's run folder.
- `seed=42` throughout; cast `dirty` arrays to `float32` on load.

---

## Status and history

`progress.md` (newest-first run log) and `context.md` (project state as of the midterm) track
the project week by week from community bonding through submission. Condensed:

| Phase | Outcome |
|---|---|
| Weeks 2–3 | Continuum baselines: classical filters, autoencoder, VAE, patch U-Net (best SSIM 0.76) |
| Week 4 | Patch DDPM underperformed classical filtering → **mentor pivot** to full-image line emission |
| Week 5 | Line-emission U-Net; found and fixed the normalisation bug (M0 −6402% → recovered) |
| Weeks 5–6 | Continuum subtraction (mentor suggestion) → M0 −1672% → **+84.9%** |
| Weeks 6–7 | V7/V9 variance discovery → 5-cube evaluation protocol; **V12** reference established |
| Weeks 7–8 | Beam conditioning, 12-run sweep (37.11 dB, later traced to seed luck), DDPM retune |
| Weeks 8–9 | Seed-validated seed bands, classical baselines on line emission, architecture comparison, conditional DDPM rebuilt and diagnosed, midterm write-up |
