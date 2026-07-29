# EXXA — Denoising Astronomical Observations of Protoplanetary Disks

Machine-learning pipeline for denoising synthetic ALMA observations of protoplanetary disks,
evaluated on the moment maps astronomers actually use rather than pixel metrics alone.

**GSoC 2026 · ML4Sci · EXXA** — Krishan Yadav, mentored by Sergei Gleyzer and Jason Terry.
Active work is on the [`line-emission`](https://github.com/KrishanYadav333/EXXA/tree/line-emission)
branch.

---

## Result

A U-Net denoiser (reference checkpoint **V12**) improves all three moment maps on **every one of
5 held-out cubes**, with no cube made worse:

| | PSNR | SSIM | MSE | M0 (intensity) | M1 (velocity) | M2 (dispersion) |
|---|---|---|---|---|---|---|
| **V12** | 32.95 dB | 0.9857 | 0.000681 | **+69.8% ± 15.2%** | **+17.5% ± 7.8%** | **+20.1% ± 14.3%** |

Moment improvement is `100 × (1 − |denoised−clean| / |dirty−clean|)` (mean absolute difference
over finite pixels), averaged over 5 inference-only holdout cubes ± standard deviation.

A 12-run hyperparameter sweep has since reached **37.11 dB** (+4.16 dB); its moment-map
validation is in progress. Full write-up: **[MIDTERM_REPORT.md](MIDTERM_REPORT.md)**.

---

## What makes this problem specific

- **The noise is structured.** A "dirty" interferometric image carries correlated PSF/sidelobe
  artifacts from sparse *uv*-coverage, not independent per-pixel noise.
- **The data is 3D.** Line-emission cubes are `(201, 600, 600)` — 201 velocity channels — and
  the scientific product is the moment maps computed over the *whole* cube, not any single
  channel.
- **Pixel metrics can mislead.** Two results in this project improved PSNR while degrading the
  moment maps (see the beam-conditioning A/B in the report). Evaluation is therefore
  moment-map-first.
- **Normalisation must be invertible at inference.** Normalising the clean target by its own
  statistics — unavailable at real inference — once produced M0 −6402% while per-channel SSIM
  looked healthy. See report §3.

---

## Layout

```
DENOISING_DIFFUSION/
├── MIDTERM_REPORT.md              # full results write-up
├── ARCHITECTURE.md                # architecture documentation
├── BEAM_AND_SWEEP.md              # beam conditioning + sweep design notes
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
├── tests/                         # 9 test modules
├── results/                       # checkpoints, CSVs, figures
└── notebooks/                     # earlier + continuum-era notebooks, Kaggle snapshots
```

Training notebooks live at the **repository root** (they are the Kaggle-linked entry points):

| Notebook | Purpose |
|---|---|
| `05-unet-line-emission.ipynb` | U-Net: sweep-winner validation, moment maps, artifact diagnostics |
| `06-ddpm-line-emission.ipynb` | Conditional DDPM comparison baseline |

---

## Setup

```bash
git clone -b line-emission https://github.com/KrishanYadav333/EXXA.git
cd EXXA/DENOISING_DIFFUSION
pip install -r ../requirements.txt
```

Key dependencies beyond the usual stack: `astropy` (FITS), `bettermoments` (moment maps),
`pytorch-msssim` (SSIM loss term).

### Tests

```bash
python -m pytest tests/ -q
python tests/test_artifacts.py     # individual modules also run standalone
```

Tests use synthetic arrays and need no data download.

### Training

The line-emission FITS cubes (14 cubes, 11 RunIDs, ~7.6 GB) are **not in git** — they are hosted
as a Kaggle Dataset. Both root notebooks bootstrap themselves by cloning this repository, so a
run is reproducible from a blank Kaggle kernel:

1. New Kaggle notebook, GPU on, Internet on.
2. `Add Input` → the line-emission Dataset (the bootstrap locates it under `/kaggle/input/`).
3. Paste in `05-unet-line-emission.ipynb` (or `06-…`) and run top to bottom.

Compute used throughout: Kaggle dual Tesla T4 with `DataParallel`.

---

## Conventions

These are project rules, not preferences — each one exists because breaking it cost a result:

- **Cube-level splitting is mandatory.** Hold out whole cubes grouped by RunID; RT variants of
  one simulation are near-duplicates and leak if split.
- **Holdout cubes are inference-only** — never touched by training or validation.
- **Moment-map results are averaged over all 5 holdout cubes with a standard deviation.** Two
  runs of one config once differed by 16 points of M2 on a single cube (report §5).
- **Normalise using only statistics available at inference time** (report §3).
- **Nothing is "done" without real execution and artifacts on disk.** Implemented-but-not-run is
  recorded as pending.
- `seed=42` throughout; cast `dirty` arrays to `float32` on load.

---

## Status and history

`progress.md` (newest-first run log) and `context.md` (current project state) track the
project in detail. Condensed timeline:

| Phase | Outcome |
|---|---|
| Weeks 2–3 | Continuum baselines: classical filters, autoencoder, VAE, patch U-Net (best SSIM 0.76) |
| Week 4 | Patch DDPM underperformed classical filtering → **mentor pivot** to full-image line emission |
| Week 5 | Line-emission U-Net; found and fixed the normalisation bug (M0 −6402% → recovered) |
| Weeks 5–6 | Continuum subtraction (mentor suggestion) → M0 −1672% → **+84.9%** |
| Weeks 6–7 | V7/V9 variance discovery → 5-cube evaluation protocol; **V12** reference established |
| Weeks 7–8 | Beam conditioning (useful negative result), 12-run sweep (37.11 dB), DDPM retune |
