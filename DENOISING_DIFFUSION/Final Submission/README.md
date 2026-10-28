# Denoising Astronomical Observations of Protoplanetary Disks

**Final submission, Google Summer of Code 2026 · ML4Sci · EXXA**
Student: Krishan Yadav. Mentors: Jason Terry and Sergei Gleyzer.

A denoiser for ALMA line-emission cubes of protoplanetary disks that preserves the small velocity perturbations astronomers look for (the gravitational-instability "wiggle", planet kinks). It is judged on moment maps and on the velocity residual after subtracting the rotation curve, not on pixel metrics alone.

## Method in one paragraph

A blind 3D Wiener filter, estimated from the cube itself, is the baseline. A U-Net learns a correction that is zero at initialisation, so the untrained model equals the Wiener filter. Four networks are averaged; a gate blends the output back toward the Wiener result on cubes that need little denoising; the Wiener filter is recomputed from the first pass; and the cube is resampled so that the beam spans about 10 pixels.

## Metrics

All scores lie between 1 (identical to the noise-free truth) and 0 (no relation).

- **Wiggle score** (`rc_r` in the code): the azimuthally averaged rotation curve is subtracted from the velocity map (M1), and the Pearson correlation of the residual map with the truth's residual is computed.
- **PV r:** the correlation between the denoised cube's position-velocity (PV) curve along the disk's minor axis and the truth's (1 px slice or 5 px slit, quadratic trend removed). The curve's standard deviation is the **PV magnitude** in m/s, always reported next to the correlation, since a correlation ignores scale.
- **Peak SNR:** the peak of the noise-free cube divided by the noise standard deviation.
- **Twin test:** the same disk with the spiral switched off. Its true minor-axis curve is flat, so the PV magnitude the denoiser produces on it measures the spurious wiggle it introduces (the detection limit is set by this).
- **M0, M1, M2:** integrated intensity, mean velocity and line width of the cube, and PSNR (in dB) of the reconstruction; improvements are quoted over the noisy cube under a signal mask and noise clip.

## Results

"Wiggle score" is the correlation between the velocity residual of the denoised cube (rotation curve subtracted) and the noise-free truth, from 0 to 1. The Wiener filter is the classical baseline the model starts from.

| test (never used for training or tuning) | model | Wiener filter |
|---|---|---|
| held-out line-emission cubes, 600 px: PSNR | 51.7 dB | 30.6 dB |
| held-out line-emission cubes: M0 / M1 / M2 improvement over the noisy cube | 62 / 89 / 91 % | |
| one-time final test (Speedie disk, 4 noise realisations): wiggle score at peak SNR 12 / 24 | 0.59 ± 0.10 / 0.79 ± 0.06 | 0.18 / 0.42 |
| 8 fresh analytic GI disks, beam to be undone: wiggle score | 0.79 | 0.47 |
| 8 fresh hydrodynamic simulations (Young et al.): wiggle score | 0.56 | 0.33 |
| 20 real disks (15 exoALMA, 5 DSHARP): position angle within 0 to 3.9 degrees of published | 19 of 20 | |
| injected GI disk in MWC 758 noise: wiggle score at SNR 12 / 24 | 0.84 / 0.93 | 0.64 / 0.83 |

**Limits** (evidence in `results/final/ANALYSIS.md`): nothing is recovered below peak SNR of about 6 or for wiggles below about 15 to 20 m/s; the model draws about 18 to 27 m/s of spurious wiggle on disks that have none; amplitudes are underestimated; flux within one beam of the star is over-estimated; nearly face-on disks and compact beam-sized features are not recovered; no per-pixel uncertainty is produced by default.

## Repository layout

```text
Final Submission/
├── README.md  MODELS.md             overview, the trained models and their availability
├── requirements.txt                dependencies (numpy >= 2 is required by bettermoments)
├── src/                            library: data loading, analytic GI disk model, U-Net, training, evaluation
├── experiments/                    training and evaluation (train_unified.py), real-cube pipeline (alma_unified.py),
│                                   PV-curve measurement (pv_curve.py), dataset generators, figure and table builders
├── tests/                          unit and end-to-end tests (plain scripts, one per module)
├── tools/                          markdown link checker and small utilities
├── docs/                           ALMA_CHECKS.md (the seven real-data checks), research.md (papers read and decisions)
└── results/
    ├── RUNS.md                     index: which run produced which number
    ├── final/                      RESULTS.md, ANALYSIS.md, RECIPES.md, detection_limit.md, figs/ (the figures)
    ├── alma/                       reports of the real-disk runs and the injection tests (JSON and Markdown; no images)
    └── unified/                    result files of the evaluation runs and the four improvement attempts; training runs have a README only
```

## Setup and tests

```bash
cd "DENOISING_DIFFUSION/Final Submission"
pip install -r requirements.txt
PYTHONPATH=. python3 tests/test_infer.py          # tests are plain scripts, one per module
PYTHONPATH=. python3 tests/test_alma_unified_smoke.py   # the real command line on a synthetic cube, CPU, about 30 s
python3 tools/check_md_links.py                   # relative links in the documents resolve
```

`numpy >= 2` is required. On numpy 1.x the four tests that compute moment maps fail with an `AttributeError` in `bettermoments`, which is the environment and not the code. `tests/test_unet.py` trains a network and takes minutes on CPU.

## Using the denoiser on a cube

The input is a CLEANed ALMA line cube in FITS, any channel width or pixel size. A raw uv-plane dirty image is not supported. It needs PyTorch; in practice a GPU.

```bash
python experiments/alma_unified.py --cube MyDisk_13CO.fits --out results/alma/mydisk \
    --ckpt unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth \
    --crop 512 --target-beam-px 10 \
    --incl 35 --pa 120 --dist 150      # optional: enables the rotation-curve and PV checks
```

Add `--spec-interp 0.1` for channels coarser than 0.1 km/s (for example DSHARP at 0.35 km/s) and `--boot 4` for an empirical M1 uncertainty map. The script finds the line-free channels and the noise level itself, decides on continuum subtraction, denoises, and returns the cube on its original grid with moment maps and a `report.json` (noise reduction, flux retained, structure generated in empty channels, and, given the geometry, the rotation-curve residual and PV curve).

**Read before using the output for science:**

- There is no per-pixel uncertainty by default. The 15 to 20 m/s figure is a detection limit from injection tests, not an error map. `--boot N` gives an empirical map that holds the noise term only.
- Do not take error bars from a fit to the denoised cube: its noise is correlated and no longer Gaussian, so fitting tools report uncertainties that are too small. Fit the raw cube for uncertainties.
- Compact, beam-sized features (a planet-like kink) are not recovered. The absence of a kink in a denoised cube is not evidence of no kink.
- One noise level per cube is assumed. Primary-beam-corrected cubes and channels near atmospheric lines break this; crop to the disk (`--crop`, `--auto-fov`).
- Continuum: `--cont-sub auto` subtracts only when the line-free planes on both sides show the same structure. Do not force a second subtraction on exoALMA cubes.
- Flux from the denoised cube should be used only beyond 3 beams from the star.

## Data and models

The training data (line-emission cubes and self-gravitating simulations) was provided by the project mentors and is not in this repository. The analytic GI disk model is in `src/data/analytic_gi.py`, and `experiments/make_analytic_gi_cubes.py` and `experiments/make_sph_cubes.py` generate the analytic and rendered test disks. The trained weights are described in [MODELS.md](MODELS.md) and are released only with the mentors' permission. Real observations (exoALMA, DSHARP) are public and were used for evaluation only; no model was trained on them.

## Reproducing the numbers

Every number reported here and in `results/final/` is traced to an archived run in [results/RUNS.md](results/RUNS.md). The training and evaluation runs were executed as Kaggle kernels; the scripts they call are in `experiments/`, and the reports they produced are under `results/`.

## Acknowledgements

Simulation data: Jason Terry and the ML4Sci EXXA mentors. External data and models used for evaluation: Young et al. (2026) GI simulations, Speedie et al. (2024), the exoALMA large program (Teague et al. 2025), DSHARP (Andrews et al. 2018).
