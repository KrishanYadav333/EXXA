# results/

One folder per notebook, one subfolder per run:

    <notebook-id>/<UTC timestamp>_<git sha>/

Each run folder carries a `manifest.json` recording the notebook, the commit, the UTC time,
and a byte count plus sha256 prefix for every file. Run folders never overwrite each other,
so two runs of the same notebook can be compared directly.

Written automatically by the last cell of each notebook
(`src/evaluation/collect_outputs.py`). On Kaggle the bundle lands in
`/kaggle/working/outputs/`; download it and commit it here.

## What is here

| Folder | Run | Status |
|---|---|---|
| `05-unet-line-emission/` | `2026-07-30T0048_2edd875` | 12-run U-Net sweep, correlation analysis, beam A/B. The notebook's own V16/V17 artifacts were never downloaded and are **not** here. |
| `08-seeds-and-augmentation/` | `2026-08-02T0333_ba328ea` | Complete — 4 arms × 3 seeds = 12/12 runs. |
| `09-architecture-comparison/` | `2026-08-02T0338_29bffd4` | Complete — 24/24 sweep runs, 3 retrains, 15 cube evaluations. |
| `_archive-continuum-era/` | `2026-07-25T1305_bf7a819` | Week 2/3 dust-continuum work at 64×64, superseded. |

Not present: **06** (DDPM section 11 has never completed a run) and **07** (classical
baselines have not been re-run at native resolution since the tuning fix).

## Two things to know before quoting these numbers

**M2 in the 08 and 09 runs predates the noise-clip fix** (`bab16d0`, `MOMENT_VER=2`).
`bettermoments.collapse_second` was being fed an unmasked cube, so background noise read as
a dispersion of roughly the velocity axis' own RMS width — far above a real line width.
M0, M1 and PSNR are unaffected; **M2 needs re-scoring**. The saturated M2 column in
`09-architecture-comparison/.../architecture_moment_maps.png` is what exposed it.

**`_archive-continuum-era/` is not comparable to anything else here.** Different data,
different resolution, no tuning. It is kept for provenance and must not be quoted alongside
line-emission results.
