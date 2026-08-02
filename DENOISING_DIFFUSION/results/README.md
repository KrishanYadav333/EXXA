# results/

One folder per notebook, one subfolder per run:

    <notebook-id>/v<kaggle version>_<UTC timestamp>_<git sha>/

The `v<N>_` prefix appears when the Kaggle version is known. Kaggle does not expose it to
the running kernel — it only shows up afterwards in the commit its GitHub integration
pushes — so **[`RUNS.md`](RUNS.md) is the index that maps every version to its run**,
including the versions that completed but whose artifacts were never downloaded.

Each run folder carries a `manifest.json` recording the notebook, the commit, the UTC time,
and a byte count plus sha256 prefix for every file. Run folders never overwrite each other,
so two runs of the same notebook can be compared directly.

Written automatically by the last cell of each notebook
(`src/evaluation/collect_outputs.py`). On Kaggle the bundle lands in
`/kaggle/working/outputs/`; download it and commit it here.

## What is here

| Folder | Run | Kaggle ver | Status |
|---|---|---|---|
| `05-unet-line-emission/` | `v7_…`, `v9_…`, `v12_…`, `v16_…`, `analysis_…` | 7, 9, 12, 16 | Five run folders. **v12 is the published reference**; v16 is the sweep-winner reproduction failure. v15 and v17 completed but were never downloaded. |
| `08-seeds-and-augmentation/` | `v4_2026-08-02T0421_1ca611f` | 4 | Complete — 4 arms × 3 seeds = 12/12 runs. |
| `09-architecture-comparison/` | `v4_2026-08-02T0313_29bffd4` | 4 | Complete — 24/24 sweep, 3 retrains, 15 cube evaluations. Versions 6 and 7 resumed and reported these same numbers. |
| `_archive-continuum-era/` | `superseded_2026-07-25T1305_bf7a819` | — | Week 2/3 dust-continuum work at 64×64, superseded. |

No folder for **06** (section 11 has never completed) or **07** (version 2 completed, but
its artifacts were never downloaded and it needs a native-resolution re-run anyway).
See [`RUNS.md`](RUNS.md) for the full version history.

## Two things to know before quoting these numbers

**M2 in the 08 and 09 runs predates the noise-clip fix** (`bab16d0`, `MOMENT_VER=2`).
`bettermoments.collapse_second` was being fed an unmasked cube, so background noise read as
a dispersion of roughly the velocity axis' own RMS width — far above a real line width.
M0, M1 and PSNR are unaffected; **M2 needs re-scoring**. The saturated M2 column in
`09-architecture-comparison/.../architecture_moment_maps.png` is what exposed it.

**`_archive-continuum-era/` is not comparable to anything else here.** Different data,
different resolution, no tuning. It is kept for provenance and must not be quoted alongside
line-emission results.
