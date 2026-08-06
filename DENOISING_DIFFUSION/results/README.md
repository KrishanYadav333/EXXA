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
| `05-unet-line-emission/` | `baseline_…` `v5_…` `v6_…` `v7_…` `v9_…` `v12_…` `v15_…` `v16_…` `v17_…` `analysis_…` | all | Every completed version. **v12 is the published reference**; v16/v17 are the sweep-winner reproduction failures. Versions never downloaded were recovered from their push commits. |
| `08-seeds-and-augmentation/` | `v4_2026-08-02T0421_1ca611f` | 4 | Complete — 4 arms × 3 seeds = 12/12 runs. |
| `09-architecture-comparison/` | `v4_2026-08-02T0313_29bffd4` | 4 | Complete — 24/24 sweep, 3 retrains, 15 cube evaluations. Versions 6 and 7 resumed and reported these same numbers. |
| `_archive-continuum-era/` | `superseded_2026-07-25T1305_bf7a819` | — | Week 2/3 dust-continuum work at 64×64, superseded. |

| `06-unet-continuum/` | `v7_…`, `v9_…` | 7, 9 | The U-Net **continuum** experiment — a different notebook from the DDPM one. Recovered from git. |
| `07-classical-baselines/` | `v2_2026-07-31T1904_eb03589` | 2 | Completed, recovered from git. **Superseded** — tuned at 256 px, applied at 600 px. |

No folder for **06-ddpm-line-emission**: section 11 has never completed a run. (`06-unet-continuum/`
above is a different notebook — do not read it as DDPM results.)
See [`RUNS.md`](RUNS.md) for the full version history.

Files named `*__2.png` come from a notebook that ran the same section twice (05 v7/v9);
`unnamed_cell*.png` were displayed without a `savefig` call. Anything recovered from a push
commit is the notebook's **display** copy of a figure, not the `savefig` artifact — lower
fidelity, different bytes. Each `manifest.json` says which is which per file.

## Two things to know before quoting these numbers

**M2 in the 08 and 09 runs predates the noise-clip fix** (`bab16d0`, `MOMENT_VER=2`).
`bettermoments.collapse_second` was being fed an unmasked cube, so background noise read as
a dispersion of roughly the velocity axis' own RMS width — far above a real line width.
M0, M1 and PSNR are unaffected; **M2 needs re-scoring**. The saturated M2 column in
`09-architecture-comparison/.../architecture_moment_maps.png` is what exposed it.

**`_archive-continuum-era/` is not comparable to anything else here.** Different data,
different resolution, no tuning. It is kept for provenance and must not be quoted alongside
line-emission results.
