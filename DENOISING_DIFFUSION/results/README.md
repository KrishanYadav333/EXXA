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
| `08-seeds-and-augmentation/` | `v4_…`, `superseded_pre-artifacts-fix/` | 4 | Complete — 4 arms × 3 seeds = 12/12. Moment/artifact CSVs from an earlier session were separated out; v4's own numbers are in its `run_log.txt`. |
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

## Before quoting any of these numbers

**Two different metrics are in play.** `bab16d0` added a 3-sigma noise clip before the
moment collapse, which shifts **M1 and M2 substantially and M0 modestly** — not M2 alone, as
an earlier version of this note claimed. Measured on a synthetic cube:

    unclipped   M0 +69.8%   M1  +8.7%   M2 +11.9%
    clipped     M0 +64.7%   M1 +31.3%   M2 +24.3%

Only **08 v4** ran on the clipped (current) metric. Every 05, 07 and 09 run predates it.
Do not compare moment numbers across that line — see [`RUNS.md`](RUNS.md). Re-scoring 05 and
09 needs no retraining.

**`_archive-continuum-era/` is not comparable to anything here.** Different data, different
resolution, no tuning. Kept for provenance only.
