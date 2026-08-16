# Run index — every Kaggle notebook version

Kaggle does not expose its version number to the running kernel, but its GitHub integration
writes it into the push commit message (`Kaggle Notebook | <name> | Version N`). The table
below is reconstructed from those commits and from the outputs stored inside each one, so
every row is evidence from the notebook itself rather than recollection.

**`code` is the repo commit the run actually executed** (from the section-0b
`HEAD is now at ...` line) — that is what reproduces the numbers. **`push`** is the commit
Kaggle created afterwards to store the notebook.

A version having completed is not the same as its artifacts being here: several finished
and wrote their CSVs to the kernel Output tab, which was never downloaded.

---

## 05 — U-Net line emission (`05_unet_line_emission`)

Full lineage, recovered from every branch (`line-emission`, `midterm-prep`, `week-4`).
`PSNR` is the holdout figure the version itself printed.

| Ver | Date (UTC) | code | push | PSNR | Holdout M0 | Artifacts |
|----:|---|---|---|---|---|---|
| — | 2026-06-24 19:42 | `a4ac1fe` | `27fa388` | 26.46 | — | [`baseline_.../`](05-unet-line-emission/baseline_2026-06-24T1942_a4ac1fe/) — first run, 30 epochs |
| #5 | 2026-06-26 17:15 | `f548947` | `d307a29` | 28.16 | — | [`v5_.../`](05-unet-line-emission/v5_2026-06-26T1715_f548947/) |
| 6 | 2026-06-27 16:00 | `5ed8fc6` | `01aaf88` | 31.73 | — | [`v6_.../`](05-unet-line-emission/v6_2026-06-27T1600_5ed8fc6/) |
| 7 | 2026-07-02 13:59 | `662fef5` | `95d9034` | 31.02 / 33.18 | — | [`v7_.../`](05-unet-line-emission/v7_2026-07-02T1359_662fef5/) · also in [`for_jason/`](for_jason/) |
| 9 | 2026-07-02 18:39 | `7fa5fce` | `d0adb14` | 32.31 / 33.04 | — | [`v9_.../`](05-unet-line-emission/v9_2026-07-02T1839_7fa5fce/) · also in [`for_jason/`](for_jason/) |
| **12** | 2026-07-10 19:28 | `c61d112` | `82abd23` | **32.95** | **+69.8% ±15.2** | [`v12_.../`](05-unet-line-emission/v12_2026-07-10T1928_c61d112/) · also in [`for_jason/`](for_jason/) |
| 15 | 2026-07-26 04:42 | `09bfcb9` | `0307cc9` | 33.70 / 34.94 | +59.8% ±33.0 | [`v15_.../`](05-unet-line-emission/v15_2026-07-26T0442_09bfcb9/) — recovered from git |
| 16 | 2026-07-30 03:34 | `ef0a37b` | `227d2fc` | **30.28** | +64.4% ±14.6 (−5.4 pt) | [`v16_.../`](05-unet-line-emission/v16_2026-07-30T0334_ef0a37b/) |
| 17 | 2026-08-02 04:13 | `227d2fc` | `7305455` | **29.92** | **+48.2% ±12.6 (−21.6 pt)** | [`v17_.../`](05-unet-line-emission/v17_2026-08-02T0413_227d2fc/) — recovered from git |
| — | 2026-07-30 00:48 | `2edd875` | — | — | — | [`analysis_.../`](05-unet-line-emission/analysis_2026-07-30T0048_2edd875/) |
| **18** | 2026-08-11 | stale | `b24c84d` | **35.94** | **+81.2% ±12.6** | [`v18_.../`](05-unet-line-emission/v18_2026-08-11_b24c84d/) |
| 19 | 2026-08-14 | `c899750` | — | — | — | [`v19_.../`](05-unet-line-emission/v19_2026-08-14_crashed/) — **CRASHED** at 4.0 h, section 6 never ran, no moment scores |
| **20** | 2026-08-14 | `c28b860` | — | 39.30 (aug) | **+29.2% ±7.2** (aug) | [`v20_.../`](05-unet-line-emission/v20_2026-08-14_c28b860/) — **mask + clip**, 15 checkpoints, **zero training** |
| 21 | 2026-08-17 | `6f5c798` | — | unchanged | unchanged | [`v21_.../`](05-unet-line-emission/v21_2026-08-17_6f5c798/) — sections 7-8 only; artifact CSV finally saved |

**v18 is the best 05 run to date, and the first with every moment positive on every cube.**
M0 +81.2% ±12.6, M1 +31.5% ±9.2, M2 +19.9% ±18.8, 5/5 cubes positive on all three, against
V12's +69.8 / +17.5 / +20.1. PSNR 35.94, +2.99 dB over V12. The sweep-time 37.109 dB
reproduces at −1.166 dB, consistent with the seed-49 explanation.

**Correction — v18 did NOT use the masked metric.** An earlier version of this entry said
its positive M2 values were the signal mask removing empty sky. That is wrong. Notebook 05
bootstraps from **`line-emission`**, not `midterm-prep`, and v18's log records
`HEAD is now at 227d2fc`. At that commit `src/evaluation/moment_maps.py` contains no
`moment_improvement`, no `signal_mask` and no `clip_sigma` — zero occurrences of each. So
v18 ran the **original whole-map, unclipped metric**, and why its M2 came out positive where
earlier runs' did not is unexplained, not attributed to masking.

What this does *not* undermine: v18 vs V12 is still like-for-like, because V12's published
figures are on that same original metric. The +81.2 / +31.5 / +19.9 against +69.8 / +17.5 /
+20.1 stands as a comparison.

What it does mean: **v18's numbers are not comparable to anything scored on the masked
metric** — notebook 08's arms, or any future 05 run once its bootstrap is moved to
`midterm-prep`. Two different measurements, and the branch is what decides which one a run
gets.

Caveat on provenance: the Kaggle notebook was also a **stale copy predating `1d3022b`** — 23
cells, no `SEEDS`, no `CONFIGS`, no V12-checkpoint restore. So v18 is a single seed on one
configuration, not the 3-seed comparison the current notebook runs, and its "vs V12" column
compares against V12's *published* numbers rather than a re-scored V12 checkpoint. The push
`b24c84d` also reverted `c1f21e1` and `718ca3e` in the repo; restored in `b8cc16d`.

### How these were attributed

**v16 — certain.** `moment_map_holdout_summary_sweepwinner.csv` reads MEAN 64.36 / 13.78 /
23.08, matching v16's printed +64.4% / +13.8% / +23.1%, and
`artifact_diagnostics_sweepwinner.csv` reproduces v16's overshoot statistics exactly
(mean 0.929, median 0.907, p90 1.162, max 1.424, 19% above 1.10). v17 printed
1.220 / 1.198 / 1.423 / 1.924 / 69%. Pixel-matching the figures had *preferred v17* — the
numbers overruled it, which is why the CSVs decided this and the images did not.

**v12 — partly certain.** `moment_map_holdout_summary.png` is written by v12 and by no
other downloaded version. (An earlier note also claimed `moment_maps_holdout.png` was
v12-unique; recovering v7 and v9 showed they write it too, so that file rests on download
clustering like the rest.)

**v7 / v9 / v12 continuum + denoised figures — moderate.** Three
`moment_maps_continuum_comparison` files were downloaded and exactly three versions write
that filename (v7, v9, v12); two `moment_maps_denoised_comparison` files and exactly two
versions write it (v7, v9). The set is therefore certain; only which file belongs to which
version rests on browser download order (`name`, `(1)`, `(2)`) plus the timestamp clustering,
which independently puts the third continuum file in the same batch as v12's two unique
figures. Pixel comparison agreed but too weakly to count (candidate diffs 44.8 / 45.0 / 45.9).

**v17 — recovered from git, not downloaded.** Its artifacts were never pulled off the
Output tab, but the executed notebook was pushed to `7305455` with its outputs still
embedded, so all four figures and the full stdout came straight out of the commit
(`src/evaluation/recover_version.py`). The moment table is `.RECONSTRUCTED.csv`: `imp_*`
are v17's own printed per-cube figures and `dirty_*` are copied from v16's committed CSV,
which they must equal — they depend only on the cubes, and v17's printed table matches them
digit for digit. `artifact_diagnostics_sweepwinner.csv` is **not** recoverable, being
per-channel with only its summary printed. The same recovery filled v16's two missing
figures; v16's downloaded files were left untouched, since a recovered figure is the
notebook's display copy and not the `savefig` artifact.

**v7 / v9 / v12 / v15 and 07 v2 — recovered from git.** Same route as v17. Recovery also
surfaced two things the first pass had lost. v7 and v9 each contain **two runs in one
notebook** — a line-emission section (cells 17–27) and an appended continuum section
(cells 46–56) — which save the same filenames; keying on basename alone kept only the later
one, so the earlier figure is now suffixed `__2`. And several cells display a figure without
any `savefig`, which were being dropped entirely; they are kept as `unnamed_cell<N>_<i>.png`.
07 assigns its path to a variable (`fig_path = os.path.join(...)`) before calling
`savefig(fig_path)`, which a literal-only pattern missed — the tool now resolves that.

Figures are named from the `savefig` call in their own cell. Naming by output order had
mislabelled three of v17's four — it writes `sweepwinner_loss.png`, not
`unet_line_emission_loss.png`, and writes no `moment_map_holdout_summary.png` at all, which
also confirms that file belongs to v12.

**`unet_line_emission_loss.png` — unattributed.** v7, v9 and v12 all write it, one copy was
downloaded, and pixel matching cannot separate them (29.1 / 29.2 / 29.6). Parked in
[`_unattributed/`](05-unet-line-emission/_unattributed/) rather than guessed.

### v20 — the first U-Net scores on the DDPM's metric

v20 trained nothing. It restored 12 checkpoints from 08 and 3 from v19's Output, scored
`winner_patch` from its stored weights (v19 died before writing its metric row), and spent
the whole session in section 6. That is why it exists: **every U-Net number before it was on
the raw or clip-only metric, and the DDPM was on mask + clip**, so no comparison between the
two families was ever like for like.

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner + D4 aug | 39.30 | **+29.2 ± 7.2** | **+74.0 ± 2.0** | **+55.0 ± 13.9** |
| winner + patience 10 | 39.27 | +33.5 ± 9.6 | +70.7 ± 6.7 | +31.8 ± 11.1 |
| sweep winner (4 seeds, incl. 49) | 37.52 | +11.4 ± 27.4 | +55.6 ± 9.0 | +6.0 ± 35.1 |
| v12_cfg | 37.60 | −4.4 ± 33.9 | +58.0 ± 20.0 | +15.5 ± 31.2 |
| winner + beam (1 seed) | 38.71 | ~~−95.7~~ **RETRACTED** | ~~+14.1~~ | ~~−27.3~~ |

**The beam arm's moment scores are withdrawn.** `denoise_cube` never passed the beam
vector, and `UNet.forward` ignores `beam=None` silently (it asserts only in the opposite
direction), so this row was measured with the model's conditioning branch dead. The PSNR
in the same row is sound, because section 4 does pass the beam, which is exactly why the
row looked like a dramatic pixel-vs-science split. On a toy beam model the vector shifts
the output by 0.025 mean absolute, so the two paths were scoring different functions.
Fixed in the notebook; the arm needs re-scoring before any of these three numbers is
quoted again. Nothing else in this table is affected -- the other arms are `beam_dim=0`.

| winner, 64px patches (1 seed) | 33.96 | −40.8 | +43.9 | +18.5 |

**These are a third metric generation and are not comparable to v2's or v4's numbers.** Use
them only against each other and against the DDPM, which shares this metric.

Two results are new here. **Beam conditioning is refuted a third time**: second-best PSNR in
the table, worst M0 of any arm by a factor of two. **64px patches fail**, which is the
cheapest possible statement of why this project abandoned patches for full images: a 64-pixel
crop cannot contain a disk, so there is no rotation field to recover.

Seed 49 is also finally on record for `sweep_winner`, at PSNR 36.16 against the sweep's
published 37.109 on the same seed. The two are not comparable (the sweep used `N_SAMPLES=50`
against this notebook's 150), so the reproduction question stays where 08 left it: the
sweep's number sits inside this config's 3-seed band, and that is as far as the evidence goes.

`unet_loss.png` was dropped: byte-identical (sha `c2ae97f9`) to the continuum-era copy
already archived, so it was a stray download rather than a 05 artifact.

**v18 is the first version to beat V12 on the scientific metric**, and it does so on M0 and
M1 together (+81.2 / +31.5 vs +69.8 / +17.5), with M2 level. Its own printed verdict is
*"sweep winner supersedes V12 as reference checkpoint"*.

Before it, no version had: v16 and v17 both printed *"NOT a clean improvement — V12 stays
the reference"*, with M0 falling from +69.8% to +64.4% then +48.2%.

**All of v16, v17 and v18 ran the same unmasked metric**, because notebook 05 bootstraps
from `line-emission` and none of the moment fixes are on that branch — v18's log records
`HEAD is now at 227d2fc`, where `moment_improvement`, `signal_mask` and `clip_sigma` do not
exist. So this whole column is internally consistent and comparable to V12's published
figures, and comparable to **nothing** scored on the masked metric. An earlier version of
this paragraph credited the mask for v18's positive M2; that was wrong, since v18 had no
mask.

Re-scoring V12's own checkpoint under the masked metric is what the present notebook adds,
and it has not yet been run.

### The "7 dB reproduction failure" was a seed difference, not a failure

v16 and v17 were written up as failing to reproduce the sweep's 37.109 dB, scoring 30.28 and
29.92. They did not fail. `run_sweep` trains run *i* at **`seed + i`**, so the winning row —
run index 7 — was trained at **seed 49**. The retrain in notebook 05 calls
`train_unet(..., seed=SEED)` with **`SEED = 42`**. Same hyperparameters, different draw.

Everything else was verified identical: dataset (350 train / 100 val in both), split
(seed 42, `n_holdout=3`), early stopping (min 20 / max 60 / patience 5), and the PSNR
calculation. The trajectories agree early — the sweep's val loss at epoch 11 was 0.0038 and
v17's *best ever* was 0.0035, also at epoch 11 — then the sweep kept improving to 0.0014 by
epoch 33 while v17 stalled and early-stopped at epoch 20.

**Consequence for every other sweep row.** Each of the twelve runs used a different seed
(42–53), so the sweep confounds configuration with seed, and taking the maximum over them
selects partly on luck. The winner's "+4.16 dB over V12" is an order statistic, not a
measured effect. Notebook 08 settles it directly: the same configuration over seeds 42/43/44
gives **37.97 ± 0.90 dB**, a band that contains 37.109 comfortably — and V12's own arm gives
37.60 ± 1.00. 08's own verdict on the pair is *"INDISTINGUISHABLE from seed noise"*.

`run_sweep` now writes a `seed` column so a row is reproducible from its own record, and
migrates older CSVs on open, backfilling `seed` as `base_seed + run`.

**V12's headline moment figures are not in any committed CSV.** `+69.8% ±15.2 / +17.5% ±7.8
/ +20.1% ±14.3` is quoted in `readme.md`, `MIDTERM_REPORT.md`, `context.md` and hardcoded as
a `V12 = {...}` dict in notebooks 07/08/09, but the only moment CSV in the entire git history
is the beam variant (+59.83). What was downloaded for v12 is figures, not the table.
**Correction:** an earlier note here claimed 08 v4 superseded it with "M0 +71.9% ±10.3".
That figure is 08's **M1**, not M0 — the table it was read from had been mis-transcribed.
08 v4's v12 arm actually scores **M0 +27.7% ±17.6**, well *below* the published +69.8%, so
it does not supersede that number. The published V12 M0 remains unsourced by any CSV.

Versions 16 and 17 also report overshoot mean 0.929 and 1.220 respectively for the same
notebook, each alongside "0% invented structure" — which the floor-relative fix later showed
to be vacuous, the background mask having selected zero pixels. Neither is quotable.

## 06-unet-continuum — U-Net continuum experiment (a DIFFERENT notebook)

`06-unet-line-emission-continuum.ipynb` is **not** the DDPM notebook. It is the U-Net
continuum experiment, and its outputs were synced into a directory called 06 — the whole
source of the long-standing confusion that "folder 06 contains 05's results". Both runs are
now filed under their own notebook name.

| Ver | Date (UTC) | push | Artifacts |
|----:|---|---|---|
| 7 | 2026-07-02 14:18 | `d4ff643` | [`v7_.../`](06-unet-continuum/v7_2026-07-02T1418_d4ff643/) |
| 9 | 2026-07-02 18:44 | `5933292` | [`v9_.../`](06-unet-continuum/v9_2026-07-02T1844_5933292/) — the ablation |

Neither records a `HEAD is now at` line, so the code commit is unknown; the push commit is
the only anchor.

## 06-ddpm-line-emission — DDPM

| Ver | Date (UTC) | code | Outcome | Artifacts |
|----:|---|---|---|---|
| 11 | 2026-08-11 14:55 | `41df52d` | sweep + 60-epoch train complete; improved holdout crashed at the CSV write | [`v11_.../`](06-ddpm-line-emission/v11_2026-08-11T1455_41df52d/) |
| **13** | 2026-08-12 04:50 | `19efd47` | **complete, zero errors** — first DDPM moment maps ever produced | [`v13_.../`](06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/) |

### v13 — the DDPM matches the U-Net on PSNR and fails on the science

PSNR **38.180**, SSIM **0.9933** — within ~1 dB of the best U-Net. The moment maps, on the
same signal-masked metric 08 uses:

| | M0 | M1 | M2 |
|---|---|---|---|
| DDPM (n=5 cubes) | **−56.1% ± 152.2** | +15.0% ± 87.4 | +10.9% ± 81.7 |
| U-Net V12 (published) | +69.8% ± 15.2 | +17.5% ± 7.8 | +20.1% ± 14.3 |

The mean hides the real structure, which is bimodal — three cubes work, two collapse:

| cube | M0 | M1 | M2 |
|---|---|---|---|
| run_0002_00560_rt_00 | −84.8% | −126.5% | −130.8% |
| run_0002_00560_rt_01 | +41.9% | +61.8% | +47.8% |
| run_0002_00560_rt_04 | +18.5% | +73.5% | +51.2% |
| run_0025_01000_rt_04 | **−310.2%** | −13.2% | +15.5% |
| run_0026_00005_rt_04 | +54.1% | +79.2% | +70.7% |

**The failure is a pedestal, not lost structure.** `ddpm_moment_maps.png` shows the denoised
M0 recovering the disk and the denoised M1 recovering the rotation pattern — but the sky,
which is empty in the truth, carries a constant non-zero floor, and M2 is saturated across
the entire field. The smoke test records the cause directly:

    sampler OK: (8,1,256,256) -> (8,1,256,256), range [0.348, 0.701]

The model emits a narrow band around 0.5 rather than spanning [0,1], so inverting the
per-channel dirty-scale normalisation puts the whole field at mid-scale. `K_AVG=4` averages
four reverse draws, which shrinks the spread further — the notebook pays the full cost of
learning a distribution and then throws the distribution away.

This is a **conditional-mean/variance problem, not a capacity problem**, and it is testable:
score the same checkpoint at `K_AVG=1`, and rescale the denoised channel to the dirty
channel's own mean and standard deviation before inverting the normalisation. Neither
requires retraining.

Not obtained: 6d's baseline pass. No checkpoint was attached (`no .pth/.pth.tar anywhere
under /kaggle/input`), so it correctly skipped and trained from scratch;
`moment_map_holdout_baseline_ddpm.csv` is listed as NOT FOUND by the collector for that
reason. All five checkpoints persisted as they trained (RULES.md #1), so v13 is resumable.

The sweep reproduced v11's ranking at a second seed — C_v_cosine 37.195 / D 36.248 /
patch 35.787 / A_eps_linear **18.014** — a 19.18 dB spread against v11's 19.92. The
objective result is now measured twice.

The version number is not in the log — Kaggle does not expose it to the kernel and this
notebook is not GitHub-linked, so no push commit records it either. v11 is from the author.

**This run overturns the standing explanation for the DDPM gap.** Every earlier DDPM run
used eps-prediction on a linear schedule, and the deficit against the U-Net was attributed
to the data regime — 6 training disks being too few to learn a distribution. The 12-epoch
objective sweep says otherwise, on identical data:

| objective | pred | schedule | minSNR | PSNR | SSIM |
|---|---|---|---|---:|---:|
| C_v_cosine | v | cosine | 0.0 | **37.822** | 0.9892 |
| D_v_cosine_minsnr5 | v | cosine | 5.0 | 36.326 | 0.9919 |
| patch_view | v | cosine | 0.0 | 35.966 | **0.9928** |
| A_eps_linear | eps | linear | 0.0 | 17.907 | 0.5040 |

`A_eps_linear` is the configuration every previous run used. The 19.9 dB spread is the
objective, not the data. The full 60-epoch run at the winner reaches **PSNR 38.363 / SSIM
0.9930**, and 38.154 dB over 300 validation channels — within ~1 dB of the best U-Net
(`winner_aug`, 39.30). Caveats: 12-epoch sweep runs at one seed, scored on 8 validation
batches; 08 measured ~1 dB of pure seed spread, so only the A-vs-C gap is beyond doubt.

`patch_view` ran at all only because `295fa38` relaxed the U-Net's exact-resolution assert;
before that it raised `expected 256x256, got (64, 64)` on its first batch.

Not obtained: moment maps. Section 13 sampled every held-out cube and then died at the
write with `ValueError: dict contains fields not in fieldnames: 'imp_M0_all', ...` — fixed
in `2fe7f61`. The baseline pass was also skipped, because the run predates the
checkpoint-discovery fix and printed `no checkpoint found`, so it trained from scratch to
epoch 41 rather than resuming the epoch-99 checkpoint.

Earlier attempts stopped at 88/201 channels twice (DataParallel deadlock, fixed in
`7e49e62`), then died just after sampling (host RAM during the moment collapse, `aa2a41d`).
Kaggle v8 is the source of the epoch-99 / `best_val 13.4706` checkpoint.

Everything previously filed under a "06" heading belongs to `06-unet-continuum` above.
`d704a7c` ("archive Kaggle Version 6 with outputs") is not a separate run: its five figure
hashes and its log hash are identical to v6's push commit `01aaf88`.

## 07 — Classical baselines

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 2 | 2026-07-31 19:04 | `eb03589` | `7dbd17b` | completed. Headline M0 +69.8% (V12) vs +11.7% (best classical) → +58.1 pp. | [`v2_.../`](07-classical-baselines/v2_2026-07-31T1904_eb03589/) — recovered from git |

Superseded: this run tuned filters at 256 px and applied them at 600 px, and its optimum sat
on the edge of the σ grid. Both handicapped the classical side, so the gap above is an
overstatement. Needs a ~10 min CPU re-run at native resolution.

## 08 — Seeds and augmentation (`08-seeds-and-augmentationa369c56f22`)

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 2 | 2026-07-31 | `eb03589` | — | complete — 4 arms × 3 seeds; **raw metric**, broken artifact detector | [`v2_.../`](08-seeds-and-augmentation/v2_2026-07-31_eb03589/) |
| 4 | 2026-08-02 04:21 | `1ca611f` | `ba328ea` | **complete** — 4 arms × 3 seeds = 12/12; **clip metric**, working detector | [`v4_.../`](08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/) |

### v2 and v4 are the same 12 checkpoints scored twice — take a different half from each

Both versions reuse the identical 12 checkpoints, so their PSNR/SSIM/MSE rows are
byte-identical. What differs is the evaluation code, and the difference is large enough that
neither version is usable whole:

```
                     v2 (eb03589)                    v4 (1ca611f)
moment metric        raw: no clip, no mask           3-sigma clip, no mask
artifact detector    BROKEN -- 0.0% for all arms     working

config           v2: M0      M1      M2         v4: M0      M1      M2
v12                 +82.0   +28.2   +26.4          +27.7   +71.9   -10.5
winner              +85.5   +29.1   +27.0          +18.1   +71.3    -2.7
winner_aug          +87.5   +25.4   +26.7          +38.7   +75.4   +40.3
winner_p10          +85.8   +35.9   +24.1          +11.6   +71.1   -28.3
```

**Take v2's moments.** They are on the same raw metric as V12's published +69.8 / +17.5 /
+20.1, so they are the only seed-validated numbers in this project directly comparable to the
reference checkpoint. On that metric `winner_aug` scores **M0 +87.5% ±4.2** — the highest M0
anywhere here, at the *lowest* variance, over 3 seeds. Every arm is positive on M0 on all five
cubes; nothing goes negative under this metric.

**But v2's own promotion check says none of it is established.** The notebook compares each
arm to `v12` on a matched schedule and reports the gap against the cube-to-cube spread:

```
winner_aug vs v12       M0  +5.6 pp  (spread  9.5 pp)  -> within noise
                        M1  -2.9 pp  (spread 31.7 pp)  -> within noise
                        M2  +0.3 pp  (spread 25.5 pp)  -> within noise
winner vs v12           M0  +3.5 pp  (spread 12.6 pp)  -> within noise
                        M1  +0.9 pp  (spread 22.7 pp)  -> within noise
                        M2  +0.6 pp  (spread 23.6 pp)  -> within noise
```

So `winner_aug` has the **highest M0 and the tightest spread**, which makes it the one to
use — but "augmentation beats V12 on the science" is **not** demonstrated at n=3. The PSNR
gain is on firmer ground and still not settled: `winner_aug − winner` is +1.322 dB against a
combined spread of 1.005, which the notebook grades *suggestive, not established*.

**v2 also settles the V16 shortfall, and the answer is early stopping rather than seed
luck.** The sweep recorded 37.11 dB for this configuration; V16's retrain got 30.28 dB. On a
matched schedule:

```
patience  5  ->  mean 27.3 epochs,  PSNR 37.974 ±0.896   113% of the shortfall closed
patience 10  ->  mean 48.3 epochs,  PSNR 39.273 ±0.484   132% of the shortfall closed
```

Both clear the notebook's own >50% threshold for "early-stopping artifact", and `winner_p10`
overshoots the sweep figure outright. Patience 10 buys +21 epochs, +1.298 dB, and *halves*
the seed spread (0.896 → 0.484). This does not contradict the seed-49 explanation — 37.11
also sits inside `winner`'s 3-seed band — but the dominant term is the early stop, not the
draw.

**Take v4's artifact diagnostics.** v2 reports 0.0% invented structure for all four arms,
which is RULES.md #8: the background mask selected zero pixels, so the detector could not
fire. v2's overshoot figures (0.89–0.93, all *below* 1.0) are the same artefact. v4's
22.3–39.0% and its 7× low-SNR concentration are the real measurement.

Do not mix a v2 moment number with a v4 moment number, and do not quote v2's artifact row at
all.

`seed_repeats.csv` and `seed_spread.png` are v4's (their PSNRs match its log exactly — every
run reuses the same 12 checkpoints, so training numbers never change). The moment and
artifact CSVs previously filed here did **not** match v4's log and came from an earlier
session; they are in [`superseded_pre-artifacts-fix/`](08-seeds-and-augmentation/superseded_pre-artifacts-fix/).
v4's own values are in its `run_log.txt` and in the tables below, pending a download of the
CSVs themselves.

| arm | PSNR | M0 | M1 | M2 (pre-fix) |
|---|---|---|---|---|
| v12 | 37.60 ± 1.00 | +27.7 ± 17.6 | +71.9 ± 10.3 | −10.5 ± 39.7 |
| winner | 37.97 ± 0.90 | +18.1 ± 45.9 | +71.3 ± 16.3 | −2.7 ± 63.1 |
| winner_aug | **39.30 ± 0.46** | **+38.7 ± 47.6** | **+75.4 ± 13.9** | +40.3 ± 30.7 |
| winner_p10 | 39.27 ± 0.48 | +11.6 ± 58.7 | +71.1 ± 18.8 | −28.3 ± 83.6 |

**Correction.** An earlier version of this table listed 71.9 / 71.3 / 75.4 / 71.1 under an
"M0" heading. Those are **M1**; the M0 column was omitted and the numbers shifted left. The
real M0 values are far lower and every arm sits *below* the published V12 M0 of +69.8%.

### Invented structure — the first real measurement

The artifact diagnostics finally return evidence rather than an artefact of an empty mask
(they ran at `1ca611f`, after the floor-relative fix `04b3f2c`):

| arm | channels with an invented blob | blobs/channel | peak overshoot |
|---|---|---|---|
| v12 | 39.0% | 0.897 | 2.741 |
| winner | 37.7% | 0.940 | 2.349 |
| **winner_aug** | **29.7%** | **0.743** | 1.736 |
| winner_p10 | 22.3% | 0.857 | 1.889 |

Two results worth reporting. **Invented structure is concentrated in faint channels** —
1.580 blobs/channel below the median SNR against 0.213 above it, a ~7x difference at a split
of SNR 3.9. That answers the open question of whether hallucination is a low-SNR-specific
failure: it is. And **augmentation reduces it**, 37.7% → 29.7% of channels (−8.0 pp), which
is the hypothesis this notebook was built to test.

The earlier "0% invented structure, 0.00 blobs/channel" for every arm was the empty-mask
artefact, and those CSVs are now in `superseded_pre-artifacts-fix/`.

## 09 — Architecture comparison (`09-architecture-comparison2`)

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 4 | 2026-08-02 03:13 | `29bffd4` | `cb51254` | **complete** — 24/24 sweep, 3 retrains, 15 cube evals. **Origin of the numbers.** | [`v4_2026-08-02T0313_29bffd4/`](09-architecture-comparison/v4_2026-08-02T0313_29bffd4/) |
| 6 | 2026-08-02 04:04 | `bab16d0` | `1ca611f` | resumed; reused v4's rows unchanged | [`v6_.../`](09-architecture-comparison/v6_2026-08-02T0404_bab16d0/) |
| 7 | 2026-08-02 17:21 | `ee491fc` | `ecaa540` | resumed; reused v4's rows unchanged | [`v7_.../`](09-architecture-comparison/v7_2026-08-02T1721_ee491fc/) |
| — | 2026-08-03 | `d07f8f4` | — | input not attached → swept from scratch; U-Net 8 runs in 299 min, died in autoencoder run 3 | none (partial) |

Versions 6 and 7 print identical numbers to v4 because section 6 skipped everything already
scored. That is the resume working, but it also meant the M2 noise-clip fix pulled in at v6
and v7 never actually ran — which is why `MOMENT_VER` was added in `ee491fc`.

---

## The metric changed — which runs are comparable

`bab16d0` added a 3-sigma noise clip before the collapse. It was introduced to fix M2, but
it changes **M1 and M2 substantially and M0 modestly**, because zeroing sub-3-sigma channels
alters the integrated intensity and the intensity-weighted velocity as well. On a synthetic
cube with a known answer:

    unclipped   M0 +69.8%   M1  +8.7%   M2 +11.9%
    clipped     M0 +64.7%   M1 +31.3%   M2 +24.3%

So runs sit on one of two metrics, and **numbers may only be compared within a group**:

| Metric | Runs |
|---|---|
| **clipped** (current) | 08 v4 (`1ca611f`) |
| unclipped (old) | 05 all versions, 07 v2, 09 v4 / v6 / v7 |

`1ca611f` is *after* `bab16d0`, so 08 v4 is the only run already on the current metric — the
opposite of what an earlier note here claimed. Its M0 +27.7 / M1 +71.9 cannot be set beside
05's +69.8 / +17.5 or 09's +77.0 / +22.2; the gap is the metric, not the models.

`d1f0f66` (strip-wise collapse) changes memory and speed, not values: M0 is bit-identical
and bright pixels agree to 1e-10, so it does not split the groups further.

**To restore comparability, 05 and 09 must be re-scored.** Neither needs retraining —
checkpoints exist for both, and 09's `MOMENT_VER` already forces its cube rows to be redone.

## M2 is genuinely weak, and that survives the fix

On the corrected metric 08 v4 still reports M2 −10.5 / −2.7 / +40.3 / −28.3 with standard
deviations up to 83.6 — three of four arms make dispersion *worse* than the dirty cube. The
clip made M2 measure line width rather than band width; it did not make the result good.
That is consistent with 09, where M2 was the U-Net's weakest moment (+8.5 ± 18.1) and the
only one where it lost to both baselines.

Each 08 arm is a single seed evaluated over 5 cubes, so the spread is cube-to-cube. Treat
M2 as an open weakness of the method, not as a metric artefact.
