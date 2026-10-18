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
| 22 | 2026-08-17 | `6f5c798` | — | unchanged | unchanged | [`v21_.../`](05-unet-line-emission/v22_2026-08-17_6f5c798/) — sections 7-8 only; artifact CSV finally saved |
| — | — | — | — | — | — | **v23-v29 not entered here** — docs debt, not a gap in the runs themselves (each is archived under `results/05-unet-line-emission/` or noted in PROGRESS.md; `v_pending_2026-09-17_1832c54` still needs its real version number). Close before Block 3. |
| 30 | 2026-09-19 | `22b01ae` | — | 40.32 (`winner_starlet_ft`) | not re-scored this version | [`v30_.../`](05-unet-line-emission/v30_2026-09-19_22b01ae/) — loss-sweep arms first clean finish: `winner_{mae,wavelet,starlet}_ft` 39.78/40.13/40.32 |
| 33 | 2026-09-20 | `fe52be5` | — | 40.18 (`winner_res480` ft, old recipe) | — | [`v33_.../`](05-unet-line-emission/v33_2026-09-20_crashed/) — **CRASHED**, host RAM exhaustion, root cause found this run (DataLoader fork-storm) |
| 34 | 2026-09-20 | `c7083fb` | — | 40.14 (`winner_gradient_ft`) | — | [`v34_.../`](05-unet-line-emission/v34_2026-09-20_crashed/) — **CRASHED**, same cause as v33, `persistent_workers` fix landed after |
| 36 | 2026-09-24 | `1832c54` (stale — see README) | — | 39.93/40.18/40.14 (`winner_{mae,wavelet,starlet}_ft`, reconfirmed) | — | [`v36_.../`](05-unet-line-emission/v36_2026-09-24_stale-cells/) — **CRASHED**, `KeyError: 'M0'`, Kaggle ran cells frozen 3 fix-generations behind HEAD; not a code bug, see README |
| 38 | 2026-09-24 | `9856f17` | — | 40.18 (`winner_aug_res480`), 37.36 (`winner_aug_res320`) | — | [`v38_.../`](05-unet-line-emission/v38_2026-09-24_crashed/) — **CRASHED**, host RAM at epoch 19 of the third arm; first winner_aug-recipe res arms, PSNR not comparable across resolutions; RAM leak NOT fixed by `persistent_workers`, see PROGRESS.md |
| 40 | 2026-09-24 | `90f2390` | — | 40.41 (`winner_wavelet_p10_ft`) | best M0 +44.5 (`winner_starlet_p10_ft`) in v42's table | [`v40_.../`](05-unet-line-emission/v40_2026-09-24_90f2390/) — clean, 3 new arms: `mae`/`wavelet`/`starlet`_`p10_ft` |
| 41 | 2026-09-25 | `d923d28` | — | 40.12 (`winner_gradient_p10_ft`) | — | [`v41_.../`](05-unet-line-emission/v41_2026-09-25_d923d28/) — clean, 3 new arms: `gradient_p10_ft`, `mae_beam_ft`, `wavelet_beam_ft` |
| 42 | 2026-09-25 | `ad600d2` | — | 42.59 (`winner_k1`) | **moment table for 25 arms**; res320/res480 rows INVALID (scored at 256 px) | [`v42_.../`](05-unet-line-emission/v42_2026-09-25_ad600d2/) — clean, 3 new arms: `starlet_beam_ft`, `gradient_beam_ft`, `winner_k1`. Only `winner_k2` left, plus two control arms added afterwards |
| 43 | 2026-09-25 | `6fb324d` | — | 42.81 (`winner_k2`) | `winner_k2` M0 +20.2 / M1 +65.9 / M2 +36.8; res320/res480 rows still INVALID | [`v43_.../`](05-unet-line-emission/v43_2026-09-25_6fb324d/) — clean, 1 new arm `winner_k2`, ran pre-fix cells (`ad600d2`); Kaggle's push clobbered the fixes, restored by `tools/reapply_05_fixes.py` |
| 44 | 2026-09-25 | `1b1ee7f` | — | 40.25 (`winner_hybrid_p10_ft`) | control arms: hybrid_ft +39.7/+80.9/+69.9, hybrid_p10_ft +40.4/+76.2/+68.4 (M0/M1/M2); res320 scored at 320 | [`v44_.../`](05-unet-line-emission/v44_2026-09-25_failed_oom/) — **FAILED (CUDA OOM)** scoring `winner_aug_res480` at 480 px, after both control arms trained and persisted; fix `bs` scales with size |
| 47 | 2026-09-25 | `499030b` | — | 40.25 (`winner_hybrid_p10_ft`) | res320 -17.8/+48.0/-3.9, res480 +39.7/+73.5/+67.3 (M0/M1/M2, own size); controls as v44 | [`v47_.../`](05-unet-line-emission/v47_2026-09-25_773b0ce/) — clean, no training, scoring only; first valid 320/480 rows. v45, v46 not in git |

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

### v26 — the band-limit idea is refuted; spectral context is a pixel win only

First run with the CONFIGS-derived moment table and the out-of-band check, so the first that
shows the spectral arms and the first that can answer whether a band-limit loss is worth
building. See [`v26_.../`](05-unet-line-emission/v26_2026-08-21_df38336/).

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner + 1 spectral neighbour | 41.61 | +17.7 | +63.4 | +38.0 |
| winner + 2 spectral neighbours | 41.96 | +17.0 | +72.4 | +33.0 |

Pooling all three runs of each arm at seed 42 (this, v25, and the first): k1 at
41.26 ± 0.92 dB / 21.0 ± 9.3 / 63.4 ± 4.9 / 33.0 ± 16.3, k2 at 41.81 ± 0.55 / 16.6 ± 2.1 /
67.6 ± 6.1 / 23.8 ± 12.3. Against `winner_aug`'s 39.30 / 29.2 / 74.0 / 55.0, that is **+2.4 dB
of PSNR bought at the cost of M0 and M2**. Three runs each, so not seed noise.

**Out-of-band error: median excess 0.5x**, 1.1% of residual power above the beam cutoff. The
model's error is if anything smoother than the truth, so the invented structure is beam-scale
rather than sharp and no band constraint can see it. With Phase 0's `A = I`, DDRM, VIREO-lite's
consistency term and the band-limit variant are all refuted by measurement.

### v25 — spectral context, and a reproducibility problem

Two new arms, `winner_k1` and `winner_k2`: the sampled channel plus k neighbours along
velocity, aimed at M1 and M2, which lag because every channel was denoised in isolation.
Archived at [`v25_.../`](05-unet-line-emission/v25_2026-08-21_889cd44/).

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner_k1 (1 seed) | 40.219 | 13.9 ± 21.1 | 58.5 ± 23.1 | 14.8 ± 27.9 |
| winner_k2 (1 seed) | 42.269 | 18.5 ± 62.2 | 69.6 ± 20.5 | 28.6 ± 34.5 |

Spreads are **across the 5 holdout cubes at one seed**, not across seeds, so they are not
comparable to the multi-seed rows in the v20 table above.

**These arms were run twice and did not reproduce.** k1 gave 41.948 dB and M2 +46.2% the
first time, 40.219 dB and M2 +14.8% here, at the same seed with the same code. Three
fixed-seed runs span 1.73 dB, standard deviation 0.865, against `winner_aug`'s across-SEED
spread of 0.455. GPU nondeterminism, not a bug. See `PROGRESS.md` for which numbers this
touches; the short version is that every single-seed row in this file, including
`winner_beam` and `winner_patch`, has no error bar and should not be read at this resolution.

**What holds:** the PSNR gain over the un-augmented baseline, 40.2 to 41.9 against 37.5,
across all three runs. **What does not:** any moment claim from a single spectral run.

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
| winner + beam (1 seed) | 38.71 | **+9.6** | **+63.2** | **+20.9** |

**The beam arm's moment scores were re-measured in Kaggle Version 24 and the row above is
the corrected one.** The original −95.7 / +14.1 / −27.3 came from scoring a beam-conditioned
model without its beam vector: `denoise_cube` never passed one and `UNet.forward` ignores
`beam=None` silently. Section 4 did pass it, which is why the PSNR was sound and the row
looked like a dramatic pixel-vs-science split. Corrected, M0 moves **+105 points** and M2
changes sign. Beam conditioning is therefore **not refuted**, it is positive on all three
moments and simply below the winner at one seed. Four of five cubes are strongly positive;
`run_0025_01000_rt_04` at −133.8% carries the whole deficit.
See [`v24_.../`](05-unet-line-emission/v24_2026-08-17_beamfix/).

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

## 13-checkpoint-loss-sweep — MAE/wavelet/starlet/gradient sweep on the 4 non-05 checkpoints

`kin_gamma0`, `sg_k3_fresh` (31/7-channel spectral U-Nets, `LOSS_REGISTRY` losses) and
`ddpm_seed42`, `ddrm_prior` (`loss_type='l1'`/`'l2'` + optional wavelet/starlet/gradient
aux on `x0_hat`, `src/training/diffusion.py`). Every arm both fine-tuned and fresh.
Resumes across sessions (`_import_prior_nb13`, `MAX_NEW_ARMS_PER_SESSION=2`).

| Ver | Date (UTC) | push | status | Artifacts |
|----:|---|---|---|---|
| 5 | 2026-09-18 | `27b6f7a` | kin 1/8, first clean finish | [`v5_.../`](13-checkpoint-loss-sweep/v5_2026-09-19_6b3e6ea/) |
| 6-12 | 2026-09-19/20 | — | kin 8/8, sg 8/8, ddpm ramping up (RAM-leak sessions, capped) | not yet archived — see `v_pending`/PROGRESS.md |
| 17 | 2026-09-24 | `aca35c5` | **kin 8/8, sg 8/8, ddpm 8/8, ddrm 4/4 — all 28 arms complete** | not yet archived; per-arm PSNR/`best_val_loss` live in the 4 CSVs inside this version's Kaggle Output, not pulled into git yet |

**Read from cell output, not re-derived:** `ddrm_l2_fresh` (last arm to finish)
`best_val_loss` 24.35863, 50 epochs, no early stop. The other 27 arms' numbers are not
in this session's log capture (resumed arms print `SKIPPED, already done`, not their
value) — pull `nb13_*.csv` from the Output before quoting anything else from v17.
Same caveat as 05: v17's cells matched `aca35c5`, which predates today's lr-comment and
kin-epoch-budget fixes (behaviour-neutral — see PROGRESS.md 2026-09-24) — reapplied on
top in the merge that landed this row, not lost.

## Learned correction on 3D Wiener, unified model, fine-tunes (private script kernels, midterm-prep code, not GitHub-linked)

Cells/code ship in the private dataset `exxa-wiggle-code` (sha256 manifest), not via GitHub, so RULES.md #2 does not apply; the commit that holds this code is still pending (PROGRESS.md 2026-10-09). Score = `wiggle_scorecard` rc_r plus
`checkpoint_eval.score` (nb16 definitions). One run per row, no seeds.

| Kernel | Ver | Date | what | outcome | Artifacts |
|---|---:|---|---|---|---|
| `wiggle-correction` | 1 | 2026-10-09 | smoke, fold 9074, 80 steps | COMPLETE 2.7 min; rc_r 9074 0.16 to 0.26 | (log in PROGRESS.md) |
| `wiggle-correction` | 2 | 2026-10-09 | leave-one-hydro-disk-out, 5 folds x 9000 steps | COMPLETE 605 min; 9074 0.13/0.16/**0.44**, 9025 0.13/0.21/**0.44** (dirty/Wiener/corrected) | [`v2_.../`](wiggle-correction/v2_2026-10-09_loo/) |
| `unified-smoke` | 1 | 2026-10-09 | plumbing | COMPLETE 14.8 min; loss NaN (9019 float16 overflow), fixed | none |
| `unified-wiener` | 1 | 2026-10-09 | one model, all data kinds, 5 sizes, 12000 steps | COMPLETE 197 min; line 600 px PSNR 46.3, M1 85, M2 83, rc_r 0.70; 4 scoring failures | [`unified-wiener_v1_.../`](unified/unified-wiener_v1_2026-10-09/) |
| `unified-ft-kin` | 1 | 2026-10-10 | fine-tune kin_gamma0_{mae,starlet}_ft, mixed data | COMPLETE ~220 min; see PROGRESS.md 2026-10-10 | [`ft-kin_v1_.../`](unified/ft-kin_v1_2026-10-10/) |
| `unified-ft-single` | 1 | 2026-10-10 | fine-tune res480, p10_600, sg_k3_fresh | COMPLETE ~250 min; res480/p10 refuted (line 600 px M0 -99/-140 %), sg_k3_fresh line PSNR 40.9, rc_r 0.55; 6 scoring failures (old scorer) | [`ft-single_v1_.../`](unified/ft-single_v1_2026-10-10/) |
| `speedie-prep` | 1 | 2026-10-10 | fetch + prepare Speedie GI disk (CPU) | cancelled: sequential download needed ~20 h, superseded by `speedie-prep-fast` | none |
| `speedie-prep-fast` | 1 | 2026-10-10 | same, 24 parallel range requests (the first kernel needed ~20 h) | COMPLETE 58 min; noise rms 2.12 / 1.26 mJy/beam, lag-1 channel corr. +0.64; FITS headers carry wrong geometry | [`speedie/`](speedie/) |
| `unified-wiener-k15` | 1 | 2026-10-10 05:44 IST | round 2: Wiener + correction, window +-1.5 km/s (31 channels), deconvolution AND noise-only tasks (p_ai 0.5), 12000 steps | ERROR at step 1532 (round 5/40, 33 min): one non-finite fp16 forward, trainer raised on it | [`unified-wiener-k15_v1_..._failed/`](unified/unified-wiener-k15_v1_2026-10-10_failed/) |
| `unified-wiener-k15` | 2 | 2026-10-10 07:14 IST | same config, trainer now skips isolated non-finite steps | ERROR at step ~1532 again: CUDA OOM in forward, round 6 (log in PROGRESS.md) | none (round-5 checkpoint only) |
| `unified-wiener-k15` | 3 | 2026-10-10 ~12:25 IST | same config; empty_cache after each redraw, expandable_segments, OOM steps skipped, bad batches logged | ERROR at step 5032 (round 13): diverged at step 1532 (L1 0.030 to 0.052, never recovered), then non-finite steps multiplied; see PROGRESS.md | [`unified-wiener-k15_v3_..._failed/`](unified/unified-wiener-k15_v3_2026-10-10_failed/) |
| `eval-speedie` | 1 | 2026-10-10 06:24 IST | score unified-wiener, kin_mae_ft (mixed fine-tune) and nb16 kin_gamma0_mae_ft on Speedie GI/Kep r0.5/r1.5 | COMPLETE; learned models worse than dirty on M0/M2 at r1.5, plain Wiener best on wiggle; see PROGRESS.md 2026-10-10 | [`speedie/eval_v1_.../`](speedie/eval_v1_2026-10-10/) |
| `unified-wiener-real` | 1 | 2026-10-10 ~12:55 IST | R3 (PLAN.md): Wiener + correction, K=3, 12000 steps, realistic noise (peak SNR log-uniform 10-150, channel kernel [a,1,a] a~U(0,0.5), beam x U(1,1.6) in noise-only tasks), velocity-aware Wiener; Speedie scored after training | ERROR at step 2242 (round 7): fp16 overflow of the residual stream on ordinary inputs; see PROGRESS.md | [`unified-wiener-real_v1_..._failed/`](unified/unified-wiener-real_v1_2026-10-10_failed/) |
| `sph-validation` | 1 | 2026-10-10 ~13:00 IST | PLAN.md B2 (CPU): render Speedie dump sg_00005 with `src/data/sph_cube.py`, compare with MCFOST clean (r0.5), GI + Keplerian twin, tau0 1-100 | COMPLETE 12 min; gate NOT passed (rc_r -0.05) but bulk M1 matches (corr 0.92); cause: PA offset of ~24 deg, found afterwards | [`sph-validation/v1_.../`](sph-validation/v1_2026-10-10_first_run/) |
| `sph-validation` | 2 | 2026-10-10 ~14:10 IST | same, PA fitted continuously, spin / mirror chosen on the high-pass M1 | COMPLETE 11 min; bulk M1 corr 0.996, slope 1.015; wiggle rc_r 0.05 (gate not passed); twin rc residual 60 vs 17.5 m/s: shot noise | [`sph-validation/v2_.../`](sph-validation/v2_2026-10-10_pa_fixed_no_smoothing/) |
| `sph-validation` | 3 | 2026-10-10 ~15:50 IST | same + SPH-kernel smoothing of each particle's contribution | COMPLETE 23 min; gate as written NOT met (rc_r -0.05..0.07, twin 52-63 vs 17.5 m/s); large-scale M1 structure matches (high-pass corr 0.89 at 100 au, amplitude 77 vs 78 m/s); see PROGRESS.md | [`sph-validation/v3_.../`](sph-validation/v3_2026-10-10_smoothing/) |
| `unified-wiener-real-k15` | 1 | 2026-10-10 14:20 IST | R3b: R3 recipe at K=15 (+-1.5 km/s), lr 1.5e-4, `--clip-in 10`; Speedie scored after training | COMPLETE 247 min, 12000 steps, 0 skipped; line 600 px PSNR 48.4, M0/M1/M2 41 / 86 / 88; Speedie GI r0.5 M0/M1/M2 18 / 83 / 88, rc_r 0.87, twin spurious -2 m/s; see PROGRESS.md | [`real-k15_v1_.../`](unified/real-k15_v1_2026-10-10/) |
| `unified-wiener-real-sgv2` | 1 | 2026-10-10 15:32 IST | R3c: R3 recipe (K=3, lr 3e-4) + SG v2 in training; unfixed network | COMPLETE 179 min, 12000 steps, 0 skipped; line 600 px PSNR 48.5, 45 / 87 / 68; Speedie GI r0.5 21 / 66 / -96 (K=3 window too narrow for M2); SG v2 itself rc_r 0.82 (dirty 0.85) | [`real-k3-sgv2_v1_.../`](unified/real-k3-sgv2_v1_2026-10-10/) |
| `unified-wiener-real-fp32` | 1 | 2026-10-10 23:16 IST | R3d: K=3, R3 recipe + `--fp32-stream`, no SG v2 | COMPLETE, 12000 steps, 0 skipped; line 600 px 47.8 dB, 45 / 88 / 84; Speedie GI r0.5 0 / 76 / -13, rc_r 0.84 | [`real-k3-fp32_v1_.../`](unified/real-k3-fp32_v1_2026-10-11/) |
| `unified-wiener-real-fp32-sgv2` | 1 | 2026-10-10 23:16 IST | R3e: R3d + SG v2 in training | COMPLETE, 12000 steps, 0 skipped; SG v2 HURTS M2 (Speedie r0.5 -145 vs -13; analytic -89 vs 63) | [`real-k3-fp32-sgv2_v1_.../`](unified/real-k3-fp32-sgv2_v1_2026-10-11/) |
| `unified-smoke-backup` | 1 | 2026-10-11 ~02:00 IST | account `backup` (deepsceneloc): smoke (2 rounds x 40 steps), datasets and speedie-prep-fast shared from the main account; tests cross-account mounting | COMPLETE 10.6 min: shared datasets mounted from the main account, 19 cubes loaded, 80 steps, eval on run_0002 rt_00 / an07 at 256 and 600 px (smoke numbers, not results). Cross-account `kernel_sources` NOT exercised (this kernel attaches none); the SNR ladder will test it | none (smoke) |
| `eval-ladder` | 1 | 2026-10-11 ~19:10 IST | SNR ladder (3/6/12/24) on Speedie for R3b (K=15, SNR floor 10) and R3d (K=3): compare with R4 (floor 3) | COMPLETE (45.7 + 36.9 min); at SNR 12 wiggle rc_r R4 0.61-0.62 vs R3b 0.38, R3d 0.21: training to SNR 3 wins | [`eval-ladder_v1_.../`](unified/eval-ladder_v1_2026-10-11/) |
| `unified-r4-k15-s0` (deepsceneloc) | 1 | 2026-10-11 02:23 IST | R4 seed 0: K=15, lr 1.5e-4, clip 10, training SNR 3-150, fp32 stream; Speedie + SNR ladder | COMPLETE, 12000 steps, 0 skipped; see PROGRESS.md 2026-10-11 | [`r4-k15-s0_v1_.../`](unified/r4-k15-s0_v1_2026-10-11/) |
| `unified-r4-k15-s1` (deepsceneloc) | 1 | 2026-10-11 02:23 IST | R4 seed 1: K=15, lr 1.5e-4, clip 10, training SNR 3-150, fp32 stream; Speedie + SNR ladder | COMPLETE, 12000 steps, 0 skipped; see PROGRESS.md 2026-10-11 | [`r4-k15-s1_v1_.../`](unified/r4-k15-s1_v1_2026-10-11/) |
| `unified-r4-k15-s2` (deepsceneloc) | 1 | 2026-10-11 ~23:25 IST | R4 recipe, seed 2 (third seed for the error bar / ensemble) | COMPLETE; Speedie r0.5 wiggle 0.85, 21 / 78 / 76; ladder SNR12 0.50 (s0 / s1: 0.61 / 0.62) | [`r4-k15-s2_v1_.../`](unified/r4-k15-s2_v1_2026-10-12/) |
| `unified-r4-k15-long` (deepsceneloc) | 1 | 2026-10-11 ~23:25 IST | R4 recipe, seed 3, 80 rounds = 24000 steps (R4's loss was still falling at 12000); eval incl. ladder | ERROR at step 575: round 1 L1 1.06 (normal 0.04), 197 non-finite steps: Wiener-peak normalisation collapses at SNR 3 (targets 16x, inputs 29x) | [`r4-k15-long_v1_..._failed/`](unified/r4-k15-long_v1_2026-10-12_failed/) |
| `unified-r4n-k15-long` (deepsceneloc) | 1 | 2026-10-12 ~03:05 IST | R4 + `--norm robust`, 24000 steps, seed 3 | COMPLETE, 0 skipped; BEST so far: Speedie r0.5 wiggle 0.87, ladder SNR24 0.81 / SNR12 0.60, line 600 px 49.30 dB 55/87/89 | [`r4n-k15-long_v1_.../`](unified/r4n-k15-long_v1_2026-10-12/) |
| `eval-ensemble` | 1 | 2026-10-12 ~03:10 IST | (a) mean of R4 s0 + s1 on everything; (b) + 4-flip TTA on Speedie + ladder; (c) + R3b with TTA | STOPPED at the 12 h limit: (a) and (b) complete, (c) partial (Speedie r0.5 / r1.5 / twin and ladder SNR3-6); averaging matches the best seed on wiggle, does not exceed it; TTA adds nothing | [`eval-ensemble_v1_.../`](unified/eval-ensemble_v1_2026-10-12/) |
| `sph-cubes` (deepsceneloc) | 1 | 2026-10-12 ~04:00 IST | CPU: 22 Young et al. 2026 GI runs (16 train / 6 held-out simulations) rendered by `make_sph_cubes.py` (last dump, random view, 13CO tau0 3-50, beam 5-12 px, 301 px, 0.1 km/s) | ERROR after 5 of 22: M0.2_R200_Mu1.2's dump has neither `temperature` nor `u` (KeyError) | none |
| `sph-cubes` (deepsceneloc) | 2 | 2026-10-12 ~04:10 IST | same; temperature fallback 20 K (R/100 au)^-0.5 when a dump has none (recorded per cube as `temperature_source`); a failing archive is skipped and logged instead of stopping the job | COMPLETE ~20 min: 19 of 22 (14 train mu 0.18-0.97, 5 held-out simulations); 3 failed (`KeyError: 'vx'`: last dump has no velocities) | [`sph-cubes/v2_2026-10-12/`](sph-cubes/v2_2026-10-12/) |
| R5 seed 0 = `deepsceneloc/unified-wiener` (slug by mistake) | 1 | 2026-10-12 11:07 IST | R5: R4 recipe + `--norm robust` + 14 Young GI cubes | COMPLETE, 0 skipped; Speedie r0.5 wiggle 0.84, ladder SNR12 0.29 (worse than R4), 9074 0.41 | [`r5-k15-s0_v1_.../`](unified/r5-k15-s0_v1_2026-10-12/) |
| `aditisah11/sph-cubes` | 1 | 2026-10-12 11:10 IST | the same render on account aditi; made redundant minutes later when `deepsceneloc/sph-cubes` was shared with aditi (CPU only, not used) | running | none |
| `aditisah11/unified-r5-k15-s{1,2}` | 1 | 2026-10-12 11:14 IST | R5 seeds 1 and 2 | s1 COMPLETE: Speedie 0.85, ladder SNR12 0.49, line 49.65 dB; s2 COMPLETE (0 skipped): Speedie 0.83, SNR24 0.75, SNR12 0.56, line 47.15 dB 48/86/83 | [`r5-k15-s1_v1_.../`](unified/r5-k15-s1_v1_2026-10-12/), [`r5-k15-s2_v1_.../`](unified/r5-k15-s2_v1_2026-10-12/) |
| `aditisah11/unified-r4n-k15-long-s11` | 1 | 2026-10-12 ~18:50 IST | second seed of the R4n-long recipe (seed 11) | COMPLETE, 0 skipped: Speedie 0.88, SNR24 0.82, SNR12 0.41, line 600 px 47.91 dB 34/84/85, 256 px 47.38 53/87/87 | [`r4n-k15-long-s11_v1_.../`](unified/r4n-k15-long-s11_v1_2026-10-13/) |
| `aditisah11/alma-exoalma` | 1 | 2026-10-13 ~00:50 IST | FIRST REAL-DATA RUN (inference only): exoALMA MWC 758 13CO fiducial, full 1024 px, `alma_unified.py` with R5 seed 1 (aditi's own checkpoint; R4n-long is not visible to aditi and main's weekly GPU quota is used up); checks S1, S2, S6; for Gate A (preprocessing sanity), rerun with the final model later | ERROR before any cube: Dataverse API answers HTTP 403 to Python's default `Python-urllib` user agent (curl / browser agents get 200); fixed with an explicit User-Agent | [`alma/mwc758-13co_r5_v1_..._failed/`](alma/mwc758-13co_r5_v1_2026-10-13_failed/) |
| `chiragbhayal/alma-exoalma-r4n` | 1 | 2026-10-13 ~03:35 IST | FIRST REAL-DATA RUN: exoALMA MWC 758 13CO, full 1024 px, R4n-long (inference only) | COMPLETE 5.4 min: off-line noise 30.7x lower, M0 total ratio 0.96, |M1 shift| on bright px median 83 / p95 267 m/s, >4 sigma in line-free planes 2.3e-4 raw vs 1.2e-5 denoised; S6 INVALID (field 95 % empty sky) | [`alma/mwc758-13co_r4n_v1_.../`](alma/mwc758-13co_r4n_v1_2026-10-13/) |
| `chiragbhayal/alma-exoalma-r4n` | 2 | 2026-10-13 ~03:50 IST | same, cropped to 256 px (6.4 arcsec) around the disc so S6 and the moment figures measure the disc | COMPLETE: noise 24.6x lower, M0 ratio 0.98, |M1 shift| median 76 m/s, S2 below raw; S6 STILL INVALID (noise blobs of the raw reference enter the geometry mask) | [`alma/mwc758-13co_r4n_v2_..._crop256/`](alma/mwc758-13co_r4n_v2_2026-10-13_crop256/) |
| `chiragbhayal/unified-r4n-k15-long-s23` | 1 (+ resume v2) | 2026-10-13 ~03:25 IST | third seed of the R4n-long recipe (seed 23), first run on the RULES.md #14 kernel | COMPLETE: trained 24000 steps, stopped at its 11 h budget with 3 cases unscored, resumed (v2) and finished all 72; 0 skipped steps; Speedie 0.87, SNR24 0.80, SNR12 0.56, line 600 px 48.07 dB 30/83/85 | [`r4n-k15-long-s23_v1_.../`](unified/r4n-k15-long-s23_v1_2026-10-13/) |
| `aditisah11/alma-exoalma-s5` | 1 | 2026-10-13 ~13:20 IST | MWC 758 13CO, 512 px, R4n-long seeds 3 + 11 averaged; S6 on the disc mask; S5 injection-recovery (Speedie GI / twin in the cube's own line-free noise, SNR 12 / 24 / 36) | COMPLETE: S5 model line-centre error vs truth 14-25 m/s (GI), 6-8 m/s (twin) vs dirty 20-67; wiggle 0.75 / 0.89 / 0.93; twin spurious -0.1 to -1.3 m/s; real cube S6 r 0.82 vs raw, M0 ratio 1.02 | [`alma/mwc758-13co_s5_v1_.../`](alma/mwc758-13co_s5_v1_2026-10-13/) |
| `aditisah11/alma-exoalma-all13co` | 1 | 2026-10-13 ~15:05 IST | the 14 other exoALMA discs, 13CO fiducial, 512 px, R4n-long seeds 3 + 11 averaged, S1 / S2 / S6 (geometry from exoALMA I Tables 1-2); inference only | running | pending |
| `aditisah11/eval-gate` | 1 | 2026-10-13 ~15:30 IST | do-no-harm gate sweep (a * model + (1 - a) * Wiener, a = 0.25 / 0.5 / 0.75) with blind `snr_est`, R4n-long seeds 3 + 11 averaged, on ladder 12 / 24, Speedie r0.5, SG v2, Speedie r1.5 | COMPLETE: at est. 121-133 (true SNR ~135) a = 0.25 gives M0 / M1 / M2 -4 / +37 / +50 (plain -76 / +40 / -257) with the wiggle at dirty's level; a stays 1 below est. 40; gate = (40, 120, 0.25) | [`eval-gate_v1_.../`](unified/eval-gate_v1_2026-10-13/) |
| `chiragbhayal/eval-final` | 1 | 2026-10-13 ~15:35 IST | THE FINAL EVALUATION: R4n-long seeds 3 + 11 + 23 averaged; line holdouts x 5 sizes, 9074, SG v2, an07 / an08, 5 held-out Young simulations x 5 sizes, Speedie GI / twin r0.5 + r1.5, SNR ladder 3 / 6 / 12 / 24 (GI + twin); every case also scored at gate mixes a = 0.25 / 0.5 / 0.75 with `snr_est`, so the gate is chosen afterwards; RULES.md #14 stop-and-resume | complete 2026-10-14, 362 min, 97 cases; `results/unified/eval-final_v1_2026-10-14/`, report `results/final/RESULTS.md` | Kaggle version 1 (author to confirm) |
| `jensipaneliya/unified-r7-k15-snr40-s{41,42,43}` | 1 | 2026-10-13 16:55 IST | R7: R4n-long recipe with training peak SNR 3-40 instead of 3-150 (the gate now covers high SNR), 3 seeds, for the SNR-12 wiggle; account jensi (shares done 16:55); s42 complete 2026-10-14 (`results/unified/r7-k15-snr40-s42_v1_2026-10-14/`: SNR-12 wiggle 0.53, SNR 6 0.24, line slightly lower; not adoptable on one seed); s41 stopped at budget, needs resume after quota reset Oct 17; s43 running; RULES.md #14 kernels | partial | pending |
| `chiragbhayal/unified-r6-k15-crop256` | 1 | 2026-10-13 ~13:25 IST | R6 (PLAN.md track C, last wiggle experiment): R4n-long recipe with 256 px training crops, batch 4 (~1.3x the pixels per step), seed 31, 24000 steps; RULES.md #14 kernel (expected to stop near 11 h and resume) | complete 2026-10-15 (v1 stopped at 11 h budget, v2 resumed, 80/80 rounds, 72/72 cases); `results/unified/r6-k15-crop256_v2_2026-10-15/`; 9074 wiggle 0.43-0.47 (R4n 0.26-0.36), twin spurious SNR 12 11.9 m/s, line@600 45.8 dB | v1 + v2 (author to confirm) |
| `aditisah11/size-test` | 1 | 2026-10-15 ~04:45 IST | Size fix (PLAN.md 0b item 7): R4n-long seeds 3 + 11 averaged (aditi cannot see seed 23), gate on; 5 line holdouts, 9074 (+ _ai), 5 held-out Young sims at 480 / 600 px; variants plain, `--infer-beam-px 10 --infer-mode corr`, `10 full`, `8 corr` (network run where the beam spans ~10 px); RULES.md #14 | complete 2026-10-15 (30 min per variant x 4); `results/unified/size-test_v1_2026-10-15/`; line wiggle @600 0.67 -> 0.77 (b10 full), Young M0 @600 12 -> 29 %, line PSNR -0.5 dB; 9074 0.31 -> 0.25 | v1 (author to confirm) |
| `chiragbhayal17/unified-r8-m1-r4n` | 1 | 2026-10-15 ~05:00 IST | R8: fine-tune of R4n-long seed 3 with the line-centre (M1) loss (`--warm unified_r4n_k15_long.pth --w-m1 0.01`, band +-1 km/s, 2 x 96 px crops per step), lr 5e-5, 20 rounds x 300 steps, seed 52, R4n's crop 128 / batch 12; eval as R4n (Speedie, ladder 3/6/12/24); RULES.md #14 | complete 2026-10-15 (349 min); `results/unified/r8-m1-r4n_v1_2026-10-15/`; ladder SNR 12 0.35 (WORSE than R4n 0.41-0.60), line @600 M0 58 wiggle 0.73; not adopted | v1 (author to confirm) |
| `chiragbhayal17/unified-r8-m1-r6` | - | prepared 2026-10-15 | R8 from R6 (crop 256, batch 4, seed 51), otherwise as above; waits for R6 (`chiragbhayal/unified-r6-k15-crop256`) to be shared with chirag2 | complete 2026-10-15 (348 min, launched ~05:05 after the R6 share); `results/unified/r8-m1-r6_v1_2026-10-15/`; ladder SNR 12 / 24 0.64 / 0.83, twin spurious SNR 6 293 -> 60 m/s, 9074 0.38-0.40 (R6 0.43-0.47), PSNR 45.8 | v1 (author to confirm) |
| `krishanyadav333/oracle-speedie` | 1, 2 | 2026-10-15 ~05:05 IST (v1 died writing its first figure: folder not created; v2 ~05:20) | CPU: oracle linear filter (knows clean's and the noise's power spectra) on Speedie GI / twin r0.5 + r1.5 and the SNR ladder 3/6/12/24, same scorecard: where linear filtering stops on the test disc | complete 2026-10-15 (v2; `results/oracle-speedie_v2_2026-10-15/`): oracle linear on Speedie GI r0.5 native 0.80, ladder SNR 3 / 6 / 12 / 24: 0.02 / 0.20 / 0.45 / 0.68 | v2 (author to confirm) |
| `anshi2904/unified-r6-s32` | 1 | 2026-10-16 ~15:40 IST | R6 recipe (R4n flags, `--crop 256 --batch 4`) seed 32: repeat of R6 (seed 31) to see whether its real-GI wiggle gain (9074 0.43-0.47) is reproducible; account anshika (fresh quota); RULES.md #14 | v1 stopped at its 12 h deadline after round 70 of 80 (560 min, RESUME_NEEDED, no eval); v2 = resume with its own output attached (02:30 IST), COMPLETE 231 min, eval done; 9074 600 px rc_r 0.38 (s31 0.43): gain not repeated, not adopted | [`r6-k15-s32_v1-v2_2026-10-17/`](unified/r6-k15-s32_v1-v2_2026-10-17/) |
| `krishanyadav333/unified-r6-s33` | v1, v2, v3 | 2026-10-17 05:55 IST | R6 recipe seed 33 (third R6 seed) | v1 stopped at round 67 of 80 (deadline); v2 and v3 resumes crashed at startup (no inputs mounted); not pursued, partial checkpoint kept | [`r6-k15-s33_v1-v3_2026-10-17_stopped/`](unified/r6-k15-s33_v1-v3_2026-10-17_stopped/) |
| `chiragbhayal17/eval-ladder-sg` | 1 | 2026-10-18 03:05 IST | hydro SNR ladder of recipe B: 9074 (held out) and 9025 (TRAINING disc, diagnostic only), native and SNR 11 / 20 / 40 / 80, deconvolution + noise-only, 300 / 600 px | COMPLETE, 40 cases; 9074_ai rc_r 0.40 / 0.48 / 0.57 / 0.70 at SNR 11 / 20 / 40 / 80 (oracle 0.93); 9025_ai 0.57 / 0.66 / 0.78 / 0.82 | [`eval-ladder-sg_v1_2026-10-18/`](unified/eval-ladder-sg_v1_2026-10-18/) |
| `chiragbhayal17/eval-final-test` | 1 | 2026-10-18 03:05 IST | THE ONE-TIME FINAL TEST of frozen recipe B: Speedie r1.5 GI + twin at SNR 6 / 12 / 24 (pre-registered 2026-10-15) | COMPLETE; rc_r 0.06 / 0.44 / 0.70 (Wiener 0.05 / 0.17 / 0.41), PV r 0.16 / 0.35 / 0.71 (Wiener 0.04 / 0.01 / 0.27) | [`eval-final-test_v1_2026-10-18/`](unified/eval-final-test_v1_2026-10-18/) |
| `chiragbhayal17/sph-cubes-test2` | 1 | 2026-10-18 03:30 IST | CPU: render 8 more held-out Young et al. simulations (TEST2 in `make_sph_cubes.py`) | running | pending |
| `chiragbhayal17/alma-mwc758-zoom512` (MISNAMED: ran the Young test2 evaluation) | 1 | 2026-10-18 03:58 IST | recipe B on the 8 TEST2 Young discs, 300 / 600 px (the watcher pushed `kaggle/eval_young_test2` under the zoom slug: bash 3.2 bug) | COMPLETE 52.8 min, 16 cases; rc_r mean 0.35 (Wiener 0.24, dirty 0.20) at 600 px | [`eval-young-test2_v1_2026-10-18/`](unified/eval-young-test2_v1_2026-10-18/) |
| `chiragbhayal/eval-young-test2` | 1 | 2026-10-18 04:18 IST | same code, the intended run (duplicate; reproducibility check) | COMPLETE; every number identical to the misnamed run (max abs difference 0) | same archive as the misnamed run |
| `aditisah11/alma-mwc758-zoom256` | 1 | 2026-10-18 04:18 IST | MWC 758, 6.4 arcsec field + `--scale 1.6667` (beam 6 to 10 px), recipe B, S5 | COMPLETE; flux 0.991, S5 GI rc_r 0.81 / 0.91 / 0.94 (256 px native 0.76 / 0.88 / 0.92) | [`mwc758-13co_zoom256_v1_2026-10-18/`](alma/mwc758-13co_zoom256_v1_2026-10-18/) |
| `chiragbhayal17/alma-mwc758-zoom512-v2` | 1 | 2026-10-18 04:55 IST | MWC 758, 512 px field + `--scale 1.6667` (the real zoom-512 run) | COMPLETE 67 min; flux 1.037, S5 GI rc_r 0.85 / 0.93 / 0.95, twin error 9 / 7 / 5 m/s, twin invented 0.0 / 1.4 / 2.0 m/s | [`mwc758-13co_zoom512_v2_2026-10-18/`](alma/mwc758-13co_zoom512_v2_2026-10-18/) |
| `jensipaneliya/unified-r6-s33b` | 1 | 2026-10-18 ~03:15 IST | R6 recipe seed 33, restarted from scratch (the first s33 could not resume) | STOPPED at round 65 / 80 (deadline, RESUME_NEEDED), 560.7 min | [`r6-k15-s33b_v1_2026-10-18_round65/`](unified/r6-k15-s33b_v1_2026-10-18_round65/) |
| `jensipaneliya/unified-r6-s33b` | 2 | 2026-10-18 ~12:45 IST | resume of v1 (own output attached): rounds 66 to 80 + evaluation | COMPLETE (80 rounds + eval); 9074@600 rc_r 0.41, 9074_ai 0.40, line @600 PSNR 48.1, Speedie r0.5 SNR 12 / 24 0.63 / 0.80 | [`r6-k15-s33b_v2_2026-10-18/`](unified/r6-k15-s33b_v2_2026-10-18/) |
| `chiragbhayal17/alma-exoalma-final-p0` | 1 | 2026-10-18 ~10:25 IST | FINAL ALMA run, recipe B, exoALMA 13CO: MWC 758 (S5), HD 34282, HD 135344B (S5), V4046 Sgr; 512 px + `--target-beam-px 10`, PA from Table 2 | COMPLETE; PA 0.0 to 1.7 deg on MWC 758 / HD 135344B / V4046 Sgr; HD 34282 fit FAILED (refit); S5 HD 135344B GI rc_r 0.86 / 0.93 / 0.95; V4046 Sgr continuum auto-SUBTRACTED (to check) | [`exoalma-final-p0_v1_2026-10-18/`](alma/exoalma-final-p0_v1_2026-10-18/) |
| `chiragbhayal17/alma-exoalma-final-p1` | 1 | 2026-10-18 ~10:25 IST | FINAL ALMA run, recipe B, exoALMA 13CO: DM Tau (S5), CQ Tau, HD 143006, J1842; same settings | COMPLETE; PA 0.4 to 2.4 deg on DM Tau / CQ Tau / J1842; HD 143006 fit FAILED (face-on, M* at bound); S5 DM Tau GI rc_r 0.68 / 0.84 / 0.90; flux 1.06 to 1.15 | [`exoalma-final-p1_v1_2026-10-18/`](alma/exoalma-final-p1_v1_2026-10-18/) |
| `chiragbhayal/alma-exoalma-final-p2` | 1 | 2026-10-18 ~10:25 IST | FINAL ALMA run, recipe B, exoALMA 13CO: AA Tau, SY Cha, J1604, J1852; same settings | COMPLETE ~3.8 h; PA within 0.5 to 2.6 deg on AA Tau / SY Cha / J1852 (J1604 face-on, 11.7); flux 1.065 to 1.090 on the three faint discs, 0.992 on J1604 | [`exoalma-final-p2_v1_2026-10-18/`](alma/exoalma-final-p2_v1_2026-10-18/) |
| `chiragbhayal/alma-exoalma-final-p3` | 1 | 2026-10-18 ~10:25 IST | FINAL ALMA run, recipe B, exoALMA 13CO: LkCa 15 (S5), PDS 66, J1615; same settings | COMPLETE; PA 0.4 / 0.9 deg on J1615 / LkCa 15; PDS 66 fit FAILED (M* at bound); S5 LkCa 15 GI rc_r 0.75 / 0.88 / 0.92; flux 1.02 / 1.02 / 1.29 (PDS 66) | [`exoalma-final-p3_v1_2026-10-18/`](alma/exoalma-final-p3_v1_2026-10-18/) |
| `aditisah11/alma-dsharp-final-p0` | 1 | 2026-10-18 ~10:25 IST | FINAL ALMA run, recipe B, DSHARP 12CO, `--spec-interp 0.1`: HD 143006 (S5), HD 142666, IM Lup; `--target-beam-px 10 --max-scaled-px 1024`, PA Huang 2018 | COMPLETE; IM Lup flux 0.977, S6 0.93, PV5 r 0.94, PA 0.1 deg from Huang; HD 143006 PA 3.9 deg off, S5 rc_r invalid (negative for all methods); HD 142666 geometry fit FAILED (M* 10.3) | [`dsharp-final-p0_v1_2026-10-18/`](alma/dsharp-final-p0_v1_2026-10-18/) |
| `aditisah11/alma-dsharp-final-p1` | 1 | 2026-10-18 ~10:25 IST | FINAL ALMA run, recipe B, DSHARP 12CO, `--spec-interp 0.1`: AS 209 (S5), HD 163296; same settings | AS 209 COMPLETE (flux 1.030, S6 0.85, PV5 r 0.63, sky PA 84.9 vs 85.8, S5 GI rc_r 0.26 / 0.55 / 0.86 / 0.95 at SNR 6 / 8 / 12 / 24); HD 163296 FAILED (no BMAJ keyword: multi-beam cube) | [`dsharp-final-p1_v1_2026-10-18/`](alma/dsharp-final-p1_v1_2026-10-18/) |
| `aditisah11/alma-dsharp-final-hd163296` | 1 | 2026-10-18 ~13:45 IST | HD 163296 alone (`PART, NPART = 3, 5`), with the `beam_of` fix; same settings | COMPLETE 20 min; flux 0.991, noise reduction 21x, S6 0.98, PV5 r 0.97, PA 0.6 deg from Huang | [`dsharp-final-hd163296_v1_2026-10-18/`](alma/dsharp-final-hd163296_v1_2026-10-18/) |
| `aditisah11/alma-dsharp-s5fix` | 1 | 2026-10-18 ~20:40 IST | DSHARP HD 143006 + AS 209 S5 rerun after the fix: noise from line-free planes without compact sources, truth framed as in exoALMA (`--inject-field-frac 0.625`) | running | pending |
| `aditisah11/eval-ladder-sg` | 1 | 2026-10-18 13:43 IST | rerun of `chiragbhayal17/eval-ladder-sg` v1 with the beam-matched PV score (`pv_r_vs_beam`, `pv5_r_vs_beam`) for 9074 and 9025 (new slug: aditi had the free slot) | COMPLETE 100 min; 9074 deconvolution PV5 vs beam truth: model 0.45, dirty 0.51 (native); noise-only PV5 vs truth: model 0.27 / 0.36 / 0.43 / 0.53 / 0.80, Wiener 0.77 / 0.34 / 0.62 / 0.51 / 0.78 | [`eval-ladder-sg_v2_2026-10-18/`](unified/eval-ladder-sg_v2_2026-10-18/) |
| `aditisah11/eval-planet` | 1 | 2026-10-18 14:06 IST | planet check: recipe B on the 3 held-out run_0002_00560 views + Young test sims at 600 px, `--save-pred` (M1 maps, +-4 km/s channels) | COMPLETE 45 min, 8 cases; run_0002_00560 rt_00 / 01 / 04 rc_r model 0.88 / 0.75 / 0.58 (dirty 0.76 / 0.04 / 0.03); planet check: see PROGRESS | [`eval-planet_v1_2026-10-18/`](unified/eval-planet_v1_2026-10-18/) |
| `chiragbhayal/eval-planet` | 1 | 2026-10-18 14:24 IST | DUPLICATE of `aditisah11/eval-planet` (pushed by mistake: a scratch `queue.py` shadowed the stdlib module and ran); same code, reproducibility check | COMPLETE; identical to `aditisah11/eval-planet` (1304 numbers, max abs difference 0) | same archive as the original |
| `chiragbhayal17/alma-aatau-s5flux` | 1 | 2026-10-18 14:56 IST | S5 FLUX TEST: AA Tau with injection at SNR 6 / 8 / 12 / 24, flux of dirty / Wiener / model against the known truth (decides whether the faint-disc flux excess is the model) | COMPLETE; model flux / truth 1.085 / 1.039 / 1.016 / 1.017 (GI) and 1.103 / 1.060 / 1.025 / 1.023 (twin) at SNR 6 / 8 / 12 / 24; dirty 0.98 to 1.00 | [`aatau-s5flux_v1_2026-10-18/`](alma/aatau-s5flux_v1_2026-10-18/) |
| `anshi2904/unified-r9-s61` | 1 | 2026-10-16 ~15:40 IST | R9: R6 recipe with `--train-beam-px 8,12` (each redraw resampled so the beam spans 8-12 px, where `--infer-beam-px 10` runs the network), seed 61, evaluated with the levers (`--wiener2 --infer-beam-px 10 --infer-mode full`) | COMPLETE 644 min, 80 rounds, eval done; with the levers: line 600 px 51.37 dB, M0/M1/M2 56.6 / 89.9 / 90.3, 9074 spine 0.41 (A and B 0.72), Speedie SNR 12 0.61; not adopted (rule of PLAN.md 0f) | [`r9-k15-s61_v1_2026-10-17/`](unified/r9-k15-s61_v1_2026-10-17/) |
| `deepsceneloc/unified-r9-s62` | v2 (v1 stopped by Krishan, see README) | 2026-10-17 08:47 IST | R9 recipe (R6 flags, `--train-beam-px 8,12`), seed 62, levers on in eval | COMPLETE 604 min, eval done; line 600 px 50.81 dB, Speedie SNR 12 wiggle 0.32 (s61 0.61, B 0.60), hydro M0 minus 13; not adopted | [`r9-k15-s62_v2_2026-10-17/`](unified/r9-k15-s62_v2_2026-10-17/) |
| `chiragbhayal17/eval-final-b` | 1 | 2026-10-17 10:30 IST | FINAL EVALUATION of recipe B at five sizes + PV curves (Speedie r1.5 kept out); script `kaggle/eval_final_b/kernel_final_b.py` | COMPLETE, 95 cases, none failed; line PSNR 51.2 / 51.8 / 51.7 / 51.7 / 51.7 dB at 256 / 300 / 320 / 480 / 600 px, M0 / M1 / M2 about 62 / 89 / 91 at every size | [`eval-final-b_v1_2026-10-17/`](unified/eval-final-b_v1_2026-10-17/) |
| `chiragbhayal/unified-r8-w003` | 1 | 2026-10-17 05:55 IST | R8 from R6 with M1 loss weight 0.003 (weight 0.01 is in recipe B), 20 rounds, lever-free eval | COMPLETE, eval done; Speedie SNR 12 wiggle 0.30 vs 0.64 for weight 0.01; 9074 600 px rc_r 0.39 (R6 s31 0.43); not adopted | [`r8-m1-r6-w003_v1_2026-10-17/`](unified/r8-m1-r6-w003_v1_2026-10-17/) |
| `chiragbhayal/unified-r8-w030` | 1 | 2026-10-17 05:55 IST | R8 from R6 with M1 loss weight 0.03 (weight 0.01 is in recipe B), 20 rounds, lever-free eval | COMPLETE, eval done; Speedie SNR 12 wiggle 0.42 vs 0.64 for weight 0.01; 9074 600 px rc_r 0.39 (R6 s31 0.43); not adopted | [`r8-m1-r6-w030_v1_2026-10-17/`](unified/r8-m1-r6-w030_v1_2026-10-17/) |
| `chiragbhayal17/combo-a`, `combo-b` | 1 | 2026-10-15 ~12:00 IST | The two inference levers together (`--infer-beam-px 10 --infer-mode full --wiener2`, gate on) on the full decision set at 256 / 600 px + Speedie + ladder + Young: A = R4n-long seeds 3 + 11 + 23, B = A + R8-from-R6 (4 nets); C (R8-from-R6 alone) not launched | complete 2026-10-15 ~17:30; `results/unified/combo-{a,b}_v1_2026-10-15/`; B best: line @600 51.7 dB, M0/M1/M2 62/89/91, rc_r 0.76; Young M0 @600 47 %; Speedie SNR 12 0.60, twin SNR 6 +25 m/s; costs: analytic rc_r 0.88 -> 0.83, SG v2 M1/M2 @600 17/28 -> 11/20, 9074 rc_r @600 0.33 -> 0.30 (spine 0.60 -> 0.67) | v1 (author to confirm) |
| `aditisah11/alma-mwc758-recheck` | 1 | 2026-10-15 ~13:00 IST | CPU kernel (aditi's GPU quota is spent): MWC 758 13CO, R4n-long seeds 3 + 11, with the FIXED line-free finder, twice on the same cube: `fixed_only` (no levers) and `levers` (size fix + two-pass Wiener); S1, S2, S5 (SNR 12 / 24 / 36), S6; crop 512 | partial 2026-10-16: fixed_only complete (S5 line-centre error 14-25 m/s vs dirty 26-67; S2 0), levers config cancelled by Kaggle after S5 GI at 164 min; `results/alma/mwc758-13co_recheck_v1_2026-10-16_partial/` | v1 (author to confirm) |
| `aditisah11/wiener2-test` | 1 | 2026-10-15 ~05:30 IST | Two-pass Wiener (`--wiener2`, `prior_wiener`): the network re-run from a Wiener whose signal spectrum comes from its own first pass; R4n-long seeds 3 + 11 averaged, gate on; Speedie GI / twin r0.5 native and ladder 6 / 12 / 24, 9074 (+ _ai), 5 line holdouts at 256 / 600 px; variants `w2_plain` and `w2_second_pass` in the same run | complete 2026-10-15 (163 min); `results/unified/wiener2-test_v1_2026-10-15/`; line PSNR +2.9 / +2.0 dB (256 / 600 px), M0 +8-10, wiggle @600 0.67 -> 0.71; ladder SNR 12 0.50 -> 0.57; 9074_ai 0.34 -> 0.43 | v1 (author to confirm) |

---

## 16-checkpoint-evaluation: one protocol over every checkpoint, as numbers and images

Scores every checkpoint it finds under `/kaggle/input` with one harness (`src/evaluation/checkpoint_eval.py`) on the same cubes, and writes paged image sheets (`checkpoint_figures.py`).
Line-emission means depend on how many cubes the run used: **only compare rows from runs with the same cube set.**

| Ver | Date | code | scope | Artifacts |
|----:|---|---|---|---|
| 5 | 2026-09-25 | `6e4e139` | QUICK: 34 checkpoints, 2 line-emission cubes (`rt_00`, `rt_01`) + SG v2, 102 rows in 49 min; display cell crashed after all scoring | [`v5_.../`](16-checkpoint-evaluation/v5_2026-09-25_quick_6e4e139/): notebook, log, CSV, inventory, 12 `rt_00` sheets |

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

## 09-wiggle-scoring: GI wiggle, GPU (a DIFFERENT notebook from `09 — Architecture
comparison` above; root-level, Kaggle-linked, unrelated lineage)

No training. GPU port of `experiments/wiggle_all_methods.py`: dirty / U-Net / DDRM /
beam-only against clean, scored with the corrected `compare_wiggles()`. Downloaded manually
rather than auto-pushed through Kaggle's GitHub link, so there is no `push` commit; `code` is
the commit confirmed pulled from the run's own cell 0/0b log.

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 2 | 2026-08-29 | `7d73e2e` | `ce1b6ae` | complete, both configs, 9.0 min total (19x the 173 min local CPU run): **third independent confirmation of the retraction's corrected table** | [`v2_2026-08-29_ce1b6ae/`](09-wiggle-scoring/v2_2026-08-29_ce1b6ae/) |
| 3 | 2026-09-11 | `5c51ad7` | -- (downloaded manually, no push commit yet) | complete, both configs, 9.3 min total: **first GPU confirmation at the corrected frac=0.05**, matches the local CPU frac sweep to 3 decimal places | [`v3_2026-09-11_5c51ad7/`](09-wiggle-scoring/v3_2026-09-11_5c51ad7/) |

240-360 step 1 (121 channels): resid r dirty 0.891 / beam-only 0.920 / U-Net 0.804 /
DDRM 0.583, matching the original correction and the local CPU rerun to within fit noise.
Step 4 (31 channels) bunches all methods near 1.0, a known `quadratic_moment1` coarse-sampling
artifact, not a real finding. `wiggle_all_methods.png` was regenerated on Kaggle per the run
log but not downloaded from the Output tab; the version committed at
`results/self-gravitating/wiggle_all_methods.png` is from the local CPU rerun, not this run.

**Version 3** reran at the corrected `frac=0.05` after `613aa0e` found v2's table had been
computed at the flagged `frac=0.02`. Result: dirty 0.862 / beam-only 0.888 / U-Net 0.760 /
DDRM 0.568, mstar=0.644 (not degenerate). Matches the same-day local CPU sweep at frac=0.05
(dirty 0.862 / beam-only 0.888 / U-Net 0.760 / DDRM 0.571 / mstar=0.643) to within fit noise
-- independent code path (GPU vs CPU), independent cube read. The frac=0.05 table is no
longer provisional. This run's own `wiggle_all_methods.png` also surfaced a second smoothing
source in U-Net/DDRM inference (a 600x600->256->600 resize round trip); see PROGRESS.md
2026-09-11.

An earlier interactive run on the same kernel, downloaded manually rather than committed, gave
the same numbers (0.5835 vs this run's 0.5833 on DDRM) but has no Kaggle version number to
attribute it to, so only this auto-pushed run is archived.

---

## 10-sg-training: training on self-gravitating data

First training run on SG data. Uses the pairs synthesized in `experiments/synthesize_sg_pairs.py`,
not the ones shipped, which differ clean-to-dirty by 0.4-7% RMS and would teach the identity.

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 1 | 2026-09-04 | `be616fd` | `cc194ef` | complete, 2 arms + frozen baseline, ~18 min each | [`v1_2026-09-04_cc194ef/`](10-sg-training/v1_2026-09-04_cc194ef/) |
| **4** | 2026-09-04 | `340e26c` | `ee040ae` | complete, identical settings. **`fresh` collapsed; V1's headline does not reproduce** | [`v4_2026-09-04_ee040ae/`](10-sg-training/v4_2026-09-04_ee040ae/) |

Holdout `run_9074`, moment improvement at frac=0.05:

| arm | PSNR (V1 / V4) | M0 | M1 | M2 |
|---|---|---|---|---|
| frozen | 29.032 / 29.032 | -10.3 / -10.3 | -0.6 / -0.6 | -43.6 / -43.6 |
| finetune | 30.859 / 30.874 | -6.5 / -7.0 | **+36.5 / +36.7** | +15.6 / +15.3 |
| fresh | 30.024 / 29.258 | **+5.0 / -61.1** | +21.8 / -27.3 | +26.0 / -42.2 |

**SG training closes the domain gap**, and that part holds across both runs: `frozen` is
negative on all three moments while `finetune` is positive on M1 and M2 in V1 and V4 alike.

**`fresh` beats `finetune` is WITHDRAWN.** V4 re-ran identical settings and `fresh` collapsed
from +5.0 / +21.8 / +26.0 to **-61.1 / -27.3 / -42.2**, worse than the untrained baseline.
`frozen` reproduced exactly and `finetune` to within 0.5 pp in the same run, so the 66 pp
swing belongs to that arm, not the setup. V1's headline was one draw of a high-variance arm.

Cause is v25's finding recurring: early stopping on a noisy val set turns run-to-run
nondeterminism into large swings. `fresh` stopped at epoch 23 (best 15) in V4 against epoch 33
(best 25) in V1; `finetune` stopped at 32 (best 24) both times, being already near a solution.

PSNR is nearly tied (30.86 vs 30.02) while the moments differ by 10-15 pp in opposite
directions, and finetune had the better val loss while losing M0 and M2. RULES.md #4 again.

---

## 11-sg-loo: leave-one-out over the five SG disks

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 2 (author-reported, no push commit to verify against) | 2026-09-10 | `b7b140d` | -- | complete, 5 folds x 2 arms, 128 min, checkpoints downloaded and stored (`models/11-loo/`) | [`v2_2026-09-10_b7b140d/`](11-sg-loo/v2_2026-09-10_b7b140d/) |

Seed fixed at 42 across folds, every cube holds out exactly once.

| arm | PSNR mean +/- std | M0 mean +/- std | M1 mean +/- std | M2 mean +/- std |
|---|---|---|---|---|
| frozen | 29.53 +/- 2.98 | -13.93 +/- 56.28 | -52.70 +/- 129.56 | -52.91 +/- 41.98 |
| finetune | 30.88 +/- 3.03 | -11.84 +/- 39.27 | -6.30 +/- 50.18 | -11.64 +/- 26.42 |
| fresh | 28.27 +/- 1.36 | -46.55 +/- 136.27 | -66.56 +/- 160.08 | -83.89 +/- 109.02 |

**`fresh - finetune` is smaller than the fold-to-fold spread on all three moments** (M0
-34.7+/-104.4 pp, M1 -60.3+/-111.1 pp, M2 -72.3+/-86.6 pp). Not separable at n=5.

**Fold 3 (`run_9032` as holdout) is catastrophic for every arm**, predicted in the notebook
before the run from that cube's own diagnostics (rmsdiff 0.107 vs the others' 0.46-0.54).

**Follow-up (same day): scored the actual GI wiggle across all 5 folds, not just the
moments** (`experiments/score_sg_wiggle_loo.py`, `results/self-gravitating/sg_loo_wiggle.json`).
Opposite conclusion from the moment table above:

| method | resid r mean +/- std |
|---|---|
| dirty | 0.664 +/- 0.207 |
| frozen | 0.641 +/- 0.213 |
| finetune | 0.564 +/- 0.265 |
| fresh | 0.522 +/- 0.282 |

Doing nothing wins on average; training degrades the wiggle monotonically in 3 of 5 folds.
M0/M1/M2 improving with SG training does not mean the kinematic signature came back, on this
evidence it means the opposite. Detail and per-fold table in PROGRESS.md 2026-09-10.

**`frozen`'s spread is the only clean cube-variance measurement** -- it never trains. Against
it, `finetune` has lower std on M0/M2 and the best mean on all three moments: "more stable and
not worse," not "better." Design caveat: seed was fixed to isolate cube variance, but notebook
10 V1 vs V4 showed `fresh` alone can move 66 pp between identical runs on ONE cube, so `fresh`'s
column here mixes cube and training variance.

---

## 12-sg-spectral-context: does feeding neighbouring channels fix the SG wiggle collapse

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| (author-reported, no push commit) | 2026-09-11 | `07f047f` | -- | complete, 4 arms (`k=0/1/2/3`), all `fresh` init, checkpoints downloaded and stored (`models/12-spectral/`) | no dedicated `results/12-*/` archive yet -- notebook itself not yet pulled into a run folder, see open item below |

Direct response to 11-sg-loo's wiggle follow-up: SG training improves moments and damages the
wiggle. Same split as notebook 10 (`train {9015, 9019, 9032}`, `val 9025`, `holdout 9074`),
`n_neighbors=k` (`in_channels=2k+1`) fed to a fresh U-Net, scored on the wiggle directly rather
than assumed from moments.

| k | in_channels | epoch | val_loss | PSNR | M0/M1/M2 | wiggle resid_r |
|---|---|---|---|---|---|---|
| 0 (control) | 1 | 46 | 0.003133 | 30.053 | -26.5 / +4.2 / -44.4 | 0.366 |
| 1 | 3 | 33 | 0.002004 | 32.828 | +29.2 / +31.0 / +62.5 | 0.590 |
| 2 | 5 | 29 | 0.001779 | 33.576 | +61.4 / +31.4 / +67.6 | 0.506 |
| **3** | 7 | 60 | 0.001366 | **35.114** | **+53.0 / +38.8 / +70.1** | **0.681** |

**`k=3` beats that holdout's own dirty (0.594) -- the first SG-trained arm in the project to
exceed doing nothing on the wiggle, not just approach it.** Not cleanly monotonic (`k=2` dips
below `k=1`), but the trend is upward. All four checkpoints verified single-root,
strict-`load_state_dict`-compatible at their `in_channels`, epochs matching the run log.

**Does not transfer to the SG v2 cube.** `sg_k3_fresh` scored on the separate SG v2 cube
(2026-09-11, `wiggle_domain_split.py`) gets 0.7053, worse than the line-emission-trained
`winner_aug` (0.7603) there, both below that cube's dirty at 0.862. Same checkpoint, opposite
verdict depending on which cube -- the variable that matters is how degraded the specific
cube's dirty already is, not which model or domain trained it. This triggered the full
six-test chain closing Phase J; see `context.md`'s Phase J closing summary and `PROGRESS.md`
2026-09-11 (six entries, rendering audit through the headroom scatter) for the complete
picture, not just this notebook's own holdout number.

**Open item:** unlike every other notebook in this project, 12 has never been archived per
RULES.md #10 (the notebook itself, outputs intact, plus log/figures/README under
`results/12-sg-spectral-context/v<N>_.../`). The numbers above are attributed only through
`models/README.md` and `PROGRESS.md`. Close this before Block 3's cleanup pass at the latest.

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
