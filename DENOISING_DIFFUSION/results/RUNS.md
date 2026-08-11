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

**v18 is the best 05 run to date, and the first with every moment positive on every cube.**
M0 +81.2% ±12.6, M1 +31.5% ±9.2, M2 +19.9% ±18.8, 5/5 cubes positive on all three, against
V12's +69.8 / +17.5 / +20.1. PSNR 35.94, +2.99 dB over V12. The sweep-time 37.109 dB
reproduces at −1.166 dB, consistent with the seed-49 explanation.

The negative M2 values that dogged earlier runs are gone. They were the unmasked metric
scoring empty sky, where dispersion is a ratio with a vanishing denominator; the signal
mask removed them without changing any ranking.

Caveat on provenance: the Kaggle notebook was a **stale copy predating `1d3022b`** — 23
cells, no `SEEDS`, no `CONFIGS`, no V12-checkpoint restore. So v18 is a single seed on one
configuration, not the 3-seed comparison the current notebook runs, and its "vs V12" column
compares against V12's *published* numbers rather than a re-scored V12 checkpoint. The push
`b24c84d` also reverted `c1f21e1` and `718ca3e` in the repo; restored in `f0a2e31`.

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

`unet_loss.png` was dropped: byte-identical (sha `c2ae97f9`) to the continuum-era copy
already archived, so it was a stray download rather than a 05 artifact.

**No version of 05 has beaten V12 on the scientific metric.** Both v16 and v17 print
*"NOT a clean improvement — V12 stays the reference"* as their own verdict, with M0 falling
from V12's +69.8% to +64.4% then +48.2%. **V12 remains the reference model.**

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
| ? | 2026-08-11 14:55 | `41df52d` | sweep + 60-epoch train complete; improved holdout crashed at the CSV write | [`run_.../`](06-ddpm-line-emission/run_2026-08-11T1455_41df52d/) |

The Kaggle version number is not in the log; the folder is named by the code commit.

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
| 4 | 2026-08-02 04:21 | `1ca611f` | `ba328ea` | **complete** — 4 arms × 3 seeds = 12/12 | [`v4_.../`](08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/) |

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
