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
| — | 2026-06-24 19:42 | `a4ac1fe` | `27fa388` | 26.46 | — | first run, 30-epoch baseline |
| #5 | 2026-06-26 17:15 | `f548947` | `d307a29` | 28.16 | — | — |
| 6 | 2026-06-27 16:00 | `5ed8fc6` | `01aaf88` | 31.73 | — | — |
| 7 | 2026-07-02 13:59 | `662fef5` | `95d9034` | 31.02 / 33.18 | — | [`v7_.../`](05-unet-line-emission/v7_2026-07-02T1359_662fef5/) · also in [`for_jason/`](for_jason/) |
| 9 | 2026-07-02 18:39 | `7fa5fce` | `d0adb14` | 32.31 / 33.04 | — | [`v9_.../`](05-unet-line-emission/v9_2026-07-02T1839_7fa5fce/) · also in [`for_jason/`](for_jason/) |
| **12** | 2026-07-10 19:28 | `c61d112` | `82abd23` | **32.95** | **+69.8% ±15.2** | [`v12_.../`](05-unet-line-emission/v12_2026-07-10T1928_c61d112/) · also in [`for_jason/`](for_jason/) |
| 15 | 2026-07-26 04:42 | `09bfcb9` | `0307cc9` | 33.70 / 34.94 | +59.8% ±33.0 | [`v15_.../`](05-unet-line-emission/v15_2026-07-26T0442_09bfcb9/) — recovered from git |
| 16 | 2026-07-30 03:34 | `ef0a37b` | `227d2fc` | **30.28** | +64.4% ±14.6 (−5.4 pt) | [`v16_.../`](05-unet-line-emission/v16_2026-07-30T0334_ef0a37b/) |
| 17 | 2026-08-02 04:13 | `227d2fc` | `7305455` | **29.92** | **+48.2% ±12.6 (−21.6 pt)** | [`v17_.../`](05-unet-line-emission/v17_2026-08-02T0413_227d2fc/) — recovered from git |
| — | 2026-07-30 00:48 | `2edd875` | — | — | — | [`analysis_.../`](05-unet-line-emission/analysis_2026-07-30T0048_2edd875/) |

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

**No version of 05 has beaten V12 on the scientific metric.** The sweep found 37.11 dB;
v16 retrained it to 30.28 and v17 to 29.92, with M0 falling from V12's +69.8% to +64.4%
then +48.2%. Both later versions print *"NOT a clean improvement — V12 stays the reference"*
as their own verdict. **V12 remains the reference model.**

**V12's headline moment figures are not in any committed CSV.** `+69.8% ±15.2 / +17.5% ±7.8
/ +20.1% ±14.3` is quoted in `readme.md`, `MIDTERM_REPORT.md`, `context.md` and hardcoded as
a `V12 = {...}` dict in notebooks 07/08/09, but the only moment CSV in the entire git history
is the beam variant (+59.83). What was downloaded for v12 is figures, not the table.
**08 v4 supersedes it anyway**, retraining that configuration over 3 seeds for
**M0 +71.9% ±10.3** — a stronger number than the single run it replaces.

Versions 16 and 17 also report overshoot mean 0.929 and 1.220 respectively for the same
notebook, each alongside "0% invented structure" — which the floor-relative fix later showed
to be vacuous, the background mask having selected zero pixels. Neither is quotable.

## 06 — DDPM line emission

No Kaggle version has completed section 11. Attempts stopped at 88/201 channels twice
(DataParallel deadlock, fixed in `7e49e62`), then died just after sampling finished
(host RAM during the moment collapse, addressed in `aa2a41d`). **No artifacts.**

Note that the *continuum-era* folder 06 was populated by syncing **05's** outputs into it
(`d4ff643` from 05 v7, `5933292` from 05 v9, snapshots in `800474c`/`2024360`). Those are
05 results filed under a 06 directory and are not DDPM line-emission runs — do not read
them as such.

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
| 4 | 2026-08-02 04:21 | `1ca611f` | `ba328ea` | **complete** — 4 arms × 3 seeds = 12/12 | [`v4_2026-08-02T0421_1ca611f/`](08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/) |

| arm | PSNR | M0 | M2 |
|---|---|---|---|
| v12 | 37.60 ± 1.00 | 71.9 ± 10.3 | −10.5 ± 39.7 |
| winner | 37.97 ± 0.90 | 71.3 ± 16.3 | −2.7 ± 63.1 |
| winner_aug | **39.30 ± 0.46** | **75.4 ± 13.9** | 40.3 ± 30.7 |
| winner_p10 | 39.27 ± 0.48 | 71.1 ± 18.8 | −28.3 ± 83.6 |

## 09 — Architecture comparison (`09-architecture-comparison2`)

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 4 | 2026-08-02 03:13 | `29bffd4` | `cb51254` | **complete** — 24/24 sweep, 3 retrains, 15 cube evals. **Origin of the numbers.** | [`v4_2026-08-02T0313_29bffd4/`](09-architecture-comparison/v4_2026-08-02T0313_29bffd4/) |
| 6 | 2026-08-02 04:04 | `bab16d0` | `1ca611f` | resumed; reused v4's rows unchanged | same as v4 |
| 7 | 2026-08-02 17:21 | `ee491fc` | `ecaa540` | resumed; reused v4's rows unchanged | same as v4 |
| — | 2026-08-03 | `d07f8f4` | — | input not attached → swept from scratch; U-Net 8 runs in 299 min, died in autoencoder run 3 | none (partial) |

Versions 6 and 7 print identical numbers to v4 because section 6 skipped everything already
scored. That is the resume working, but it also meant the M2 noise-clip fix pulled in at v6
and v7 never actually ran — which is why `MOMENT_VER` was added in `ee491fc`.

---

## The M2 caveat

Every M2 figure above predates the noise-clip fix (`bab16d0`, `MOMENT_VER=2`).
`bettermoments.collapse_second` was fed an unmasked cube, so pure background noise reads a
dispersion near the velocity axis' own RMS width — far above a real line width, and
uniform across the whole background rather than confined to a few outliers. **M0, M1 and
PSNR are unaffected. Every M2 needs re-scoring.**
