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

| Ver | Date (UTC) | code | push | PSNR | Holdout M0 | Note |
|----:|---|---|---|---|---|---|
| — | 2026-06-24 19:42 | `a4ac1fe` | `27fa388` | 26.46 | — | first run, 30-epoch baseline |
| #5 | 2026-06-26 17:15 | `f548947` | `d307a29` | 28.16 | — | |
| 6 | 2026-06-27 16:00 | `5ed8fc6` | `01aaf88` | 31.73 | — | |
| 7 | 2026-07-02 13:59 | `662fef5` | `95d9034` | 31.02 / 33.18 | — | outputs also synced into folder 06 (`d4ff643`) |
| 9 | 2026-07-02 18:39 | `7fa5fce` | `d0adb14` | 32.31 / 33.04 | — | ablation; outputs synced to folder 06 (`5933292`) |
| **12** | 2026-07-10 19:28 | `c61d112` | `82abd23` | **32.95** | **+69.8% ±15.2** | **the published V12 reference** |
| 15 | 2026-07-26 04:42 | `09bfcb9` | `0307cc9` | 33.70 / 34.94 | +59.8% ±33.0 | 12-run sweep + beam A/B |
| 16 | 2026-07-30 03:34 | `ef0a37b` | `227d2fc` | **30.28** | +64.4% ±14.6 (−5.4 pt) | sweep winner failed to reproduce: 37.11 → 30.28 |
| 17 | 2026-08-02 04:13 | `227d2fc` | `7305455` | **29.92** | **+48.2% ±12.6 (−21.6 pt)** | worse again |
| — | 2026-07-30 00:48 | `2edd875` | — | — | — | sweep/correlation/beam analysis, committed directly → [`analysis_.../`](05-unet-line-emission/analysis_2026-07-30T0048_2edd875/) |

**No version of 05 has beaten V12 on the scientific metric.** The sweep found 37.11 dB;
v16 retrained it to 30.28 and v17 to 29.92, with M0 falling from V12's +69.8% to +64.4%
then +48.2%. Both later versions print *"NOT a clean improvement — V12 stays the reference"*
as their own verdict. **V12 remains the reference model**, and none of v15–v17 artifacts
were ever downloaded.

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
| 2 | 2026-07-31 19:04 | `eb03589` | `7dbd17b` | completed. Headline M0 +69.8% (V12) vs +11.7% (best classical) → +58.1 pp. | not downloaded |

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
