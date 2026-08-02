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

| Ver | Date (UTC) | code | push | Outcome | Artifacts |
|----:|---|---|---|---|---|
| 15 | 2026-07-26 04:42 | `09bfcb9` | `0307cc9` | completed | not downloaded |
| 16 | 2026-07-30 03:34 | `ef0a37b` | `227d2fc` | completed. Holdout: M2 +23.1%±13.0 vs V12 +20.1%±14.3. Verdict *not a clean improvement* — M0 regressed. Overshoot mean **0.929**, 19% of channels >10%. | not downloaded |
| 17 | 2026-08-02 04:13 | `227d2fc` | `7305455` | completed. Same verdict. Overshoot mean **1.220**, 69% of channels >10%. | not downloaded |
| — | 2026-07-30 00:48 | `2edd875` | — | 12-run sweep + correlation analysis + beam A/B, committed directly (not a Kaggle run) | [`analysis_2026-07-30T0048_2edd875/`](05-unet-line-emission/analysis_2026-07-30T0048_2edd875/) |

Versions 16 and 17 are the *same notebook* reporting overshoot 0.929 against 1.220 — the
artifact metrics changed underneath them between runs. Both report **0% invented structure**,
which the floor-relative fix later showed to be vacuous: the background mask was selecting
zero pixels. Neither number should be quoted.

## 06 — DDPM line emission

No version has completed section 11. Attempts stopped at 88/201 channels twice
(DataParallel deadlock, fixed in `7e49e62`), then died just after sampling finished
(host RAM during the moment collapse, addressed in `aa2a41d`). **No artifacts.**

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
