# 08 — code commit `eb03589` (2026-07-31)

Two downloads of 08 from the same code commit, both showing the four arms across seeds
42/43/44 with v4's per-run PSNRs (unchanged across versions: every run after the first
reuses the same twelve checkpoints).

| file | cells | note |
|---|---|---|
| `08-seeds-and-augmentation.ipynb` | 33 | matches v4's cell structure |
| `08-seeds-and-augmentation__35cell-variant.ipynb` | 35 | two extra cells; previously sat at the repo root |

The 35-cell file lived beside the live notebook as `08-seeds-and-augmentation.ipynb` — the
duplicate-name copy RULES.md #7 exists to prevent. The Kaggle-linked notebook is
`08-seeds-and-augmentationa369c56f22.ipynb`, and only that file receives runs.

The exact Kaggle version numbers are not recorded in either notebook's output; `v2_` is the
folder's ordering label, not a verified version.

## Files

| file | what |
|---|---|
| `08-seeds-and-augmentation.ipynb` | the notebook as executed, outputs intact |
| `run_log.txt` | full stdout, **extracted from the notebook's cell outputs** — this run's log was never downloaded from Kaggle, so the notebook was its only copy |
| `figure_cell24.png` | the per-seed spread plot |

## Why this run still matters

v2 and v4 are the **same twelve checkpoints**. v4 reused v2's weights and re-scored them, so
their PSNR/SSIM/MSE rows are byte-identical. Only the evaluation code differs:

```
v2 (eb03589)   raw:  no noise clip, no signal mask
v4 (1ca611f)   clip: 3-sigma noise clip added
```

**v2's moment scores are the ones comparable to V12.** V12's published +69.8 / +17.5 / +20.1
was measured on this same raw metric, so v2 is the only seed-validated result in the project
that can be placed beside the reference checkpoint. On that footing `winner_aug` scores
**M0 +87.5% ±4.2** — the highest M0 anywhere here, at the lowest variance, over 3 seeds.

**v2's artifact diagnostics are not usable.** It reports 0.0% invented structure for all four
arms, which is RULES.md #8: the background mask selected zero pixels, so the detector could
not fire. Its overshoot figures (0.89–0.93, all *below* 1.0) are the same artefact. Use v4's
22.3–39.0% instead.

So: take v2's moments, take v4's artifact rates, and never mix a v2 moment number with a v4
one.
