# 06-ddpm-line-emission — Kaggle Version 13

Code `19efd47`, 2026-08-12 04:50 UTC. **Complete, zero errors** — the first DDPM run to
produce moment maps. ~9 h GPU: sweep 2.6 h + training 4.5 h + holdout 1.8 h.

## Headline

PSNR **38.180**, SSIM **0.9933** — within ~1 dB of the best U-Net. Moment maps on the
signal-masked metric: **M0 −56.1% ± 152.2**, M1 +15.0% ± 87.4, M2 +10.9% ± 81.7, against
V12's +69.8 / +17.5 / +20.1. See RUNS.md for the per-cube breakdown and the diagnosis.

## Files

| file | what |
|---|---|
| `06-ddpm-line-emission.ipynb` | the notebook as executed, outputs intact |
| `run_log.txt` | full stdout |
| `ddpm_objective_sweep.csv` | the four objective arms |
| `ddpm_seed_repeats.csv` | the 60-epoch run |
| `moment_map_holdout_summary_ddpm.csv` | per-cube moments, masked **and** unmasked |
| `ddpm_moment_maps.png`, `moment_map_holdout_summary_ddpm.png`, `ddpm_line_emission_loss.png` | the `savefig` artifacts |
| `line_emission_ddpm_comparison.png` | five validation channels: dirty, DDPM-denoised, clean truth |
| `figure_cell*.png` | the notebook's inline display copies of the same figures |
| `manifest.json` | collect_outputs record: commit `19efd47`, 2026-08-12T04:50:41Z, Batch run, sha256 per file |

Every file the manifest declares is verified present and hash-matching, except
`ddpm_moment_maps.npz` — see below.

## Not here

`ddpm_moment_maps.npz` (14 MB) — the M0/M1/M2 arrays for cube 1, which let the moment
figure be redrawn without re-running. `*.npz` is gitignored repo-wide; the file is in
[`/06`](../../../../06/) locally and in Kaggle Version 13's Output.

Five 332 MB checkpoints — `ddpm_seed42.pth` (epoch 58, the trained model) and four sweep
arms — are gitignored and live in [`/06`](../../../../06/) locally, plus Kaggle Version 13's
Output. RULES.md #5: `collect_outputs` writes into the same Output, so naming checkpoints
there would duplicate 1.6 GB.
