# 06-ddpm-line-emission, Kaggle Version 13

Code `19efd47`, 2026-08-12 04:50 UTC. Completed with zero errors, and the first DDPM run to
produce moment maps at all. About 9 hours of GPU time: 2.6 h of sweep, 4.5 h of training,
1.8 h on the holdout.

## Headline

PSNR **38.180** at SSIM **0.9933**, within roughly 1 dB of the best U-Net. Then the moment
maps, on the signal-masked metric:

```
M0  −56.1% ± 152.2
M1  +15.0% ±  87.4
M2  +10.9% ±  81.7
```

against V12's +69.8 / +17.5 / +20.1. That gap between a competitive PSNR and a catastrophic
M0 is the whole point of the run. `RUNS.md` has the per-cube breakdown and the diagnosis.

## Files

The notebook as executed, with its full stdout in the cell outputs, is in
[`../../../notebooks/06-ddpm-line-emission.ipynb`](../../../notebooks/06-ddpm-line-emission.ipynb)
rather than duplicated here.

| file | what |
|---|---|
| `ddpm_objective_sweep.csv` | the four objective arms |
| `ddpm_seed_repeats.csv` | the 60-epoch run |
| `moment_map_holdout_summary_ddpm.csv` | per-cube moments, both masked and unmasked |
| `ddpm_moment_maps.png`, `moment_map_holdout_summary_ddpm.png`, `ddpm_line_emission_loss.png` | the `savefig` artifacts |
| `line_emission_ddpm_comparison.png` | five validation channels: dirty, DDPM-denoised, clean truth |
| `figure_cell*.png` | the notebook's inline display copies of the same figures |
| `manifest.json` | the `collect_outputs` record: commit `19efd47`, 2026-08-12T04:50:41Z, batch run, sha256 per file |

Every file the manifest declares is present and hash-matching, apart from the one noted below.

## Not here

`ddpm_moment_maps.npz`, 14 MB of M0/M1/M2 arrays for cube 1, which would let the moment
figure be redrawn without re-running anything. `*.npz` is gitignored across the repo. The file
is in Kaggle Version 13's Output.

The five 332 MB checkpoints are also gitignored: `ddpm_seed42.pth` at epoch 58, which is the
trained model, plus the four sweep arms. They live in Kaggle Version 13's Output too. See
[`../../../MODELS.md`](../../../MODELS.md) for the full checkpoint inventory and how to get
them.
