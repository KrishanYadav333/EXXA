# Trained models

The final model averages four U-Net checkpoints. Each network learns a correction on top of a 3D Wiener filter (`--arm wiener`), with a spectral context of K = 15 channels (plus and minus 1.5 km/s at 0.1 km/s per channel).

| file | how it was trained | evaluation of the run |
|---|---|---|
| `unified_r4n_k15_long.pth` | 80 rounds (24,000 steps), learning rate 1.5e-4, seed 3 | [`results/10-training-runs/r4n-k15-long_v1_2026-10-12/`](results/10-training-runs/r4n-k15-long_v1_2026-10-12/) |
| `unified_r4n_k15_long_s11.pth` | the same, seed 11 | [`results/10-training-runs/r4n-k15-long-s11_v1_2026-10-13/`](results/10-training-runs/r4n-k15-long-s11_v1_2026-10-13/) |
| `unified_r4n_k15_long_s23.pth` | the same, seed 23 | [`results/10-training-runs/r4n-k15-long-s23_v1_2026-10-13/`](results/10-training-runs/r4n-k15-long-s23_v1_2026-10-13/) |
| `unified_r8_m1_r6.pth` | 80 rounds on 256 px crops (batch 4, seed 31), then 20 rounds of fine-tuning with a line-centre (M1) penalty of weight 0.01 (learning rate 5e-5, seed 51) | [`results/10-training-runs/r8-m1-r6_v1_2026-10-15/`](results/10-training-runs/r8-m1-r6_v1_2026-10-15/) |

Exact commands: [`REPRODUCE.md`](REPRODUCE.md). Inference settings (gate `40,120,0.25`, second Wiener pass, size normalisation to a 10 px beam) are the defaults of `experiments/alma_unified.py`; see [`results/final/RECIPES.md`](results/final/RECIPES.md).

Checkpoint format: a PyTorch dictionary holding the weights (`model_state_dict`) and the settings needed to rebuild the network (`K`, `arm`, `base`, `base_channels`, `channel_multipliers`, `in_channels`, `out_channels`, `args`). Networks 1 and 2 are about 116 MB. Networks 3 and 4 are about 349 MB because they also store the optimiser, scheduler and random-number state used to resume training across Kaggle sessions; inference ignores those.

## Availability

The weights are not in this repository. They were trained on synthetic disks that are not public, and will be released when that is permitted.
