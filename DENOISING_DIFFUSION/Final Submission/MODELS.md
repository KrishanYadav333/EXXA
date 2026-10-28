# Trained models

The final denoiser is an average of four U-Net checkpoints (see the report in `BLOG.md`, section "Method"). Each was trained as a correction on top of a 3D Wiener filter, with spectral context K = 15 channels.

| file | what it is |
|---|---|
| `unified_r4n_k15_long.pth` | long training run, seed 3 |
| `unified_r4n_k15_long_s11.pth` | the same run, seed 11 |
| `unified_r4n_k15_long_s23.pth` | the same run, seed 23 |
| `unified_r8_m1_r6.pth` | fine-tuned with an extra penalty on the line centre (M1), weight 0.01 |

Inference uses all four, the do-no-harm gate (`--gate 40,120,0.25`), two Wiener passes and the size normalisation, which are the defaults of `experiments/alma_unified.py`.

## Availability

The weights are **not in this repository**. They were trained on data provided by the project mentors, and are released only with the mentors' permission. Until then, `experiments/train_unified.py` and the analytic-disk generator (`experiments/make_analytic_gi_cubes.py`) reproduce the training from scratch on public inputs, apart from the mentor-provided cubes.

Checkpoint format: a PyTorch dictionary with the keys `K`, `arm`, `base`, `model_state_dict` and `args`.
