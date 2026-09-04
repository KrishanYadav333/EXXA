"""
The PSNR/SSIM the notebook-10 results table printed as `nan` for the `frozen` arm.

`frozen` never goes through `train_unet`, so it never got that function's final fixed-metric
evaluation, and the results cell hardcoded nan. That left the hole exactly where the baseline
belongs: without it, "training improved PSNR from X to 30.86" cannot be stated at all.

Scored with `val_metrics` itself, not a reimplementation, on the same val split (seed 42,
n_holdout=1, val_fraction=0.25) -- a baseline measured with a different metric would not be
comparable (RULES.md #4).

Run: PYTHONPATH=.. python3 experiments/frozen_val_metrics.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cube_split import split_cubes
from src.data.fits_cube_dataset import FITSChannelDataset
from src.training.sweep import val_metrics
from src.training.architectures import build_model

DATA = "self-gravitating cube and dirty cube/sg_synth"
SEED, TARGET_SIZE, N_SAMPLES = 42, 256, 120
WINNER = dict(base_channels=48, channel_multipliers=(1, 2, 4, 8))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

_, val_cubes, _ = split_cubes(data_dir=DATA, n_holdout=1, val_fraction=0.25, seed=SEED)
val_ds = FITSChannelDataset(val_cubes, n_samples=N_SAMPLES, target_size=TARGET_SIZE,
                            seed=SEED, subtract_continuum=False, verbose=False)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
print(f"val: {len(val_ds)} items from {[c['folder'] for c in val_cubes]}\n")

ARMS = {
    "frozen":   "models/08-seeds/winner_aug_seed43.pth",
    "finetune": "models/10-sg/sg_finetune.pth",
    "fresh":    "models/10-sg/sg_fresh.pth",
}

rows = {}
for name, path in ARMS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_model("unet", base_channels=WINNER["base_channels"],
                      channel_multipliers=WINNER["channel_multipliers"],
                      use_beam=False, n_neighbors=0, out_channels=1, latent_dim=128).to(device)
    missing, unexpected = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not missing and not unexpected, f"{name}: state dict mismatch"
    m = val_metrics(net, val_loader, device, use_beam=False, arch="unet")
    rows[name] = m
    print(f"{name:9s} PSNR {m['psnr']:7.3f}  SSIM {m['ssim']:.5f}  MSE {m['mse']:.6f}")

print()
base = rows["frozen"]["psnr"]
for n in ("finetune", "fresh"):
    print(f"{n:9s} {rows[n]['psnr'] - base:+.3f} dB against frozen")

with open("results/self-gravitating/frozen_val_metrics.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nsaved -> results/self-gravitating/frozen_val_metrics.json")
