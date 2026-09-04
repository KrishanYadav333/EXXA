"""
Can the existing training pipeline actually consume the synthesized SG pairs, unchanged?

Three things have to hold before any of this is worth a GPU hour:
  1. `split_cubes` discovers the run folders and groups them leakage-safely.
  2. `FITSChannelDataset` loads items with matching dirty/clean shapes and no NaNs.
  3. The pairs are actually hard enough to be worth learning -- a model that copies its input
     should score badly, which is the failure the shipped pairs would have rewarded.

Run: PYTHONPATH=.. python3 experiments/check_sg_synth_pipeline.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.cube_split import list_cubes, split_cubes
from src.data.fits_cube_dataset import FITSChannelDataset

SYNTH = "self-gravitating cube and dirty cube/sg_synth"

print("1. discovery")
cubes = list_cubes(SYNTH)
for c in cubes:
    print(f"   run_id={c['run_id']:6s} {os.path.basename(c['folder'])}")
assert cubes, "split_cubes found nothing -- folder naming does not match run_<id>_<step>_rt_<pp>"
ids = [c["run_id"] for c in cubes]
assert len(set(ids)) == len(ids), f"run_ids collide {ids}: cube_split would group these as one disk"
print(f"   {len(cubes)} pairs, {len(set(ids))} distinct run ids\n")

print("2. split")
train, val, holdout = split_cubes(data_dir=SYNTH, n_holdout=1, val_fraction=0.25, seed=42)
print(f"   train {len(train)} | val {len(val)} | holdout {len(holdout)}")
tr_ids = {os.path.basename(c["folder"]) for c in train}
ho_ids = {os.path.basename(c["folder"]) for c in holdout}
assert not (tr_ids & ho_ids), "leakage: a cube is in both train and holdout"
print("   no cube appears in both train and holdout\n")

print("3. dataset load")
ds = FITSChannelDataset(train, n_samples=8, target_size=256, seed=42,
                        subtract_continuum=False, verbose=False)
d, c = ds[0]
print(f"   item: dirty {tuple(d.shape)} clean {tuple(c.shape)}")
assert d.shape == c.shape, "dirty/clean shape mismatch"
assert torch.isfinite(d).all() and torch.isfinite(c).all(), "non-finite values in a sample"
print(f"   dirty range [{d.min():.4f}, {d.max():.4f}]  clean range [{c.min():.4f}, {c.max():.4f}]\n")

print("4. is the task actually hard?")
# The identity baseline: predict the input. On the shipped pairs this would have scored
# almost perfectly, which is exactly why they were unusable.
mses, psnrs = [], []
for i in range(len(ds)):
    d, c = ds[i]
    mse = float(torch.mean((d - c) ** 2))
    rng = float(c.max() - c.min())
    mses.append(mse)
    if mse > 0 and rng > 0:
        psnrs.append(10 * np.log10(rng ** 2 / mse))
print(f"   identity baseline over {len(ds)} samples: MSE {np.mean(mses):.5f}, "
      f"PSNR {np.mean(psnrs):.2f} dB")
print("   (for scale: the trained U-Net reaches ~39 dB on the line-emission task, and its")
print("    identity baseline there is far below that -- a high number here would mean the")
print("    synthesized corruption is too weak to learn anything from)")

if np.mean(psnrs) > 35:
    print("\n   WARNING: identity already scores >35 dB. Corruption is too weak, raise NOISE_FRAC.")
else:
    print(f"\n   OK: identity is {np.mean(psnrs):.1f} dB, there is real signal to recover.")
