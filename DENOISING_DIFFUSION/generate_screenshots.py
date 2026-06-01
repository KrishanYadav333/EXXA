"""
Generate screenshots for GSoC proposal PDF.

Produces:
  1. q_sample noise progression plot  -> screenshots/noise_progression.png
  2. compare_denoisers PSNR/SSIM table -> screenshots/metrics_comparison.png

Run from DENOISING_DIFFUSION/:
    python ../generate_screenshots.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DENOISING_DIFFUSION"))

from src.models.noise_scheduler import NoiseScheduler


# inline metrics (no dependency on untracked src/utils)
def _mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

def _psnr(a, b):
    err = _mse(a, b)
    return float("inf") if err == 0 else float(10.0 * np.log10(1.0 / err))

def _ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a.astype(np.float64), b.astype(np.float64), data_range=1.0))

def compare_denoisers(noisy, target, outputs):
    all_outputs = {"noisy": noisy, **outputs}
    return {
        name: {"psnr": _psnr(img, target), "ssim": _ssim(img, target), "mse": _mse(img, target)}
        for name, img in all_outputs.items()
    }

os.makedirs("screenshots", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Noise progression plot
# ---------------------------------------------------------------------------

def synthetic_disk(size=64):
    """Simple synthetic protoplanetary disk: bright center + faint ring."""
    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2)
    disk = np.exp(-r**2 / (2 * 8**2))           # central star
    ring = np.exp(-(r - 20)**2 / (2 * 3**2)) * 0.4  # ring at r=20
    img = disk + ring
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float32)

scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear")
img = synthetic_disk(64)
x0 = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)

timesteps = [0, 250, 500, 750, 999]
fig, axes = plt.subplots(1, len(timesteps), figsize=(14, 3))
fig.patch.set_facecolor("#0d1117")

for ax, t in zip(axes, timesteps):
    t_tensor = torch.tensor([t])
    noisy, _ = scheduler.q_sample(x0, t_tensor)
    noisy = noisy.squeeze().numpy()
    ax.imshow(noisy, cmap="inferno", origin="lower")
    ax.set_title(f"t = {t}", color="white", fontsize=11)
    ax.axis("off")

fig.suptitle("Forward Diffusion: q_sample at increasing timesteps",
             color="white", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("screenshots/noise_progression.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved: screenshots/noise_progression.png")

# ---------------------------------------------------------------------------
# 2. compare_denoisers table
# ---------------------------------------------------------------------------

from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(42)
clean = synthetic_disk(64)
noisy = np.clip(clean + rng.normal(0, 0.15, clean.shape).astype(np.float32), 0, 1)

results = compare_denoisers(noisy, clean, {
    "gaussian_σ1": gaussian_filter(noisy, sigma=1).astype(np.float32),
    "gaussian_σ2": gaussian_filter(noisy, sigma=2).astype(np.float32),
    "gaussian_σ3": gaussian_filter(noisy, sigma=3).astype(np.float32),
})

methods = list(results.keys())
psnr_vals = [results[m]["psnr"] for m in methods]
ssim_vals = [results[m]["ssim"] for m in methods]
mse_vals  = [results[m]["mse"]  for m in methods]

fig, ax = plt.subplots(figsize=(8, 2.4))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.axis("off")

col_labels = ["Method", "PSNR (dB) ↑", "SSIM ↑", "MSE ↓"]
rows = [
    [m, f"{p:.2f}", f"{s:.4f}", f"{e:.6f}"]
    for m, p, s, e in zip(methods, psnr_vals, ssim_vals, mse_vals)
]

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor("#161b22" if row % 2 == 0 else "#0d1117")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#30363d")
    if row == 0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")

ax.set_title("Denoiser Comparison (baseline: Gaussian filters)",
             color="white", fontsize=11, pad=12)

plt.tight_layout()
plt.savefig("screenshots/metrics_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved: screenshots/metrics_comparison.png")

print("\nDone. Screenshots saved to DENOISING_DIFFUSION/screenshots/")
