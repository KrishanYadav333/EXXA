"""
Figures for the SG training result. Notebook 10 produced a table and no images.

Two figures, rebuilt after V4 withdrew V1's headline:

  1. The result that REPRODUCED: clean / dirty / frozen / finetune on the holdout cube.
     `fresh` is deliberately not shown here -- the only weights we hold are V1's, which are
     its good draw, and V4 showed that draw is not representative (M0 +5.0 -> -61.1).
  2. The instability itself: every arm's V1 vs V4 result. This is now the more useful figure,
     and it is the reason figure 1 shows what it shows.

Run: PYTHONPATH=.. python3 experiments/figures_sg_training.py
"""
import os, sys, json, math, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as Fn
from astropy.io import fits

from src.models.unet import UNet
from src.evaluation.moment_maps import generate_moment_maps, signal_mask, moment_improvement

HOLD = "self-gravitating cube and dirty cube/sg_synth/run_9074_00025_rt_00"
ARMS = {
    "frozen":   "models/08-seeds/winner_aug_seed43.pth",
    "finetune": "models/10-sg/sg_finetune.pth",
}
BASE, MULTS, SIZE, FRAC = 48, (1, 2, 4, 8), 256, 0.05
OUT = "results/self-gravitating"
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def denoise(path, cube):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UNet(in_channels=1, out_channels=1, base_channels=BASE,
               channel_multipliers=MULTS, time_emb_dim=128, num_res_blocks=2,
               groups=math.gcd(8, BASE), beam_dim=0).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not miss and not unexp, f"{path}: state dict mismatch"
    net.eval()
    C, H, W = cube.shape
    los = cube.reshape(C, -1).min(axis=1); his = cube.reshape(C, -1).max(axis=1)
    rng = his - los
    out = np.empty_like(cube, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, C, 8):
            blk = cube[s:s + 8].astype(np.float64)
            lo, hi = los[s:s + 8], his[s:s + 8]
            den = np.where((hi - lo) > 0, hi - lo, 1)[:, None, None]
            t = torch.from_numpy((blk - lo[:, None, None]) / den)[:, None].float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(b.shape[0]):
                out[s + k] = b[k] * rng[s + k] + los[s + k] if rng[s + k] > 0 else los[s + k]
    return out


clean_p = f"{HOLD}/run_9074_00025_rt_00_clean.fits"
dirty_p = f"{HOLD}/run_9074_00025_rt_00_dirty.fits"
with fits.open(clean_p, memmap=True) as h:
    hdr, clean = h[0].header, h[0].data[:]
with fits.open(dirty_p, memmap=True) as h:
    dirty = h[0].data[:]
velax = (hdr["CRVAL3"] + (np.arange(clean.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
print(f"holdout {clean.shape}, dv {hdr['CDELT3']:.4f} km/s, device {dev}")

mom = {"clean": generate_moment_maps("", data_velax=(clean.astype(np.float64), velax)),
       "dirty": generate_moment_maps("", data_velax=(dirty.astype(np.float64), velax))}
for name, path in ARMS.items():
    t0 = time.time()
    den = denoise(path, dirty)
    mom[name] = generate_moment_maps("", data_velax=(den.astype(np.float64), velax))
    imp = moment_improvement(mom["clean"], mom["dirty"], mom[name], frac=FRAC)
    print(f"  {name:9s} M0 {imp['M0']:+7.1f} M1 {imp['M1']:+7.1f} M2 {imp['M2']:+7.1f} "
          f"({(time.time()-t0)/60:.1f} min)")

mask = signal_mask(mom["clean"][0], frac=FRAC)
scores = {n: moment_improvement(mom["clean"], mom["dirty"], mom[n], frac=FRAC) for n in ARMS}

# ---------------------------------------------------------------- figure 1: moment maps
cols = ["clean", "dirty", "frozen", "finetune"]
titles = {"clean": "clean (truth)", "dirty": "dirty (input)",
          "frozen": "frozen\nno SG training", "finetune": "finetune\ntrained on SG data"}
rows = [("M0", 0, "inferno"), ("M1", 1, "RdBu_r"), ("M2", 2, "viridis")]

# Crop to the mask's bounding box. The disk covers a small part of the 301x301 field, so
# plotting the whole frame leaves ~95% blank and the spiral structure invisible -- which is
# the entire thing these panels exist to show.
_ys, _xs = np.where(mask)
_pad = 12
y0, y1 = max(0, _ys.min() - _pad), min(mask.shape[0], _ys.max() + _pad + 1)
x0, x1 = max(0, _xs.min() - _pad), min(mask.shape[1], _xs.max() + _pad + 1)
print(f"cropping to mask bbox: y {y0}-{y1}, x {x0}-{x1} "
      f"(from {mask.shape}, {100*mask.sum()/mask.size:.1f}% masked in)")

def _cut(a):
    return a[y0:y1, x0:x1]

fig, ax = plt.subplots(3, 4, figsize=(14.5, 10.5))
for ri, (label, mi, cmap) in enumerate(rows):
    ref = _cut(np.where(mask, mom["clean"][mi], np.nan))
    if label == "M1":
        lim = np.nanpercentile(np.abs(ref), 98)
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = np.nanpercentile(ref, 2), np.nanpercentile(ref, 98)
    for ci, c in enumerate(cols):
        a = ax[ri, ci]
        im = a.imshow(_cut(np.where(mask, mom[c][mi], np.nan)), cmap=cmap,
                      vmin=vmin, vmax=vmax, origin="lower")
        a.set_xticks([]); a.set_yticks([])
        if ri == 0:
            a.set_title(titles[c], fontsize=12.5, pad=8)
        if ci == 0:
            a.set_ylabel(label, fontsize=16, labelpad=10)
        if c in scores:
            v = scores[c][label]
            a.text(0.5, -0.075, f"{v:+.1f}%", transform=a.transAxes, ha="center",
                   fontsize=12.5, color=("#1a7f37" if v > 0 else "#b3261e"), weight="bold")
    # one shared colourbar per row, since the row shares vmin/vmax
    plt.colorbar(im, ax=list(ax[ri, :]), fraction=0.018, pad=0.012)

fig.suptitle("Training on self-gravitating data closes the domain gap\n"
             "held-out disk; % = improvement over the dirty cube, signal-masked (frac=0.05). "
             "Reproduced in Kaggle V1 and V4.",
             fontsize=13.5, y=0.985)
p1 = f"{OUT}/sg_training_moments.png"
plt.savefig(p1, dpi=130); plt.close()
print("saved ->", p1)

# --------------------------------------------------------- figure 2: the instability
# V1 and V4 ran identical settings: same seed, same split, same data. frozen is inference
# only so it MUST reproduce, and does -- that is the control that makes the rest readable.
V1 = {"frozen": (-10.3, -0.6, -43.6), "finetune": (-6.5, +36.5, +15.6), "fresh": (+5.0, +21.8, +26.0)}
V4 = {"frozen": (-10.3, -0.6, -43.6), "finetune": (-7.0, +36.7, +15.3), "fresh": (-61.1, -27.3, -42.2)}
names = ["frozen", "finetune", "fresh"]
moments = ["M0", "M1", "M2"]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
for k, m in enumerate(moments):
    a = axes[k]
    x = np.arange(len(names))
    a.bar(x - 0.19, [V1[n][k] for n in names], width=0.38, label="run 1", color="#2f5f96")
    a.bar(x + 0.19, [V4[n][k] for n in names], width=0.38, label="run 2", color="#c9603f")
    a.axhline(0, color="black", lw=1)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=11)
    a.set_title(m, fontsize=13)
    a.grid(axis="y", alpha=0.25)
    for i_, n in enumerate(names):
        d = V4[n][k] - V1[n][k]
        if abs(d) > 5:
            a.annotate("", xy=(i_ + 0.19, V4[n][k]), xytext=(i_ - 0.19, V1[n][k]),
                       arrowprops=dict(arrowstyle="->", color="#b3261e", lw=1.6))
            a.text(i_, max(V1[n][k], V4[n][k]) + 4, f"{d:+.0f} pp",
                   ha="center", fontsize=10.5, color="#b3261e", weight="bold")
axes[0].set_ylabel("improvement over dirty (%)")
axes[0].legend(frameon=False, fontsize=10.5)
fig.suptitle("Two runs, identical settings (same seed, split and data)\n"
             "frozen reproduces exactly and finetune to within 0.5 pp -- fresh moves 66 pp, "
             "so the variance is in that arm, not the setup",
             fontsize=12.5, y=1.02)
plt.tight_layout()
p2 = f"{OUT}/sg_training_run_to_run.png"
plt.savefig(p2, dpi=140, bbox_inches="tight"); plt.close()
print("saved ->", p2)
