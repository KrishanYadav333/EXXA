#!/usr/bin/env python
"""
src/evaluation/moment_maps.py
=============================
Moment-map generation for line-emission cubes via the `bettermoments` package.

Moment maps are the actual scientific data product (mentor, 2026-06-18): an individual
denoised channel means little; the value is in the collapsed cube. We generate:
  - Moment 0 : integrated intensity            (collapse_zeroth)
  - Moment 1 : intensity-weighted velocity     (collapse_first)   -- the rotation/kinematics map
  - Moment 2 : intensity-weighted dispersion    (collapse_second)

The ultimate denoising test: build moment maps from a DIRTY cube vs the model's DENOISED cube
and compare against the CLEAN-cube moment maps. This module provides the generation + a
clean/dirty comparison visualization to confirm the package works.

bettermoments API (v1.10):
  data, velax = bm.load_cube(path)              # data (C,H,W), velax (C,)
  rms          = bm.estimate_RMS(data, N=...)   # noise from N edge channels
  Mx, dMx      = bm.collapse_{zeroth,first,second}(velax, data, rms)
"""

import os
from typing import Optional, Tuple

import numpy as np


def generate_moment_maps(
    fits_path: str,
    rms_n_channels: int = 5,
    save_path: Optional[str] = None,
    data_velax: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
    clip_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Moment 0/1/2 maps for a line-emission FITS cube.

    Args:
        fits_path: path to the FITS cube (ignored if `data_velax` is supplied).
        rms_n_channels: number of edge channels used to estimate the noise RMS.
        save_path: if given, save a 3-panel (M0|M1|M2) visualization here.
        data_velax: optional pre-loaded (data, velax) to collapse instead of reading
            from disk — used to make moment maps from an in-memory DENOISED cube.
        title: optional suptitle for the figure.
        clip_sigma: channels below `clip_sigma * rms` are zeroed before collapsing.
            `bettermoments.collapse_{first,second}` divide by the per-pixel flux sum
            with no threshold of their own; on a spectral background where flux
            oscillates near zero, that sum crosses zero and the moment blows up
            (M2 is a sqrt of a ratio, so it is unbounded, not just noisy). One cube's
            worth of these pixels is enough to saturate a shared colour scale and to
            dominate a mean-absolute-difference metric taken over the whole map. Set
            to 0 to disable and reproduce the old unclipped behaviour.

    Returns:
        (moment0, moment1, moment2) as numpy arrays, each shape (H, W).
    """
    import bettermoments as bm

    if data_velax is not None:
        data, velax = data_velax
    else:
        data, velax = bm.load_cube(fits_path)

    data = np.asarray(data)
    velax = np.asarray(velax)

    # noise estimate from the line-free edge channels
    rms = bm.estimate_RMS(data=data, N=rms_n_channels)

    if clip_sigma > 0:
        data = np.where(np.abs(data) >= clip_sigma * rms, data, 0.0)

    moment0, _ = bm.collapse_zeroth(velax=velax, data=data, rms=rms)
    moment1, _ = bm.collapse_first(velax=velax, data=data, rms=rms)
    moment2, _ = bm.collapse_second(velax=velax, data=data, rms=rms)

    if save_path is not None:
        _plot_moments(moment0, moment1, moment2, save_path, title=title or os.path.basename(fits_path))

    return moment0, moment1, moment2


def _plot_moments(m0, m1, m2, save_path, title=""):
    """Save a 3-panel M0/M1/M2 figure with appropriate colormaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (m0, "Moment 0 (integrated intensity)", "inferno"),
        (m1, "Moment 1 (velocity)", "RdBu_r"),          # diverging: rotation red/blue
        (m2, "Moment 2 (dispersion)", "viridis"),
    ]
    for a, (m, name, cmap) in zip(ax, panels):
        im = a.imshow(m, origin="lower", cmap=cmap)
        a.set_title(name); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def _plot_clean_vs_dirty(clean_maps, dirty_maps, save_path, tag=""):
    """2x3 grid: clean (top) vs dirty (bottom) moment maps, shared scale per column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names = ["Moment 0 (intensity)", "Moment 1 (velocity)", "Moment 2 (dispersion)"]
    cmaps = ["inferno", "RdBu_r", "viridis"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    for col in range(3):
        cm, dm = clean_maps[col], dirty_maps[col]
        # shared color scale per column so clean/dirty are directly comparable
        vmin = float(np.nanmin([np.nanmin(cm), np.nanmin(dm)]))
        vmax = float(np.nanmax([np.nanmax(cm), np.nanmax(dm)]))
        for row, m in [(0, cm), (1, dm)]:
            im = ax[row, col].imshow(m, origin="lower", cmap=cmaps[col], vmin=vmin, vmax=vmax)
            ax[row, col].axis("off")
            fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.04)
        ax[0, col].set_title(names[col])
    ax[0, 0].set_ylabel("CLEAN", fontsize=13)
    ax[1, 0].set_ylabel("DIRTY", fontsize=13)
    # ylabels get cleared by axis('off'); add text labels instead
    fig.text(0.02, 0.74, "CLEAN", fontsize=14, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.26, "DIRTY", fontsize=14, fontweight="bold", rotation=90, va="center")
    fig.suptitle(f"Moment maps — clean vs dirty  {tag}", fontweight="bold", fontsize=14)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    plt.savefig(save_path, dpi=140)
    plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test bettermoments on a held-out clean/dirty pair")
    ap.add_argument("--clean", default="data/Line Emission Data/run_0002_00560_rt_00/run_0002_00560_rt_00_clean.fits")
    ap.add_argument("--dirty", default="data/Line Emission Data/run_0002_00560_rt_00/run_0002_00560_rt_00_dirty.fits")
    ap.add_argument("--out", default="results/moment_maps_test.png")
    args = ap.parse_args()

    print("Generating moment maps (clean)...")
    cm = generate_moment_maps(args.clean)
    print("Generating moment maps (dirty)...")
    dm = generate_moment_maps(args.dirty)

    # quantitative sanity: how different are clean vs dirty moment maps?
    def stats(name, c, d):
        diff = np.abs(c - d)
        denom = np.nanmax(np.abs(c)) or 1.0
        print(f"  {name}: clean[min={np.nanmin(c):.4g}, max={np.nanmax(c):.4g}]  "
              f"dirty[min={np.nanmin(d):.4g}, max={np.nanmax(d):.4g}]  "
              f"mean|diff|={np.nanmean(diff):.4g} ({100*np.nanmean(diff)/denom:.1f}% of clean max)")

    print("\nClean vs dirty moment-map comparison:")
    stats("M0", cm[0], dm[0])
    stats("M1", cm[1], dm[1])
    stats("M2", cm[2], dm[2])

    tag = os.path.basename(os.path.dirname(args.clean))
    _plot_clean_vs_dirty(cm, dm, args.out, tag=tag)
    print(f"\nsaved comparison -> {args.out}")
