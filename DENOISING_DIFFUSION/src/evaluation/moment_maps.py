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
    tile_bytes: int = 8 << 20,
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
        tile_bytes: size of one horizontal strip. The collapse is done in strips because
            `bettermoments` is extremely allocation-hungry. Measured on a 201x600x600
            float32 cube (0.27 GiB): `collapse_zeroth` peaks at 2x the cube,
            `collapse_first` at 8x and `collapse_second` at 12x, for 3.54 GiB of transient
            per call — and there are three calls per cube. That, not the cubes themselves,
            is what exhausted the Kaggle host and killed the kernel with no traceback
            across three runs; trimming the 289 MB cubes around it was never going to be
            enough.

            All three collapses are per-pixel along the spectral axis, so strips are exact
            rather than an approximation. Peak scales linearly with this value (it is
            amplified ~13x by the collapse), and strips are also *faster* — better cache
            locality:

                untiled   3.54 GiB peak   5.24 s
                8 MB      0.11 GiB peak   1.97 s      <- default

            M0 comes out bit-identical and the NaN pattern is unchanged. Over the brightest
            10% of pixels — the region anything is reported from — M1 agrees to 9e-6
            relative and M2 to 5e-11. Larger relative differences occur only on near-zero
            background pixels, where the denominator is ~0 and the value is meaningless
            anyway; that is the instability `clip_sigma` exists to suppress.

            Set to 0 to collapse in one shot.

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
    C, H, W = data.shape

    # One global noise estimate from the line-free edge channels. It must be computed on
    # the whole cube and shared by every tile: a per-tile rms would make the clip
    # threshold, and therefore the moments, depend on how the image was subdivided.
    rms = bm.estimate_RMS(data=data, N=rms_n_channels)

    rows = H if tile_bytes <= 0 else max(1, min(H, int(tile_bytes // (C * W * 4))))

    moment0 = np.empty((H, W), dtype=np.float64)
    moment1 = np.empty((H, W), dtype=np.float64)
    moment2 = np.empty((H, W), dtype=np.float64)

    for y0 in range(0, H, rows):
        y1 = min(y0 + rows, H)
        blk = np.asarray(data[:, y0:y1, :])
        if clip_sigma > 0:
            blk = np.where(np.abs(blk) >= clip_sigma * rms, blk, 0.0)
        moment0[y0:y1], _ = bm.collapse_zeroth(velax=velax, data=blk, rms=rms)
        moment1[y0:y1], _ = bm.collapse_first(velax=velax, data=blk, rms=rms)
        moment2[y0:y1], _ = bm.collapse_second(velax=velax, data=blk, rms=rms)
        del blk

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


# --------------------------------------------------------------------------- #
# Scoring: where a moment map should actually be compared                      #
# --------------------------------------------------------------------------- #
SIGNAL_FRAC = 0.05   # of the clean M0 peak


def signal_mask(clean_m0: np.ndarray, frac: float = SIGNAL_FRAC) -> np.ndarray:
    """Pixels where the TRUTH has real emission.

    Defined from the clean cube alone, so it is identical for every method being
    compared and cannot be tuned per model. This uses ground truth and is therefore a
    benchmark construct: on real data the mask would have to come from the observation.
    """
    finite = np.isfinite(clean_m0)
    if not finite.any():
        return finite
    peak = float(np.nanmax(np.abs(clean_m0[finite])))
    if not np.isfinite(peak) or peak <= 0:
        return finite
    return finite & (np.abs(clean_m0) > frac * peak)


def moment_improvement(clean, dirty, denoised, frac: float = SIGNAL_FRAC) -> dict:
    """
    Percent improvement of `denoised` over `dirty` against `clean`, per moment.

    `clean`, `dirty`, `denoised` are each a (M0, M1, M2) triple.

    Scored over the signal mask rather than the whole map. Averaging over every finite
    pixel — the previous behaviour — lets empty sky dominate, and dispersion in a pixel
    with no line is not a quantity anyone reports. It penalised M2 hardest because M2 is
    a ratio whose denominator vanishes exactly there.

    The mask does not change which method wins. On a synthetic benchmark with three
    denoisers of known quality the ranking is identical at every threshold from 0.5% to
    10% of peak; only the scale moves (M2 for the best model, 54.1% unmasked vs 68-77%
    masked). `frac` is deliberately taken from a plateau rather than its maximum.

    Returns per-moment improvements plus `n_px`, and the unmasked values under
    `M0_all`/`M1_all`/`M2_all` so the effect of masking is always visible alongside it.
    """
    mask = signal_mask(clean[0], frac)
    out = {"n_px": int(mask.sum()), "frac": frac}
    for i, nm in enumerate(("M0", "M1", "M2")):
        cl, di, no = np.asarray(clean[i]), np.asarray(dirty[i]), np.asarray(denoised[i])
        base = np.isfinite(cl) & np.isfinite(di) & np.isfinite(no)
        for suffix, m in ((nm, base & mask), (nm + "_all", base)):
            if not m.any():
                out[suffix] = float("nan")
                continue
            dd = float(np.abs(cl[m] - di[m]).mean())
            nn = float(np.abs(cl[m] - no[m]).mean())
            out[suffix] = 100.0 * (1.0 - nn / dd) if dd > 0 else float("nan")
    return out
