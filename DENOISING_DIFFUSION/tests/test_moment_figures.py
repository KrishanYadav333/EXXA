#!/usr/bin/env python
"""
Tests for the moment-map comparison figure.

The failure this guards against is silent and cosmetic-looking but not cosmetic: if the
off-source pixels are allowed into the colour limits, M1 and M2 are fits to noise out
there and span tens of km/s, so the disk gets a few percent of the ramp and the figure
reads as static. Nothing errors -- it just stops showing the result.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.evaluation.moment_maps import (SIGNAL_FRAC, _limits, plot_moment_comparison,
                                        signal_mask)

print("=" * 60)
print("Moment Figure Tests")
print("=" * 60)

# A ring disk with solid-body-ish rotation, systemic velocity deliberately NOT zero so the
# centring logic is actually exercised.
N = 128
SYSTEMIC = 4.2
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:N, 0:N]
r = np.hypot(yy - N / 2, xx - N / 2)
th = np.arctan2(yy - N / 2, xx - N / 2)

m0 = np.exp(-((r - 28) ** 2) / (2 * 9.0 ** 2))
on_source = m0 > SIGNAL_FRAC * m0.max()
m1 = np.where(on_source, SYSTEMIC + 3.0 * np.cos(th) * (r / 30), rng.uniform(-40, 40, (N, N)))
m2 = np.where(on_source, 0.4 + 0.25 * np.exp(-r / 40), rng.uniform(0, 25, (N, N)))
clean = (m0, m1, m2)
dirty = tuple(a + rng.normal(0, s, (N, N)) for a, s in zip(clean, (0.08, 1.2, 0.5)))
den = tuple(a + rng.normal(0, s, (N, N)) for a, s in zip(clean, (0.02, 0.4, 0.15)))

mask = signal_mask(clean[0])


# ------------------------------------------------------------------ [1] masking matters
# This is the whole point: the masked scale must be dramatically tighter than the range
# the raw data spans, or the disk is being squeezed into a sliver of the colormap.
for name, kind, i, min_gain in (("M1", "velocity", 1, 3.0), ("M2", "dispersion", 2, 3.0)):
    maps = [clean[i], dirty[i], den[i]]
    lo, hi = _limits(maps, mask, kind)
    full = np.concatenate([m.ravel() for m in maps])
    gain = (full.max() - full.min()) / (hi - lo)
    assert gain > min_gain, f"{name}: masking only tightened the scale {gain:.1f}x"
    print(f"[1] {name}: masked span {hi - lo:6.2f} vs raw {full.max() - full.min():7.2f} "
          f"-> {gain:.1f}x more of the colormap spent on the disk")


# ------------------------------------------------------------------ [2] velocity centred
# RdBu_r is diverging; its white point must sit at the systemic velocity, otherwise
# red/blue does not mean receding/approaching.
lo, hi = _limits([clean[1], dirty[1], den[1]], mask, "velocity")
centre = (lo + hi) / 2
assert abs(centre - SYSTEMIC) < 0.5, f"velocity scale centred at {centre:.2f}, not {SYSTEMIC}"
print(f"[2] velocity scale centred at {centre:.2f} km/s (systemic {SYSTEMIC}), symmetric")


# ------------------------------------------------------------------ [3] outliers clipped
# One hot pixel must not set the ceiling.
spiked = clean[0].copy()
spiked[0, 0] = 1e6
lo_s, hi_s = _limits([spiked], mask, "intensity")
assert hi_s < 100, f"a single 1e6 pixel set the ceiling to {hi_s}"
print(f"[3] a 1e6 spike does not set the ceiling (vmax {hi_s:.2f}) -- percentiles, not max")


# ------------------------------------------------------------------ [4] figure builds
fig = plot_moment_comparison(clean, dirty, den, tag="unit test")
# 9 map panels + 3 colorbars
assert len(fig.axes) == 12, len(fig.axes)
images = [im for a in fig.axes for im in a.images]
assert len(images) == 9, len(images)
# every column shares one scale across all three rows, or the eye cannot compare them
for col in range(3):
    clims = {images[row * 3 + col].get_clim() for row in range(3)}
    assert len(clims) == 1, f"column {col} has {len(clims)} different colour scales"
print("[4] 3x3 panels + 3 colorbars; each column shares one scale across all rows")


# ------------------------------------------------------------------ [5] degenerate input
# An empty clean map must not raise -- it happens on a channel with no emission.
blank = (np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)))
fig2 = plot_moment_comparison(blank, blank, blank, tag="empty")
assert len(fig2.axes) == 12
print("[5] an all-zero (no-emission) cube renders instead of raising")

print("\n" + "=" * 60)
print("All moment figure tests PASSED")
print("=" * 60)
