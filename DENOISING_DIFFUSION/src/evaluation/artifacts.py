#!/usr/bin/env python
"""
src/evaluation/artifacts.py
===========================
Per-channel artifact diagnostics for a denoised line-emission channel.

Three artifacts have been observed across V7/V9/V12 but only ever documented
from a single channel (channel 100), which the V7/V9 variance lesson says is not
enough to characterise anything:

  * **peak overshoot** -- denoised peak brighter than clean (1.151x at ch 100).
  * **negative floor leak** -- denoised background dips below clean's ~0 floor.
  * **invented structure** ("hallucination") -- the model asserts signal where
    the ground truth has none. Scientifically the worst of the three: in a
    protoplanetary-disk map, invented structure reads as a false detection.

`channel_artifacts` scores one channel so a caller can run it over every
validation channel and report distributions instead of anecdotes. It is
model-agnostic (takes arrays, not a model), so the same numbers can be produced
for the U-Net and the DDPM and compared directly.
"""

from typing import Dict

import numpy as np
from scipy import ndimage

# Defaults chosen so "background" and "asserted signal" are separated by a clear
# margin: a pixel the truth puts below 10% of peak, the model puts above 20%.
FLOOR_FRAC = 0.10   # clean below this fraction of its own peak counts as background
INVENT_FRAC = 0.20  # denoised above this fraction of clean's peak = asserted signal
BLOB_MIN_PX = 20    # ignore single-pixel specks; count only structures this large


def channel_artifacts(
    clean: np.ndarray,
    dirty: np.ndarray,
    denoised: np.ndarray,
    *,
    floor_frac: float = FLOOR_FRAC,
    invent_frac: float = INVENT_FRAC,
    blob_min_px: int = BLOB_MIN_PX,
) -> Dict[str, float]:
    """
    Score overshoot / floor leak / invented structure for one channel.

    All three inputs are 2D arrays on the same scale (the shared dirty-scale
    normalisation, so clean may exceed 1).

    Returns a dict with:
        snr:            clean peak / std of dirty over clean's background pixels.
                        NaN when the background is empty or noiseless.
        overshoot:      denoised.max() / clean.max(); 1.0 is perfect.
        floor_leak:     denoised.min(); clean's floor is ~0, so negative is leak.
        invented_frac:  fraction of background pixels pushed above the assert
                        threshold.
        invented_blobs: count of connected invented regions of >= blob_min_px,
                        i.e. how many distinct fake structures.

    Raises ValueError if clean has no positive peak (nothing to normalise against).
    """
    cmax = float(clean.max())
    if not np.isfinite(cmax) or cmax <= 0:
        raise ValueError("clean channel has no positive peak; cannot score artifacts")

    background = clean < floor_frac * cmax
    # Noise proxy from DIRTY over clean-background pixels: measures what the model
    # had to see through, not what it produced.
    noise = float(dirty[background].std()) if background.any() else float("nan")
    snr = cmax / noise if np.isfinite(noise) and noise > 0 else float("nan")

    invented = background & (denoised > invent_frac * cmax)
    labels, n_labels = ndimage.label(invented)
    if n_labels:
        sizes = ndimage.sum(invented, labels, range(1, n_labels + 1))
        n_blobs = int((np.asarray(sizes) >= blob_min_px).sum())
    else:
        n_blobs = 0

    return {
        "snr": snr,
        "overshoot": float(denoised.max() / cmax),
        "floor_leak": float(denoised.min()),
        "invented_frac": float(invented.sum() / max(int(background.sum()), 1)),
        "invented_blobs": n_blobs,
    }


def summarise(rows) -> Dict[str, float]:
    """
    Aggregate `channel_artifacts` dicts into report-ready statistics.

    Splits the invented-structure and overshoot rates at the median SNR, which is
    what answers the open question flagged to the mentor: is invented structure a
    low-SNR-specific failure, or does it happen everywhere?
    """
    if not rows:
        return {}
    ov = np.array([r["overshoot"] for r in rows], dtype=float)
    leak = np.array([r["floor_leak"] for r in rows], dtype=float)
    inv = np.array([r["invented_frac"] for r in rows], dtype=float)
    blob = np.array([r["invented_blobs"] for r in rows], dtype=float)
    snr = np.array([r["snr"] for r in rows], dtype=float)

    out = {
        "n_channels": len(rows),
        "overshoot_mean": float(ov.mean()),
        "overshoot_median": float(np.median(ov)),
        "overshoot_p90": float(np.percentile(ov, 90)),
        "overshoot_max": float(ov.max()),
        "frac_over_10pct": float((ov > 1.10).mean()),
        "floor_leak_mean": float(leak.mean()),
        "floor_leak_min": float(leak.min()),
        "invented_frac_mean": float(inv.mean()),
        "invented_frac_max": float(inv.max()),
        "frac_channels_with_blob": float((blob > 0).mean()),
        "blobs_per_channel": float(blob.mean()),
    }

    ok = np.isfinite(snr)
    if ok.sum() > 4:
        thr = float(np.median(snr[ok]))
        lo, hi = ok & (snr <= thr), ok & (snr > thr)
        out.update({
            "snr_median": thr,
            "low_snr_blobs_per_channel": float(blob[lo].mean()),
            "high_snr_blobs_per_channel": float(blob[hi].mean()),
            "low_snr_invented_frac": float(inv[lo].mean()),
            "high_snr_invented_frac": float(inv[hi].mean()),
            "low_snr_overshoot": float(ov[lo].mean()),
            "high_snr_overshoot": float(ov[hi].mean()),
        })
    return out
