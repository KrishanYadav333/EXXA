#!/usr/bin/env python
"""
Smoke tests for the per-channel artifact diagnostics:
  1. a perfect prediction scores no artifacts at all.
  2. a known 15% overshoot is measured as 1.15 (the ch-100 figure was 1.151).
  3. an invented source in the background is counted as one blob.
  4. two invented sources are counted as two.
  5. a speck smaller than BLOB_MIN_PX is ignored (not a "detection").
  6. a depressed background surfaces as a negative floor leak.
  7. a faint channel scores lower SNR than a bright one, so the low/high-SNR
     split in `summarise` means what it claims.
"""

import numpy as np

from src.evaluation.artifacts import BLOB_MIN_PX, channel_artifacts, summarise

print("=" * 60)
print("Artifact Diagnostics Smoke Test")
print("=" * 60)

H = 64
yy, xx = np.mgrid[0:H, 0:H]
rng = np.random.default_rng(0)


def blob(cy, cx, amp, s=4.0):
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))


clean = blob(20, 20, 1.0)
dirty = clean + rng.normal(0, 0.01, (H, H))

# [1] perfect prediction -> nothing flagged
r = channel_artifacts(clean, dirty, clean.copy())
assert abs(r["overshoot"] - 1.0) < 1e-6, r
assert r["invented_blobs"] == 0 and r["invented_frac"] == 0.0, r
print(f"[1] perfect prediction OK: overshoot {r['overshoot']:.4f}, blobs {r['invented_blobs']}")

# [2] 15% overshoot must read as 1.15
r = channel_artifacts(clean, dirty, clean * 1.15)
assert 1.14 < r["overshoot"] < 1.16, r
print(f"[2] 15% overshoot detected: {r['overshoot']:.4f}")

# [3] one invented source away from the real one -> exactly one blob
r = channel_artifacts(clean, dirty, clean + blob(45, 45, 0.6))
assert r["invented_blobs"] == 1, r
assert r["invented_frac"] > 0, r
print(f"[3] invented source counted: {r['invented_blobs']} blob, "
      f"{r['invented_frac']:.3%} of background")

# [4] two invented sources -> two blobs
r = channel_artifacts(clean, dirty, clean + blob(45, 45, 0.6) + blob(45, 15, 0.6))
assert r["invented_blobs"] == 2, r
print(f"[4] two invented sources counted: {r['invented_blobs']} blobs")

# [5] a 2x2 speck is below BLOB_MIN_PX and must not count as a detection
speck = clean.copy()
speck[50:52, 50:52] = 0.9
r = channel_artifacts(clean, dirty, speck)
assert r["invented_blobs"] == 0, r
print(f"[5] sub-{BLOB_MIN_PX}px speck ignored: {r['invented_blobs']} blobs")

# [6] depressed background -> negative floor leak
r = channel_artifacts(clean, dirty, clean - 0.005)
assert r["floor_leak"] < -0.004, r
print(f"[6] floor leak surfaced: {r['floor_leak']:+.4f}")

# [7] SNR ordering: a faint channel must score lower than a bright one
faint = blob(20, 20, 0.05)
r_lo = channel_artifacts(faint, faint + rng.normal(0, 0.01, (H, H)), faint)
r_hi = channel_artifacts(clean, dirty, clean)
assert r_lo["snr"] < r_hi["snr"], (r_lo["snr"], r_hi["snr"])
print(f"[7] SNR ordering OK: faint {r_lo['snr']:.1f} < bright {r_hi['snr']:.1f}")

# [8] summarise aggregates and splits by SNR without crashing on mixed rows
rows = []
for amp in [0.05, 0.1, 0.3, 0.6, 1.0, 1.5]:
    cl = blob(20, 20, amp)
    dt = cl + rng.normal(0, 0.01, (H, H))
    pred = cl * 1.12 + blob(45, 45, 0.3 * amp)   # both overshoot and invention
    rows.append(channel_artifacts(cl, dt, pred))
s = summarise(rows)
assert s["n_channels"] == 6
assert 1.0 < s["overshoot_mean"] < 1.3, s
assert "low_snr_blobs_per_channel" in s and "high_snr_blobs_per_channel" in s, s
print(f"[8] summarise OK: overshoot mean {s['overshoot_mean']:.3f}, "
      f"channels with a blob {s['frac_channels_with_blob']:.0%}, "
      f"SNR split at {s['snr_median']:.1f}")

# empty input must not explode
assert summarise([]) == {}

print("\n" + "=" * 60)
print("All artifact-diagnostic tests PASSED")
print("=" * 60)
