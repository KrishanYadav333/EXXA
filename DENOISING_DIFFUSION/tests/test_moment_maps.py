#!/usr/bin/env python
"""
Smoke test for the M2 (dispersion) noise-clip fix.

Unmasked, bettermoments' collapse_second computes an intensity-weighted variance over
the WHOLE velocity axis with no threshold of its own. Fed pure noise, that estimator does
not blow up to infinity -- it converges to roughly the RMS width of the velocity axis
itself (~577 m/s here, for a +/-1000 m/s axis), because a zero-mean noise weighting still
samples the full axis roughly evenly. That value is uniform across the whole background,
not a few outlier pixels, and it sits far ABOVE a real disk's actual line width (150 m/s
injected here) -- which is what turned architecture_moment_maps.png's entire M2 column
(everything except the noiseless clean truth) into a solid saturated block: real disk
structure and pure background noise both compare in the hundreds of m/s, so a colour scale
that has to show both loses the real structure entirely.
"""

import numpy as np

from src.evaluation.moment_maps import generate_moment_maps

print("=" * 60)
print("Moment Map Clip Smoke Test")
print("=" * 60)

H, C = 32, 40
rng = np.random.default_rng(0)
velax = np.linspace(-1000, 1000, C).astype(np.float32)
axis_rms = float(np.std(velax))

yy, xx = np.mgrid[0:H, 0:H]
r2 = (yy - H / 2) ** 2 + (xx - H / 2) ** 2
disk = 20.0 * np.exp(-r2 / (2 * 6.0 ** 2))       # 20-sigma peak: well above the noise
line_center = velax[C // 2]
line_width = 150.0

data = np.zeros((C, H, H), dtype=np.float32)
for i, v in enumerate(velax):
    data[i] = disk * np.exp(-((v - line_center) ** 2) / (2 * line_width ** 2))
data += rng.normal(0, 1.0, data.shape).astype(np.float32)

# [1] unclipped: PURE noise (no disk) must read a dispersion comparable to the velocity
# axis itself -- this is the bug. A metric that reports ~axis-width for empty sky is not
# measuring dispersion, it is measuring how wide the observing band is.
noise_only = rng.normal(0, 1.0, data.shape).astype(np.float32)
_, _, m2_noise = generate_moment_maps(None, data_velax=(noise_only, velax), clip_sigma=0)
assert 0.5 * axis_rms < np.nanmedian(m2_noise) < 1.5 * axis_rms, \
    f"expected pure-noise M2 near the axis RMS ({axis_rms:.0f}), got {np.nanmedian(m2_noise):.0f}"
print(f"[1] unclipped pure noise reads M2 {np.nanmedian(m2_noise):.0f} "
      f"~= axis RMS {axis_rms:.0f} -- indistinguishable from a real wide line")

# [2] same field: unclipped, the disk's OWN M2 is dragged far off its true injected width
# by noise in its wings (channels where the line is faint), toward the background's
# axis-RMS value -- a ~2x inflation despite this pixel being a 20-sigma detection.
cy, cx = H // 2, H // 2
_, _, m2_raw = generate_moment_maps(None, data_velax=(data, velax), clip_sigma=0)
disk_raw = float(np.mean(m2_raw[cy - 2:cy + 2, cx - 2:cx + 2]))
bg_raw = float(np.mean(m2_raw[:4, :4]))
assert disk_raw > 1.5 * line_width, \
    f"expected unclipped disk M2 ({disk_raw:.0f}) inflated well past the true width ({line_width:.0f})"
print(f"[2] unclipped disk-centre M2 {disk_raw:.0f} vs true width {line_width:.0f} "
      f"(background corner reads {bg_raw:.0f}) -- noise wings inflate even a strong detection")

# [3] clipped: background must drop out (masked to NaN, not a false dispersion reading),
# and the disk must recover close to its TRUE injected width.
_, _, m2_clip = generate_moment_maps(None, data_velax=(data, velax), clip_sigma=3.0)
bg_clip = m2_clip[:4, :4]
assert np.isnan(bg_clip).mean() > 0.5, \
    f"clip should mask most of the noise-only background, only {np.isnan(bg_clip).mean():.0%} masked"
disk_clip = float(np.nanmean(m2_clip[cy - 2:cy + 2, cx - 2:cx + 2]))
assert abs(disk_clip - line_width) < 60, \
    f"clipped disk M2 ({disk_clip:.0f}) should be near the injected width ({line_width:.0f})"
print(f"[3] clipped: background {np.isnan(bg_clip).mean():.0%} masked, "
      f"disk-centre M2 {disk_clip:.0f} ~= injected width {line_width:.0f}")

print("\n" + "=" * 60)
print("All moment-map clip tests PASSED")
print("=" * 60)
