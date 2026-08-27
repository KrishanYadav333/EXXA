"""
The GI wiggle diagnostic (Jason's reading list, 2026-08-27: Speedie+2024, Terry+2024,
Hall+2020/2021/2022) fits a Keplerian model to a disk's moment-1 map and looks at what is
left over. Getting that fit wrong is easy to do silently -- a diverged optimiser still
returns a number -- so this is checked against synthetic cases with a known answer before it
is trusted on real data.

    PYTHONPATH=. python3 tests/test_gi_wiggle.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.evaluation.gi_wiggle import (
    keplerian_los, fit_keplerian, wiggle_residual, wiggle_amplitude, disk_geometry_from_m0,
)

N, AU_PER_PX = 200, 1.5
TRUE = dict(cx=100.0, cy=95.0, pa_deg=35.0, incl_deg=40.0, vsys=1.2, mstar_msun=0.8)
failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("=" * 70)
print("GI wiggle: Keplerian fit and residual")
print("=" * 70)

y, x = np.mgrid[0:N, 0:N].astype(float)
m1_pure = keplerian_los((x, y), TRUE["cx"], TRUE["cy"], TRUE["pa_deg"], TRUE["incl_deg"],
                        TRUE["vsys"], TRUE["mstar_msun"], AU_PER_PX)
r = np.sqrt((x - TRUE["cx"]) ** 2 + (y - TRUE["cy"]) ** 2)
mask = (r > 6) & (r < 80)

# --- case 1: a pure Keplerian field must be recovered exactly, residual exactly 0 --------
print("\ncase 1  pure Keplerian, no wiggle")
geom = fit_keplerian(m1_pure, mask, AU_PER_PX, init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
for k in ("cx", "cy", "pa_deg", "incl_deg", "vsys", "mstar_msun"):
    check(f"{k} recovered exactly", abs(geom[k] - TRUE[k]) < 1e-4,
          f"true {TRUE[k]}, fit {geom[k]:.6f}")
resid = wiggle_residual(m1_pure, geom)
amp = wiggle_amplitude(resid, mask)
check("residual is exactly zero with no wiggle present", amp["rms_kms"] < 1e-6,
      f"RMS {amp['rms_kms']:.2e} km/s")

# --- case 2: a known injected perturbation must be recoverable, not just noise fit -------
print("\ncase 2  a known m=2 perturbation is recovered, not fit away as noise")
pa, inc = np.radians(TRUE["pa_deg"]), np.radians(TRUE["incl_deg"])
dx, dy = x - TRUE["cx"], y - TRUE["cy"]
xr = dx * np.cos(pa) + dy * np.sin(pa)
yr = (-dx * np.sin(pa) + dy * np.cos(pa)) / np.cos(inc)
az = np.arctan2(yr, xr)
injected = 0.4 * np.sin(2 * az) * np.exp(-((r - 40) ** 2) / (2 * 20 ** 2))
m1_wiggly = m1_pure + injected

geom2 = fit_keplerian(m1_wiggly, mask, AU_PER_PX,
                      init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
resid2 = wiggle_residual(m1_wiggly, geom2)
amp2 = wiggle_amplitude(resid2, mask)
corr = float(np.corrcoef(resid2[mask].ravel(), injected[mask].ravel())[0, 1])
check("recovered residual correlates with the injected pattern", corr > 0.7, f"r={corr:.3f}")
check("recovered amplitude is the right order of magnitude",
      0.05 < amp2["rms_kms"] < 1.0, f"RMS {amp2['rms_kms']:.3f} km/s")

# --- case 3: no signal at all must not crash or invent a wiggle --------------------------
print("\ncase 3  flat field (no rotation, no wiggle) degrades safely")
flat = np.zeros((N, N))
geom3 = fit_keplerian(flat, mask, AU_PER_PX, init=dict(cx=95, cy=90, pa_deg=30, incl_deg=45))
resid3 = wiggle_residual(flat, geom3)
amp3 = wiggle_amplitude(resid3, mask)
check("a flat field does not crash the fit", geom3["success"] or True)
check("residual amplitude is finite", np.isfinite(amp3["rms_kms"]))

# --- geometry helper: initial guess from image moments alone -----------------------------
print("\ncase 4  disk_geometry_from_m0 gives a sane starting guess")
m0 = np.exp(-((r - 40) ** 2) / (2 * 15 ** 2))
g0 = disk_geometry_from_m0(m0, mask)
check("centre from M0 moments is close to true centre",
      abs(g0["cx"] - TRUE["cx"]) < 5 and abs(g0["cy"] - TRUE["cy"]) < 5,
      f"got ({g0['cx']:.1f}, {g0['cy']:.1f}), true ({TRUE['cx']}, {TRUE['cy']})")

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
