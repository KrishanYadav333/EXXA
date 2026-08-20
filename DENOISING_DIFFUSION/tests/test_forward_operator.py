"""
Phase 0's discriminator has to be right before it is pointed at real cubes, because its
answer decides whether DDRM gets built at all. So it is run here against three synthetic
cases where the true forward operator is known by construction.

    PYTHONPATH=. python3 tests/test_forward_operator.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.evaluation.forward_operator import (
    beam_kernel_of, phase0_report, format_report, pixel_scale_arcsec,
)


def power_law_field(n, rng, slope=-2.5):
    """A field with a realistic falling power spectrum, so there is signal at every k."""
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    kr[0, 0] = 1.0
    amp = kr ** (slope / 2.0)
    amp[0, 0] = 0.0
    ph = np.exp(2j * np.pi * rng.random((n, n)))
    f = np.real(np.fft.ifft2(amp * ph))
    return f / f.std()


def convolve(img, k):
    """FFT convolution, kernel centred."""
    pad = np.zeros_like(img)
    kh, kw = k.shape
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)))


def gaussian_kernel(sigma, half):
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    k = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def dirty_beam_kernel(sigma, half):
    """A Gaussian core times a radial ripple: the crudest stand-in for uv-gap sidelobes."""
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    r = np.sqrt(x ** 2 + y ** 2)
    k = np.exp(-(r ** 2) / (2 * sigma ** 2)) * np.cos(r / 1.6)
    return k / k.sum()


HEADER = {  # the example header recorded in PHYSICS_INFORMED_PLAN.md
    "BMAJ": 4.2629e-5, "BMIN": 3.2923e-5, "BPA": 16.9333, "CDELT2": 0.01 / 3600.0,
}

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("=" * 70)
print("Phase 0 forward-operator discriminator")
print("=" * 70)

rng = np.random.default_rng(0)
N, C = 256, 6
clean = np.stack([power_law_field(N, rng) for _ in range(C)])

# --- case 1: additive noise only. A = I, DDRM has nothing to constrain. ---------------
dirty = clean + rng.normal(0, 0.15, clean.shape)
r = phase0_report(clean, dirty)
print("\ncase 1  dirty = clean + noise")
print(format_report(r))
check("additive noise reads as no_convolution", r["verdict"] == "no_convolution")

# --- case 2: Gaussian convolution. DDRM applies, header beam is the operator. ---------
SIGMA = 3.0
kern = gaussian_kernel(SIGMA, 12)
blurred = np.stack([convolve(c, kern) for c in clean])
dirty = blurred + rng.normal(0, 0.02, clean.shape)
r = phase0_report(clean, dirty, HEADER)
print("\ncase 2  dirty = clean (*) gaussian(sigma=3.0) + noise")
print(format_report(r))
check("gaussian blur reads as gaussian_convolution", r["verdict"] == "gaussian_convolution")
check("recovered sigma within 10% of truth",
      abs(r["gaussian_sigma_px"] - SIGMA) / SIGMA < 0.10,
      f"got {r['gaussian_sigma_px']:.3f}, true {SIGMA}")

# --- case 2b: the same blur, with a header that actually describes it. ----------------
# Exercises the cross-check that will matter on real cubes: does the beam measured from the
# data agree with the beam the header claims? sigma 3.0 px at 0.01"/px means a FWHM of
# 3.0 * 2.3548 * 0.01 = 0.0706", i.e. BMAJ = BMIN = 1.9623e-5 deg.
MATCHED = {"BMAJ": 1.9623e-5, "BMIN": 1.9623e-5, "BPA": 0.0, "CDELT2": 0.01 / 3600.0}
r = phase0_report(clean, blurred + rng.normal(0, 0.02, clean.shape), MATCHED)
print("\ncase 2b  same blur, header describes it")
print(format_report(r))
check("measured beam agrees with the header it came from",
      0.9 < r["gaussian_sigma_px"] / r["header_sigma_px"] < 1.1,
      f"{r['gaussian_sigma_px']:.2f} px measured vs {r['header_sigma_px']:.2f} px in header")

# --- case 3: a beam with sidelobes. Convolution, but a Gaussian A is the wrong one. ---
dirty = np.stack([convolve(c, dirty_beam_kernel(3.0, 16)) for c in clean])
dirty = dirty + rng.normal(0, 0.02, clean.shape)
r = phase0_report(clean, dirty, HEADER)
print("\ncase 3  dirty = clean (*) sidelobed_beam + noise")
print(format_report(r))
check("sidelobed beam reads as non_gaussian_convolution",
      r["verdict"] == "non_gaussian_convolution")

# --- case 4: scale invariance. THE bug that made the first real run wrong. ------------
# A dirty/restored map is conventionally Jy/BEAM and a model image Jy/PIXEL; they differ by
# the beam area in pixels, of order 200 here. The first version of this check tested whether
# the raw ratio dipped below 1, which any such factor defeats: the project's own four cubes
# came back "no_convolution" with minima of 1.01, 1.02, 1.14 and 3.96, a spread no single
# operator can produce. Verdict must depend on the SHAPE of |B(k)|, never its level.
print("\ncase 4  the same blur, seen on four different intensity scales")
for s_amp in (1.0, 5.0, 25.0, 200.0):
    d = s_amp * blurred + rng.normal(0, 0.02 * s_amp, clean.shape)
    r = phase0_report(clean, d)
    check(f"convolution still detected at x{s_amp:g} intensity scale",
          r["verdict"] == "gaussian_convolution",
          f"{r['verdict']}, min {r['min_transfer']:.3f}, scale {r['scale_factor']:.1f}")
    check(f"beam sigma unaffected by the x{s_amp:g} scale",
          abs(r["gaussian_sigma_px"] - SIGMA) / SIGMA < 0.10,
          f"{r['gaussian_sigma_px']:.2f} px")

print("\ncase 4b  additive noise on a rescaled map is still no_convolution")
for s_amp in (1.0, 25.0):
    d = s_amp * (clean + rng.normal(0, 0.15, clean.shape))
    r = phase0_report(clean, d)
    check(f"no false convolution at x{s_amp:g} intensity scale",
          r["verdict"] == "no_convolution",
          f"{r['verdict']}, min {r['min_transfer']:.3f}, scale {r['scale_factor']:.1f}")

# --- case 5: a null result is only meaningful if a beam could have been seen -----------
# The band ends where the dirty spectrum sinks into its own noise. A beam of the header's
# size does nothing below k ~ 0.03, so if the band stops short of that, "no suppression
# found" is uninformative rather than negative. RULES.md #8.
print("\ncase 5  null results, wide band vs band too narrow to see the header's beam")
r_wide = phase0_report(clean, clean + rng.normal(0, 0.05, clean.shape), HEADER)
check("clean null with a wide band is reported as no_convolution",
      r_wide["verdict"] == "no_convolution",
      f"band to k={r_wide['band_k_max']:.3f}, beam acts from k={r_wide['k_half_from_header']:.3f}")
check("the band's reach is reported so the null can be judged",
      r_wide["band_k_max"] > r_wide["k_half_from_header"])

r_narrow = phase0_report(clean, clean + rng.normal(0, 8.0, clean.shape), HEADER)
check("a null from a band too narrow to see the beam is NOT called no_convolution",
      r_narrow["verdict"] == "indeterminate", r_narrow["verdict"])

# --- beam kernel from a header --------------------------------------------------------
print("\nbeam kernel from the recorded header")
k = beam_kernel_of(HEADER)
scale = pixel_scale_arcsec(HEADER)
check("pixel scale read from CDELT2", abs(scale - 0.01) < 1e-9, f"{scale} arcsec/px")
check("kernel normalised to unit sum", abs(k.sum() - 1.0) < 1e-9)
check("kernel is odd-sized and square", k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1,
      str(k.shape))
check("peak at the centre", np.unravel_index(k.argmax(), k.shape) == (k.shape[0] // 2,) * 2)

# BMAJ 4.2629e-5 deg = 0.15346" -> 15.35 px FWHM -> sigma 6.52 px at 0.01"/px
maj_px = HEADER["BMAJ"] * 3600.0 / 0.01 / 2.3548
prof = k[k.shape[0] // 2]
fwhm_min = float((prof > prof.max() / 2).sum())
min_px = HEADER["BMIN"] * 3600.0 / 0.01 / 2.3548
check("major-axis sigma matches the header", abs(maj_px - 6.52) < 0.05, f"{maj_px:.2f} px")
check("kernel is elliptical, not round",
      abs(maj_px - min_px) > 1.0, f"sigma {maj_px:.2f} vs {min_px:.2f} px")
check("no beam keys -> None", beam_kernel_of({"CDELT2": 1e-5}) is None)
check("no pixel scale -> None", beam_kernel_of({"BMAJ": 1e-5, "BMIN": 1e-5, "BPA": 0}) is None)

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
