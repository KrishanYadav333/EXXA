"""
Phase 0's discriminator decides whether DDRM and VIREO-lite get built at all, so it is run
here against synthetic cases whose true forward operator is known by construction.

Three earlier versions of this check each produced a confident WRONG verdict on the project's
real cubes, and each failure is preserved below as a case:

  1. Testing whether the raw ratio P_d/P_c dips below 1. Fails whenever the two maps are on
     different intensity scales -- a dirty map is conventionally Jy/beam and a model image
     Jy/pixel, differing by the beam area in pixels. A real convolution then sits above 1 and
     reads as no_convolution. (case 3)
  2. Normalising that ratio by its own low-k level, then taking the minimum over the whole
     band. For a monotonically RISING ratio the lowest bin is by construction below the median
     of the lowest bins, so a dip appears where nothing was suppressed and additive noise
     reads as a convolution. (case 1)
  3. Deciding by whether a Gaussian beam beats the no-beam model. A sidelobed beam is not
     Gaussian, so neither model fits, the comparison ties, and a real convolution reads as
     no_convolution. (case 4)

What survives all three is fitting the forward model
`P_d(k) = A exp(-4 pi^2 sigma^2 k^2) P_c(k) + N` and asking whether the no-beam version of it
is adequate on its own. A is a free parameter, so intensity scale cannot mislead it.

    PYTHONPATH=. python3 tests/test_forward_operator.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.evaluation.forward_operator import (
    beam_kernel_of, phase0_report, format_report, pixel_scale_arcsec,
)

SIGMA, N, C = 3.0, 256, 6
failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def power_law_field(n, rng, slope=-2.5):
    """A field with a realistic falling spectrum, so there is signal at every k."""
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    kr[0, 0] = 1.0
    amp = kr ** (slope / 2.0)
    amp[0, 0] = 0.0
    f = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    return f / f.std()


def gaussian_blur(img, sigma=SIGMA):
    n = img.shape[-1]
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img)
                                * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (ky ** 2 + kx ** 2))))


def sidelobed_blur(img, sigma=SIGMA):
    """Gaussian core times a radial ripple: the crudest stand-in for uv-gap sidelobes."""
    n = img.shape[-1]
    y, x = np.mgrid[-24:25, -24:25].astype(float)
    r = np.sqrt(x ** 2 + y ** 2)
    k = np.exp(-r ** 2 / (2 * sigma ** 2)) * np.cos(r / 1.6)
    k /= k.sum()
    pad = np.zeros((n, n))
    pad[:49, :49] = k
    pad = np.roll(pad, (-24, -24), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)))


HEADER = {  # the example header recorded in PHYSICS_INFORMED_PLAN.md
    "BMAJ": 4.2629e-5, "BMIN": 3.2923e-5, "BPA": 16.9333, "CDELT2": 0.01 / 3600.0,
}
MATCHED = {  # a header describing SIGMA exactly, at 0.01"/px
    "BMAJ": SIGMA * 2.3548 * 0.01 / 3600.0, "BMIN": SIGMA * 2.3548 * 0.01 / 3600.0,
    "BPA": 0.0, "CDELT2": 0.01 / 3600.0,
}

print("=" * 70)
print("Phase 0 forward-operator discriminator")
print("=" * 70)

rng = np.random.default_rng(0)
clean = np.stack([power_law_field(N, rng) for _ in range(C)])
clean /= clean.std()
blurred = np.stack([gaussian_blur(c) for c in clean])

# --- case 1: additive noise only. A = I, nothing for DDRM to constrain. ----------------
# Regression for failure 2: the ratio here RISES monotonically with k, and a low-k
# normalisation used to turn that into a false dip.
print("\ncase 1  dirty = clean + noise")
r = phase0_report(clean, clean + rng.normal(0, 0.15, clean.shape))
print(format_report(r))
check("additive noise reads as no_convolution", r["verdict"] == "no_convolution")
check("a rising ratio is not mistaken for suppression",
      r["residual_no_convolution"] < 0.05, f"resid {r['residual_no_convolution']:.4f}")
check("best-fit beam is sub-pixel", r["fit_sigma_px"] < 0.5, f"{r['fit_sigma_px']:.2f} px")

# --- case 2: Gaussian convolution. DDRM applies. ---------------------------------------
print("\ncase 2  dirty = clean (*) gaussian(sigma=3.0) + noise")
r = phase0_report(clean, blurred + rng.normal(0, 0.02, clean.shape), MATCHED)
print(format_report(r))
check("gaussian blur reads as gaussian_convolution", r["verdict"] == "gaussian_convolution")
check("recovered sigma within 5% of truth", abs(r["fit_sigma_px"] - SIGMA) / SIGMA < 0.05,
      f"got {r['fit_sigma_px']:.3f}, true {SIGMA}")
check("measured beam agrees with the header describing it",
      0.9 < r["fit_sigma_px"] / r["header_sigma_px"] < 1.1,
      f"{r['fit_sigma_px']:.2f} px vs header {r['header_sigma_px']:.2f} px")

# --- case 3: scale invariance. Failure 1. ----------------------------------------------
print("\ncase 3  the same blur, seen on four different intensity scales")
for amp in (1.0, 5.0, 25.0, 200.0):
    r = phase0_report(clean, amp * blurred + rng.normal(0, 0.02 * amp, clean.shape))
    check(f"convolution still found at x{amp:g} intensity scale",
          r["verdict"] == "gaussian_convolution",
          f"{r['verdict']}, sigma {r['fit_sigma_px']:.2f}, A {r['scale_factor']:.1f}")
    check(f"sigma unaffected by the x{amp:g} scale",
          abs(r["fit_sigma_px"] - SIGMA) / SIGMA < 0.05, f"{r['fit_sigma_px']:.2f} px")
for amp in (1.0, 25.0):
    r = phase0_report(clean, amp * (clean + rng.normal(0, 0.15, clean.shape)))
    check(f"no false convolution at x{amp:g} intensity scale",
          r["verdict"] == "no_convolution", r["verdict"])

# --- case 4: a beam with sidelobes. Failure 3. -----------------------------------------
# A Gaussian cannot fit this, so a "does a Gaussian beat no-beam" test ties and calls it
# no_convolution. The no-beam model must be judged on its own adequacy instead.
print("\ncase 4  dirty = clean (*) sidelobed_beam + noise")
r = phase0_report(clean, np.stack([sidelobed_blur(c) for c in clean])
                  + rng.normal(0, 0.02, clean.shape), HEADER)
print(format_report(r))
check("sidelobed beam reads as non_gaussian_convolution",
      r["verdict"] == "non_gaussian_convolution")
check("no-beam model is clearly inadequate here",
      r["residual_no_convolution"] > 0.05, f"resid {r['residual_no_convolution']:.4f}")

# --- case 5: a null result must prove a beam could have been seen ----------------------
print("\ncase 5  null results, wide band vs band too narrow to see the header's beam")
r_wide = phase0_report(clean, clean + rng.normal(0, 0.05, clean.shape), HEADER)
check("clean null with a wide band is reported as no_convolution",
      r_wide["verdict"] == "no_convolution",
      f"band to k={r_wide['band_k_max']:.3f}, beam acts from k={r_wide['k_half_from_header']:.3f}")
check("the band's reach is reported so the null can be judged",
      r_wide["band_k_max"] > r_wide["k_half_from_header"])
r_narrow = phase0_report(clean, clean + rng.normal(0, 8.0, clean.shape), HEADER)
check("a null from a band too narrow to see the beam is NOT no_convolution",
      r_narrow["verdict"] == "indeterminate", r_narrow["verdict"])

# --- beam kernel from a header ---------------------------------------------------------
print("\nbeam kernel from the recorded header")
k = beam_kernel_of(HEADER)
check("pixel scale read from CDELT2", abs(pixel_scale_arcsec(HEADER) - 0.01) < 1e-9)
check("kernel normalised to unit sum", abs(k.sum() - 1.0) < 1e-9)
check("kernel is odd-sized and square",
      k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1, str(k.shape))
check("peak at the centre",
      np.unravel_index(k.argmax(), k.shape) == (k.shape[0] // 2,) * 2)
maj = HEADER["BMAJ"] * 3600.0 / 0.01 / 2.3548
mn = HEADER["BMIN"] * 3600.0 / 0.01 / 2.3548
check("major-axis sigma matches the header", abs(maj - 6.52) < 0.05, f"{maj:.2f} px")
check("kernel is elliptical, not round", abs(maj - mn) > 1.0, f"{maj:.2f} vs {mn:.2f} px")
check("no beam keys -> None", beam_kernel_of({"CDELT2": 1e-5}) is None)
check("no pixel scale -> None", beam_kernel_of({"BMAJ": 1e-5, "BMIN": 1e-5, "BPA": 0}) is None)

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
