"""
Phase 0 of PHYSICS_INFORMED_PLAN.md: what actually produced the dirty cubes?

DDRM constrains the reverse diffusion chain with a known forward operator `A` such that
`dirty = A(clean) + noise`. The method is only worth building if `A` is a non-trivial
operator we can write down. Two possibilities with opposite conclusions:

  1. dirty = clean (*) beam + noise   -> a real convolution, DDRM applies, build it.
  2. dirty = clean + noise            -> A = I, DDRM degenerates to the conditional DDPM
                                         that already exists, and there is nothing to gain.

BMAJ/BMIN/BPA sitting in the headers and the MCFOST provenance both point at (1), but that
is an inference and this module is the check.

The discriminator is the azimuthally averaged power spectrum. Write P_d and P_c for the
dirty and clean spectra and |B(k)| for the beam's transfer function:

    convolution:  P_d(k) = |B(k)|^2 P_c(k) + N     -> the ratio DIPS below 1 at mid-k,
                                                      because |B(k)| < 1 there
    additive:     P_d(k) =            P_c(k) + N   -> the ratio only ever RISES above 1

A convolution suppresses intermediate spatial frequencies. Additive noise cannot suppress
anything; it only adds power. That asymmetry is what makes the two easy to tell apart, and
it survives not knowing the noise level, because the noise term is estimated and removed
before the ratio is formed.

There is a third answer the plan cares about. If a convolution is present but its transfer
function is not Gaussian, a true interferometric dirty beam was used, sidelobes and all, and
a Gaussian `A` would make DDRM enforce the wrong constraint. That case needs the PSF image
from whoever generated the cubes, so it is reported separately rather than folded into (1).
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 0.42466


def pixel_scale_arcsec(header) -> float | None:
    """
    Arcsec per pixel from CDELT1/CDELT2, or CD1_1/CD2_2 for headers written with a CD
    matrix instead. None when the header carries neither.

    Nothing in src/ read CDELT before this: `beam_features_of` takes only BPA/BMAJ/BMIN,
    which is enough to describe a beam in angular units but not to build a kernel in
    pixels.
    """
    if header is None:
        return None
    for k in ("CDELT2", "CDELT1", "CD2_2", "CD1_1"):
        if k in header:
            v = abs(float(header[k])) * 3600.0
            if v > 0:
                return v
    return None


def beam_kernel_of(header, shape=None) -> np.ndarray | None:
    """
    Elliptical Gaussian beam kernel in pixel units, rotated by BPA, normalised to unit sum.

    This is the `A` of Phase 1, and the restoring beam rather than the dirty beam: a real
    interferometric PSF has sidelobes from incomplete uv coverage and a Gaussian has none.
    Whether that approximation is safe here is exactly what `phase0_report` decides.

    BMAJ/BMIN are FWHM in degrees; BPA is degrees east of north, so the major axis runs
    along (-sin BPA, cos BPA) in (x, y). Returns None when the header lacks the beam keys
    or the pixel scale.
    """
    if header is None or not all(k in header for k in ("BMAJ", "BMIN", "BPA")):
        return None
    scale = pixel_scale_arcsec(header)
    if scale is None:
        return None

    sig_maj = float(header["BMAJ"]) * 3600.0 / scale * FWHM_TO_SIGMA
    sig_min = float(header["BMIN"]) * 3600.0 / scale * FWHM_TO_SIGMA
    if sig_maj <= 0 or sig_min <= 0:
        return None

    # Six sigma of support on the major axis, forced odd so the peak sits on a pixel.
    half = max(1, int(np.ceil(3.0 * sig_maj)))
    if shape is not None:
        half = min(half, min(shape) // 2)
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)

    bpa = np.deg2rad(float(header["BPA"]))
    # Rotate into the beam frame: u along the major axis, v along the minor axis.
    u = -x * np.sin(bpa) + y * np.cos(bpa)
    v = x * np.cos(bpa) + y * np.sin(bpa)

    k = np.exp(-0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2))
    return (k / k.sum()).astype(np.float64)


def radial_power_spectrum(img: np.ndarray, n_bins: int = 60):
    """
    Azimuthally averaged power spectrum. Returns (k, power) with k in cycles per pixel,
    running from 0 to 0.5 (Nyquist).

    A Hann window is applied first. Without it the image edges act as a step function and
    leak broadband power across every frequency, which is indistinguishable from the
    high-frequency noise floor this module has to measure.
    """
    img = np.asarray(img, dtype=np.float64)
    img = img - img.mean()

    h, w = img.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(img * win)) ** 2)

    ky = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)

    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(kr.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)

    power = np.bincount(idx[valid], weights=f.ravel()[valid], minlength=n_bins)
    count = np.bincount(idx[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        power = np.where(count > 0, power / np.maximum(count, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), power


def transfer_function(clean, dirty, n_bins: int = 60):
    """
    Estimate |B(k)| from a clean/dirty pair, plus the k grid and the noise power removed.

    Both inputs may be 2D (one channel) or 3D (C, H, W), in which case spectra are averaged
    over channels, which is the cheapest way to beat down the variance of a single channel's
    periodogram.

    Returns `(k, b_raw, b, noise, pc, pd)` with two transfer estimates, because the two
    questions Phase 0 asks want different ones:

      b_raw = sqrt(P_d / P_c)                  -- for "is there a convolution at all"
      b     = sqrt(max(P_d - N, 0) / P_c)      -- for "what shape is it"

    `b_raw` carries the noise floor, but it is the honest statistic for the dip test:
    additive noise can only ADD power, so P_d = P_c + N gives a raw ratio at or above 1
    everywhere, no matter how large N is or how badly it was estimated. A dip below 1 is
    therefore proof of suppression rather than an artefact of the noise model.

    `b` removes an estimated floor so the beam's shape can be fitted. N is the median of
    P_dirty over the top decile of k. That estimate is biased high whenever the clean field
    still has power up there, and subtracting a too-large N pushes the ratio below 1 where
    nothing was suppressed -- which is exactly why the verdict must not be read off `b`.
    """
    clean = np.asarray(clean, dtype=np.float64)
    dirty = np.asarray(dirty, dtype=np.float64)
    if clean.ndim == 2:
        clean, dirty = clean[None], dirty[None]

    k = None
    pc = np.zeros(n_bins)
    pd = np.zeros(n_bins)
    for c, d in zip(clean, dirty):
        k, p = radial_power_spectrum(c, n_bins); pc += np.nan_to_num(p)
        _, p = radial_power_spectrum(d, n_bins); pd += np.nan_to_num(p)
    pc /= len(clean)
    pd /= len(dirty)

    tail = k >= 0.45
    noise = float(np.median(pd[tail])) if tail.any() else 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(pc > 0, pd / pc, np.nan)
        ratio = np.where(pc > 0, np.maximum(pd - noise, 0.0) / pc, np.nan)
    return k, np.sqrt(raw), np.sqrt(ratio), noise, pc, pd


def phase0_report(clean, dirty, header=None, n_bins: int = 60) -> dict:
    """
    Settle Phase 0. Returns a dict with a `verdict` of:

      "no_convolution"           -> A = I. DDRM degenerates to the existing conditional
                                    DDPM. Stop, per the plan.
      "gaussian_convolution"     -> DDRM applies and the header's beam is the operator.
      "non_gaussian_convolution" -> a real dirty beam with sidelobes. DDRM still applies but
                                    a Gaussian A enforces the wrong constraint; the PSF
                                    image is needed from whoever generated the cubes.

    `min_transfer` is the decisive number: the minimum of |B(k)| over the band where the
    clean signal actually has power. A convolution drives it well below 1; additive noise
    leaves it at 1.
    """
    k, b_raw, b, noise, pc, pd = transfer_function(clean, dirty, n_bins)

    # |B(k)| is only measurable where the dirty spectrum still stands above its own noise
    # floor. Beyond that the ratio divides noise by noise and says nothing.
    #
    # The band must not be set as a fraction of the peak power instead. A disk's spectrum is
    # steeply red, so "1% of the maximum" lands around k = 0.04 and everything above it is
    # discarded -- which is precisely the range where a Gaussian beam and a sidelobed one
    # stop looking alike. Cutting there makes every beam look Gaussian.
    band = np.isfinite(b_raw) & (pc > 0) & (pd > 3.0 * noise) & (k < 0.45)
    if band.sum() < 5:
        return {"verdict": "indeterminate",
                "reason": "too few bins where the dirty spectrum rises above its noise floor",
                "k": k, "transfer": b, "noise_power": noise}

    kb, bb = k[band], b[band]
    raw_b = b_raw[band]

    # Normalise by the low-k level before testing for a dip.
    #
    # The raw ratio only means "1 where nothing was suppressed" if clean and dirty share an
    # intensity scale. They frequently do not: a dirty or restored map is conventionally in
    # Jy/BEAM while a model image is in Jy/PIXEL, and the two differ by the beam area in
    # pixels, of order 200 for this project's headers. Any such factor s multiplies the whole
    # ratio, so a real convolution that should fall to 0.06 sits above 1 instead and reads as
    # no_convolution. That is a measured failure of the first version of this function against
    # the project's own cubes, not a hypothetical.
    #
    # A convolution kernel has unit sum, so |B(k)| -> 1 as k -> 0 and the ratio's low-k level
    # estimates s by itself. Dividing it out leaves the SHAPE, which is what separates the two
    # cases: a convolution falls away from its own low-k level, additive noise only rises
    # above it.
    n_lo = max(3, int(0.1 * raw_b.size))
    scale = float(np.nanmedian(raw_b[:n_lo]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    shape_b = raw_b / scale
    min_transfer = float(np.nanmin(shape_b))

    out = {
        "k": k, "transfer": b, "transfer_raw": b_raw, "noise_power": noise,
        "clean_power": pc, "dirty_power": pd,
        "min_transfer": min_transfer,
        "scale_factor": scale,
        "k_at_min": float(kb[int(np.nanargmin(shape_b))]),
        "band_k_max": float(kb.max()), "band_bins": int(band.sum()),
    }

    # Before believing "no convolution", check the measurement could have SEEN one.
    #
    # The band ends where the dirty spectrum sinks into its own noise floor. A beam only
    # suppresses above roughly k_half = sqrt(ln 2 / (2 pi^2 sigma^2)); for this project's
    # headers sigma is about 6.5 px, so nothing happens below k ~ 0.03. If the band stops
    # short of that, a beam of the header's own size would be invisible and a null result
    # says nothing at all. That is RULES.md #8: check the denominator before believing a
    # clean result.
    sigma_hdr = _header_sigma_px(header) if header is not None else None
    if sigma_hdr:
        k_half = float(np.sqrt(np.log(2.0) / (2.0 * np.pi ** 2 * sigma_hdr ** 2)))
        out["k_half_from_header"] = k_half
        if min_transfer > 0.9 and out["band_k_max"] < k_half:
            out["verdict"] = "indeterminate"
            out["reason"] = (
                f"no suppression seen, but the band only reaches k = {out['band_k_max']:.3f} "
                f"and the header's own beam (sigma {sigma_hdr:.1f} px) would not act until "
                f"k = {k_half:.3f}. The measurement could not have detected this beam, so "
                f"the null result is uninformative rather than negative.")
            return out

    # A convolution suppresses relative to its own low-k level. Additive noise cannot: it only
    # adds power, so the normalised shape sits at or above 1 everywhere. Read the verdict off
    # this shape, never off the noise-subtracted transfer, whose floor estimate can
    # manufacture a dip on its own.
    if min_transfer > 0.9:
        out["verdict"] = "no_convolution"
        out["reason"] = (f"the shape of |B(k)| never falls below {min_transfer:.3f} of its "
                         f"low-k level; the dirty cube adds power without suppressing any, "
                         f"so A = I")
        return out

    # A Gaussian transfer function is itself Gaussian and therefore monotonically decreasing
    # in k. A true dirty beam's is not: uv gaps put ripples in it. Count sign changes of the
    # slope over the signal band, ignoring ones smaller than the local scatter.
    d = np.diff(bb)
    tol = 0.02 * max(np.nanmax(bb), 1e-12)
    sign = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    sign = sign[sign != 0]
    reversals = int(np.sum(sign[1:] != sign[:-1])) if sign.size > 1 else 0
    out["slope_reversals"] = reversals

    # Fit the shape too: an unremoved scale factor would bias the fitted sigma and the
    # header cross-check along with it.
    fitted = fit_gaussian_transfer(kb, bb / max(float(np.nanmedian(bb[:n_lo])), 1e-12))
    out.update(fitted)

    if reversals <= 1 and fitted["gaussian_rms_error"] < 0.05:
        out["verdict"] = "gaussian_convolution"
        out["reason"] = (f"|B(k)| falls to {min_transfer:.3f} of its low-k level, Gaussian to "
                         f"{fitted['gaussian_rms_error']:.4f} RMS")
    else:
        out["verdict"] = "non_gaussian_convolution"
        out["reason"] = (f"|B(k)| falls to {min_transfer:.3f} of its low-k level but departs "
                         f"from a Gaussian "
                         f"({reversals} slope reversals, {fitted['gaussian_rms_error']:.4f} "
                         f"RMS); a real dirty beam, so the PSF image is needed")

    if header is not None:
        out["header_sigma_px"] = _header_sigma_px(header)
    return out


def fit_gaussian_transfer(k: np.ndarray, b: np.ndarray) -> dict:
    """
    Fit |B(k)| = exp(-2 pi^2 sigma^2 k^2) by least squares on log|B|, which is linear in
    k^2. Returns the implied beam sigma in pixels and the RMS error of the fit.

    Only bins with |B| above a floor are used: below it the log is dominated by the noise
    subtraction and would drag the fit.
    """
    m = np.isfinite(b) & (b > 0.05)
    if m.sum() < 4:
        return {"gaussian_sigma_px": float("nan"), "gaussian_rms_error": float("inf")}

    x = k[m] ** 2
    y = np.log(b[m])
    slope = float(np.sum(x * y) / np.sum(x * x))  # forced through log|B(0)| = 0
    sigma_sq = -slope / (2.0 * np.pi ** 2)
    sigma = float(np.sqrt(sigma_sq)) if sigma_sq > 0 else float("nan")

    rms = float(np.sqrt(np.mean((b[m] - np.exp(slope * x)) ** 2)))
    return {"gaussian_sigma_px": sigma, "gaussian_rms_error": rms}


def _header_sigma_px(header) -> float | None:
    """Geometric-mean beam sigma in pixels predicted by the header, for cross-checking."""
    scale = pixel_scale_arcsec(header)
    if scale is None or not all(k in header for k in ("BMAJ", "BMIN")):
        return None
    maj = float(header["BMAJ"]) * 3600.0 / scale * FWHM_TO_SIGMA
    mn = float(header["BMIN"]) * 3600.0 / scale * FWHM_TO_SIGMA
    return float(np.sqrt(maj * mn))


def format_report(r: dict) -> str:
    sig = r.get("gaussian_sigma_px")
    have_sig = sig is not None and np.isfinite(sig)

    lines = [f"verdict: {r['verdict'].upper()}", f"  {r.get('reason', '')}"]
    if "min_transfer" in r:
        lines.append(f"  min |B(k)|/|B(0)| = {r['min_transfer']:.4f} at k = {r['k_at_min']:.3f} cyc/px")
    if r.get("scale_factor") is not None:
        sc = r["scale_factor"]
        note = ("   <- clean and dirty are NOT on one intensity scale (Jy/beam vs Jy/pixel?)"
                if (sc > 3.0 or sc < 0.33) else "")
        lines.append(f"  dirty/clean low-k amplitude = {sc:.3f}{note}")
    if have_sig:
        lines.append(f"  measured beam sigma  = {sig:.2f} px "
                     f"(Gaussian fit RMS {r['gaussian_rms_error']:.4f})")
    if "band_k_max" in r:
        line = (f"  band reaches k = {r['band_k_max']:.3f} cyc/px over {r['band_bins']} bins")
        if r.get("k_half_from_header"):
            kh = r["k_half_from_header"]
            line += (f"; header's beam would act from k = {kh:.3f}"
                     f"{'  <- band too narrow to see it' if r['band_k_max'] < kh else '  <- wide enough'}")
        lines.append(line)
    if r.get("header_sigma_px"):
        lines.append(f"  header beam sigma    = {r['header_sigma_px']:.2f} px")
        if have_sig:
            ratio = sig / r["header_sigma_px"]
            lines.append(f"  measured / header    = {ratio:.2f}"
                         f"{'  <- consistent' if 0.8 < ratio < 1.25 else '  <- MISMATCH'}")
    return "\n".join(lines)


def phase0_from_fits(clean_path: str, dirty_path: str, max_channels: int = 8) -> dict:
    """
    Run Phase 0 on a clean/dirty FITS pair. Channels are subsampled evenly to `max_channels`
    because the spectra are averaged and a handful is already enough to beat the periodogram
    variance down.
    """
    from astropy.io import fits

    with fits.open(clean_path, memmap=False) as hdul:
        clean = np.squeeze(hdul[0].data).astype(np.float64)
    with fits.open(dirty_path, memmap=False) as hdul:
        dirty = np.squeeze(hdul[0].data).astype(np.float64)
        header = hdul[0].header

    if clean.ndim == 3 and clean.shape[0] > max_channels:
        idx = np.linspace(0, clean.shape[0] - 1, max_channels).astype(int)
        clean, dirty = clean[idx], dirty[idx]
    return phase0_report(clean, dirty, header)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python -m src.evaluation.forward_operator <clean.fits> <dirty.fits>")
        raise SystemExit(2)
    print(format_report(phase0_from_fits(sys.argv[1], sys.argv[2])))
