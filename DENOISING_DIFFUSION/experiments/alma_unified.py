"""
Real exoALMA cube -> the unified denoiser (Wiener + learned correction, train_unified.py) -> denoised cube, moment maps, checks. ALMA_PLAN.md.

INFERENCE ONLY: exoALMA data are never trained on (Jason). There is no truth on a real cube, so this measures what the model CHANGES and whether
it invents anything (ALMA_PLAN.md section 6):
  S1  noise off the line (rms of line-free planes, raw vs denoised), M0 total ratio inside the signal mask, M1 shift on the brightest pixels (m/s)
  S2  invented structure: fraction of line-free-plane pixels above 4 sigma_raw, raw vs denoised (the raw rate is the denominator; RULES.md #8)
  S6  kinematics: rotation-curve residual of M1 (wiggle_scorecard), raw vs denoised: correlation and rms
Same preprocessing as training and the synthetic tests: continuum removed with the line-free planes' mean, the velocity-aware Wiener on the native grid
with its noise from the line-free planes, the model's own normalisation (--norm and --clip-in read back from the checkpoint), optional resampling to
a pixel scale (--scale) and back. Several checkpoints are averaged (same K / arm / base).

  python experiments/alma_unified.py --cube MWC_758_13CO_fiducial.image.fits --ckpt a.pth[,b.pth] --out results/alma/mwc758_13co
         [--incl 21 --pa 240 --dist 155.9] [--crop 1024] [--scale 1.0] [--budget-h 11]
"""
import argparse, json, math, os, sys, time
import numpy as np

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import torch
from astropy.io import fits
import train_unified as tu
import wiggle_scorecard as ws

C_KMS = 299792.458


def read_cube(path, crop):
    """(C, n, n) float32 with velocity ascending (km/s), the header, pixel (arcsec), beam (arcsec). Accepts a truncated download (whole planes only)."""
    with fits.open(path, memmap=True, ignore_missing_end=True) as h:
        hdr = h[0].header
        off = h[0]._data_offset
    shape = tuple(int(hdr[f"NAXIS{k}"]) for k in range(int(hdr["NAXIS"]), 0, -1))
    dtype = {-32: ">f4", -64: ">f8"}[int(hdr["BITPIX"])]
    plane = int(np.prod(shape[-2:])) * np.dtype(dtype).itemsize
    nplanes = min(int(np.prod(shape[:-2])), (os.path.getsize(path) - off) // plane)   # a truncated download keeps its whole planes
    if nplanes < int(np.prod(shape[:-2])):
        print(f"  {os.path.basename(path)} is truncated: {nplanes} of {int(np.prod(shape[:-2]))} planes present", flush=True)
    data = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=(nplanes,) + shape[-2:])
    cube = np.asarray(data, dtype=np.float32)          # 4D (Stokes, chan, y, x) with one Stokes plane reads as (chan, y, x) here
    C = cube.shape[0]
    nu0 = float(hdr.get("RESTFRQ") or hdr.get("RESTFREQ"))
    ax = float(hdr["CRVAL3"]) + (np.arange(C) + 1 - float(hdr["CRPIX3"])) * float(hdr["CDELT3"])
    if str(hdr["CTYPE3"]).upper().startswith("FREQ"):
        velax = C_KMS * (nu0 - ax) / nu0                                      # radio convention
    else:
        velax = ax / (1.0 if str(hdr.get("CUNIT3", "m/s")).lower().startswith("km") else 1000.0)
    if velax[1] < velax[0]:
        cube, velax = cube[::-1], velax[::-1]
    n = cube.shape[-1]
    c = min(crop or n, n); c -= c % 2
    o = (n - c) // 2
    cube = np.nan_to_num(np.ascontiguousarray(cube[:, o:o + c, o:o + c]), nan=0.0)   # NaN outside the primary beam -> 0
    pix = abs(float(hdr["CDELT1"])) * 3600
    return cube, np.ascontiguousarray(velax), hdr, pix, (float(hdr["BMAJ"]) * 3600, float(hdr["BMIN"]) * 3600)


def line_free(cube, nsig=3.0):
    """Planes without line emission: iterative sigma clipping of the spectrum of the central half of the field (median / MAD, 3 sigma, until stable).
    Line planes are outliers whatever their position, so this works when the edges hold line emission (a truncated download, a narrow band)."""
    C, n = cube.shape[0], cube.shape[-1]
    q = n // 4
    spec = cube[:, q:n - q, q:n - q].astype(np.float64).sum((1, 2))
    keep = np.ones(C, bool)
    for _ in range(20):
        med = np.median(spec[keep]); mad = 1.4826 * np.median(np.abs(spec[keep] - med)) + 1e-30
        new = np.abs(spec - med) < nsig * mad
        if (new == keep).all():
            break
        keep = new
    lf = np.where(keep)[0]
    return (lf if len(lf) >= 6 else np.r_[0:max(3, C // 10), C - max(3, C // 10):C]), spec


def load_models(paths):
    ck = torch.load(paths[0], map_location="cpu", weights_only=False)
    tu.K, tu.CLIP_IN = int(ck["K"]), float((ck.get("args") or {}).get("clip_in", 0.0))
    tu.NORM = (ck.get("args") or {}).get("norm", "wiener")
    arm, base = ck["arm"], int(ck["base"])
    nets = []
    for p in paths:
        c = torch.load(p, map_location="cpu", weights_only=False)
        assert (int(c["K"]), c["arm"], int(c["base"]), (c.get("args") or {}).get("norm", "wiener")) == (tu.K, arm, base, tu.NORM), f"{p}: members must match"
        net = tu.build_net(arm, base); net.load_state_dict(c["model_state_dict"], strict=True); nets.append(net)
    print(f"models: {len(nets)} x arm {arm}, K {tu.K}, base {base}, norm {tu.NORM}, clip {tu.CLIP_IN}", flush=True)
    return tu.Ensemble(nets, False), arm


def moments(cube, velax, mask_sig, sigma):
    """M0 (K km/s or Jy/beam km/s), quadratic M1 (km/s), M2 (km/s) with a sigma clip, the project's own estimators."""
    dv = float(velax[1] - velax[0])
    clip = np.where(cube > mask_sig * sigma, cube, 0.0)
    m0 = clip.sum(0) * dv
    m1 = ws.quad_m1(cube, velax)
    w = np.clip(clip, 0, None)
    m1w = (w * velax[:, None, None]).sum(0) / np.maximum(w.sum(0), 1e-30)
    m2 = np.sqrt((w * (velax[:, None, None] - m1w) ** 2).sum(0) / np.maximum(w.sum(0), 1e-30))
    return m0, m1, m2


def regrid_velocity(cube, velax, dv_new):
    """Average a cube onto a coarser velocity grid (bins of width dv_new centred on 0); returns (cube, velax)."""
    lo, hi = velax[0], velax[-1]
    edges = np.arange(np.floor(lo / dv_new) - 0.5, np.ceil(hi / dv_new) + 0.5 + 1e-9) * dv_new
    idx = np.digitize(velax, edges) - 1
    k = np.unique(idx)
    out = np.stack([cube[idx == j].mean(0) for j in k]).astype(np.float32)
    return out, ((edges[k] + edges[k + 1]) / 2).astype(np.float64)


def inject(a, model, arm, cube, lf, velax_real, pix, beam):
    """S5: a cube with KNOWN truth in this observation's REAL noise. Truth = Speedie et al.'s GI disc and its Keplerian twin (speedie_prep clean cubes,
    test-only data), moved to this cube's pixel scale (area resampling) and channel width (velocity binning), centred, then added to blocks of the real
    cube's own line-free planes (real noise, its spatial and channel correlation kept) scaled to each target peak SNR. Same denoising path as the real cube."""
    import speedie_prep  # noqa: F401  (keeps the import surface of the test-only data in one place)
    out = {}
    dv = float(velax_real[1] - velax_real[0])
    n_real = cube.shape[-1]
    noise_planes = cube[lf]
    sig = float(noise_planes.std())
    for name in [x for x in a.inject_cases.split(",") if x]:
        base = os.path.join(a.inject_dir, f"{name}_clean.fits")
        h = fits.getheader(base); tr = fits.getdata(base).astype(np.float32)
        v = (h["CRVAL3"] + (np.arange(tr.shape[0]) + 1 - h["CRPIX3"]) * h["CDELT3"]) / 1000.0
        tr, v = regrid_velocity(tr, v, dv)
        S = int(round(tr.shape[-1] * abs(float(h["CDELT1"])) * 3600 / pix)); S -= S % 2
        tr = tu.resample(tr, S)
        if S >= n_real:
            o = (S - n_real) // 2; tr = tr[:, o:o + n_real, o:o + n_real]
        else:
            o = (n_real - S) // 2; tr = np.pad(tr, ((0, 0), (o, n_real - S - o), (o, n_real - S - o)))
        pad = int(min(20, max(5, (len(noise_planes) - tr.shape[0]) // 2)))   # line-free planes each side, as a real cube has
        tr = np.concatenate([np.zeros((pad,) + tr.shape[1:], np.float32), tr, np.zeros((pad,) + tr.shape[1:], np.float32)])
        v = np.r_[v[0] - dv * np.arange(pad, 0, -1), v, v[-1] + dv * np.arange(1, pad + 1)]
        C = tr.shape[0]
        if len(noise_planes) < C:
            print(f"  S5 {name}: only {len(noise_planes)} line-free planes for {C} needed; skipped", flush=True); continue
        nz = noise_planes[:C]
        for snr in [float(x) for x in a.inject_snr.split(",") if x]:
            truth = tr * np.float32(snr * sig / max(float(tr.max()), 1e-30))
            dirty = truth + nz
            lf_i, _ = line_free(dirty)
            w = tu.native_wiener(dirty - dirty[lf_i].mean(0)[None], "line", lf_i)
            pred = model.denoise(arm, dirty - dirty[lf_i].mean(0)[None], w, dv)
            d = {"clean": truth, "dirty": dirty, "velax": v, "au": pix * 155.9, "incl": 23.2, "crop": slice(None), "beam_au": beam[0] * 155.9}
            ws.METHODS["wiener"] = lambda _d, w=w: w; ws.METHODS["model"] = lambda _d, p=pred: p
            sc = ws.score_cube(f"S5_{name}_snr{snr:g}", d, ["dirty", "wiener", "model"], a.out)["methods"]
            m1d, m1p = ws.quad_m1(dirty, v), ws.quad_m1(pred, v)
            m0t = np.clip(truth, 0, None).sum(0); br = m0t >= np.percentile(m0t[m0t > 0], 90)
            row = {k: {"rc_r": sc[k]["rc_r"], "dv_med_ms": sc[k]["dv_med_ms"], "spurious_ms": sc[k]["rc_rms_ms"] - sc[k]["rc_rms_clean_ms"]} for k in ("dirty", "wiener", "model")}
            row["m1_shift_model_vs_dirty_bright_median_ms"] = float(np.nanmedian(np.abs(m1p - m1d)[br]) * 1000)
            out[f"{name}@snr{snr:g}"] = row
            print(f"  S5 {name} SNR {snr:g}: rc_r dirty {row['dirty']['rc_r']:.2f} wiener {row['wiener']['rc_r']:.2f} MODEL {row['model']['rc_r']:.2f} | "
                  f"line-centre error vs truth dirty {row['dirty']['dv_med_ms']:.0f} MODEL {row['model']['dv_med_ms']:.0f} m/s | "
                  f"|M1 model - M1 dirty| on bright px {row['m1_shift_model_vs_dirty_bright_median_ms']:.0f} m/s", flush=True)
    return out


def write(path, data, hdr, keep3d=True):
    h = fits.Header()
    for k in ("CTYPE1", "CTYPE2", "CDELT1", "CDELT2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", "BMAJ", "BMIN", "BPA", "BUNIT", "OBJECT", "RESTFRQ"):
        if k in hdr:
            h[k] = hdr[k]
    n = data.shape[-1]
    h["CRPIX1"] = h["CRPIX2"] = n / 2 + 0.5
    h["ORIGIN"] = "EXXA unified denoiser (train_unified.py); inference only"
    fits.writeto(path, data.astype(np.float32), h, overwrite=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--ckpt", required=True, help="comma list: averaged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=1.0, help="resample the cube by this factor before the model (and back); 1 = native pixels")
    ap.add_argument("--incl", type=float, default=None)
    ap.add_argument("--pa", type=float, default=None)
    ap.add_argument("--dist", type=float, default=150.0)
    ap.add_argument("--inject-dir", default="", help="S5: folder with speedie_prep clean cubes (GI_r0.5_clean.fits ...); empty = skip")
    ap.add_argument("--inject-cases", default="GI_r0.5,Kep_r0.5")
    ap.add_argument("--inject-snr", default="12,24,36")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cube, velax, hdr, pix, beam = read_cube(a.cube, a.crop)
    C, n = cube.shape[0], cube.shape[-1]
    dv = float(velax[1] - velax[0])
    lf, spec = line_free(cube)
    cont = cube[lf].mean(0)
    sigma_raw = float(cube[lf].std())
    cont_sig = float(np.abs(cont).max() / (sigma_raw / math.sqrt(len(lf))))
    cube = cube - cont[None]                      # removes a residual continuum / static pattern; ~0 if the image is already continuum-subtracted
    print(f"cube {os.path.basename(a.cube)}: {C} x {n} x {n}, {pix * 1000:.1f} mas/px, beam {beam[0]:.3f} x {beam[1]:.3f} arcsec "
          f"({beam[0] / pix:.1f} x {beam[1] / pix:.1f} px), dv {dv * 1000:.0f} m/s, line-free planes {len(lf)}, rms {sigma_raw:.3g}, "
          f"residual continuum peak {cont_sig:.1f} sigma_mean", flush=True)
    model, arm = load_models([p for p in a.ckpt.split(",") if p])
    w = tu.native_wiener(cube, "line", lf)
    if a.scale != 1.0:
        S = int(round(n * a.scale)); S -= S % 2
        d_in, w_in = tu.resample(cube, S), tu.resample(w, S)
    else:
        d_in, w_in = cube, w
    pred = model.denoise(arm, d_in, w_in, dv)
    if a.scale != 1.0:
        pred = tu.resample(pred, n) * np.float32(1.0)                 # area / bilinear back to the native grid
    print(f"denoised in {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- S1 / S2
    sig_den = float(pred[lf].std())
    m0r, m1r, m2r = moments(cube, velax, 3.0, sigma_raw)
    m0d, m1d, m2d = moments(pred, velax, 3.0, sigma_raw)
    s_m0 = sigma_raw * dv * math.sqrt(max(1, C - len(lf)))
    mask = m0r > 5 * s_m0
    bright = mask & (m0r >= np.percentile(m0r[mask], 90)) if mask.any() else mask
    dm1 = np.abs(m1d - m1r)[bright & np.isfinite(m1d) & np.isfinite(m1r)] * 1000
    inv_raw = float((np.abs(cube[lf]) > 4 * sigma_raw).mean())
    inv_den = float((np.abs(pred[lf]) > 4 * sigma_raw).mean())
    rep = {"cube": os.path.basename(a.cube), "shape": [C, n, n], "pix_mas": pix * 1000, "beam_arcsec": beam, "beam_px": [beam[0] / pix, beam[1] / pix],
           "dv_ms": dv * 1000, "line_free_planes": len(lf), "sigma_raw": sigma_raw, "residual_continuum_peak_sigma": cont_sig, "scale": a.scale,
           "S1": {"offline_rms_raw": sigma_raw, "offline_rms_den": sig_den, "noise_reduction": sigma_raw / max(sig_den, 1e-30),
                  "m0_total_ratio": float(m0d[mask].sum() / max(m0r[mask].sum(), 1e-30)) if mask.any() else None, "mask_px": int(mask.sum()),
                  "m1_shift_bright_median_ms": float(np.median(dm1)) if dm1.size else None, "m1_shift_bright_p95_ms": float(np.percentile(dm1, 95)) if dm1.size else None},
           "S2": {"frac_above_4sigma_raw_lineplanes": inv_raw, "frac_above_4sigma_den_lineplanes": inv_den}}
    # ---- S6: rotation-curve residual (needs the geometry; the fit uses the raw M1)
    if a.incl is not None:
        au = pix * a.dist
        # the scorecard builds its geometry mask from the reference M0 (5 % of its peak); with the RAW cube as reference that mask took in noise blobs over
        # the whole field (v1 / v2, 2026-10-13). Restrict both cubes to the S1 signal mask grown by one beam, so the fit, R_out and the spine see the disc only.
        from scipy import ndimage
        keep = ndimage.binary_dilation(mask, iterations=max(1, int(round(beam[0] / pix))))
        raw_m, pred_m = cube * keep[None], pred * keep[None]
        d = {"clean": raw_m, "dirty": raw_m, "velax": velax, "au": au, "incl": a.incl, "crop": slice(None), "beam_au": beam[0] * a.dist}
        ws.METHODS["denoised"] = lambda _d: pred_m
        try:
            sc = ws.score_cube(os.path.splitext(os.path.basename(a.cube))[0], d, ["denoised"], a.out)["methods"]["denoised"]
            rep["S6"] = {"rc_r_denoised_vs_raw": sc["rc_r"], "rc_rms_raw_ms": sc["rc_rms_clean_ms"], "rc_rms_den_ms": sc["rc_rms_ms"],
                         "note": "raw is the reference here (no truth): correlation of the denoised residual with the RAW residual"}
        except Exception as e:
            rep["S6"] = {"failed": f"{type(e).__name__}: {e}"}
    if a.inject_dir:
        rep["S5"] = inject(a, model, arm, cube, lf, velax, pix, beam)
    rep["minutes"] = (time.time() - t0) / 60
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    base = os.path.join(a.out, os.path.splitext(os.path.basename(a.cube))[0].replace(".image", ""))
    write(base + "_denoised.fits", pred, hdr)
    for tag, arr in (("m0_raw", m0r), ("m1_raw", m1r), ("m2_raw", m2r), ("m0_den", m0d), ("m1_den", m1d), ("m2_den", m2d)):
        write(f"{base}_{tag}.fits", arr, hdr)
    figure(base + "_moments.png", cube, pred, velax, m0r, m1r, m2r, m0d, m1d, m2d, lf, rep)
    print(json.dumps({k: rep[k] for k in ("S1", "S2") }, indent=1, default=float), flush=True)
    print("ALMA DONE", flush=True)


def figure(path, raw, den, velax, m0r, m1r, m2r, m0d, m1d, m2d, lf, rep):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    line = np.setdiff1d(np.arange(raw.shape[0]), lf)
    mid = int(line[len(line) // 2]) if len(line) else raw.shape[0] // 2
    vmin1, vmax1 = np.nanpercentile(m1r[m0r > 0], [2, 98]) if (m0r > 0).any() else (-1, 1)
    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    for r, (lab, cube, m0, m1, m2) in enumerate((("raw", raw, m0r, m1r, m2r), ("denoised", den, m0d, m1d, m2d))):
        axs[r, 0].imshow(cube[mid], origin="lower", cmap="inferno", vmin=0, vmax=np.percentile(raw[mid], 99.9)); axs[r, 0].set_title(f"{lab}: channel {velax[mid]:.2f} km/s")
        axs[r, 1].imshow(m0, origin="lower", cmap="inferno", vmin=0, vmax=np.percentile(m0r, 99.5)); axs[r, 1].set_title(f"{lab}: M0")
        axs[r, 2].imshow(np.where(m0r > 0, m1, np.nan), origin="lower", cmap="RdBu_r", vmin=vmin1, vmax=vmax1); axs[r, 2].set_title(f"{lab}: M1 (quadratic)")
        axs[r, 3].imshow(np.where(m0r > 0, m2, np.nan), origin="lower", cmap="viridis", vmin=0, vmax=np.nanpercentile(m2r[m0r > 0], 98) if (m0r > 0).any() else 1); axs[r, 3].set_title(f"{lab}: M2")
        axs[r, 4].imshow(cube[lf[len(lf) // 2]], origin="lower", cmap="RdBu_r", vmin=-4 * rep["sigma_raw"], vmax=4 * rep["sigma_raw"]); axs[r, 4].set_title(f"{lab}: a line-free plane (+-4 sigma_raw)")
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    s1 = rep["S1"]
    fig.suptitle(f"{rep['cube']}: noise x{s1['noise_reduction']:.1f} lower off-line, M0 total ratio {s1['m0_total_ratio']}, "
                 f"M1 shift on bright px median {s1['m1_shift_bright_median_ms']} m/s; >4 sigma in line-free planes raw {rep['S2']['frac_above_4sigma_raw_lineplanes']:.2e} "
                 f"denoised {rep['S2']['frac_above_4sigma_den_lineplanes']:.2e}")
    fig.savefig(path, dpi=70, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
