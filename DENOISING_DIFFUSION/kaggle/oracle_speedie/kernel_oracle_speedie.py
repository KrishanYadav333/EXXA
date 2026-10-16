"""
Kaggle CPU kernel (PLAN.md 0b item 7, 2026-10-15): how much of the Speedie GI wiggle is recoverable at each peak SNR?
The oracle LINEAR filter (experiments/oracle_limit.py, noise-only regime): per velocity frequency, Ps / (Ps + Pn) with Ps and Pn the radially averaged power
of clean and of the noise. It knows both spectra (no real method does), so it is the best shift-invariant linear filter possible; R6 beat it on 9074
(0.45 vs 0.38), so it is not an upper bound for a non-linear model, but it tells where linear filtering stops. Cases exactly as train_unified.eval_cases
builds them (Speedie native noise, and the SNR ladder: Speedie's own noise rescaled to peak SNR 3 / 6 / 12 / 24), scored with the same scorecard.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/oracle_speedie"

if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO], check=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
os.chdir(PKG)
sys.path[:0] = [PKG, f"{PKG}/experiments"]
os.environ["EXXA_ROOT"] = PKG
import numpy as np
import scipy.fft as sfft
from astropy.io import fits
import wiggle_scorecard as ws

SPD = os.path.dirname(glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)[0])
os.makedirs(f"{OUT}/figs", exist_ok=True)      # score_cube writes its figure there and does not create the folder


def oracle_ai(clean, dirty):
    C, n, _ = clean.shape
    kr = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]); nb = 120
    rb = np.minimum((kr / kr.max() * nb).astype(int), nb - 1); cnt = np.maximum(np.bincount(rb.ravel(), minlength=nb), 1)
    S = sfft.fftn(clean.astype(np.float32), workers=-1)
    N = sfft.fftn((dirty - clean).astype(np.float32), workers=-1)
    D = sfft.fftn(dirty.astype(np.float32), workers=-1)
    for i in range(C):
        Ps = ws._radial_avg(np.abs(S[i]) ** 2, rb, cnt, nb); Pn = ws._radial_avg(np.abs(N[i]) ** 2, rb, cnt, nb)
        D[i] *= (Ps / (Ps + Pn + 1e-30)).astype(np.float32)
    del S, N
    return sfft.ifftn(D, workers=-1).real.astype(np.float32)


def cases():
    for name in ("GI_r0.5", "Kep_r0.5", "GI_r1.5", "Kep_r1.5"):
        cl = fits.getdata(f"{SPD}/{name}_clean.fits").astype(np.float32)
        dr = fits.getdata(f"{SPD}/{name}_dirty.fits").astype(np.float32)
        yield f"speedie_{name}", name, cl, dr
        if name.endswith("r0.5"):
            nz = dr - cl
            for snr in (3, 6, 12, 24):
                f = float(cl.max()) / snr / max(float(nz.std()), 1e-30)
                yield f"speedie_ladder_{name}_snr{snr}", name, cl, cl + f * nz


res = {}
for key, base, cl, dr in cases():
    t = time.time()
    hdr = fits.getheader(f"{SPD}/{base}_clean.fits")
    velax = (hdr["CRVAL3"] + (np.arange(cl.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) / 1000.0
    d = {"clean": cl, "dirty": dr, "velax": velax, "au": abs(hdr["CDELT1"]) * 3600 * 155.9, "incl": 23.2, "crop": slice(None),
         "beam_au": float(hdr["BMAJ"]) * 3600 * 155.9}
    ws.METHODS["oracle_linear"] = lambda _d, cl=cl, dr=dr: oracle_ai(cl, dr)
    sc = ws.score_cube(key, d, ["dirty", "oracle_linear"], f"{OUT}/figs")["methods"]
    res[key] = {m: {k: sc[m][k] for k in ("rc_r", "rc_rms_ms", "rc_rms_clean_ms", "dv_med_ms")} for m in sc}
    json.dump(res, open(f"{OUT}/oracle_speedie.json", "w"), indent=1)
    o, dd = sc["oracle_linear"], sc["dirty"]
    print(f"{key:36s} rc_r dirty {dd['rc_r']:.3f} ORACLE {o['rc_r']:.3f} | spurious (rc_rms - clean) dirty {dd['rc_rms_ms'] - dd['rc_rms_clean_ms']:+.1f} "
          f"oracle {o['rc_rms_ms'] - o['rc_rms_clean_ms']:+.1f} m/s | dv_med {dd['dv_med_ms']:.0f}/{o['dv_med_ms']:.0f} m/s [{time.time() - t:.0f}s]", flush=True)
print("ORACLE SPEEDIE DONE", flush=True)
