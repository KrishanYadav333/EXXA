"""
Does 3D Wiener (no training) beat dirty on the GI wiggle across ALL 5 SG disks?

The 2026-10-02 result (0.925 vs dirty 0.892, U-Net 0.805) was one cube. This repeats it on the
five sg_synth disks, scored exactly as score_sg_wiggle_loo.py does: shared Keplerian fit from the
clean cube with the true inclination held fixed, signal_mask frac 0.05, resid_r against clean.

The Wiener filter uses NO clean data: the beam is the recovered v2 beam (the same one that made
these pairs), the noise spectrum is beam^2 * flat level estimated from the line-free edge
channels of the dirty cube itself. Noise here is white-then-beam-convolved (see
synthesize_sg_pairs.py), independent between channels, so spectral Wiener filtering applies.

Run (CPU, ~minutes/disk):  PYTHONPATH=.. python3 experiments/wiggle_wiener3d_loo.py
"""
import os, sys, json, time, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from astropy.io import fits

spec = importlib.util.spec_from_file_location("gi_wiggle", os.path.join(ROOT, "src/evaluation/gi_wiggle.py"))
gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)

SYN = os.path.join(ROOT, "self-gravitating cube and dirty cube/sg_synth")
BEAM = os.path.join(ROOT, "results/self-gravitating/dirty_beam_recovered_v2.fits")
FOLDS = [("run_9015_00370_rt_00", 20.0), ("run_9019_00019_rt_00", 30.0), ("run_9025_00370_rt_00", 20.0),
         ("run_9032_00020_rt_00", 20.0), ("run_9074_00025_rt_00", 20.0)]
FRAC = 0.05
EDGES = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.5])


def quad_m1(cube, velax):
    C = cube.shape[0]
    idx = np.clip(np.argmax(cube, axis=0), 1, C - 2)
    yy, xx = np.indices(idx.shape)
    ym, y0, yp = cube[idx - 1, yy, xx], cube[idx, yy, xx], cube[idx + 1, yy, xx]
    den = ym - 2 * y0 + yp
    off = np.clip(np.where(den != 0, 0.5 * (ym - yp) / den, 0.0), -0.5, 0.5)
    return velax[idx] + off * (velax[1] - velax[0])


def radial(P, rb, cnt, nb):
    return (np.bincount(rb.ravel(), P.ravel(), nb) / cnt)[rb]


def coherence(x, ref, mask, kr):
    def prep(a):
        a = np.where(mask & np.isfinite(a), a, 0.0)
        return a - a[mask].mean() * mask
    X, Rf = np.fft.fft2(prep(x)), np.fft.fft2(prep(ref))
    out = []
    for e1, e2 in zip(EDGES[:-1], EDGES[1:]):
        b = (kr >= e1) & (kr < e2)
        out.append(float(np.sum(X[b] * np.conj(Rf[b])).real / np.sqrt(np.sum(abs(X[b]) ** 2) * np.sum(abs(Rf[b]) ** 2))))
    return out


beam = fits.getdata(BEAM).astype(np.float64)
beam = beam / beam.sum()   # unit DC gain, as synthesize_sg_pairs.py built the pairs
rows = []
t0 = time.time()
for run, incl in FOLDS:
    d = f"{SYN}/{run}"
    with fits.open(f"{d}/{run}_clean.fits", memmap=True) as h:
        hdr, clean = h[0].header, h[0].data[:].astype(np.float64)
    with fits.open(f"{d}/{run}_dirty.fits", memmap=True) as h:
        dirty = h[0].data[:].astype(np.float64)
    C, n, _ = clean.shape
    velax = hdr["CRVAL3"] + (np.arange(C) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    au = abs(hdr["CDELT1"]) * 3600.0 * float(hdr["DIST_PC"])

    pad = np.zeros((n, n)); hb = beam.shape[0] // 2; c = n // 2
    pad[c - hb:c + hb + 1, c - hb:c + hb + 1] = beam
    H = np.fft.fft2(np.fft.ifftshift(pad))
    Hs = np.where(np.abs(H) > 1e-6 * np.abs(H).max(), H, 1e-6 * np.abs(H).max())
    kr = np.sqrt(np.fft.fftfreq(n)[None, :] ** 2 + np.fft.fftfreq(n)[:, None] ** 2)
    NB = 100; rb = np.minimum((kr / kr.max() * NB).astype(int), NB - 1); cnt = np.maximum(np.bincount(rb.ravel(), minlength=NB), 1)

    # Noise = beam (*) white noise, so P_noise(k) = s^2 |H(k)|^2. Estimate s^2 from the dirty
    # cube's faintest channels at high k, where signal is negligible (no clean data), and
    # separately build an ORACLE spectrum from dirty - beam (*) clean to tell "method fails"
    # apart from "noise estimate is off".
    Hm2 = np.abs(H) ** 2
    ch_pow = np.abs(dirty).reshape(C, -1).mean(1)
    quiet = np.argsort(ch_pow)[:max(10, C // 8)]
    Pq = np.mean(np.abs(np.fft.fft2(dirty[quiet] - dirty[quiet].mean(axis=(1, 2), keepdims=True))) ** 2, axis=0)
    hi = (kr > 0.12) & (kr < 0.35) & (Hm2 > 1e-8 * Hm2.max())
    s2 = float(np.median(Pq[hi] / Hm2[hi]))
    Pn_est = radial(s2 * Hm2, rb, cnt, NB)
    nz = dirty - np.real(np.fft.ifft2(np.fft.fft2(clean) * H[None]))
    Pn_orc = radial(np.mean(np.abs(np.fft.fft2(nz[::4])) ** 2, axis=0), rb, cnt, NB)
    del nz
    D = np.fft.fftn(dirty)
    Pd = np.stack([radial(p, rb, cnt, NB) for p in np.abs(D) ** 2]) / C
    cubes = {"clean": clean, "dirty": dirty}
    for tag, Pn in (("est", Pn_est), ("orc", Pn_orc)):
        for al in (1.0, 2.0):
            G = np.clip(1 - al * Pn[None] / np.maximum(Pd, 1e-30), 0, 1)
            cubes[f"wiener3d_{tag}_a{al}"] = np.real(np.fft.ifftn(D * G / Hs[None]))
    del D, Pd

    m1 = {k: quad_m1(v, velax) / (1.0 if abs(velax).max() < 50 else 1000.0) for k, v in cubes.items()}
    m0 = clean.sum(0)
    mask = np.isfinite(m0) & (np.abs(m0) > FRAC * np.abs(m0).max())
    geom = gw.fit_keplerian(m1["clean"], mask, au, fix_incl_deg=incl)
    ref = gw.wiggle_residual(m1["clean"], geom)
    row = {"holdout": run, "geom_at_bound": bool(geom["mstar_at_bound"]), "methods": {}}
    for k in cubes:
        if k == "clean":
            continue
        r = gw.wiggle_residual(m1[k], geom)
        ok = mask & np.isfinite(ref) & np.isfinite(r)
        row["methods"][k] = {"resid_r": float(np.corrcoef(ref[ok], r[ok])[0, 1]),
                             "coh": coherence(r, ref, mask, kr),
                             "psnr": float(20 * np.log10(clean.max() / np.sqrt(np.mean((cubes[k] - clean) ** 2))))}
    rows.append(row)
    print(f"{run}  bound={row['geom_at_bound']}  " + "  ".join(f"{k} r={v['resid_r']:.3f} psnr={v['psnr']:.1f}" for k, v in row["methods"].items()),
          f" [{(time.time() - t0) / 60:.1f} min]", flush=True)
    json.dump(rows, open(os.path.join(ROOT, "results/self-gravitating/wiener3d_loo.json"), "w"), indent=1)

print("\nmean resid_r over 5 disks")
for k in rows[0]["methods"]:
    v = np.array([r["methods"][k]["resid_r"] for r in rows])
    print(f"  {k:16s} {v.mean():.3f} +/- {v.std(ddof=1):.3f}   per-disk wins vs dirty: "
          f"{int(np.sum(v > np.array([r['methods']['dirty']['resid_r'] for r in rows])))}/5")
