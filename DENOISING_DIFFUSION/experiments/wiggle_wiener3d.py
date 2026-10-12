import numpy as np, json, os
from wiener_wiggle import read_fits, quad_m1, band_stats, resid_r, SG, ROOT, NPZ, MASK, VKEP

OUT = os.path.dirname(os.path.abspath(__file__))
BLK = np.arange(180, 421)                 # processing block (241 ch), scored on 240-360
CROP = slice(240 - 180, 361 - 180)

clean, hc = read_fits(f"{SG}/clean_sg.fits", BLK)
dirty, _ = read_fits(f"{SG}/dirty_sg.fits", BLK)
crval, cdelt, crpix = float(hc["CRVAL3"]), float(hc["CDELT3"]), float(hc["CRPIX3"])
velax = crval + (np.arange(240, 361) + 1 - crpix) * cdelt

lf, _ = read_fits(f"{SG}/dirty_sg.fits", np.r_[0:60, 541:601])
static = lf.mean(0)
lf_ind = lf - static
a, b = lf_ind[:59].ravel(), lf_ind[1:60].ravel()
print("independent-noise corr at lag 1 after static removal:", round(np.corrcoef(a, b)[0, 1], 3))
dcs = dirty - static[None]

n = dirty.shape[-1]
beam, _ = read_fits(f"{ROOT}/results/self-gravitating/dirty_beam_recovered_v2.fits")
pad = np.zeros((n, n)); h = beam.shape[0] // 2; c = n // 2
pad[c - h:c + h + 1, c - h:c + h + 1] = beam
H = np.fft.fft2(np.fft.ifftshift(pad))
Hs = np.where(np.abs(H) > 1e-6 * np.abs(H).max(), H, 1e-6 * np.abs(H).max())
kr = np.sqrt(np.fft.fftfreq(n)[None, :] ** 2 + np.fft.fftfreq(n)[:, None] ** 2)
NB = 120
rb = np.minimum((kr / kr.max() * NB).astype(int), NB - 1)


def rad2(P):  # radial average of a 2D power map
    s = np.bincount(rb.ravel(), P.ravel(), NB) / np.maximum(np.bincount(rb.ravel(), minlength=NB), 1)
    return s[rb]


def rad3(P):  # radial average per spectral frequency
    out = np.empty_like(P)
    for i in range(P.shape[0]):
        out[i] = rad2(P[i])
    return out


Pn2 = rad2(np.mean(np.abs(np.fft.fft2(lf_ind)) ** 2, axis=0))      # independent noise, per channel
del lf, lf_ind

edges = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.5])
res = {}


def score(name, cube):
    m1 = quad_m1(cube[CROP], velax)
    r, rr = resid_r(m1)
    coh, pw = band_stats(rr, NPZ["clean_r"], MASK, kr, edges)
    res[name] = {"resid_r": r, "coh": coh, "pow": pw}
    np.save(os.path.join(OUT, f"m1_{name}.npy"), m1)
    print(f"{name:28s} resid_r {r:.3f}  coh " + " ".join(f"{x:+.2f}" for x in coh), flush=True)


score("dirty", dirty)
score("dirty_static_sub", dcs)

# 2D (spatial) Wiener after static subtraction, data-driven
D2 = np.fft.fft2(dcs)
Pd2 = rad2(np.mean(np.abs(D2) ** 2, axis=0))
for al in (1.0, 2.0):
    G = np.clip(1 - al * Pn2 / np.maximum(Pd2, 1e-30), 0, 1)
    score(f"wiener2d_static_a{al}", np.real(np.fft.ifft2(D2 * (G / Hs)[None])))
del D2

# 3D (spatial + spectral) Wiener, data-driven: noise white along spectrum after static removal
D3 = np.fft.fftn(dcs, axes=(0, 1, 2))
Pd3 = rad3(np.abs(D3) ** 2) / dcs.shape[0]
Pn3 = Pn2[None]
for al in (1.0, 2.0, 4.0):
    G = np.clip(1 - al * Pn3 / np.maximum(Pd3, 1e-30), 0, 1)
    score(f"wiener3d_a{al}", np.real(np.fft.ifftn(D3 * G / Hs[None], axes=(0, 1, 2))))

# ORACLE 3D Wiener: true signal power spectrum from clean (phases still from dirty).
# A learned prior can approach this; it is the ceiling for any LINEAR method.
C3 = np.fft.fftn(clean, axes=(0, 1, 2))
S3 = rad3(np.abs(C3) ** 2) / clean.shape[0]
del C3
Hm2 = np.abs(H) ** 2
score("oracle_wiener3d", np.real(np.fft.ifftn(D3 * (np.conj(H) * S3) / (Hm2 * S3 + Pn3), axes=(0, 1, 2))))

json.dump(res, open(os.path.join(OUT, "wiener3d_results.json"), "w"), indent=1)
