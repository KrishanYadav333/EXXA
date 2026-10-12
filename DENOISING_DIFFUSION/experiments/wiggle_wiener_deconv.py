import numpy as np, os, sys, json

ROOT = "/Users/Krishan/Projects/EXXA/DENOISING_DIFFUSION"
SG = f"{ROOT}/self-gravitating cube and dirty cube/kinematic_data_v2"
NPZ = np.load(f"{ROOT}/experiments/wiggle_all_methods_step1.npz")
MASK = NPZ["mask"]
VKEP = NPZ["clean_m1"] - NPZ["clean_r"]
CH = np.arange(240, 361)
OUT = os.path.dirname(os.path.abspath(__file__))


def read_fits(path, chans=None):
    f = open(path, "rb")
    hdr = b""
    while True:
        blk = f.read(2880)
        hdr += blk
        if b"END" + b" " * 77 in blk or not blk:
            break
    cards = {}
    for i in range(0, len(hdr), 80):
        c = hdr[i:i + 80].decode("ascii", "replace")
        if "=" in c[:10]:
            k = c[:8].strip()
            v = c[10:].split("/")[0].strip().strip("'").strip()
            cards[k] = v
    bitpix = int(cards["BITPIX"])
    dt = {-32: ">f4", -64: ">f8"}[bitpix]
    shape = [int(cards[f"NAXIS{i}"]) for i in range(int(cards["NAXIS"]), 0, -1)]
    off = f.tell()
    f.close()
    mm = np.memmap(path, dtype=dt, mode="r", offset=off, shape=tuple(shape))
    bscale = float(cards.get("BSCALE", 1)); bzero = float(cards.get("BZERO", 0))
    a = mm if chans is None else mm[chans]
    return np.asarray(a, dtype=np.float64) * bscale + bzero, cards


def quad_m1(cube, velax):
    """Parabola through peak channel and its two neighbours (bettermoments.collapse_quadratic)."""
    C = cube.shape[0]
    idx = np.argmax(cube, axis=0)
    idx = np.clip(idx, 1, C - 2)
    yy, xx = np.indices(idx.shape)
    ym, y0, yp = cube[idx - 1, yy, xx], cube[idx, yy, xx], cube[idx + 1, yy, xx]
    den = ym - 2 * y0 + yp
    off = np.where(den != 0, 0.5 * (ym - yp) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    dv = velax[1] - velax[0]
    return velax[idx] + off * dv


def radial(P, kr, nb=150):
    bins = np.linspace(0, kr.max() + 1e-9, nb + 1)
    which = np.digitize(kr.ravel(), bins) - 1
    s = np.bincount(which, P.ravel(), nb); n = np.bincount(which, minlength=nb)
    prof = s / np.maximum(n, 1)
    return prof[np.clip(which, 0, nb - 1)].reshape(P.shape)


def band_stats(x, ref, mask, kr, edges):
    def prep(a):
        a = np.where(mask & np.isfinite(a), a, 0.0)
        return a - a[mask].mean() * mask
    X, R = np.fft.fft2(prep(x)), np.fft.fft2(prep(ref))
    coh, pw = [], []
    for e1, e2 in zip(edges[:-1], edges[1:]):
        b = (kr >= e1) & (kr < e2)
        coh.append(float(np.sum(X[b] * np.conj(R[b])).real / np.sqrt(np.sum(abs(X[b]) ** 2) * np.sum(abs(R[b]) ** 2))))
        pw.append(float(np.sum(abs(X[b]) ** 2) / np.sum(abs(R[b]) ** 2)))
    return coh, pw


def resid_r(m1):
    r = m1 - VKEP
    ok = MASK & np.isfinite(r)
    return float(np.corrcoef(r[ok], NPZ["clean_r"][ok])[0, 1]), r


if __name__ == "__main__":
    clean, hc = read_fits(f"{SG}/clean_sg.fits", CH)
    dirty, _ = read_fits(f"{SG}/dirty_sg.fits", CH)
    crval, cdelt, crpix = float(hc["CRVAL3"]), float(hc["CDELT3"]), float(hc["CRPIX3"])
    velax = crval + (CH + 1 - crpix) * cdelt
    print("velax km/s", velax[0], velax[-1], "dv", velax[1] - velax[0])

    # 1) sanity: my quadratic centroid vs the stored bettermoments maps
    for name, cube in (("clean", clean), ("dirty", dirty)):
        m = quad_m1(cube, velax)
        ref = NPZ[f"{name}_m1"]
        ok = MASK & np.isfinite(ref)
        print(f"sanity {name}: corr {np.corrcoef(m[ok], ref[ok])[0,1]:.5f}  median|diff| {np.median(abs(m[ok]-ref[ok])):.2e} km/s")

    # 2) beam transfer function
    beam, _ = read_fits(f"{ROOT}/results/self-gravitating/dirty_beam_recovered_v2.fits")
    n = clean.shape[-1]
    pad = np.zeros((n, n)); h = beam.shape[0] // 2; c = n // 2
    pad[c - h:c + h + 1, c - h:c + h + 1] = beam
    H = np.fft.fft2(np.fft.ifftshift(pad))
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.fftfreq(n)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)

    # 3) noise PSD from line-free channels of the full dirty cube (no clean used)
    nf, _ = read_fits(f"{SG}/dirty_sg.fits", np.r_[0:60, 541:601])
    Pn = radial(np.mean(np.abs(np.fft.fft2(nf, axes=(-2, -1))) ** 2, axis=0), kr)
    del nf
    D = np.fft.fft2(dirty, axes=(-2, -1))
    Pd = radial(np.mean(np.abs(D) ** 2, axis=0), kr)
    print("noise/dirty power at k=0.05,0.15,0.3:",
          [round(float(np.median((Pn / Pd)[(kr > k - .005) & (kr < k + .005)])), 3) for k in (0.05, 0.15, 0.3)])

    edges = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.5])
    results = {}
    for name in ("dirty", "beam-only", "U-Net", "DDRM"):
        m1 = NPZ[f"{name}_m1"]
        r, _ = resid_r(m1)
        coh, pw = band_stats(m1 - VKEP, NPZ["clean_r"], MASK, kr, edges)
        results[name] = {"resid_r": r, "coh": coh, "pow": pw}

    best = None
    for alpha in (0.5, 1.0, 2.0, 4.0, 8.0):
        gain = np.clip(1.0 - alpha * Pn / np.maximum(Pd, 1e-30), 0.0, 1.0)
        Hs = np.where(np.abs(H) > 1e-8 * np.abs(H).max(), H, 1e-8 * np.abs(H).max())
        W = gain / Hs
        Xw = np.real(np.fft.ifft2(D * W[None], axes=(-2, -1)))
        m1 = quad_m1(Xw, velax)
        r, res = resid_r(m1)
        coh, pw = band_stats(res, NPZ["clean_r"], MASK, kr, edges)
        results[f"wiener_a{alpha}"] = {"resid_r": r, "coh": coh, "pow": pw}
        print(f"wiener alpha={alpha}: resid_r {r:.3f}")
        if best is None or r > best[0]:
            best = (r, alpha, m1)
    np.save(os.path.join(OUT, "wiener_best_m1.npy"), best[2])
    lab = [f"{1/e2:.0f}-{(1/e1 if e1 > 0 else np.inf):.0f}px" for e1, e2 in zip(edges[:-1], edges[1:])]
    print("\nbands:", lab)
    for k, v in results.items():
        print(f"{k:14s} r={v['resid_r']:.3f} coh=" + " ".join(f"{c:+.2f}" for c in v["coh"]) + "  pow=" + " ".join(f"{p:.2f}" for p in v["pow"]))
    json.dump({"bands": lab, "best_alpha": best[1], "results": results}, open(os.path.join(OUT, "wiener_results.json"), "w"), indent=1)
