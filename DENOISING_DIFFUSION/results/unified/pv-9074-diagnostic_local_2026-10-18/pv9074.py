"""9074 PV diagnostic: why does even the beam-only oracle score negative? Profiles of clean / oracle / dirty and alternative detrends."""
import sys, os, json
import numpy as np
os.chdir("/Users/Krishan/Projects/EXXA/DENOISING_DIFFUSION"); sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import scipy.fft as sfft
from scipy.ndimage import gaussian_filter1d
import wiggle_scorecard as ws
import pv_curve as pv
from train_unified import beam_otf

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pv9074"
os.makedirs(OUT, exist_ok=True)
name = sys.argv[2] if len(sys.argv) > 2 else "9074"
d = ws.load(name)
cl, dr = d["clean"].astype(np.float32), d["dirty"].astype(np.float32)
n = cl.shape[-1]
orc = sfft.ifft2(sfft.fft2(cl, workers=-1) * beam_otf(n)[None], workers=-1).real.astype(np.float32)
velax, au = d["velax"], d["au"]
m1 = {k: ws.quad_m1(v, velax) for k, v in (("clean", cl), ("oracle", orc), ("dirty", dr))}
m0 = cl.sum(0)
mask = np.isfinite(m0) & (np.abs(m0) > ws.FRAC * np.abs(m0).max())
geom = ws.gw.fit_keplerian(m1["clean"], mask, au, m0=m0, fix_incl_deg=d["incl"])
r_in, r_out = 2 * d["beam_au"], 0.0
yy, xx = np.indices((n, n), dtype=np.float64)
pa, ci = np.radians(geom["pa_deg"]), np.cos(np.radians(geom["incl_deg"]))
dx, dy = xx - geom["cx"], yy - geom["cy"]
rdep = np.hypot(dx * np.cos(pa) + dy * np.sin(pa), (-dx * np.sin(pa) + dy * np.cos(pa)) / ci) * au
w = np.clip(m0, 0, None).ravel(); o = np.argsort(rdep.ravel())
r_out = float(rdep.ravel()[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.95)])
edge = min(geom["cx"], geom["cy"], n - 1 - geom["cx"], n - 1 - geom["cy"]) - 2
r_out = min(r_out, edge * au / ci)
print(f"{name}: n {n}, au/px {au:.2f}, incl {d['incl']}, beam {d['beam_au']:.1f} au, R {r_in:.0f}-{r_out:.0f} au, geom pa {geom['pa_deg']:.1f} vsys {geom['vsys']:.3f}")

res = {}
for wd in (1, 5):
    r_grid, _ = pv.pv_profile(m1["clean"], geom, au, r_in, r_out)
    prof = {k: pv.pv_profile(v, geom, au, r_in, r_out, width_px=wd, mask=mask)[1] for k, v in m1.items()}
    for deg in (2, 4, 6):
        sc = pv.pv_scores(prof, deg=deg)
        res[f"w{wd}_poly{deg}"] = {k: round(sc[k]["pv_r"], 3) for k in sc}

    def hp(y, sig_au):
        g = np.isfinite(y)
        yy_ = np.interp(np.arange(len(y)), np.where(g)[0], y[g]) if g.sum() > 3 else y
        return np.where(g, yy_ - gaussian_filter1d(yy_, sig_au / au, mode="nearest"), np.nan)
    for sig in (100, 150, 250):
        hpd = {k: [hp(h, sig) for h in v] for k, v in prof.items()}
        sc = pv.pv_scores(hpd, deg=0)
        res[f"w{wd}_hp{sig}au"] = {k: round(sc[k]["pv_r"], 3) for k in sc}
    if wd == 1:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(14, 7))
        for j in range(2):
            for k, c in (("clean", "k"), ("oracle", "C2"), ("dirty", "C0")):
                ax[0, j].plot(r_grid, prof[k][j], c, lw=2 if k == "clean" else 1, label=k)
                ax[1, j].plot(r_grid, hp(prof[k][j], 150), c, lw=2 if k == "clean" else 1, label=k)
            ref = prof["clean"][j]; lo, hi = np.nanmin(ref), np.nanmax(ref); pad = max(hi - lo, 60)
            ax[0, j].set_ylim(lo - pad, hi + pad); ax[0, j].set_title(f"{'+' if j == 0 else '-'} half, raw profile (m/s)"); ax[0, j].legend()
            ax[1, j].set_title("after a 150 au Gaussian high-pass"); ax[1, j].legend()
        fig.suptitle(f"{name} PV: clean vs beam-only oracle vs dirty"); fig.tight_layout(); fig.savefig(f"{OUT}/{name}_pv_diag.png", dpi=80)
print(json.dumps(res, indent=1))
json.dump({"geom": {k: float(v) for k, v in geom.items() if isinstance(v, (int, float))}, "r_in": r_in, "r_out": r_out, "scores": res},
          open(f"{OUT}/{name}_pv_diag.json", "w"), indent=1)
