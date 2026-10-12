"""
PLAN.md B2: does src/data/sph_cube.py reproduce Speedie et al.'s own MCFOST cube of the same SPH dump?

Input: the Speedie dump sg_00005 (Zenodo 11668694) and speedie_prep.py's clean cubes ({GI,Kep}_r0.5_clean.fits: the MCFOST 13CO model,
1024 px crop, beam-convolved, on the synthetic ALMA velocity grid). Our render gets the same pixel grid, velocity axis, geometry
(155.9 pc, i = 23.2 deg) and beam. Sky-orientation conventions (PA sense, mirror, spin) are not documented in a form we can trust, so the
8 combinations are tried at reduced size and the one whose M1 correlates best with MCFOST's is kept: a convention choice, not a fit of
the physics (the geometry itself stays fixed). The Keplerian twin is rendered from the same dump with radial and vertical velocities
removed, as MCFOST's -no_vr -no_vz did.

Pass (PLAN.md B2): rc_r of our render against MCFOST's rotation-curve residual (the wiggle) well above what the noisy dirty cube
reaches (0.65 mean over r0.5/r1.5, PROGRESS.md 2026-10-10), and the Keplerian twin wiggle-free. Optical depth tau0 is the one free
emission parameter; every value tried is reported, not only the best.

Run: python experiments/validate_sph_speedie.py --dump sg_00005 --clean-dir <speedie out> --out results/sph-validation
"""
import os, sys, json, math, time, argparse
import numpy as np
from astropy.io import fits

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from src.data import sph_cube as sc
import speedie_prep as sp
import wiggle_scorecard as ws

DIST, INCL, PA = 155.9, 23.2, 236.7


def keplerian_twin(d):
    """Same particles, velocity reduced to its azimuthal component (MCFOST -no_vr -no_vz)."""
    R = np.hypot(d["x"], d["y"])
    vphi = (d["x"] * d["vy"] - d["y"] * d["vx"]) / np.maximum(R, 1e-9)
    out = dict(d)
    out["vx"], out["vy"], out["vz"] = -vphi * d["y"] / R, vphi * d["x"] / R, np.zeros_like(R)
    return out


def m1_of(cube, velax):
    w = np.clip(cube, 0, None)
    return (w * velax[:, None, None]).sum(0) / np.maximum(w.sum(0), 1e-30), w.sum(0)


def corr(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/sph-validation"))
    ap.add_argument("--tau", default="1,3,10,30,100")
    ap.add_argument("--which", default="GI,Kep")
    ap.add_argument("--length-scale", type=float, default=1.5, help="MCFOST was run with -scale_length_units 1.5 (GIdisk_13C16O_mcfost.para)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = sc.rescale(sc.load_phantom(a.dump), length=a.length_scale)     # velocities x sqrt(1/L): G M / r kept
    print(f"dump: {len(d['m']):,} gas particles, star {d['mstar']:.3f} Msun, disc {d['m'].sum():.3f} Msun, T {np.percentile(d['T'], [5, 50, 95]).round(1)} K", flush=True)
    report = {"dist_pc": DIST, "incl_deg": INCL, "pa_deg": PA}
    best_orient = None
    for which in a.which.split(","):
        path = os.path.join(a.clean_dir, f"{which}_r0.5_clean.fits")
        h = fits.getheader(path)
        ref = fits.getdata(path).astype(np.float32)
        C, n = ref.shape[0], ref.shape[-1]
        velax = (float(h["CRVAL3"]) + (np.arange(C) + 1 - float(h["CRPIX3"])) * float(h["CDELT3"])) / 1000.0
        pix_deg = abs(float(h["CDELT1"]))
        au = pix_deg * 3600 * DIST
        kern = sp.beam_kernel(float(h["BMAJ"]), float(h["BMIN"]), float(h["BPA"]), pix_deg, np.sign(float(h["CDELT1"])) or -1)
        print(f"{which}: ref {ref.shape}, {au:.3f} au/px (MCFOST grid 6385/2048 = 3.118), v {velax[0]:.2f}..{velax[-1]:.2f} km/s, "
              f"beam {float(h['BMAJ']) * 3600:.3f}x{float(h['BMIN']) * 3600:.3f} arcsec PA {float(h['BPA']):.1f}", flush=True)
        dd = d if which == "GI" else keplerian_twin(d)
        m1r, m0r = m1_of(ref, velax)
        mask = m0r > 0.05 * m0r.max()
        if best_orient is None:
            # Orientation convention, chosen once on the GI disk at 1/4 size. MCFOST's PA is not the sky PA of our projection by a fixed offset (first
            # run 2026-10-10: discrete PA +-236.7 / +180 gave M1 corr 0.91 and a large-scale dipole residual; a rotation of -24 deg gave 0.995), and the
            # bulk rotation cannot tell the 4 (spin flip, mirror) combinations apart (all tie at M1 corr 0.99, but the spiral handedness differs and the
            # wiggle correlation of the wrong one is ~0, planted-offset test). So: PA is a rotation of the sky image, scanned continuously on the
            # coarse M1 for each combination; the combination is chosen by the correlation of the HIGH-PASS M1 (spiral structure), not the bulk.
            from scipy import ndimage
            f = 4; ns = n // f
            refs = ref[:, : ns * f, : ns * f].reshape(C, ns, f, ns, f).mean((2, 4))
            m1s, m0s = m1_of(refs, velax); masks = m0s > 0.05 * m0s.max()
            yy, xx = np.indices((ns, ns)); core = masks & (np.hypot(yy - ns / 2, xx - ns / 2) < 0.33 * ns)
            hp = lambda m: m - ndimage.gaussian_filter(m, ns / 12)
            rot = lambda m, t: ndimage.rotate(m, t, reshape=False, order=1, mode="nearest")
            scan = []
            for flip in (False, True):
                for mirror in (False, True):
                    cub = sc.render(dd, ns, au * f, velax, INCL, pa_deg=0.0, flip=flip, tau0=10.0)
                    m1c = m1_of(cub[:, :, ::-1] if mirror else cub, velax)[0]
                    th = max(np.arange(-180, 180, 3.0), key=lambda t: corr(rot(m1c, t), m1s, core))
                    th = max(th + np.arange(-2.0, 2.1, 1.0), key=lambda t: corr(rot(m1c, t), m1s, core))
                    r = rot(m1c, th)
                    scan.append({"flip": flip, "mirror": mirror, "rot_deg": float(th), "m1_corr": corr(r, m1s, core), "hp_corr": corr(hp(r), hp(m1s), core)})
                    print(f"  flip {flip!s:5} mirror {mirror!s:5}: rotation {th:+6.1f} deg  M1 corr {scan[-1]['m1_corr']:.3f}  high-pass corr {scan[-1]['hp_corr']:+.3f}", flush=True)
            top = max(scan, key=lambda e: e["hp_corr"])
            cand = []
            for sgn in (1.0, -1.0):                  # image rotation -> pa_deg sense: confirm by rendering
                pa = sgn * top["rot_deg"]
                cub = sc.render(dd, ns, au * f, velax, INCL, pa_deg=pa, flip=top["flip"], tau0=10.0)
                m1c = m1_of(cub[:, :, ::-1] if top["mirror"] else cub, velax)[0]
                cand.append((corr(hp(m1c), hp(m1s), core), pa))
                print(f"  render pa {pa:+7.1f}: high-pass corr {cand[-1][0]:+.3f}", flush=True)
            c0, pa1 = max(cand)
            for dpa in (-2.0, -1.0, 1.0, 2.0):
                cub = sc.render(dd, ns, au * f, velax, INCL, pa_deg=pa1 + dpa, flip=top["flip"], tau0=10.0)
                m1c = m1_of(cub[:, :, ::-1] if top["mirror"] else cub, velax)[0]
                c1 = corr(hp(m1c), hp(m1s), core)
                if c1 > c0:
                    c0, pa1 = c1, pa1 + dpa
            best_orient = (c0, pa1, top["flip"], top["mirror"])
            report["orientation_scan"] = scan
            report["orientation"] = {"hp_corr": c0, "pa": pa1, "flip": top["flip"], "mirror": top["mirror"]}
            print(f"  kept: {report['orientation']}", flush=True)
        _, pa, flip, mirror = best_orient
        rows = {}
        for tau in [float(x) for x in a.tau.split(",")]:
            cub = sc.render(dd, n, au, velax, INCL, pa_deg=pa, flip=flip, tau0=tau)
            if mirror:
                cub = cub[:, :, ::-1]
            cub = sp.convolve_cube(cub, kern)
            cub *= float(np.percentile(ref, 99.9)) / max(float(np.percentile(cub, 99.9)), 1e-30)    # brightness scale is free
            m1o, _ = m1_of(cub, velax)
            dsc = {"clean": ref, "dirty": ref, "velax": velax, "au": au, "incl": INCL, "crop": slice(None), "beam_au": float(h["BMAJ"]) * 3600 * DIST}
            ws.METHODS["sph"] = lambda _d, c=cub: c
            sco = ws.score_cube(f"{which}_tau{tau:g}", dsc, ["sph"], a.out)["methods"]["sph"]
            slope = float(np.polyfit(m1r[mask], m1o[mask], 1)[0])           # 1 if the velocity scale matches MCFOST's
            rows[f"tau{tau:g}"] = {"m1_corr": corr(m1o, m1r, mask), "m1_slope": slope, **{k: (float(v) if np.isscalar(v) else v) for k, v in sco.items() if np.isscalar(v)}}
            r = rows[f"tau{tau:g}"]
            print(f"{which} tau0 {tau:6g}: M1 corr {r['m1_corr']:.3f} slope {r['m1_slope']:.3f}  rc_r {r.get('rc_r', float('nan')):.3f}  resid_r {r.get('resid_r', float('nan')):.3f}  "
                  f"rc_rms {r.get('rc_rms_ms', float('nan')):.1f} vs clean {r.get('rc_rms_clean_ms', float('nan')):.1f} m/s  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            if tau == 10.0:
                np.save(os.path.join(a.out, f"{which}_sph_tau10_m1.npy"), m1o); np.save(os.path.join(a.out, f"{which}_mcfost_m1.npy"), m1r)
        report[which] = rows
        json.dump(report, open(os.path.join(a.out, "validation.json"), "w"), indent=1, default=str)
    print(f"VALIDATION DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
