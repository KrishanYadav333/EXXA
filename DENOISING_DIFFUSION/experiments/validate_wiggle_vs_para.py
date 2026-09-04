"""
Does `fit_keplerian` actually recover the true stellar mass? Five cubes, stated ground truth.

Until now this had exactly ONE validation point: 0.639 Msun recovered from the v2 cube against
Hall+2020's 0.6 Msun founding simulation -- and that 0.6 was inferred from a paper, not from
the data. The September batch ships MCFOST `.para` files that STATE the star's mass and the
distance, across two different configurations (0.6 Msun at 140 pc, 1.0 Msun at 175.178 pc).

So this is the first real test of the diagnostic the whole GI wiggle line of work rests on.
Clean cubes only: no noise, no beam, no model. If the fit cannot recover a known mass from a
noiseless velocity field, nothing downstream of it means anything.

Mask at frac=0.05 (the project default in moment_maps.py). The 0.02 used by the older
self-gravitating scripts pulls in a low-SNR halo -- Jason flagged it, and the sweep confirmed
it: at 0.02 the mask is 39% of the field, most of it near-threshold.

Run: PYTHONPATH=.. python3 experiments/validate_wiggle_vs_para.py
"""
import os, sys, glob, re, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
from astropy.io import fits

from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_amplitude, wiggle_residual

ROOT = "self-gravitating cube and dirty cube/_v3_extract/kinematic_data"
FRAC = 0.05
MSTAR_BOUND = 50.0
MAX_CHANNELS = 240        # cap for the 601-channel cube; the 201-channel ones use all of theirs


def parse_para(path):
    """MCFOST parameter file: distance on its own line, star line is 'Teff R M x y z bb?'."""
    out = {}
    with open(path, errors="ignore") as f:
        lines = f.readlines()
    for ln in lines:
        if "distance (pc)" in ln:
            out["distance_pc"] = float(ln.split()[0])
        if "Temp, radius" in ln and "solar mass" in ln:
            parts = ln.split()
            out["teff"] = float(parts[0])
            out["rstar_rsun"] = float(parts[1])
            out["mstar_msun"] = float(parts[2])
    return out


def trim_padding(cube, n=45):
    """Leading/trailing channels that are byte-identical repeats are padding, not baseline."""
    lead = 0
    for i in range(1, min(n, cube.shape[0])):
        if np.array_equal(cube[i], cube[0]):
            lead = i
        else:
            break
    trail = 0
    for i in range(2, min(n, cube.shape[0])):
        if np.array_equal(cube[-i], cube[-1]):
            trail = i - 1
        else:
            break
    return lead + 1 if lead else 0, cube.shape[0] - (trail + 1 if trail else 0)


rows = []
cleans = sorted(glob.glob(f"{ROOT}/*/*clean*.fits"))

for cpath in cleans:
    run = os.path.basename(os.path.dirname(cpath))
    paras = glob.glob(os.path.join(os.path.dirname(cpath), "*.para"))
    truth = parse_para(paras[0]) if paras else {}

    print("=" * 78)
    print(run)
    print("=" * 78)

    with fits.open(cpath, memmap=True) as h:
        hdr, data = h[0].header, h[0].data
        lo, hi = trim_padding(data)
        n = hi - lo
        step = max(1, int(np.ceil(n / MAX_CHANNELS)))
        chans = list(range(lo, hi, step))
        cube = np.stack([np.asarray(data[c], np.float64) for c in chans])

    dist = float(hdr.get("DIST_PC", truth.get("distance_pc", np.nan)))
    au_per_px = abs(hdr["CDELT1"]) * 3600.0 * dist
    velax = (hdr["CRVAL3"] + (np.array(chans) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0

    print(f"  channels {lo}-{hi} step {step} ({len(chans)} used), dv={hdr['CDELT3']*step:.4f} km/s")
    print(f"  distance {dist} pc (header) / {truth.get('distance_pc')} pc (.para), "
          f"{au_per_px:.3f} AU/px")

    m0, _, _ = generate_moment_maps("", data_velax=(cube, velax))
    mask = signal_mask(m0, frac=FRAC)
    v0, _ = quadratic_moment1(cube, velax)
    m1 = v0 / 1000.0

    geom = fit_keplerian(m1, mask, au_per_px)
    resid = wiggle_residual(m1, geom)
    amp = wiggle_amplitude(resid, mask)

    got = geom["mstar_msun"]
    want = truth.get("mstar_msun", float("nan"))
    err = 100.0 * (got - want) / want if want else float("nan")
    degen = got > 0.9 * MSTAR_BOUND

    print(f"  mask {mask.sum()} px ({100*mask.sum()/mask.size:.1f}% of field)")
    print(f"  fit: incl={geom['incl_deg']:.1f} pa={geom['pa_deg']:.1f} vsys={geom['vsys']:.3f}")
    print(f"  MSTAR recovered {got:.3f}  |  .para truth {want:.3f}  |  error {err:+.1f}%"
          + ("   ** DEGENERATE (at bound) **" if degen else ""))
    print(f"  residual RMS {amp['rms_kms']:.3f} km/s")
    print()

    rows.append(dict(run=run, mstar_fit=got, mstar_true=want, err_pct=err,
                     incl=geom["incl_deg"], pa=geom["pa_deg"], vsys=geom["vsys"],
                     dist_pc=dist, au_per_px=au_per_px, resid_rms=amp["rms_kms"],
                     n_channels=len(chans), mask_px=int(mask.sum()), degenerate=bool(degen)))

print("=" * 78)
print(f"{'run':26s} {'truth':>7} {'fit':>8} {'err%':>8} {'incl':>6} {'residRMS':>9}")
for r in rows:
    flag = "  DEGEN" if r["degenerate"] else ""
    print(f"{r['run'][:26]:26s} {r['mstar_true']:7.2f} {r['mstar_fit']:8.3f} "
          f"{r['err_pct']:+8.1f} {r['incl']:6.1f} {r['resid_rms']:9.3f}{flag}")

good = [r for r in rows if not r["degenerate"] and np.isfinite(r["err_pct"])]
if good:
    errs = np.array([r["err_pct"] for r in good])
    print(f"\nnon-degenerate fits: {len(good)}/{len(rows)}, "
          f"mean |error| {np.mean(np.abs(errs)):.1f}%, worst {np.max(np.abs(errs)):.1f}%")

os.makedirs("results/self-gravitating", exist_ok=True)
with open("results/self-gravitating/wiggle_mass_validation.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nsaved -> results/self-gravitating/wiggle_mass_validation.json")
