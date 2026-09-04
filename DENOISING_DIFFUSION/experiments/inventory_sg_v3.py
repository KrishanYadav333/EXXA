"""
First look at the September SG batch: what each pair actually is, before designing anything
around it.

The two earlier SG deliveries were NOT the same kind of data (original: clean Jy/pixel, 51.55%
negative pixels in dirty; v2 replacement: both Jy/beam, 19.25% negative, different header keys
entirely). So the regime has to be measured per pair, not assumed, which is what this does.

Per pair: header inventory, negative-pixel fraction, end-channel padding check, and the Phase 0
forward-operator verdict (is there a beam between clean and dirty, or is A = I).

Run: PYTHONPATH=.. python3 experiments/inventory_sg_v3.py
"""
import os, sys, glob, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
from astropy.io import fits

from src.evaluation.forward_operator import phase0_from_fits

ROOT = "self-gravitating cube and dirty cube/_v3_extract/kinematic_data"


def find_pairs(root):
    """Pairs are <run>/<name>_clean.fits + <name>_dirty.fits, plus the loose clean_sg/dirty_sg."""
    pairs = []
    for c in sorted(glob.glob(os.path.join(root, "**", "*clean*.fits"), recursive=True)):
        d = c.replace("clean", "dirty")
        if os.path.exists(d):
            pairs.append((os.path.relpath(c, root), os.path.relpath(d, root)))
    return pairs


def end_padding(cube, n=40):
    """Channels at the cube ends that are byte-identical repeats (found in the v2 cube)."""
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
    return lead, trail


def hdr_summary(h):
    keys = ["NAXIS1", "NAXIS2", "NAXIS3", "BUNIT", "BMAJ", "BMIN", "BPA",
            "CDELT1", "CDELT3", "CRVAL3", "CRPIX3", "RESTFRQ", "DISTPC", "DIST_PC",
            "RMS", "PBCOR", "SEED"]
    return {k: h[k] for k in keys if k in h}


rows = []
for cpath, dpath in find_pairs(ROOT):
    print("=" * 78)
    print(cpath.split("/")[0] if "/" in cpath else cpath)
    print("=" * 78)

    cfull, dfull = os.path.join(ROOT, cpath), os.path.join(ROOT, dpath)
    with fits.open(cfull, memmap=True) as h:
        ch, cdata = h[0].header, h[0].data
        cshape = cdata.shape
        chdr = hdr_summary(ch)
        lead, trail = end_padding(cdata)
        cmid = np.asarray(cdata[cshape[0] // 2], np.float64)
    with fits.open(dfull, memmap=True) as h:
        dh, ddata = h[0].header, h[0].data
        dhdr = hdr_summary(dh)
        dmid = np.asarray(ddata[ddata.shape[0] // 2], np.float64)
        dneg = float((ddata[::37] < 0).mean())      # strided, whole cube is too big to load

    print(f"  shape        {cshape}")
    print(f"  clean BUNIT  {chdr.get('BUNIT')!r}   dirty BUNIT {dhdr.get('BUNIT')!r}")
    print(f"  clean beam   BMAJ={chdr.get('BMAJ')} BMIN={chdr.get('BMIN')}")
    print(f"  dirty beam   BMAJ={dhdr.get('BMAJ')} BMIN={dhdr.get('BMIN')}")
    print(f"  dv           {chdr.get('CDELT3')}   px {chdr.get('CDELT1')}")
    print(f"  negative px in dirty (strided)  {100*dneg:.2f}%")
    print(f"  end-channel padding: lead={lead} trail={trail}")
    print(f"  clean-only header keys: {sorted(set(chdr) - set(dhdr))}")
    print(f"  dirty-only header keys: {sorted(set(dhdr) - set(chdr))}")
    print(f"  peak clean {np.nanmax(cmid):.4g}   peak dirty {np.nanmax(dmid):.4g}   "
          f"ratio {np.nanmax(dmid)/max(np.nanmax(cmid), 1e-12):.3g}")

    try:
        r = phase0_from_fits(cfull, dfull, max_channels=8)
        print(f"  PHASE 0: {r['verdict'].upper()}  "
              f"(A={r.get('amplitude', float('nan')):.4g}, "
              f"sigma={r.get('sigma_px', float('nan')):.2f} px)")
        verdict = r["verdict"]
    except Exception as e:
        print(f"  PHASE 0 FAILED: {type(e).__name__}: {e}")
        verdict = f"error: {type(e).__name__}"

    rows.append(dict(run=cpath.split("/")[0], shape=list(cshape),
                     clean_bunit=chdr.get("BUNIT"), dirty_bunit=dhdr.get("BUNIT"),
                     neg_frac=dneg, lead_pad=lead, trail_pad=trail, verdict=verdict))
    print()

os.makedirs("results/self-gravitating", exist_ok=True)
with open("results/self-gravitating/sg_v3_inventory.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"{len(rows)} pairs -> results/self-gravitating/sg_v3_inventory.json")
