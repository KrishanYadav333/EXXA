"""
Score notebook 08's kinematic_gamma sweep (0/0.1/1/10) on the 5 line-emission holdout cubes:
moment improvement AND the GI wiggle, not just the val_loss already on record.

Model shape is the 31-channel stack (n_neighbors=15, stack_target=True): input is 31
neighbouring dirty channels, output is 31 channels, of which only the CENTRE one is used per
sliding-window position -- the standard readout for this kind of stack, one prediction per
true target position rather than trusting the whole predicted stack at once.

Normalisation matches training exactly (src/data/fits_cube_dataset.py): continuum-subtract
first (mean of the first/last 5 channels, mentor's convention), then min-max both dirty and
clean using the CENTRE dirty channel's (lo, hi) -- shared across the whole neighbour stack,
never per-channel, which is what keeps the un-normalisation at inference invertible.

Wiggle: unlike the self-gravitating disks, these cubes have no stated ground-truth
inclination, so the geometry fit is free -- flag `mstar_at_bound` per cube rather than assume
it converged (RULES.md #8). One geometry per cube, fit on clean, shared across every gamma
(compare_wiggles' own logic, applied by hand since these are held out per notebook 05/08's
split, not the compare_wiggles() self-gravitating helper's cube list).

Run: PYTHONPATH=.. python3 experiments/score_08_kinematic.py [--cubes N] [--time-only]
"""
import os, sys, math, time, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import torch
import torch.nn.functional as Fn
from astropy.io import fits

from src.data.cube_split import split_cubes
from src.data.fits_cube_dataset import continuum_of
from src.training.architectures import build_model
from src.evaluation.moment_maps import generate_moment_maps, signal_mask, moment_improvement
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual, wiggle_amplitude

DATA_DIR = "Line Emission Data"
BASE, MULTS, K, TARGET_SIZE, CONTINUUM_N, FRAC = 48, (1, 2, 4, 8), 15, 256, 5, 0.05
GAMMAS = [0.0, 0.1, 1.0, 10.0]
CKPTS = {g: f"models/08-kinematic/kin_gamma{g if g != int(g) else int(g)}.pth" for g in GAMMAS}
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_net(gamma):
    ck = torch.load(CKPTS[gamma], map_location=dev, weights_only=False)
    net = build_model("unet", base_channels=BASE, channel_multipliers=MULTS, use_beam=False,
                      n_neighbors=K, out_channels=2 * K + 1, latent_dim=128).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not miss and not unexp, f"gamma={gamma}: state dict mismatch"
    net.eval()
    return net


def denoise_stack(net, dirty_csub, batch=8):
    """Full cube, continuum-subtracted, sliding 31-channel window, centre-channel readout."""
    C, H, W = dirty_csub.shape
    out = np.empty((C, H, W), dtype=np.float32)
    idx_all = np.arange(C)
    with torch.no_grad():
        for s in range(0, C, batch):
            centres = idx_all[s:s + batch]
            nb = np.clip(centres[:, None] + np.arange(-K, K + 1)[None, :], 0, C - 1)  # (b, 2K+1)
            stack = dirty_csub[nb].astype(np.float64)                                  # (b, 2K+1, H, W)
            lo = dirty_csub[centres].reshape(len(centres), -1).min(axis=1)
            hi = dirty_csub[centres].reshape(len(centres), -1).max(axis=1)
            rng = np.where((hi - lo) > 0, hi - lo, 1.0)
            norm = (stack - lo[:, None, None, None]) / rng[:, None, None, None]
            t = torch.from_numpy(norm).float().to(dev)
            t = Fn.interpolate(t, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            p = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False).cpu().numpy()
            centre_out = p[:, K]  # (b, H, W), the centre of the predicted 31-channel stack
            for j, c in enumerate(centres):
                out[c] = centre_out[j] * rng[j] + lo[j] if rng[j] > 0 else lo[j]
    return out


def csub(cube):
    return cube - continuum_of(cube, CONTINUUM_N)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", type=int, default=5)
    ap.add_argument("--time-only", action="store_true")
    args = ap.parse_args()

    _, _, holdout_cubes = split_cubes(data_dir=DATA_DIR, n_holdout=3, val_fraction=0.2, seed=42)
    holdout_cubes = holdout_cubes[:args.cubes]
    print(f"device {dev} | {len(holdout_cubes)} holdout cube(s): "
          f"{[c['folder'] for c in holdout_cubes]}")

    nets = {g: load_net(g) for g in GAMMAS}
    print("loaded 4 checkpoints")

    if args.time_only:
        ho = holdout_cubes[0]
        with fits.open(ho["dirty"], memmap=True) as h:
            dirty_raw = np.asarray(h[0].data[:40], np.float64)
        t0 = time.time()
        denoise_stack(nets[0.0], dirty_raw, batch=8)
        dt = time.time() - t0
        per_channel = dt / 40
        full = per_channel * 201
        print(f"\n40 channels in {dt:.1f}s -> {per_channel:.2f}s/channel")
        print(f"projected: one full cube (201ch) x one gamma  = {full/60:.1f} min")
        print(f"projected: {len(holdout_cubes)} cubes x 4 gammas            = {full/60*4*len(holdout_cubes):.0f} min")
        return

    all_rows = []
    t_start = time.time()
    for ho in holdout_cubes:
        print(f"\n{'='*70}\n{ho['folder']}\n{'='*70}")
        with fits.open(ho["clean"], memmap=True) as h:
            hdr = h[0].header
            clean_raw = np.asarray(h[0].data, np.float64)
        with fits.open(ho["dirty"], memmap=True) as h:
            dirty_raw = np.asarray(h[0].data, np.float64)
        velax = (hdr["CRVAL3"] + (np.arange(clean_raw.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
        au_per_px = abs(hdr.get("CDELT1", 0)) * 3600.0 * float(hdr.get("DIST_PC", 140.0))

        clean_c = csub(clean_raw)
        dirty_c = csub(dirty_raw)

        m_clean = generate_moment_maps("", data_velax=(clean_c, velax))
        m_dirty = generate_moment_maps("", data_velax=(dirty_c, velax))
        mask = signal_mask(m_clean[0], frac=FRAC)

        v0, _ = quadratic_moment1(clean_c, velax)
        m1_clean = v0 / 1000.0
        geom = fit_keplerian(m1_clean, mask, au_per_px)
        flag = " DEGEN" if geom["mstar_at_bound"] else ""
        print(f"  geometry (free fit): mstar={geom['mstar_msun']:.3f} incl={geom['incl_deg']:.1f}{flag}")
        ref_resid = wiggle_residual(m1_clean, geom)

        v0d, _ = quadratic_moment1(dirty_c, velax)
        m1_dirty = v0d / 1000.0
        dirty_resid = wiggle_residual(m1_dirty, geom)
        ok = np.isfinite(ref_resid[mask]) & np.isfinite(dirty_resid[mask])
        dirty_r = float(np.corrcoef(ref_resid[mask][ok], dirty_resid[mask][ok])[0, 1]) if ok.sum() > 10 else float("nan")
        print(f"  dirty   resid_r={dirty_r:.4f}")

        row = dict(cube=ho["folder"], geom=geom, dirty_resid_r=dirty_r, gammas={})
        for g in GAMMAS:
            t0 = time.time()
            den = denoise_stack(nets[g], dirty_c)
            m_den = generate_moment_maps("", data_velax=(den.astype(np.float64), velax))
            imp = moment_improvement(m_clean, m_dirty, m_den)

            v0g, _ = quadratic_moment1(den.astype(np.float64), velax)
            m1_g = v0g / 1000.0
            resid = wiggle_residual(m1_g, geom)
            ok2 = np.isfinite(ref_resid[mask]) & np.isfinite(resid[mask])
            resid_r = float(np.corrcoef(ref_resid[mask][ok2], resid[mask][ok2])[0, 1]) if ok2.sum() > 10 else float("nan")

            row["gammas"][g] = dict(M0=imp["M0"], M1=imp["M1"], M2=imp["M2"], resid_r=resid_r)
            print(f"  gamma={g:<5} M0 {imp['M0']:+7.1f}  M1 {imp['M1']:+7.1f}  M2 {imp['M2']:+7.1f}  "
                  f"resid_r {resid_r:.4f}  ({(time.time()-t0)/60:.1f} min)")

        all_rows.append(row)
        with open("results/self-gravitating/nb08_kinematic_wiggle.json", "w") as f:
            json.dump(all_rows, f, indent=2, default=str)
        print(f"  [saved {len(all_rows)}/{len(holdout_cubes)}, {(time.time()-t_start)/60:.0f} min elapsed]")

    print("\n" + "=" * 70)
    print("SUMMARY across holdout cubes (mean +/- std)")
    print("=" * 70)
    for g in GAMMAS:
        m0 = np.array([r["gammas"][g]["M0"] for r in all_rows])
        m1 = np.array([r["gammas"][g]["M1"] for r in all_rows])
        m2 = np.array([r["gammas"][g]["M2"] for r in all_rows])
        rr = np.array([r["gammas"][g]["resid_r"] for r in all_rows])
        n = len(all_rows)
        std = lambda a: np.std(a, ddof=1) if n > 1 else 0.0
        print(f"gamma={g:<5} M0 {m0.mean():+7.1f}+/-{std(m0):5.1f}  M1 {m1.mean():+7.1f}+/-{std(m1):5.1f}  "
              f"M2 {m2.mean():+7.1f}+/-{std(m2):5.1f}  resid_r {rr.mean():.4f}+/-{std(rr):.4f}")
    dr = np.array([r["dirty_resid_r"] for r in all_rows])
    print(f"dirty (no model)        resid_r {dr.mean():.4f}+/-{np.std(dr, ddof=1) if len(dr)>1 else 0:.4f}")

    print("\nsaved -> results/self-gravitating/nb08_kinematic_wiggle.json")


if __name__ == "__main__":
    main()
