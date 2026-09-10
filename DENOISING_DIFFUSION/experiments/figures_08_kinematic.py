"""
Figures for notebook 08's kinematic_gamma sweep, scored in score_08_kinematic.py.

Two figures:
  1. Moment maps on one representative holdout cube: clean / dirty / gamma=0/0.1/1/10.
     Needs a fresh denoise pass (the scoring script doesn't cache the denoised cubes),
     reusing the exact same sliding-window / normalisation logic.
  2. Wiggle resid_r and M1 vs gamma, across all 5 holdout cubes -- reads the JSON the
     scoring script already saved, no re-denoising needed.

Run: PYTHONPATH=.. python3 experiments/figures_08_kinematic.py [--cube N]
"""
import os, sys, math, time, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Fn
from astropy.io import fits

from src.data.cube_split import split_cubes
from src.data.fits_cube_dataset import continuum_of
from src.training.architectures import build_model
from src.evaluation.moment_maps import generate_moment_maps, signal_mask, moment_improvement
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual

DATA_DIR = "Line Emission Data"
OUT = "results/self-gravitating"
JSON_PATH = f"{OUT}/nb08_kinematic_wiggle.json"
BASE, MULTS, K, TARGET_SIZE, CONTINUUM_N, FRAC = 48, (1, 2, 4, 8), 15, 256, 5, 0.05
GAMMAS = [0.0, 0.1, 1.0, 10.0]
CKPTS = {g: f"models/08-kinematic/kin_gamma{g if g != int(g) else int(g)}.pth" for g in GAMMAS}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_net(gamma):
    ck = torch.load(CKPTS[gamma], map_location=dev, weights_only=False)
    net = build_model("unet", base_channels=BASE, channel_multipliers=MULTS, use_beam=False,
                      n_neighbors=K, out_channels=2 * K + 1, latent_dim=128).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=False)
    assert not miss and not unexp
    net.eval()
    return net


def denoise_stack(net, dirty_csub, batch=8):
    C, H, W = dirty_csub.shape
    out = np.empty((C, H, W), dtype=np.float32)
    idx_all = np.arange(C)
    with torch.no_grad():
        for s in range(0, C, batch):
            centres = idx_all[s:s + batch]
            nb = np.clip(centres[:, None] + np.arange(-K, K + 1)[None, :], 0, C - 1)
            stack = dirty_csub[nb].astype(np.float64)
            lo = dirty_csub[centres].reshape(len(centres), -1).min(axis=1)
            hi = dirty_csub[centres].reshape(len(centres), -1).max(axis=1)
            rng = np.where((hi - lo) > 0, hi - lo, 1.0)
            norm = (stack - lo[:, None, None, None]) / rng[:, None, None, None]
            t = torch.from_numpy(norm).float().to(dev)
            t = Fn.interpolate(t, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            p = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False).cpu().numpy()
            centre_out = p[:, K]
            for j, c in enumerate(centres):
                out[c] = centre_out[j] * rng[j] + lo[j] if rng[j] > 0 else lo[j]
    return out


def csub(cube):
    return cube - continuum_of(cube, CONTINUUM_N)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, default=0, help="index into the 5 holdout cubes")
    args = ap.parse_args()

    _, _, holdout_cubes = split_cubes(data_dir=DATA_DIR, n_holdout=3, val_fraction=0.2, seed=42)
    ho = holdout_cubes[args.cube]
    print(f"device {dev} | figure cube: {ho['folder']}")

    with fits.open(ho["clean"], memmap=True) as h:
        hdr = h[0].header
        clean_raw = np.asarray(h[0].data, np.float64)
    with fits.open(ho["dirty"], memmap=True) as h:
        dirty_raw = np.asarray(h[0].data, np.float64)
    velax = (hdr["CRVAL3"] + (np.arange(clean_raw.shape[0]) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    au_per_px = abs(hdr.get("CDELT1", 0)) * 3600.0 * float(hdr.get("DIST_PC", 140.0))

    clean_c = csub(clean_raw)
    dirty_c = csub(dirty_raw)

    mom = {"clean": generate_moment_maps("", data_velax=(clean_c, velax)),
           "dirty": generate_moment_maps("", data_velax=(dirty_c, velax))}
    m1 = {}
    v0, _ = quadratic_moment1(clean_c, velax)
    m1["clean"] = v0 / 1000.0
    v0d, _ = quadratic_moment1(dirty_c, velax)
    m1["dirty"] = v0d / 1000.0

    for g in GAMMAS:
        t0 = time.time()
        net = load_net(g)
        den = denoise_stack(net, dirty_c).astype(np.float64)
        mom[g] = generate_moment_maps("", data_velax=(den, velax))
        v0g, _ = quadratic_moment1(den, velax)
        m1[g] = v0g / 1000.0
        del net
        print(f"  gamma={g} denoised in {(time.time()-t0)/60:.1f} min")

    mask = signal_mask(mom["clean"][0], frac=FRAC)
    geom = fit_keplerian(m1["clean"], mask, au_per_px)
    print(f"geometry: mstar={geom['mstar_msun']:.3f} incl={geom['incl_deg']:.1f} "
          f"at_bound={geom['mstar_at_bound']}")

    scores = {}
    for key in ["dirty"] + GAMMAS:
        imp = moment_improvement(mom["clean"], mom["dirty"], mom[key]) if key != "dirty" else None
        scores[key] = imp

    # -------------------------------------------------------------- figure 1: moment maps
    cols = ["clean", "dirty"] + GAMMAS
    titles = {"clean": "clean", "dirty": "dirty", 0.0: "gamma=0", 0.1: "gamma=0.1",
             1.0: "gamma=1", 10.0: "gamma=10"}
    rows = [("M0", 0, "inferno"), ("M1", 1, "RdBu_r"), ("M2", 2, "viridis")]

    ys, xs = np.where(mask)
    pad = 15
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)

    def cut(a):
        return a[y0:y1, x0:x1]

    fig, ax = plt.subplots(3, 6, figsize=(21, 10.5))
    for ri, (label, mi, cmap) in enumerate(rows):
        ref = cut(np.where(mask, mom["clean"][mi], np.nan))
        if label == "M1":
            lim = np.nanpercentile(np.abs(ref), 98)
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = np.nanpercentile(ref, 2), np.nanpercentile(ref, 98)
        for ci, c in enumerate(cols):
            a = ax[ri, ci]
            im = a.imshow(cut(np.where(mask, mom[c][mi], np.nan)), cmap=cmap,
                          vmin=vmin, vmax=vmax, origin="lower")
            a.set_xticks([]); a.set_yticks([])
            if ri == 0:
                a.set_title(titles[c], fontsize=12, pad=8)
            if ci == 0:
                a.set_ylabel(label, fontsize=15, labelpad=10)
            if c in scores and scores[c] is not None:
                v = scores[c][label]
                a.text(0.5, -0.07, f"{v:+.1f}%", transform=a.transAxes, ha="center",
                       fontsize=11.5, color=("#1a7f37" if v > 0 else "#b3261e"), weight="bold")
        plt.colorbar(im, ax=list(ax[ri, :]), fraction=0.015, pad=0.01)

    fig.suptitle(f"Notebook 08 kinematic_gamma sweep: moment maps on {ho['folder']}\n"
                 f"% = improvement over dirty, signal-masked (frac={FRAC})",
                 fontsize=13.5, y=0.99)
    p1 = f"{OUT}/nb08_kinematic_moments.png"
    plt.savefig(p1, dpi=125)
    plt.close()
    print("saved ->", p1)

    # -------------------------------------------------- figure 2: wiggle + M1 vs gamma, all cubes
    if os.path.exists(JSON_PATH):
        rows_all = json.load(open(JSON_PATH))
        gvals = [str(g) for g in GAMMAS]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        a0 = axes[0]
        for r in rows_all:
            ys_ = [r["gammas"][g]["resid_r"] for g in gvals]
            a0.plot(GAMMAS, ys_, "o-", alpha=0.6, label=r["cube"][:20])
        mean_r = [np.mean([r["gammas"][g]["resid_r"] for r in rows_all]) for g in gvals]
        a0.plot(GAMMAS, mean_r, "ks-", lw=2.5, ms=8, label="mean")
        dirty_mean = np.mean([r["dirty_resid_r"] for r in rows_all])
        a0.axhline(dirty_mean, color="#a04a30", ls="--", label=f"dirty mean ({dirty_mean:.3f})")
        a0.set_xscale("symlog", linthresh=0.05)
        a0.set_xlabel("kinematic_gamma"); a0.set_ylabel("wiggle resid_r")
        a0.set_title("Wiggle vs gamma, 5 holdout cubes")
        a0.legend(fontsize=7, loc="best")
        a0.grid(alpha=0.25)

        a1 = axes[1]
        mean_m1 = [np.mean([r["gammas"][g]["M1"] for r in rows_all]) for g in gvals]
        std_m1 = [np.std([r["gammas"][g]["M1"] for r in rows_all], ddof=1) for g in gvals]
        a1.errorbar(GAMMAS, mean_m1, yerr=std_m1, fmt="o-", capsize=4, color="#2f5f96")
        a1.axhline(0, color="black", lw=1)
        a1.set_xscale("symlog", linthresh=0.05)
        a1.set_xlabel("kinematic_gamma"); a1.set_ylabel("M1 improvement (%)")
        a1.set_title("M1 vs gamma, mean +/- std across 5 cubes")
        a1.grid(alpha=0.25)

        fig.suptitle("Does kinematic_gamma help the wiggle-adjacent signal? (line emission, n=5)",
                     fontsize=12.5)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p2 = f"{OUT}/nb08_kinematic_vs_gamma.png"
        plt.savefig(p2, dpi=135)
        plt.close()
        print("saved ->", p2)
    else:
        print(f"\n{JSON_PATH} not found yet -- figure 2 skipped, run score_08_kinematic.py first")


if __name__ == "__main__":
    main()
