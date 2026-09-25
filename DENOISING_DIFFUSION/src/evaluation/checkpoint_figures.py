"""
Every comparison in checkpoint_eval as an image, so nothing lives only in a table.

Two kinds of figure, both built from what `checkpoint_eval` keeps:

  From the saved per-checkpoint maps (`<map_dir>/<case>__<label>.npz` and `<case>__REF.npz`), for each case in
  `figure_cases`. Sheets tile EVERY checkpoint next to clean and dirty on one shared colour scale, so they can be
  compared by eye; a title carries the number that goes with each tile.
      moments       M0, M1, M2 of every checkpoint
      errors        denoised minus clean, M0 and M1, symmetric, scaled by DIRTY's own error so it reads as "better or
                    worse than doing nothing"
      wiggle        the Keplerian residual of M1, the GI wiggle itself, not just its correlation
      channels      the blue-side / systemic / red-side channel at the line peak, and the error at the systemic one
      sharpness     M1 gradient magnitude: smoothing is visible as a fading of the fine structure
      invented      where each checkpoint asserts signal that clean does not have
      spectra       line profiles at the disk peak, the disk edge and an off-source pixel
      radial        M0 radial profile and M1 power spectrum against clean's
      topk          a large-format detail figure for the best checkpoints by M0, one row each

  From the results table alone (no maps needed), for the whole run:
      score tables as images, a scoreboard of every metric with error bars, the loss x source matrix (Jason's
      asks 1 and 3), and a checkpoint x cube heatmap.

Colour scales are shared within a sheet and come from CLEAN (or from DIRTY's error, for error maps), never from a
checkpoint, so no tile can look better by being scaled differently.
"""
from __future__ import annotations

import glob
import math
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

FAM = {"unet": "#3C78B4", "stack_kin": "#D0743C", "stack_sg": "#4EA8A0", "stack": "#8E6CC0", "ddpm": "#C75E8A"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def short(label: str) -> str:
    s = re.sub(r"_seed\d+$", "", label)
    s = re.sub(r"^(winner_|sweep_)", "", s)
    return s


# ------------------------------------------------------------------------------------------------ #
# Loading                                                                                           #
# ------------------------------------------------------------------------------------------------ #
def load_case(map_dir: str, case: str):
    """(ref, {label: artifacts}) for one case, or (None, {}) if nothing was saved for it."""
    rp = os.path.join(map_dir, f"{case}__REF.npz")
    if not os.path.exists(rp):
        return None, {}
    ref = dict(np.load(rp, allow_pickle=False))
    arts = {}
    for p in sorted(glob.glob(os.path.join(map_dir, f"{case}__*.npz"))):
        lab = os.path.basename(p)[len(case) + 2:-4]
        if lab != "REF":
            arts[lab] = dict(np.load(p, allow_pickle=False))
    return ref, arts


def _crop_box(mask: np.ndarray, pad: float = 0.14):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * (1 + 2 * pad))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    Y0, X0 = max(0, cy - side // 2), max(0, cx - side // 2)
    return Y0, min(mask.shape[0], Y0 + side), X0, min(mask.shape[1], X0 + side)


def _order(labels: Sequence[str], rows, key: str = "M0") -> List[str]:
    """Best first by `key` on this case; labels with no row go last."""
    def k(l):
        v = rows.get(l, {}).get(key, np.nan)
        return -v if np.isfinite(v) else 1e9
    return sorted(labels, key=k)


def _rows_for(df, case: str) -> Dict[str, dict]:
    if df is None or df.empty:
        return {}
    d = df[df["case"] == case]
    return {r["checkpoint"]: r for r in d.to_dict("records")}


# ------------------------------------------------------------------------------------------------ #
# Contact sheet                                                                                     #
# ------------------------------------------------------------------------------------------------ #
def contact_sheet(panels, *, title, path, cmap, vlim, cbar_label, ncols=7, tile=2.1, fs=6.2, blank=None, suptitle_fs=9):
    """`panels`: [(title, 2D array)]. One shared colour scale, `blank` masks off-source pixels."""
    plt = _plt()
    n = len(panels)
    ncols = min(ncols, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * tile, nrows * (tile + 0.34) + 0.7), squeeze=False)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#111111")
    for ax in axes.ravel():
        ax.axis("off")
    im = None
    for ax, (t, arr) in zip(axes.ravel(), panels):
        a = np.array(arr, dtype=np.float64)
        if blank is not None:
            a = np.where(blank, a, np.nan)
        im = ax.imshow(a, origin="lower", cmap=cm, vmin=vlim[0], vmax=vlim[1], interpolation="nearest")
        ax.set_title(t, fontsize=fs, linespacing=1.15)
    fh = fig.get_figheight()
    fig.suptitle(title, fontsize=suptitle_fs, y=0.995)
    fig.tight_layout(rect=(0, 0.75 / fh, 1, 0.975))
    cax = fig.add_axes([0.25, 0.42 / fh, 0.5, 0.11 / fh])          # inches, so the tick labels always fit below the bar
    fig.colorbar(im, cax=cax, orientation="horizontal").set_label(cbar_label, fontsize=7)
    cax.tick_params(labelsize=6.5)
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def _p(a, mask, q):
    v = np.asarray(a)[mask]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else 1.0


# ------------------------------------------------------------------------------------------------ #
# Sheets from saved maps                                                                            #
# ------------------------------------------------------------------------------------------------ #
def sheets_for_case(case: str, ref: dict, arts: dict, rows: dict, out_dir: str) -> List[str]:
    out: List[str] = []
    labels = _order(list(arts), rows)
    mask = ref["mask"].astype(bool)
    y0, y1, x0, x1 = _crop_box(mask)
    sl = (slice(y0, y1), slice(x0, x1))
    mk = mask[sl]
    base = os.path.join(out_dir, f"nb16_sheet_{case}")

    def tiles(get_ref_clean, get_ref_dirty, get_art, fmt):
        t = [("CLEAN (truth)", get_ref_clean()[sl]), ("DIRTY (input)", get_ref_dirty()[sl])]
        for l in labels:
            t.append((fmt(l), get_art(arts[l])[sl]))
        return t

    r = lambda l, k, f: f"{f.format(rows[l][k])}" if l in rows and np.isfinite(rows[l].get(k, np.nan)) else "n/a"

    # ---- moments ----
    m0c, m1c, m2c = ref["clean_m0"], ref["clean_m1"] / 1000.0, ref["clean_m2"] / 1000.0
    vs = float(np.nanmedian(m1c[mask]))
    L1 = _p(np.abs(m1c - vs), mask, 99)
    specs = [("M0", "m0", (0, _p(m0c, mask, 99.5)), "inferno", 1.0, "integrated intensity"),
             ("M1", "m1", (vs - L1, vs + L1), "RdBu_r", 1000.0, "velocity (km/s)"),
             ("M2", "m2", (0, _p(m2c, mask, 99)), "viridis", 1000.0, "velocity dispersion (km/s)")]
    for name, key, vl, cm, scale, unit in specs:
        out.append(contact_sheet(
            tiles(lambda: ref[f"clean_{key}"] / scale, lambda: ref[f"dirty_{key}"] / scale, lambda a: a[key] / scale,
                  lambda l: f"{short(l)}\n{name} {r(l, name, '{:+.0f}')}%"),
            title=f"{case}: {name}, every checkpoint on one scale (sorted by M0 improvement, best first). {unit}",
            path=f"{base}_{name}.png", cmap=cm, vlim=vl, cbar_label=unit, blank=mk))

    # ---- errors, scaled by dirty's own error ----
    for name, key, scale in (("M0", "m0", 1.0), ("M1", "m1", 1000.0)):
        cl = ref[f"clean_{key}"] / scale
        lim = _p(np.abs(ref[f"dirty_{key}"] / scale - cl), mask, 95)
        t = [("DIRTY - CLEAN\n(the error to beat)", (ref[f"dirty_{key}"] / scale - cl)[sl])]
        for l in labels:
            e = arts[l][key] / scale - cl
            mae = float(np.nanmean(np.abs(e[mask])))
            t.append((f"{short(l)}\nmean |err| {mae:.3g}", e[sl]))
        out.append(contact_sheet(t, title=f"{case}: {name} error (denoised - clean), symmetric, scaled by dirty's own error. "
                                          f"Red/blue = wrong; pale = right.",
                                 path=f"{base}_err_{name}.png", cmap="RdBu_r", vlim=(-lim, lim),
                                 cbar_label=f"{name} error", blank=mk))

    # ---- wiggle residual ----
    rc = ref["clean_resid"]
    lim = max(3.0 * float(np.sqrt(np.nanmean(rc[mask] ** 2))), 1e-6)
    out.append(contact_sheet(
        tiles(lambda: ref["clean_resid"], lambda: ref["dirty_resid"], lambda a: a["resid"],
              lambda l: f"{short(l)}\nresid r {r(l, 'resid_r', '{:.3f}')}"),
        title=(f"{case}: the GI wiggle, M1 minus the fitted Keplerian (km/s), same geometry for every tile. r = correlation with clean's."
               + ("" if bool(ref.get("geom_ok", True)) else "   !! GEOMETRY FIT FAILED on this cube: residuals are not a wiggle, r is blanked")),
        path=f"{base}_wiggle.png", cmap="RdBu_r", vlim=(-lim, lim), cbar_label="M1 residual (km/s)", blank=mk))

    # ---- channels at the line peak ----
    names = ["blue side (20% of flux)", "systemic (50%)", "red side (80%)"]
    for i in range(3):
        cc = ref["clean_chan"][i]
        L = _p(np.abs(cc), np.ones_like(cc, bool), 99.7)
        out.append(contact_sheet(
            [("CLEAN", cc[sl]), ("DIRTY", ref["dirty_chan"][i][sl])] + [(short(l), arts[l]["chan"][i][sl]) for l in labels],
            title=f"{case}: channel {int(ref['chan_idx'][i])}, {names[i]}", path=f"{base}_chan{i}.png",
            cmap="RdBu_r", vlim=(-L, L), cbar_label="Jy/beam (continuum-subtracted)"))
    cc = ref["clean_chan"][1]
    L = _p(np.abs(ref["dirty_chan"][1] - cc), np.ones_like(cc, bool), 97)
    out.append(contact_sheet(
        [("DIRTY - CLEAN\n(the error to beat)", (ref["dirty_chan"][1] - cc)[sl])] +
        [(f"{short(l)}\nrms {float(np.sqrt(np.mean((arts[l]['chan'][1] - cc) ** 2))):.3g}", (arts[l]["chan"][1] - cc)[sl]) for l in labels],
        title=f"{case}: error at the systemic channel {int(ref['chan_idx'][1])} (denoised - clean)",
        path=f"{base}_chan_err.png", cmap="RdBu_r", vlim=(-L, L), cbar_label="channel error"))

    # ---- sharpness ----
    def grad(m):
        gy, gx = np.gradient(np.nan_to_num(m))
        return np.hypot(gx, gy)
    gm = grad(ref["clean_m1q"])
    G = _p(gm, mask, 99) * 1.3
    out.append(contact_sheet(
        tiles(lambda: gm, lambda: grad(ref["dirty_m1q"]), lambda a: grad(a["m1q"]),
              lambda l: f"{short(l)}\ngradE/clean {r(l, 'gradE_ratio', '{:.2f}')}"),
        title=f"{case}: M1 gradient magnitude (fine velocity structure). A model that fades toward black is smoothing; "
              f"dirty is bright because noise looks sharp.",
        path=f"{base}_sharpness.png", cmap="magma", vlim=(0, G), cbar_label="|grad M1| (km/s per px)", blank=mk))

    # ---- invented structure ----
    out.append(contact_sheet(
        [("DIRTY input", ref["dirty_invented"][sl].astype(np.float32))] +
        [(f"{short(l)}\nblobs/ch {r(l, 'invented_blobs', '{:.1f}')}", arts[l]["invented"][sl].astype(np.float32)) for l in labels],
        title=f"{case}: invented structure, the fraction of channels in which a checkpoint asserts signal where clean has none",
        path=f"{base}_invented.png", cmap="magma", vlim=(0, 0.25), cbar_label="fraction of channels"))

    # ---- spectra ----
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    v = ref["velax"] / 1000.0
    for j, nm in enumerate(["disk peak", "disk edge", "off source"]):
        ax[j].plot(v, ref["dirty_spec"][j], color="#bbb", lw=.9, label="dirty")
        for l in labels:
            fam = rows.get(l, {}).get("family", "unet")
            ax[j].plot(v, arts[l]["spec"][j], color=FAM.get(fam, "#888"), lw=.7, alpha=.65)
        ax[j].plot(v, ref["clean_spec"][j], color="k", lw=1.6, label="clean")
        ax[j].set_title(f"{nm}  (y, x) = {tuple(int(t) for t in ref['px'][j])}", fontsize=10)
        ax[j].set_xlabel("velocity (km/s)"); ax[j].set_ylabel("Jy/beam")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{case}: line profiles. Coloured by family; a model that follows black and ignores grey is behaving.", fontsize=10)
    fig.tight_layout(); p = f"{base}_spectra.png"; fig.savefig(p, dpi=105); plt.close(fig); out.append(p)

    # ---- radial profile and power spectrum ----
    out.append(_radial_and_power(case, ref, arts, labels, rows, base))
    out.append(_topk(case, ref, arts, labels, rows, base, sl, mk))
    return out


def _radial(m, cx, cy, nb=60):
    H, W = m.shape
    y, x = np.mgrid[0:H, 0:W]
    r = np.hypot(x - cx, y - cy)
    edges = np.linspace(0, min(H, W) * 0.45, nb + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    z = np.nan_to_num(m).ravel()
    ok = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[ok], weights=z[ok], minlength=nb)[:nb]
    c = np.bincount(idx[ok], minlength=nb)[:nb]
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(c, 1)


def _power(m1, mask, nb=48):
    z = np.where(mask, np.nan_to_num(m1), 0.0)
    F = np.fft.fftshift(np.abs(np.fft.fft2(z)) ** 2)
    H, W = z.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H)); fx = np.fft.fftshift(np.fft.fftfreq(W))
    rr = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.digitize(rr.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[ok], weights=F.ravel()[ok], minlength=nb)[:nb]
    c = np.bincount(idx[ok], minlength=nb)[:nb]
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(c, 1)


def _radial_and_power(case, ref, arts, labels, rows, base):
    plt = _plt()
    mask = ref["mask"].astype(bool)
    cx, cy = float(ref["cx"]), float(ref["cy"])
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    r, pc = _radial(ref["clean_m0"], cx, cy)
    _, pd_ = _radial(ref["dirty_m0"], cx, cy)
    ax[0].plot(r, pc, "k", lw=1.8, label="clean"); ax[0].plot(r, pd_, color="#bbb", label="dirty")
    ax[0].set_xlim(0, float(min(mask.shape) * 0.3)); ax[1].set_xlim(0, float(min(mask.shape) * 0.3))
    for l in labels:
        _, pl = _radial(arts[l]["m0"], cx, cy)
        ax[0].plot(r, pl, color=FAM.get(rows.get(l, {}).get("family", "unet"), "#888"), lw=.7, alpha=.65)
        ax[1].plot(r, pl / np.where(pc != 0, pc, np.nan), color=FAM.get(rows.get(l, {}).get("family", "unet"), "#888"), lw=.7, alpha=.65)
    ax[1].plot(r, pd_ / np.where(pc != 0, pc, np.nan), color="#bbb", lw=1)
    ax[1].axhline(1, color="k", lw=1)
    ax[0].set_title("M0 radial profile", fontsize=10); ax[1].set_title("M0 radial profile / clean (1 = exact)", fontsize=10)
    ax[1].set_ylim(0.0, 2.2)
    for a in ax[:2]: a.set_xlabel("radius (px)")
    from matplotlib.lines import Line2D
    fams = sorted({rows.get(l, {}).get("family", "unet") for l in labels})
    ax[0].legend(handles=[Line2D([0], [0], color="k", lw=1.8, label="clean"), Line2D([0], [0], color="#bbb", label="dirty")] +
                 [Line2D([0], [0], color=FAM.get(f, "#888"), label=f) for f in fams], fontsize=8)
    f, Pc = _power(ref["clean_m1q"], mask)
    _, Pd = _power(ref["dirty_m1q"], mask)
    safe = np.where(Pc > 0, Pc, np.nan)
    ax[2].semilogy(f, Pc / safe, "k", lw=1.8, label="clean")
    ax[2].semilogy(f, Pd / safe, color="#bbb", lw=1.2, label="dirty")
    for l in labels:
        _, Pl = _power(arts[l]["m1q"], mask)
        ax[2].semilogy(f, Pl / safe, color=FAM.get(rows.get(l, {}).get("family", "unet"), "#888"), lw=.7, alpha=.65)
    ax[2].set_xlabel("spatial frequency (cycles/px)"); ax[2].set_title("M1 power / clean. Below 1 at high frequency = smoothed away", fontsize=10)
    ax[2].legend(handles=[Line2D([0], [0], color="k", lw=1.8, label="clean"), Line2D([0], [0], color="#bbb", label="dirty")] +
                 [Line2D([0], [0], color=FAM.get(f, "#888"), label=f) for f in fams], fontsize=8)
    fig.suptitle(f"{case}: does the recovered disk keep the right radial structure and the fine velocity structure?", fontsize=10)
    fig.tight_layout(); p = f"{base}_radial_power.png"; fig.savefig(p, dpi=105); plt.close(fig)
    return p


def _topk(case, ref, arts, labels, rows, base, sl, mk, k=6):
    """Large-format detail: clean, dirty and the best `k` checkpoints, seven views each."""
    plt = _plt()
    pick = labels[:k]
    for extra in ("winner_aug_seed43", "sweep_winner_aug_seed43"):
        if extra in arts and extra not in pick:
            pick = pick + [extra]
            break
    cols = ["M0", "M1 (km/s)", "M1 error", "M2 (km/s)", "wiggle residual", "systemic channel", "invented"]
    n = 2 + len(pick)
    fig, ax = plt.subplots(n, len(cols), figsize=(len(cols) * 2.35, n * 2.35 + 0.6), squeeze=False)
    mask = ref["mask"].astype(bool)
    vs = float(np.nanmedian(ref["clean_m1"][mask] / 1000.0))
    L1 = _p(np.abs(ref["clean_m1"] / 1000.0 - vs), mask, 99)
    lim_e = _p(np.abs(ref["dirty_m1"] / 1000.0 - ref["clean_m1"] / 1000.0), mask, 95)
    lim_r = max(3.0 * float(np.sqrt(np.nanmean(ref["clean_resid"][mask] ** 2))), 1e-6)
    cc = ref["clean_chan"][1]; Lc = _p(np.abs(cc), np.ones_like(cc, bool), 99.7)
    def draw(i, name, m0, m1, m2, resid, chan, inv, err_src):
        data = [(m0, "inferno", 0, _p(ref["clean_m0"], mask, 99.5), True),
                (m1 / 1000.0, "RdBu_r", vs - L1, vs + L1, True),
                ((m1 - ref["clean_m1"]) / 1000.0, "RdBu_r", -lim_e, lim_e, True),
                (m2 / 1000.0, "viridis", 0, _p(ref["clean_m2"] / 1000.0, mask, 99), True),
                (resid, "RdBu_r", -lim_r, lim_r, True),
                (chan, "RdBu_r", -Lc, Lc, False),
                (inv.astype(np.float32), "magma", 0, 0.25, False)]
        for j, (a, cm, lo, hi, bl) in enumerate(data):
            a = np.array(a[sl], dtype=np.float64)
            if bl: a = np.where(mk, a, np.nan)
            c = plt.get_cmap(cm).copy(); c.set_bad("#111111")
            ax[i, j].imshow(a, origin="lower", cmap=c, vmin=lo, vmax=hi, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(cols[j], fontsize=8)
        ax[i, 0].set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
    draw(0, "CLEAN", ref["clean_m0"], ref["clean_m1"], ref["clean_m2"], ref["clean_resid"], ref["clean_chan"][1], np.zeros_like(ref["dirty_invented"]), None)
    draw(1, "DIRTY", ref["dirty_m0"], ref["dirty_m1"], ref["dirty_m2"], ref["dirty_resid"], ref["dirty_chan"][1], ref["dirty_invented"], None)
    for i, l in enumerate(pick, 2):
        a = arts[l]
        rr = rows.get(l, {})
        nm = f"{short(l)}\nM0 {rr.get('M0', float('nan')):+.0f}  M1 {rr.get('M1', float('nan')):+.0f}\nM2 {rr.get('M2', float('nan')):+.0f}  r {rr.get('resid_r', float('nan')):.2f}"
        draw(i, nm, a["m0"], a["m1"], a["m2"], a["resid"], a["chan"][1], a["invented"], None)
    fig.suptitle(f"{case}: clean, dirty and the best checkpoints by M0. Same scales as the sheets.", fontsize=10)
    fig.tight_layout(rect=(0.06, 0, 1, 0.98))
    p = f"{base}_topk.png"; fig.savefig(p, dpi=105); plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ #
# Figures from the results table alone                                                              #
# ------------------------------------------------------------------------------------------------ #
def _domain_table(ok, domain):
    d = ok[ok.domain == domain]
    if d.empty:
        return None
    g = d.groupby("checkpoint")
    cols = ["psnr", "ssim", "M0", "M1", "M2", "resid_r", "wiggle_gain", "gradE_ratio", "lapvar_ratio", "invented_blobs", "overshoot"]
    t = g[cols].mean()
    t["M0_sd"], t["M1_sd"], t["n"] = g["M0"].std(), g["M1"].std(), g.size()
    t["family"] = g["family"].first()
    return t


def img_table(t, title, path, sort="M0", cols=None):
    """A score table as a heat-coloured image: each column coloured by its own z-score, blue = better."""
    plt = _plt()
    cols = cols or ["psnr", "ssim", "M0", "M1", "M2", "resid_r", "wiggle_gain", "gradE_ratio", "invented_blobs", "n"]
    t = t.sort_values(sort, ascending=False)
    better = {"invented_blobs": -1, "gradE_ratio": 0}      # gradE: target is 1, coloured by distance below
    z = np.zeros((len(t), len(cols)))
    for j, c in enumerate(cols):
        v = t[c].astype(float).values
        if c == "gradE_ratio":
            v = -np.abs(v - 1.0)
        elif better.get(c) == -1:
            v = -v
        sd = np.nanstd(v)
        z[:, j] = 0 if c == "n" or not sd else np.clip((v - np.nanmean(v)) / sd, -2.2, 2.2)
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 3.6, 0.245 * len(t) + 1.4))
    ax.imshow(z, cmap="RdBu", vmin=-2.4, vmax=2.4, aspect="auto")
    for i in range(len(t)):
        for j, c in enumerate(cols):
            v = t[c].iloc[i]
            txt = "" if not np.isfinite(v) else (f"{int(v)}" if c == "n" else f"{v:.2f}" if c in ("resid_r", "wiggle_gain", "gradE_ratio", "psnr", "invented_blobs") else f"{v:.4f}" if c == "ssim" else f"{v:+.0f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.6, color="k")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([f"{short(i)}  [{f}]" for i, f in zip(t.index, t["family"])], fontsize=6.6)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=9, pad=28)
    fig.tight_layout(); fig.savefig(path, dpi=105); plt.close(fig)
    return path


def fig_scoreboard(t, title, path):
    """Every metric as bars, checkpoints in the same order down every panel, error bars = spread across cubes."""
    plt = _plt()
    t = t.sort_values("M0", ascending=True)
    spec = [("M0", "M0 improvement (%)", "M0_sd", 0), ("M1", "M1 improvement (%)", "M1_sd", 0), ("M2", "M2 improvement (%)", None, 0),
            ("resid_r", "wiggle corr. with clean", None, None), ("wiggle_gain", "wiggle gain over dirty", None, 0),
            ("gradE_ratio", "sharpness / clean (1 = exact)", None, 1), ("invented_blobs", "invented blobs / channel", None, 0),
            ("psnr", "PSNR (does not rank)", None, None), ("ssim", "SSIM", None, None)]
    fig, ax = plt.subplots(1, len(spec), figsize=(2.35 * len(spec) + 2.4, 0.2 * len(t) + 1.8), sharey=True)
    y = np.arange(len(t))
    for a, (c, nm, sd, ref_line) in zip(ax, spec):
        a.barh(y, t[c].values, xerr=t[sd].values if sd else None, color=[FAM.get(f, "#888") for f in t["family"]], ecolor="#555", error_kw=dict(lw=.6))
        if ref_line is not None: a.axvline(ref_line, color="k", lw=.9)
        a.set_title(nm, fontsize=7.5); a.tick_params(labelsize=6)
    ax[0].set_yticks(y); ax[0].set_yticklabels([short(i) for i in t.index], fontsize=6.4)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=105); plt.close(fig)
    return path


_LOSS = re.compile(r"^winner_(mae|wavelet|starlet|gradient|hybrid)(?:_(p10|beam))?_(ft|fresh)_seed\d+$")


def fig_loss_source(ok, path, domain="line_emission"):
    """The loss x training-source matrix: Jason's asks 1 and 3 in one image. Mean across cubes; one seed per cell."""
    plt = _plt()
    d = ok[ok.domain == domain]
    g = d.groupby("checkpoint")[["M0", "M1", "M2", "resid_r", "gradE_ratio", "psnr"]].mean()
    losses = ["hybrid", "mae", "wavelet", "starlet", "gradient"]
    sources = [("aug", "fine-tune from aug"), ("p10", "fine-tune from p10"), ("beam", "fine-tune from beam"), ("fresh", "from scratch")]
    grids = {m: np.full((len(losses), len(sources)), np.nan) for m in g.columns}
    for lab, row in g.iterrows():
        m = _LOSS.match(lab)
        if not m:
            continue
        loss, src, mode = m.groups()
        s = "fresh" if mode == "fresh" else (src or "aug")
        i, j = losses.index(loss), [x[0] for x in sources].index(s)
        for c in g.columns:
            grids[c][i, j] = row[c]
    if all(np.isnan(v).all() for v in grids.values()):
        return None
    fig, ax = plt.subplots(1, 6, figsize=(21, 3.9))
    for a, (c, nm, cm) in zip(ax, [("M0", "M0 improvement (%)", "RdBu"), ("M1", "M1 improvement (%)", "RdBu"),
                                   ("M2", "M2 improvement (%)", "RdBu"), ("resid_r", "wiggle corr.", "RdBu"),
                                   ("gradE_ratio", "sharpness / clean", "PuOr"), ("psnr", "PSNR (dB)", "RdBu")]):
        v = grids[c]
        a.imshow(v, cmap=cm, aspect="auto", vmin=np.nanmin(v) if not np.isnan(v).all() else 0, vmax=np.nanmax(v) if not np.isnan(v).all() else 1)
        for i in range(len(losses)):
            for j in range(len(sources)):
                if np.isfinite(v[i, j]):
                    a.text(j, i, f"{v[i, j]:.2f}" if c in ("resid_r", "gradE_ratio", "psnr") else f"{v[i, j]:+.0f}", ha="center", va="center", fontsize=8)
        a.set_xticks(range(len(sources))); a.set_xticklabels([s[1] for s in sources], fontsize=7, rotation=20, ha="right")
        a.set_yticks(range(len(losses))); a.set_yticklabels(losses if a is ax[0] else [], fontsize=8)
        a.set_title(nm, fontsize=9)
    fig.suptitle("Loss x source, line-emission cubes, mean across cubes, ONE seed per cell (differences under ~10 points are inside seed spread). "
                 "'hybrid' row = the control that keeps the original loss.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=105); plt.close(fig)
    return path


def fig_cube_heatmaps(ok, path, domain="line_emission"):
    """Checkpoint x cube, so a result carried by one cube is visible."""
    plt = _plt()
    d = ok[ok.domain == domain]
    if d.empty:
        return None
    order = d.groupby("checkpoint")["M0"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(1, 4, figsize=(18, 0.22 * len(order) + 1.8), sharey=True)
    for a, (c, nm) in zip(ax, [("M0", "M0 (%)"), ("M1", "M1 (%)"), ("M2", "M2 (%)"), ("resid_r", "wiggle corr.")]):
        pv = d.pivot_table(index="checkpoint", columns="case", values=c).reindex(order)
        im = a.imshow(pv.values, cmap="RdBu", aspect="auto", vmin=np.nanmin(pv.values), vmax=np.nanmax(pv.values))
        a.set_xticks(range(pv.shape[1])); a.set_xticklabels([x.replace("run_", "")[:14] for x in pv.columns], rotation=60, fontsize=6.5, ha="right")
        a.set_title(nm, fontsize=9); fig.colorbar(im, ax=a, fraction=0.046)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([short(i) for i in order], fontsize=6.4)
    fig.suptitle("Every checkpoint on every cube: a good mean carried by one cube shows here", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=105); plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ #
def build_all(map_dir: str, ok, out_dir: str, figure_cases: Optional[Sequence[str]] = None, log=print) -> List[str]:
    """Every figure, written to `out_dir`. `ok` is the scored-rows DataFrame. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for domain, tag in (("line_emission", "line_emission"), ("sg", "sg")):
        t = _domain_table(ok, domain)
        if t is None:
            continue
        paths.append(img_table(t, f"{tag}: mean across cubes, colour = z-score per column (blue better), sorted by M0", os.path.join(out_dir, f"nb16_table_{tag}.png")))
        paths.append(fig_scoreboard(t, f"{tag}: every metric, every checkpoint, same order in every panel. Bars = mean across cubes, whiskers = spread across cubes.", os.path.join(out_dir, f"nb16_scoreboard_{tag}.png")))
    for f in (fig_loss_source(ok, os.path.join(out_dir, "nb16_loss_x_source.png")),
              fig_cube_heatmaps(ok, os.path.join(out_dir, "nb16_per_cube.png"))):
        if f:
            paths.append(f)
    if figure_cases is None:                       # every case that has saved maps
        figure_cases = sorted(os.path.basename(p)[:-len("__REF.npz")] for p in glob.glob(os.path.join(map_dir, "*__REF.npz")))
    for case in figure_cases:
        ref, arts = load_case(map_dir, case)
        if ref is None or not arts:
            log(f"  no saved maps for {case}: skipped")
            continue
        rows = _rows_for(ok, case)
        got = sheets_for_case(case, ref, arts, rows, out_dir)
        log(f"  {case}: {len(arts)} checkpoint(s) -> {len(got)} figure(s)")
        paths += got
    return paths
