import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wiener_wiggle import read_fits, SG, ROOT, NPZ, MASK, VKEP

BLK = np.arange(180, 421)
clean, hc = read_fits(f"{SG}/clean_sg.fits", BLK)
dirty, _ = read_fits(f"{SG}/dirty_sg.fits", BLK)
lf, _ = read_fits(f"{SG}/dirty_sg.fits", np.r_[0:60, 541:601])
static = lf.mean(0); lf = lf - static
dcs = dirty - static[None]
n = dirty.shape[-1]
beam, _ = read_fits(f"{ROOT}/results/self-gravitating/dirty_beam_recovered_v2.fits")
pad = np.zeros((n, n)); h = beam.shape[0] // 2; c = n // 2
pad[c - h:c + h + 1, c - h:c + h + 1] = beam
H = np.fft.fft2(np.fft.ifftshift(pad))
Hs = np.where(np.abs(H) > 1e-6 * np.abs(H).max(), H, 1e-6 * np.abs(H).max())
kr = np.sqrt(np.fft.fftfreq(n)[None, :] ** 2 + np.fft.fftfreq(n)[:, None] ** 2)
NB = 120; rb = np.minimum((kr / kr.max() * NB).astype(int), NB - 1)
cnt = np.maximum(np.bincount(rb.ravel(), minlength=NB), 1)
rad2 = lambda P: (np.bincount(rb.ravel(), P.ravel(), NB) / cnt)[rb]
Pn2 = rad2(np.mean(np.abs(np.fft.fft2(lf)) ** 2, axis=0)); del lf
D3 = np.fft.fftn(dcs, axes=(0, 1, 2))
Pd3 = np.stack([rad2(p) for p in np.abs(D3) ** 2]) / dcs.shape[0]
G = np.clip(1 - Pn2[None] / np.maximum(Pd3, 1e-30), 0, 1)
X = np.real(np.fft.ifftn(D3 * G / Hs[None], axes=(0, 1, 2)))

res = json.load(open("wiener3d_results.json")); r1 = json.load(open("wiener_results.json"))["results"]
w3 = np.load("m1_wiener3d_a1.0.npy")
cut = (slice(60, 540), slice(60, 540))

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])
rows = [("clean (target)", NPZ["clean_r"], None), ("dirty", NPZ["dirty_r"], r1["dirty"]["resid_r"]),
        ("U-Net (current best)", NPZ["U-Net_r"], r1["U-Net"]["resid_r"]),
        ("3D Wiener (space+velocity)", w3 - VKEP, res["wiener3d_a1.0"]["resid_r"])]
for i, (t, a, r) in enumerate(rows):
    ax = fig.add_subplot(gs[0, i]); ax.set_facecolor("#111")
    im = ax.imshow(np.where(MASK, a, np.nan)[cut], cmap="RdBu_r", vmin=-.5, vmax=.5, origin="lower")
    ax.set_title(f"{t}\nwiggle residual" + (f", r = {r:.3f}" if r else ""), fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=fig.axes[:4], fraction=0.012, pad=0.01).set_label("km/s")

ch = 335 - 180   # v = +1.17 km/s: one isovelocity lobe, where the zig-zag shows
vmax = np.percentile(clean[ch], 99.7)
for i, (t, cube) in enumerate([("clean", clean), ("dirty", dirty), ("dirty, static pattern removed", dcs),
                               ("3D Wiener", X)]):
    ax = fig.add_subplot(gs[1, i])
    v_ = np.percentile(cube[ch][cut], 99.7); ax.imshow(cube[ch][cut], cmap="inferno", vmin=0, vmax=v_, origin="lower")
    ax.set_title(f"{t}\nchannel map v = +1.17 km/s", fontsize=12); ax.set_xticks([]); ax.set_yticks([])

bands = json.load(open("wiener_results.json"))["bands"]; x = np.arange(len(bands))
ax = fig.add_subplot(gs[2, :2])
for k, c_, ls, lab in [("dirty", "#555", "-", "dirty"), ("U-Net", "#d62728", "-", "U-Net")]:
    ax.plot(x, r1[k]["coh"], ls, color=c_, marker="o", lw=2, label=lab)
ax.plot(x, r1["beam-only"]["coh"], "--", color="#999", marker="o", lw=2, label="noise-free beam-only")
ax.plot(x, res["wiener2d_static_a1.0"]["coh"], "-", color="#7fb3e0", marker="o", lw=2, label="2D Wiener")
ax.plot(x, res["wiener3d_a1.0"]["coh"], "-", color="#1f4e9c", marker="o", lw=3, label="3D Wiener")
ax.plot(x, res["oracle_wiener3d"]["coh"], ":", color="k", lw=2, label="linear ceiling (oracle)")
ax.axvspan(1.5, 3.5, color="#f2a65a", alpha=.18, label="GI-finger scales")
ax.set_xticks(x); ax.set_xticklabels([b.replace("-inf", "+") for b in bands], rotation=20)
ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower left")
ax.set_ylabel("coherence with clean residual"); ax.set_title("Wiggle recovered at each scale")

ax = fig.add_subplot(gs[2, 2:])
names = ["dirty", "U-Net", "2D Wiener", "3D Wiener", "linear\nceiling", "noise-free\nbeam-only"]
vals = [r1["dirty"]["resid_r"], r1["U-Net"]["resid_r"], res["wiener2d_static_a1.0"]["resid_r"],
        res["wiener3d_a1.0"]["resid_r"], res["oracle_wiener3d"]["resid_r"], r1["beam-only"]["resid_r"]]
cols = ["#555", "#d62728", "#7fb3e0", "#1f4e9c", "k", "#999"]
b = ax.bar(names, vals, color=cols)
for bb, v in zip(b, vals): ax.text(bb.get_x() + bb.get_width() / 2, v + .003, f"{v:.3f}", ha="center")
ax.set_ylim(0.75, 0.95); ax.set_ylabel("wiggle resid_r vs clean"); ax.grid(axis="y", alpha=.3)
ax.set_title("Overall wiggle correlation")
fig.suptitle("SG v2: remove the static noise pattern, then deconvolve the beam in 3D (space + velocity). "
             "Reaches the linear ceiling; the U-Net falls below doing nothing.", fontsize=14)
fig.savefig("wiggle_3d_wiener.png", dpi=100, bbox_inches="tight"); print("saved")
