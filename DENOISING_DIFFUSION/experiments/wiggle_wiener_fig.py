import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/Krishan/Projects/EXXA/DENOISING_DIFFUSION"
d = np.load(f"{R}/experiments/wiggle_all_methods_step1.npz")
m = d["mask"]
vkep = d["clean_m1"] - d["clean_r"]
w = np.load("wiener_best_m1.npy")
res = json.load(open("wiener_results.json"))

panels = [("clean (target)", d["clean_r"]), ("dirty", d["dirty_r"]),
          ("U-Net (winner_aug)", d["U-Net_r"]), ("Wiener deconv, alpha=1", w - vkep)]
rr = {"dirty": res["results"]["dirty"]["resid_r"], "U-Net (winner_aug)": res["results"]["U-Net"]["resid_r"],
      "Wiener deconv, alpha=1": res["results"]["wiener_a1.0"]["resid_r"]}

fig = plt.figure(figsize=(18, 9.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.8])
for i, (t, a) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    img = np.where(m, a, np.nan)
    ax.set_facecolor("#111")
    im = ax.imshow(img[60:540, 60:540], cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="lower")
    sub = f"\nresid_r vs clean = {rr[t]:.3f}" if t in rr else "\n "
    ax.set_title(t + sub, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=fig.axes[:4], fraction=0.015, pad=0.01)
cb.set_label("Keplerian-subtracted velocity (km/s)")

ax = fig.add_subplot(gs[1, :2])
x = np.arange(len(res["bands"]))
style = {"dirty": ("#555", "-"), "beam-only": ("#999", "--"), "U-Net": ("#d62728", "-"),
         "DDRM": ("#9467bd", ":"), "wiener_a1.0": ("#1f77b4", "-")}
names = {"dirty": "dirty", "beam-only": "noise-free beam-only (denoising ceiling)", "U-Net": "U-Net",
         "DDRM": "DDRM", "wiener_a1.0": "Wiener deconv"}
for k, (c, ls) in style.items():
    ax.plot(x, res["results"][k]["coh"], ls, color=c, marker="o", lw=2, label=names[k])
ax.axvspan(1.5, 3.5, color="#f2a65a", alpha=0.18, label="GI-wiggle finger scales")
ax.set_xticks(x); ax.set_xticklabels([b.replace("-inf", "+") for b in res["bands"]], rotation=20)
ax.set_xlabel("spatial wavelength band"); ax.set_ylabel("coherence with clean residual")
ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower left")
ax.set_title("Where each method keeps or loses the wiggle, by scale")

ax = fig.add_subplot(gs[1, 2:])
for k, (c, ls) in style.items():
    if k == "DDRM":
        continue
    ax.plot(x, res["results"][k]["pow"], ls, color=c, marker="o", lw=2, label=names[k])
ax.axhline(1, color="k", lw=1)
ax.axvspan(1.5, 3.5, color="#f2a65a", alpha=0.18)
ax.set_xticks(x); ax.set_xticklabels([b.replace("-inf", "+") for b in res["bands"]], rotation=20)
ax.set_xlabel("spatial wavelength band"); ax.set_ylabel("power / clean power")
ax.grid(alpha=.3); ax.legend(fontsize=9)
ax.set_title("Power ratio: <1 = suppressed (blur), >1 = invented structure")

fig.suptitle("SG v2 cube: GI wiggle recovery. Beam-aware Wiener deconvolution beats dirty at every scale >= 6 px; "
             "the U-Net loses exactly at the finger scales", fontsize=13)
fig.savefig("wiggle_wiener_vs_unet.png", dpi=110, bbox_inches="tight")
print("saved")
