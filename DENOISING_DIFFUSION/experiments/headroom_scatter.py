"""
Headroom scatter: model kinematic gain (model resid_r - dirty resid_r) vs input quality
(dirty's own resid_r), across every wiggle benchmark this project has run.

Thesis: model value is a function of input degradation, not of any one architecture choice.
Where dirty already sits high (little degraded), every trained model loses -- the noise
being removed is small and orthogonal to the signal, while the model's own smoothing is a
systematic error aligned with the signal. Where dirty sits low (heavily degraded), models
win by a wide margin. This is the figure that reconciles "doing nothing beats the model"
(true on the SG v2 cube) with "kin_gamma0 crushes dirty" (true on the line-emission holdouts)
without either statement contradicting the other.

Real per-point data, not aggregates: each LOO fold and each line-emission holdout cube has
its OWN dirty resid_r (the x-position), because that is the actual quantity the thesis is
about -- pooling to one mean per regime would hide exactly the within-regime spread that
makes the x-axis meaningful. Sources:
  - SG v2 cube:      6 checkpoints, one shared cube -- 2026-09-11 leaderboard + domain-split
                      runs (PROGRESS.md), hardcoded here since no JSON was written for that
                      ad-hoc series
  - SG LOO folds:     results/self-gravitating/sg_loo_wiggle.json, 5 folds x 3 trained arms
  - Line-emission:    results/self-gravitating/nb08_kinematic_wiggle.json, 5 cubes x 4 gammas

Run: PYTHONPATH=.. python3 experiments/headroom_scatter.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/self-gravitating/headroom_scatter.png"

# SG v2 cube: one cube, six checkpoints, all measured against the SAME dirty (0.8620).
# 2026-09-11 entries in PROGRESS.md ("domain split", "leaderboard extends").
SG_V2_DIRTY = 0.8620
sg_v2 = [
    ("winner_aug_seed43", 0.7603),
    ("winner_p10_seed44", 0.6912),
    ("winner_beam_seed42", 0.7113),
    ("sg_k3_fresh", 0.7053),
    ("kin_gamma0", 0.7581),
    ("DDRM", 0.568),
]

loo = json.load(open("results/self-gravitating/sg_loo_wiggle.json"))
le = json.load(open("results/self-gravitating/nb08_kinematic_wiggle.json"))

GAMMAS_TO_PLOT = ["0.0", "0.1"]  # the two arms that ever beat dirty; 1.0/10.0 collapse the axis

fig, ax = plt.subplots(figsize=(10.5, 7))

# --- SG v2 cube: one x position, six models ---
for name, r in sg_v2:
    ax.scatter(SG_V2_DIRTY, r - SG_V2_DIRTY, color="#c0392b", s=90, edgecolor="black",
               zorder=3, marker="o")
ax.scatter([], [], color="#c0392b", s=90, edgecolor="black", marker="o",
           label=f"SG v2 cube, n=6 (dirty={SG_V2_DIRTY:.3f})")

# --- SG leave-one-out: 5 folds x 3 trained arms, each fold's own dirty ---
loo_colors = {"frozen": "#7f8c8d", "finetune": "#d68910", "fresh": "#e67e22"}
seen_loo = set()
for fold in loo:
    dx = fold["methods"]["dirty"]["resid_r"]
    for arm in ("frozen", "finetune", "fresh"):
        ry = fold["methods"][arm]["resid_r"]
        lbl = f"SG LOO: {arm}" if arm not in seen_loo else None
        seen_loo.add(arm)
        ax.scatter(dx, ry - dx, color=loo_colors[arm], s=70, edgecolor="black",
                   marker="^", alpha=0.85, zorder=2, label=lbl)

# --- Line-emission holdouts: 5 cubes x {gamma=0, gamma=0.1}, each cube's own dirty ---
le_colors = {"0.0": "#1e8449", "0.1": "#28b463"}
seen_le = set()
for cube in le:
    dx = cube["dirty_resid_r"]
    for g in GAMMAS_TO_PLOT:
        ry = cube["gammas"][g]["resid_r"]
        lbl = f"Line-emission: kin_gamma={g}" if g not in seen_le else None
        seen_le.add(g)
        ax.scatter(dx, ry - dx, color=le_colors[g], s=110, edgecolor="black",
                   marker="*", zorder=4, label=lbl)

ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=1,
           label="model = dirty (no gain)")

ax.set_xlabel("Input quality: dirty's own wiggle resid_r  (higher = less degraded)",
              fontsize=12)
ax.set_ylabel("Kinematic gain: model resid_r − dirty resid_r", fontsize=12)
ax.set_title("Model kinematic value is a function of input degradation,\n"
             "not of architecture, domain, or loss choice", fontsize=13.5)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print("saved ->", OUT)

# console summary for the log entry
import numpy as np
loo_x = [f["methods"]["dirty"]["resid_r"] for f in loo]
loo_y = [f["methods"][a]["resid_r"] - f["methods"]["dirty"]["resid_r"]
         for f in loo for a in ("frozen", "finetune", "fresh")]
le_x = [c["dirty_resid_r"] for c in le]
le_y = [c["gammas"][g]["resid_r"] - c["dirty_resid_r"] for c in le for g in GAMMAS_TO_PLOT]
sg_y = [r - SG_V2_DIRTY for _, r in sg_v2]
print(f"\nn points: SG v2={len(sg_v2)}  LOO={len(loo_y)} (x range {min(loo_x):.3f}-{max(loo_x):.3f})"
      f"  line-emission={len(le_y)} (x range {min(le_x):.3f}-{max(le_x):.3f})")
print(f"mean gain: SG v2 {np.mean(sg_y):+.3f}  LOO {np.mean(loo_y):+.3f}  "
      f"line-emission {np.mean(le_y):+.3f}")
