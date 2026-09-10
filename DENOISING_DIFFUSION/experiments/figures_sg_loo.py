"""
Figure for notebook 11's leave-one-out wiggle scoring (score_sg_wiggle_loo.py).

Metrics-only: reads results/self-gravitating/sg_loo_wiggle.json directly, no re-denoise.
Each fold's checkpoint never saw that disk during training, so the 5 points are genuine
independent holdouts, not repeats of the same cube. mstar_at_bound is checked per fold
(RULES.md #8) even though these fits hold inclination fixed from the .para file.

Run: PYTHONPATH=.. python3 experiments/figures_sg_loo.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON_PATH = "results/self-gravitating/sg_loo_wiggle.json"
OUT = "results/self-gravitating/sg_loo_wiggle_vs_fold.png"
METHODS = ["dirty", "frozen", "finetune", "fresh"]
COLORS = {"dirty": "#a04a30", "frozen": "#6f6f6f", "finetune": "#2f5f96", "fresh": "#1a7f37"}


def main():
    rows = json.load(open(JSON_PATH))
    print(f"{len(rows)} folds")

    for r in rows:
        flag = " DEGEN" if r["geom"]["mstar_at_bound"] else ""
        print(f"  fold {r['fold']} {r['holdout']:24s} mstar={r['geom']['mstar_msun']:.3f}{flag}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    a0 = axes[0]
    x = np.arange(len(rows))
    w = 0.2
    for i, m in enumerate(METHODS):
        vals = [r["methods"][m]["resid_r"] for r in rows]
        a0.bar(x + (i - 1.5) * w, vals, width=w, label=m, color=COLORS[m])
    a0.set_xticks(x)
    a0.set_xticklabels([r["holdout"][:14] for r in rows], rotation=30, ha="right", fontsize=8)
    a0.set_ylabel("wiggle resid_r")
    a0.set_title("Per-fold: each disk scored by the checkpoint that never saw it")
    a0.legend(fontsize=8)
    a0.grid(alpha=0.25, axis="y")

    a1 = axes[1]
    means = [np.mean([r["methods"][m]["resid_r"] for r in rows]) for m in METHODS]
    stds = [np.std([r["methods"][m]["resid_r"] for r in rows], ddof=1) for m in METHODS]
    a1.bar(METHODS, means, yerr=stds, capsize=5, color=[COLORS[m] for m in METHODS])
    a1.set_ylabel("wiggle resid_r")
    a1.set_title(f"Mean +/- std across {len(rows)} leave-one-out folds")
    a1.grid(alpha=0.25, axis="y")

    fig.suptitle("Notebook 11: leave-one-out over the 5 SG disks (frac=0.05)", fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT, dpi=135)
    plt.close()
    print("saved ->", OUT)

    for m in METHODS:
        vals = np.array([r["methods"][m]["resid_r"] for r in rows])
        print(f"  {m:10s} resid_r {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
