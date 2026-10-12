# Mass-weighted line-of-sight velocity of the upper gas layer at incl 30 deg, minus the azimuthal mean v(R): GI kinematic residual.
import sys, numpy as np, sarracen, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[-1]; fig, axs = plt.subplots(1, len(sys.argv) - 2, figsize=(6 * (len(sys.argv) - 2), 5))
for ax, f in zip(np.atleast_1d(axs), sys.argv[1:-1]):
    sdf, sinks = sarracen.read_phantom(f, separate_types="sinks")
    if "itype" in sdf.columns: print(f.split("/")[-1], "itype counts", sdf["itype"].value_counts().to_dict())
    g = sdf[sdf["itype"] == 1] if "itype" in sdf.columns else sdf
    s = sinks.iloc[0]; x, y, z = g.x - s.x, g.y - s.y, g.z - s.z
    vx, vy, vz = g.vx - s.vx, g.vy - s.vy, g.vz - s.vz
    R = np.hypot(x, y); H = np.array([np.std(z[(R > r) & (R < r * 1.1)]) for r in R.clip(1).values[:0]])
    up = z > 1.0 * np.interp(R, *zip(*[(r, np.std(z[(R > r) & (R < r * 1.2)])) for r in np.geomspace(R.quantile(.02), R.quantile(.98), 40)]))
    i = np.radians(30); yp = y * np.cos(i) - z * np.sin(i); vlos = vy * np.sin(i) + vz * np.cos(i)
    vphi = (x * vy - y * vx) / R; Rb = np.geomspace(R.quantile(.02), R.quantile(.98), 60)
    k = np.digitize(R, Rb); mean_vphi = np.array([vphi[k == j].mean() if (k == j).any() else np.nan for j in range(len(Rb) + 1)])
    vmod = mean_vphi[k] * (x / R) * np.sin(i)          # axisymmetric model projected
    L = R.quantile(.95); bins = np.linspace(-L, L, 201)
    w = np.histogram2d(x[up], yp[up], bins)[0]; v = np.histogram2d(x[up], yp[up], bins, weights=(vlos - vmod)[up])[0]
    kms = 29.78 * np.sqrt(1)  # code velocity unit with udist=au, umass=Msun, G=1 -> 29.78 km/s
    im = ax.imshow((v / np.maximum(w, 1)).T * kms, origin="lower", cmap="RdBu_r", vmin=-0.3, vmax=0.3, extent=[-L, L, -L, L])
    ax.set_title(f"{f.split('/')[-2]}/{f.split('/')[-1]}  upper layer, i=30\nv_los - <v_phi>(R) [km/s]"); plt.colorbar(im, ax=ax)
fig.savefig(out, dpi=80, bbox_inches="tight"); print("saved", out)
