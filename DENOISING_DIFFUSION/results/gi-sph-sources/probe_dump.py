import sys, numpy as np, sarracen
sdf, sinks = sarracen.read_phantom(sys.argv[1], separate_types="sinks")
print(sys.argv[1].split("/")[-1], "particles", len(sdf), "cols", list(sdf.columns))
print("params", {k: sdf.params[k] for k in list(sdf.params)[:0]})
for k in ("time", "udist", "umass", "utime", "massoftype", "hfact", "gamma"):
    if k in sdf.params: print(" ", k, sdf.params[k])
print("sinks", len(sinks), sinks[["x", "y", "z", "m"]].to_string() if len(sinks) else "")
R = np.hypot(sdf.x, sdf.y)
print("R pct 5/50/95", np.percentile(R, [5, 50, 95]).round(1), " |z|/R median", float(np.median(np.abs(sdf.z) / R)))
m = sdf.params.get("massoftype", sdf.params.get("mass"))
print("Mdisc / M*:", len(sdf) * m / float(sinks.m.sum()) if len(sinks) else "n/a")
