"""
One report for the FROZEN final model (PLAN.md: final models, all 5 sizes, wiggle and moments, seed spread), from eval JSONs written by train_unified.py.

  python experiments/final_report.py --final final=eval_final.json --seeds s3=a.json s11=b.json s23=c.json --out results/final/RESULTS.md
         [--ladder final=eval_final.json] [--gate gate=eval_gate.json]

For each of the five sizes: the final (averaged) model next to dirty and Wiener (their rows come from the final file), per group (line, hydro, analytic, sgv2,
young_test, speedie); then, per group and size, the RANGE of every metric across the individual seeds (min .. max), because a single seed is not evidence
(PROGRESS.md 2026-10-11/12: seed spread on the SNR-12 wiggle alone is 0.41-0.62). The SNR ladder (3 / 6 / 12 / 24) comes from ladder_report.py's table,
and the gate sweep (a * model + (1 - a) * Wiener) is printed per case if the file has it. Every number is a mean over the cubes of the group, never over seeds.
"""
import argparse, json, os, subprocess, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_models as cm

SIZES = (256, 300, 320, 480, 600)


def load(spec):
    return {n: json.load(open(p)) for n, p in (x.split("=", 1) for x in spec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", nargs=1, required=True)
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--ladder", nargs="*", default=[])
    ap.add_argument("--gate", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    final, seeds = load(a.final), load(a.seeds)
    (fname, fres), = final.items()
    L = [f"# Final model `{fname}`: results at all five sizes", "",
         "Moments are signal-masked, noise-clipped improvements over dirty (RULES.md #6). `model` = the averaged final model; dirty and Wiener rows are computed on the same cubes.", ""]
    for size in SIZES:
        refs = {"dirty": cm.summarise(fres, size, "dirty"), "wiener": cm.summarise(fres, size, "wiener"), fname: cm.summarise(fres, size, "model")}
        L += [f"## {size} px", ""]
        for g in cm.GROUPS:
            if not any(g in t for t in refs.values()):
                continue
            L += [f"### {g}", "", "| model | n | " + " | ".join(l for _, l in cm.METRICS) + " |", "|" + "---|" * (len(cm.METRICS) + 2)]
            for name, t in refs.items():
                if g in t:
                    r = t[g]
                    L.append(f"| {name} | {r['n']} | " + " | ".join(cm.fmt(r[k]) for k, _ in cm.METRICS) + " |")
            if seeds:
                rng = {}
                for sn, sres in seeds.items():
                    t = cm.summarise(sres, size, "model").get(g)
                    if t:
                        for k, _ in cm.METRICS:
                            if t[k] is not None:
                                rng.setdefault(k, []).append(t[k])
                if rng:
                    L.append(f"| seed range ({len(seeds)} seeds) | | " + " | ".join(f"{min(rng[k]):.2f} .. {max(rng[k]):.2f}" if k in rng else "-" for k, _ in cm.METRICS) + " |")
            L.append("")
    for spec in a.ladder:
        n, p = spec.split("=", 1)
        L += [f"## SNR ladder (Speedie GI r0.5 with its own noise rescaled): `{n}`", "", subprocess.run([sys.executable, os.path.join(HERE, "ladder_report.py"), f"{n}={p}"],
                                                                                                          capture_output=True, text=True).stdout, ""]
    for n, res in load(a.gate).items():
        L += [f"## Gate sweep: `{n}`", "", "| case | blind SNR est | a | M0 | M1 | M2 | PSNR |", "|---|---|---|---|---|---|---|"]
        for key, v in sorted(res.items()):
            if "failed" in v or "mix" not in v:
                continue
            rows = [("1 (model)", v["model"])] + [(a_, m) for a_, m in sorted(v["mix"].items())] + [("0 (Wiener)", v["wiener"])]
            for a_, m in rows:
                L.append(f"| {key} | {v.get('snr_est', float('nan')):.0f} | {a_} | {m.get('M0', float('nan')):.0f} | {m.get('M1', float('nan')):.0f} | {m.get('M2', float('nan')):.0f} | {m.get('psnr', float('nan')):.1f} |")
        L.append("")
    txt = "\n".join(L)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(txt + "\n")
    print(txt[:3000])


if __name__ == "__main__":
    main()
