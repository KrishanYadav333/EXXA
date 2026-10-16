"""
Build a ready-to-submit unified-training kernel folder from kaggle/unified/kernel_unified.py (RULES.md #14 resume included), one job per call.

  python make_unified_job.py NAME OUTDIR [--warm FILE --warm-source owner/kernel] -- <train_unified.py flags>
  python orchestrator.py submit OUTDIR --slug NAME --account ACCT --shared-datasets --shared-kernel-sources

The Oct 17 batch (PLAN.md 0b2) is written by `python make_unified_job.py --oct17 OUTROOT`.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
R6 = ["--snr", "3,150", "--chan-corr", "0.5", "--beam-max", "1.6", "--lr", "1.5e-4", "--clip-in", "10", "--fp32-stream", "--norm", "robust",
      "--crop", "256", "--batch", "4", "--rounds", "80", "--eval-ladder", "3,6,12,24"]
LEVERS = ["--infer-beam-px", "10", "--infer-mode", "full", "--wiener2"]      # evaluation of a beam-matched model uses the frozen inference recipe
FT = ["--snr", "3,150", "--chan-corr", "0.5", "--beam-max", "1.6", "--lr", "5e-5", "--clip-in", "10", "--fp32-stream", "--norm", "robust",
      "--crop", "256", "--batch", "4", "--rounds", "20", "--eval-ladder", "3,6,12,24"]
BATCH = {   # name: (flags, warm file, warm source kernel)
    "unified-r6-s32":   (R6 + ["--seed", "32", "--tag", "r6_s32"], "", ""),
    "unified-r6-s33":   (R6 + ["--seed", "33", "--tag", "r6_s33"], "", ""),
    "unified-r9-s61":   (R6 + ["--seed", "61", "--train-beam-px", "8,12", "--tag", "r9_s61"] + LEVERS, "", ""),
    "unified-r9-s62":   (R6 + ["--seed", "62", "--train-beam-px", "8,12", "--tag", "r9_s62"] + LEVERS, "", ""),
    "unified-r8-w003":  (FT + ["--seed", "71", "--w-m1", "0.003", "--tag", "r8_w003"], "unified_r6_k15_crop256.pth", "chiragbhayal/unified-r6-k15-crop256"),
    "unified-r8-w030":  (FT + ["--seed", "72", "--w-m1", "0.03", "--tag", "r8_w030"], "unified_r6_k15_crop256.pth", "chiragbhayal/unified-r6-k15-crop256"),
}


def make(name, out, extra, warm="", warm_src=""):
    src = open(os.path.join(ROOT, "kaggle/unified/kernel_unified.py")).read()
    k = src.replace("K = 3                  #", "K = 15                 #", 1).replace("EXTRA = []             #", f"EXTRA = {json.dumps(extra)}             #", 1)
    if warm:
        k = k.replace('WARM = ""              #', f'WARM = "{warm}"              #', 1)
    assert "K = 15" in k and json.dumps(extra) in k and (not warm or warm in k)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "kernel_unified.py"), "w").write(k)
    m = json.load(open(os.path.join(ROOT, "kaggle/unified/kernel-metadata.json")))
    m.update(id=f"x/{name}", title=name, code_file="kernel_unified.py", kernel_sources=["krishanyadav333/speedie-prep-fast"] + ([warm_src] if warm_src else []))
    json.dump(m, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)


if __name__ == "__main__":
    if "--oct17" in sys.argv:
        root = sys.argv[sys.argv.index("--oct17") + 1]
        for n, (f, w, ws) in BATCH.items():
            make(n, os.path.join(root, n), f, w, ws); print(n, "->", os.path.join(root, n))
    else:
        ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("out"); ap.add_argument("--warm", default=""); ap.add_argument("--warm-source", default="")
        a, rest = ap.parse_known_args(); make(a.name, a.out, [x for x in rest if x != "--"], a.warm, a.warm_source)
