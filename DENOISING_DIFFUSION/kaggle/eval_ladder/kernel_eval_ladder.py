"""
Kaggle kernel (PLAN.md phase 1): the SNR ladder. Speedie's GI disc and its Keplerian twin (robust 0.5) with their own measured noise rescaled to peak
SNR 3, 6, 12, 24, scored for each trained model with the nb16 metrics and the wiggle scorecard (spurious wiggle on the twin). No training.
Inputs: speedie-prep-fast Output (cubes), and the Outputs of unified-wiener (v1), unified-wiener-real-k15 (R3b), unified-wiener-real-sgv2 (R3c).
Add a model by naming its checkpoint in MODELS; each takes ~20-30 min on a T4 (8 native-pixel cases).
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/eval_ladder"
LADDER = "3,6,12,24"

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)

if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO], check=True)
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-msssim", "bettermoments"], check=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
print("overlaid", list(man["files"]), "| staged from", man["local_head"], flush=True)

le = {}
for p in glob.glob("/kaggle/input/**/run_00*", recursive=True):
    if os.path.isdir(p):
        le[os.path.dirname(p)] = le.get(os.path.dirname(p), 0) + 1
LE_DIR = max(le, key=le.get)
SPD = os.path.dirname(glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)[0])
find = lambda pat: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or [None])[0]
MODELS = [("unified_wiener_v1", "unified_wiener.pth"), ("R3b_k15", "unified_real_k15.pth"), ("R3c_k3_sgv2", "unified_real_k3_sgv2.pth")]
models = [(n, ["--eval-ckpt", find(f)]) for n, f in MODELS]
print("speedie ->", SPD, "| models:", {m: a[1] for m, a in models}, flush=True)

os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
for name, a in models:
    if a[1] is None:
        print(f"MODEL {name}: checkpoint NOT FOUND, skipped", flush=True)
        continue
    t0 = time.time()
    r = sh([sys.executable, "-u", "experiments/train_unified.py", "--eval-only", *a, "--tag", name, "--speedie-dir", SPD,
            "--eval-speedie", "", "--eval-ladder", LADDER, "--n-line-eval", "-1", "--eval-sg", "", "--out", OUT], env=env)
    print(f"MODEL {name} exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
print("EVAL LADDER DONE", flush=True)
