"""
PLAN.md B3 on Kaggle (CPU, no GPU quota): download 22 Young et al. 2026 Phantom GI runs (Zenodo 22045440), render one view each with
experiments/make_sph_cubes.py (16 train / 6 held-out simulations), write them to the Output as run_young_*/ folders for train_unified.py --sph-dir.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/sph_cubes"

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "sarracen"])
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
print("overlaid", list(man["files"]), flush=True)
os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
t0 = time.time()
sh([sys.executable, "-u", "experiments/make_sph_cubes.py", "--out", OUT, "--workers", "24"], env=env)
print(f"SPH CUBES KERNEL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
