"""
PLAN.md B2 on Kaggle (CPU, no GPU quota): render the Speedie et al. 2024 dump sg_00005 with src/data/sph_cube.py and compare with
speedie_prep's MCFOST clean cubes (kernel source speedie-prep-fast). Code from the exxa-wiggle-code dataset (MANIFEST, sha256 checked).
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time

BRANCH = "midterm-prep"
REPO = "/tmp/EXXA"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/sph_validation"
DUMP_URL = "https://zenodo.org/records/11668694/files/sg_00005?download=1"

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "sarracen", "bettermoments"])
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
print("overlaid", list(man["files"]), flush=True)

clean_dir = os.path.dirname(glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)[0])
os.chdir(PKG)
sys.path[:0] = [PKG, f"{PKG}/experiments"]
import speedie_prep as sp
dump = "/tmp/sg_00005"
sp.download_parallel(DUMP_URL, dump, workers=16)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
t0 = time.time()
sh([sys.executable, "-u", "experiments/validate_sph_speedie.py", "--dump", dump, "--clean-dir", clean_dir, "--out", OUT], env=env)
print(f"SPH VALIDATION DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
