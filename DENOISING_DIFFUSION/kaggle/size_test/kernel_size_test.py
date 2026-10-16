"""
Kaggle kernel (PLAN.md 0b item 7, 2026-10-15): does running the network where the beam spans ~10 px fix the wiggle loss at 480 / 600 px?
The frozen model's seeds this account can see (R4n-long 3 / 11 / 23, averaged, gate on) scored on the 5 line holdouts, 9074 (+ _ai) and the 5 held-out Young simulations at
480 and 600 px, with train_unified.py --infer-beam-px B --infer-mode corr|full. `size_plain` is the same ensemble without the resampling, so every comparison is inside this run.
RULES.md #14 resume applies.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
os.environ.setdefault("EXXA_T0", str(time.time()))     # RULES.md #14

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/size_test"

def resume_inputs(tags, out):
    """RULES.md #14: if a previous session of THIS run is attached as input (its own kernel output), copy its checkpoint / eval JSON / status
    for these tags into `out` and return {tag: checkpoint path or None}. Only files named for our own tags are taken, never another run's."""
    found = {}
    for tag in tags:
        st = [p for p in glob.glob(f"/kaggle/input/**/status_{tag}.json", recursive=True)]
        if not st:
            continue
        src = os.path.dirname(st[0])
        os.makedirs(out, exist_ok=True)
        for name in (f"status_{tag}.json", f"eval_{tag}.json", f"unified_{tag}.pth"):
            if os.path.exists(f"{src}/{name}"):
                shutil.copy(f"{src}/{name}", f"{out}/{name}")
        found[tag] = f"{out}/unified_{tag}.pth" if os.path.exists(f"{out}/unified_{tag}.pth") else None
        print(f"RESUME {tag}: previous session found in {src}: {json.load(open(st[0]))}", flush=True)
    return found


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
DATA = f"{PKG}/self-gravitating cube and dirty cube"            # hydro + SG v2 + analytic, as kernel_unified.py does
os.makedirs(DATA, exist_ok=True)
synth = os.path.dirname(glob.glob("/kaggle/input/**/run_9015_00370_rt_00", recursive=True)[0])
v2 = os.path.dirname(glob.glob("/kaggle/input/**/clean_sg.fits", recursive=True)[0])
for link, target in (("sg_synth", synth), ("kinematic_data_v2", v2)):
    if not os.path.lexists(f"{DATA}/{link}"):
        os.symlink(target, f"{DATA}/{link}")
_env0 = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
sh([sys.executable, "experiments/make_analytic_gi_cubes.py"], env=_env0, cwd=PKG, check=True)
find = lambda pat: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or [None])[0]
FILES = ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth"]
paths = [p for p in (find(f) for f in FILES) if p]     # every seed this account can see; the plain row below is the same ensemble
sph = glob.glob("/kaggle/input/**/run_young_*/view.json", recursive=True)
common = ["--eval-ckpt", ",".join(paths), "--eval-sg", "9074", "--sizes", "480,600", "--gate", "40,120,0.25"] + \
         (["--sph-dir", os.path.dirname(os.path.dirname(sph[0])), "--eval-sph"] if sph else [])
VARIANTS = [("size_plain", []),
            ("size_b10_corr", ["--infer-beam-px", "10", "--infer-mode", "corr"]),
            ("size_b10_full", ["--infer-beam-px", "10", "--infer-mode", "full"]),
            ("size_b8_corr", ["--infer-beam-px", "8", "--infer-mode", "corr"])]
models = [(n, common + v if paths else None) for n, v in VARIANTS]
print("checkpoints:", paths, "| young sims:", bool(sph), flush=True)

os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
prev = resume_inputs([n for n, _ in models], OUT)
for name, a in models:
    stf = f"{OUT}/status_{name}.json"
    if os.path.exists(stf) and json.load(open(stf)).get("eval_complete"):
        print(f"MODEL {name}: already complete in a previous session, skipped", flush=True)
        continue
    if a is None:
        print(f"MODEL {name}: checkpoint NOT FOUND, skipped", flush=True)
        continue
    t0 = time.time()
    r = sh([sys.executable, "-u", "experiments/train_unified.py", "--eval-only", *a, "--tag", name, "--out", OUT], env=env)
    print(f"MODEL {name} exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
print("SIZE TEST DONE", flush=True)
