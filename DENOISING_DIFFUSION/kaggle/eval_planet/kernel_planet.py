"""
PLANET CHECK (Jason Terry 2026-10-16: a possible planet in run_0002_00560, and in Young M1 R100 at 7 to 8 o'clock). Recipe B (frozen) on the three held-out
run_0002_00560 views (a Phantom simulation rendered with mcfost -turn-off_planets: the planets' own emission is off, their kinks in the gas stay) and the
held-out Young sims, 600 px. --save-pred keeps the M1 maps of clean / dirty / Wiener / model and the central +-2 km/s channels of clean / dirty / model for
run_0002_00560_rt_00/01/04 and run_young_M1_R100_Mu0.4, for the local kink analysis. Derived from kernel_final_b.py. RULES.md #14 resume applies.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
os.environ.setdefault("EXXA_T0", str(time.time()))     # RULES.md #14

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/planet"

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
find = lambda pat: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or [None])[0]
FILES = ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth"]
paths = [p for p in (find(f) for f in FILES) if p]     # every seed this account can see; the plain row below is the same ensemble
VARIANT = "B"
FILES = {"A": ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth"],
         "B": ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth", "unified_r8_m1_r6.pth"],
         "C": ["unified_r8_m1_r6.pth"]}[VARIANT]
find = lambda pat: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or [None])[0]
paths = [find(f) for f in FILES]
assert all(paths), f"missing checkpoints: {dict(zip(FILES, paths))}"
sph = glob.glob("/kaggle/input/**/run_young_*/view.json", recursive=True)
SAVE = "run_0002_00560_rt_00,run_0002_00560_rt_01,run_0002_00560_rt_04,run_young_M1_R100_Mu0.4"
common = ["--eval-ckpt", ",".join(paths), "--n-line-eval", "3", "--eval-sg", "", "--sizes", "600", "--gate", "40,120,0.25",
          "--eval-speedie", "", "--eval-ladder", "", "--wiener2", "--infer-beam-px", "10", "--infer-mode", "full",
          "--save-pred", f"{OUT}/pred", "--save-pred-cases", SAVE] + (["--sph-dir", os.path.dirname(os.path.dirname(sph[0])), "--eval-sph"] if sph else [])
assert sph, "Young sims not attached"
assert "c0, h = cS.shape[0] // 2, 40" in open(f"{PKG}/experiments/train_unified.py").read(), "stale code overlay: train_unified.py lacks --save-pred (+-4 km/s)"
models = [("planet_B", common)]
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
print("PLANET EVAL DONE", flush=True)
