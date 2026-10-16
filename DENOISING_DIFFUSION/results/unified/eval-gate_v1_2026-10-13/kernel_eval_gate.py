"""
Kaggle kernel (PLAN.md, do-no-harm gate): score  a * model + (1 - a) * Wiener  for a = 0.25 / 0.5 / 0.75 next to the plain model and the plain Wiener, on cases from
peak SNR 12 to 135: SNR-ladder 12 and 24, Speedie r0.5 (~36), SG v2 (~63), Speedie r1.5 (~135, where the plain model damages M0). Each case also logs the blind
`snr_est` a gate would switch on. Model: R4n-long seeds 3 + 11 averaged. RULES.md #14 resume applies.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
os.environ.setdefault("EXXA_T0", str(time.time()))     # RULES.md #14

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/eval_gate"
LADDER = "3,6,12,24"

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
SPD = os.path.dirname(glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)[0])
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
FILES = ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth"]
paths = [find(f) for f in FILES]
models = [("gate_R4n_s3_s11", None if None in paths else ["--eval-ckpt", ",".join(paths), "--eval-speedie", "GI_r0.5,GI_r1.5,Kep_r1.5",
          "--eval-ladder", "12,24", "--ladder-base", "GI_r0.5", "--n-line-eval", "-1", "--eval-sg", "sgv2", "--sizes", "600", "--gate-sweep", "0.25,0.5,0.75"])]
print("speedie ->", SPD, "| groups:", {m: (a[1] if a else "MISSING") for m, a in models}, flush=True)

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
    r = sh([sys.executable, "-u", "experiments/train_unified.py", "--eval-only", *a, "--tag", name, "--speedie-dir", SPD, "--out", OUT], env=env)
    print(f"MODEL {name} exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
print("EVAL GATE DONE", flush=True)
