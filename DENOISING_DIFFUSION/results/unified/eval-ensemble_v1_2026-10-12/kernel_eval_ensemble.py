"""
Kaggle kernel: ensemble evaluation (PLAN.md: does averaging seeds / flips raise the wiggle at SNR 12-24?). Each GROUP is a list of checkpoints averaged
by train_unified.py --eval-ckpt a,b[,c] (same K / arm / base), optionally with --tta (4 sky-plane flips). Scored on the 5 line holdouts, 9074, SG v2,
an07 / an08 at the 5 sizes, Speedie GI / twin r0.5 and r1.5, and the SNR ladder 3/6/12/24. Inputs: speedie-prep-fast and the training kernels' Outputs
(R4 seeds on deepsceneloc must be shared with the main account).
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/eval_ensemble"
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
GROUPS = [("ens_R4s0_R4s1", ["unified_r4_k15_s0.pth", "unified_r4_k15_s1.pth"], False, "full"),
          ("ens_R4s0_R4s1_tta", ["unified_r4_k15_s0.pth", "unified_r4_k15_s1.pth"], True, "speedie"),
          ("ens_R4s0_R4s1_R3b_tta", ["unified_r4_k15_s0.pth", "unified_r4_k15_s1.pth", "unified_real_k15.pth"], True, "speedie")]
models = []
for n, files, tta, scope in GROUPS:
    paths = [find(f) for f in files]
    models.append((n, None if None in paths else ["--eval-ckpt", ",".join(paths)] + (["--tta"] if tta else []) +
                   (["--eval-speedie", "GI_r0.5,Kep_r0.5,GI_r1.5,Kep_r1.5", "--eval-ladder", LADDER] if scope == "full" else
                    ["--eval-speedie", "GI_r0.5,Kep_r0.5,GI_r1.5,Kep_r1.5", "--eval-ladder", LADDER, "--n-line-eval", "-1", "--eval-sg", ""])))
print("speedie ->", SPD, "| groups:", {m: (a[1] if a else "MISSING") for m, a in models}, flush=True)

os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
for name, a in models:
    if a is None:
        print(f"MODEL {name}: checkpoint NOT FOUND, skipped", flush=True)
        continue
    t0 = time.time()
    r = sh([sys.executable, "-u", "experiments/train_unified.py", "--eval-only", *a, "--tag", name, "--speedie-dir", SPD, "--out", OUT], env=env)
    print(f"MODEL {name} exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
print("EVAL ENSEMBLE DONE", flush=True)
