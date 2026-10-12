"""
Kaggle script kernel: learned correction on 3D Wiener, leave-one-hydro-disk-out, scored with the wiggle scorecard.
See experiments/train_wiggle_correction.py for what is trained and why; PROGRESS.md 2026-10-09 for the baseline it must beat.

Code provenance: tracked files come from a clone of branch `midterm-prep`; the new files (analytic_gi.py, the scorecard, the
cube generator and the trainer) were not yet committed when this was launched, so they travel in the private dataset
`exxa-wiggle-code` with a MANIFEST.json of sha256 sums. Output is written straight to /kaggle/working/wiggle_correction
(RULES.md #1: the checkpoint of every round is already durable).
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time

MODE = "full"           # "smoke" (plumbing + speed check, ~3 min) or "full" (5 folds, hours)
BRANCH = "midterm-prep"
REPO = "/tmp/EXXA"      # not /kaggle/working: that would put the whole clone in the notebook Output
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/wiggle_correction"

def sh(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)

# 1. repo + deps
if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-msssim", "bettermoments"])
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# 2. overlay the new (uncommitted) files from the code dataset
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
print("overlaid", list(man["files"]), "| local HEAD when staged:", man["local_head"], flush=True)

# 3. data: link the two attached datasets where the scorecard expects them
DATA = f"{PKG}/self-gravitating cube and dirty cube"
os.makedirs(DATA, exist_ok=True)
synth = os.path.dirname(glob.glob("/kaggle/input/**/run_9015_00370_rt_00", recursive=True)[0])
v2 = os.path.dirname(glob.glob("/kaggle/input/**/clean_sg.fits", recursive=True)[0])
for link, target in (("sg_synth", synth), ("kinematic_data_v2", v2)):
    if not os.path.lexists(f"{DATA}/{link}"):
        os.symlink(target, f"{DATA}/{link}")
print("sg_synth ->", synth, "| kinematic_data_v2 ->", v2, flush=True)

# 4. analytic disks (deterministic, ~2 min)
os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
sh([sys.executable, "experiments/make_analytic_gi_cubes.py"], env=env)

# 5. train + score
os.makedirs(OUT, exist_ok=True)
if MODE == "smoke":
    args = ["--folds", "9074", "--hydro", "9074,9015,9025", "--rounds", "2", "--steps", "40", "--batch", "12", "--crop", "128",
            "--base", "48", "--eval-extra", "an07"]
else:
    args = ["--rounds", "30", "--steps", "300", "--batch", "12", "--crop", "128", "--base", "48"]
t0 = time.time()
sh([sys.executable, "-u", "experiments/train_wiggle_correction.py", "--out", OUT] + args, env=env)
print(f"TRAINING SCRIPT DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
