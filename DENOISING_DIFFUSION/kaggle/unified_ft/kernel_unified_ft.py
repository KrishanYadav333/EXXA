"""
Kaggle script kernel: fine-tune existing checkpoints on the unified data mix (line emission + hydro GI + analytic GI, each drawn as the
SG deconvolution task or the ALMA-matched noise-only task, fresh noise every round, 256/300/320/480/600 px), each in its OWN
input/output format, and score it on PSNR, M0/M1/M2 and the GI wiggle before and after. See experiments/train_unified.py.

GROUP selects which checkpoints this kernel handles (two kernels fit Kaggle's 2-GPU-session limit and the 12 h session cap).
Code provenance: tracked files from a clone of `midterm-prep`; the new files ride in the private dataset `exxa-wiggle-code`
(MANIFEST.json, sha256). Output goes straight to /kaggle/working/ft_<ckpt> (RULES.md #1: checkpoint written every round).
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time

GROUP = "kin"          # "kin": kin_gamma0_mae_ft, kin_gamma0_starlet_ft | "single": winner_aug_res480_seed43, sweep_winner_p10_600, sg_k3_fresh
MODE = "full"          # "smoke": 2 rounds x 40 steps, 1 holdout, two sizes
BRANCH = "midterm-prep"
REPO = "/tmp/EXXA"     # not /kaggle/working: the clone must not land in the notebook Output
PKG = f"{REPO}/DENOISING_DIFFUSION"
CKPTS = {"kin": ["kin_gamma0_mae_ft", "kin_gamma0_starlet_ft"],
         "single": ["winner_aug_res480_seed43", "sweep_winner_p10_600", "sg_k3_fresh"]}[GROUP]
LR, ROUNDS, STEPS = "1e-4", 20, 300

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-msssim", "bettermoments"])
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
print("overlaid", list(man["files"]), "| staged from", man["local_head"], flush=True)

DATA = f"{PKG}/self-gravitating cube and dirty cube"
os.makedirs(DATA, exist_ok=True)
synth = os.path.dirname(glob.glob("/kaggle/input/**/run_9015_00370_rt_00", recursive=True)[0])
v2 = os.path.dirname(glob.glob("/kaggle/input/**/clean_sg.fits", recursive=True)[0])
for link, target in (("sg_synth", synth), ("kinematic_data_v2", v2)):
    if not os.path.lexists(f"{DATA}/{link}"):
        os.symlink(target, f"{DATA}/{link}")
le = {}
for p in glob.glob("/kaggle/input/**/run_00*", recursive=True):
    if os.path.isdir(p):
        le[os.path.dirname(p)] = le.get(os.path.dirname(p), 0) + 1
LE_DIR = max(le, key=le.get)
print("sg_synth ->", synth, "| v2 ->", v2, "| line emission ->", LE_DIR, f"({le[LE_DIR]} cubes)", flush=True)

os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
sh([sys.executable, "experiments/make_analytic_gi_cubes.py"], env=env)

for name in CKPTS:
    # exact name first, then the notebook-prefixed names the kernels wrote (nb05_<name>.ckpt, nb14_<name>.pth, ...)
    hits = [h for pat in (f"{name}.ckpt", f"{name}.pth", f"*{name}.ckpt", f"*{name}.pth")
            for h in glob.glob(f"/kaggle/input/**/{pat}", recursive=True)]
    if not hits:
        print(f"CHECKPOINT {name}: NOT FOUND under /kaggle/input, skipping", flush=True)
        continue
    print(f"CHECKPOINT {name} -> {hits[0]}", flush=True)
    out = f"/kaggle/working/ft_{name}"
    os.makedirs(out, exist_ok=True)
    args = ["--init", hits[0], "--lr", LR, "--rounds", ROUNDS, "--steps", STEPS, "--before", "--out", out]
    if MODE == "smoke":
        args = ["--init", hits[0], "--lr", LR, "--rounds", 2, "--steps", 40, "--before", "--out", out, "--n-line-eval", 1,
                "--eval-sg", "an07", "--sizes", "256,600", "--before-sizes", "256", "--before-cases", "an07_ai"]
    t0 = time.time()
    try:
        sh([sys.executable, "-u", "experiments/train_unified.py"] + [str(a) for a in args], env=env)
    except subprocess.CalledProcessError as e:           # one failed checkpoint must not cost the next one
        print(f"CHECKPOINT {name} FAILED: {e}", flush=True)
    print(f"FT {name} DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
