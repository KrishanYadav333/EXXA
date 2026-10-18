"""
ALMA_PLAN.md on Kaggle: fetch exoALMA fiducial cubes (Harvard Dataverse doi:10.7910/DVN/CFHWNH, `.image.fits` only) and run the unified denoiser on them
with experiments/alma_unified.py (INFERENCE ONLY: exoALMA is never trained on). One report, denoised cube and moment maps per cube in the Output.
RULES.md #14: clock from the first line; each cube is a unit of work, a finished cube is skipped when the kernel is rerun with its own output attached.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time, urllib.request
os.environ.setdefault("EXXA_T0", str(time.time()))

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/alma"
CKPT = "unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth"   # recipe B (results/final/RECIPES.md)
# disk, line, inclination (deg), distance (pc): exoALMA I (Teague et al. 2025) values, for the S6 rotation-curve residual only
CUBES = [("MWC_758", "13CO", 21.0, 155.9, 512)]   # + crop (px at 25 mas): 256 px = 6.4 arcsec = ~1000 au, the disc fills it; the full 1024 px field
                                                  # (25.6 arcsec) is >95 % empty sky, which made the S6 geometry fit and spine run on noise (v1, 2026-10-13)
DATAVERSE, DOI = "https://dataverse.harvard.edu", "doi:10.7910/DVN/CFHWNH"
BUDGET_H = 11.0

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
ckpts = [(glob.glob(f"/kaggle/input/**/{c}", recursive=True) or [None])[0] for c in CKPT.split(",")]
spd = glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)      # S5 truth (Speedie, test-only), from the speedie-prep-fast Output
assert None not in ckpts, f"checkpoint not attached: {CKPT}"
os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
files = None
for disk, line, incl, dist, crop in CUBES:
    name = f"{disk}_{line}_fiducial.image.fits"
    out = f"{OUT}/{disk}_{line}"
    prev = glob.glob(f"/kaggle/input/**/alma/{disk}_{line}/report.json", recursive=True)
    if prev:                                         # RULES.md #14: done in a previous session of this kernel
        shutil.copytree(os.path.dirname(prev[0]), out, dirs_exist_ok=True); print(f"{name}: done in a previous session, kept", flush=True); continue
    if time.time() > float(os.environ["EXXA_T0"]) + BUDGET_H * 3600 - 1800:
        print(f"DEADLINE: {name} not started; RESUME_NEEDED", flush=True); break
    if files is None:
        req = urllib.request.Request(f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}",
                                     headers={"User-Agent": "Mozilla/5.0 (EXXA GSoC research; inference on public exoALMA data)"})
        with urllib.request.urlopen(req, timeout=60) as r:          # Dataverse answers 403 to the default Python-urllib agent (2026-10-13)
            files = {f["dataFile"]["filename"]: f["dataFile"] for f in json.load(r)["data"]["latestVersion"]["files"]}
    assert name in files, f"{name} not in the release; e.g. {sorted(files)[:5]}"
    dst = f"/kaggle/temp/{name}"; os.makedirs("/kaggle/temp", exist_ok=True)
    t0 = time.time()
    sh(["curl", "-sL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", dst, f"{DATAVERSE}/api/access/datafile/{files[name]['id']}"], check=True)
    assert os.path.getsize(dst) == files[name]["filesize"], f"{name}: truncated download"
    print(f"{name}: {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    r = sh([sys.executable, "-u", "experiments/alma_unified.py", "--cube", dst, "--ckpt", ",".join(ckpts), "--out", out,
            "--incl", str(incl), "--dist", str(dist), "--crop", str(crop), "--auto-fov", "--cont-sub"] + (["--inject-dir", os.path.dirname(spd[0])] if spd else []), env=env)
    print(f"{name}: exit {r.returncode}", flush=True)
    os.remove(dst)
print("ALMA KERNEL DONE", flush=True)
