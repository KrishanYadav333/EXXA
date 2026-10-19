"""
ALMA S3 / S4 (PLAN.md 0h, Oct 20 to 21): the final exoALMA setting (512 px field, beam to 10 px) on four discs, with other model recipes (S3: do the
recipes agree on real data?) or another zoom (S4: is the result robust to the pixel scale, +-25 percent?). Recipe B's own runs are the final ones
(alma-exoalma-final-p0 / p1 / p3); the comparison is local (experiments/alma_agree.py). Derived from kernel_alma_all13co.py. RULES.md #14 resume per
variant and disc.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time, urllib.request
os.environ.setdefault("EXXA_T0", str(time.time()))

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/alma_var"
B = "unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth"   # recipe B (results/final/RECIPES.md)
A = "unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth"                          # recipe A (stored alternative)
R6 = "unified_r6_k15_crop256.pth"                                                                                   # R6 seed 31 (documented kinematics alternative)
VARIANTS = [("A", A, ["--target-beam-px", "10"]), ("R6", R6, ["--target-beam-px", "10"])]   # the launcher replaces this line
CKPT = ",".join(sorted({c for _, ck, _ in VARIANTS for c in ck.split(",")}))
# disk, line, inclination (deg), distance (pc): exoALMA I (Teague et al. 2025) values, for the S6 rotation-curve residual only
# exoALMA I (Teague et al. 2025, arXiv:2504.18688): distance d from Table 1, |inclination| from Table 2 (discminer fits of 12CO); crop 512 px = 12.8 arcsec
# disk, |i| (deg), PA (deg, discminer, Table 2), distance (pc, Table 1)
CUBES = [(disk, "13CO", inc, dist, pa) for disk, inc, pa, dist in [
    ("MWC_758", 19.4, 240.3, 156), ("DM_Tau", 38.7, 335.7, 144), ("AA_Tau", 58.7, 272.7, 135), ("LkCa_15", 50.3, 62.1, 157), ("HD_34282", 58.3, 117.4, 309),
    ("CQ_Tau", 36.3, 235.1, 149), ("SY_Cha", 52.4, 346.2, 182), ("PDS_66", 31.9, 189.0, 98), ("HD_135344B", 16.1, 242.9, 135), ("HD_143006", 16.9, 168.9, 167),
    ("J1604", 6.0, 258.1, 145), ("J1615", 46.5, 325.4, 156), ("V4046_Sgr", 34.1, 255.7, 72), ("J1842", 39.4, 205.9, 151), ("J1852", 32.7, 117.1, 147)]]
PART, NPART = 0, 1                                   # the launcher sets these to split the discs over several kernels (CUBES[PART::NPART])
CUBES = CUBES[PART::NPART]
CUBES = [c for c in CUBES if c[0] in ("MWC_758", "LkCa_15", "HD_135344B", "J1615")]   # four discs: i 19 / 50 / 16 / 47 deg, SNR 11 to 15
FIELD = ["--crop", "512"]                            # field of view (PLAN.md 0h: 512 px kept flux; set from the Oct 18 zoom test)
INJECT_FOR = {"MWC_758"}   # S5 injection-recovery on three discs spanning inclination 16 / 39 / 50 deg (PLAN.md 0f); MWC 758 done in alma-exoalma-s5
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
assert "target-beam-px" in open(f"{PKG}/experiments/alma_unified.py").read(), "stale code overlay: alma_unified.py lacks --target-beam-px"
path = {c: (glob.glob(f"/kaggle/input/**/{c}", recursive=True) or [None])[0] for c in CKPT.split(",")}
ckpts = list(path.values())
spd = glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)      # S5 truth (Speedie, test-only), from the speedie-prep-fast Output
assert None not in ckpts, f"checkpoint not attached: {CKPT}"
os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
files = None
for (tag, ck, extra), (disk, line, incl, dist, pa) in [(v, c) for c in CUBES for v in VARIANTS]:
    name = f"{disk}_{line}_fiducial.image.fits"
    out = f"{OUT}/{tag}/{disk}_{line}"
    prev = glob.glob(f"/kaggle/input/**/alma_var/{tag}/{disk}_{line}/report.json", recursive=True)
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
    if not (os.path.exists(dst) and os.path.getsize(dst) == files[name]["filesize"]):      # downloaded once per disc, reused by its variants
        sh(["curl", "-sL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", dst, f"{DATAVERSE}/api/access/datafile/{files[name]['id']}"], check=True)
    assert os.path.getsize(dst) == files[name]["filesize"], f"{name}: truncated download"
    print(f"{name}: {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    print(f"{name}: variant {tag}: {ck} {' '.join(extra)}", flush=True)
    r = sh([sys.executable, "-u", "experiments/alma_unified.py", "--cube", dst, "--ckpt", ",".join(path[c] for c in ck.split(",")), "--out", out,
            "--incl", str(incl), "--dist", str(dist), "--pa", str(pa)] + FIELD + extra + (["--inject-dir", os.path.dirname(spd[0])] if spd and disk in INJECT_FOR else []), env=env)
    print(f"{name} {tag}: exit {r.returncode}", flush=True)
    if tag == VARIANTS[-1][0]:
        os.remove(dst)
print("ALMA KERNEL DONE", flush=True)
