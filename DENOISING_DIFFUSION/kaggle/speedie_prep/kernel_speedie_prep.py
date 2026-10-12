"""
Kaggle CPU kernel (no GPU session used): fetch the Speedie et al. 2024 simulated GI disk and its Keplerian twin from Zenodo (11668694,
CC-BY-4.0), crop the central 1024 px, build the beam-convolved truth, measure the real noise statistics. Output: {GI,Kep}_r{0.5,1.5}_{clean,dirty}.fits
and speedie_report.json. Later kernels attach this kernel's Output (`kernel_sources`). See experiments/speedie_prep.py.
"""
import os, sys, glob, json, shutil, subprocess, hashlib

OUT, CACHE = "/kaggle/working/speedie", "/kaggle/temp/speedie_cache"
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
info = man["files"]["experiments/speedie_prep.py"]
assert hashlib.sha256(open(f"{code}/{info['flat']}", "rb").read()).hexdigest() == info["sha256"], "checksum mismatch"
shutil.copy(f"{code}/{info['flat']}", "/tmp/speedie_prep.py")
print("staged from", man["local_head"], flush=True)
subprocess.run([sys.executable, "-u", "/tmp/speedie_prep.py", "--out", OUT, "--cache", CACHE, "--crop", "1024"], check=True)
print(open(f"{OUT}/speedie_report.json").read())
