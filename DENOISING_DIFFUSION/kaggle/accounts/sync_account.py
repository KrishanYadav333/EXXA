"""
Give a Kaggle account its OWN copy of the project's datasets, using that account's key (no website sharing needed).

  python sync_account.py data  <account> [--only slug,slug] [--dry-run]   upload line-emission-data, exxa-sg-synth-pairs, self-gravitating-v2 (~11 GB)
  python sync_account.py code  <account> [--dry-run]                      (re)build and upload exxa-wiggle-code: the flat files + MANIFEST.json (sha256)
  python sync_account.py status <account>                                 creation status of each dataset

Local sources are cloned (cp -c on APFS: no extra disk) into a staging dir; subfolders are zipped (`--dir-mode zip`), which Kaggle unpacks again, so
the layout matches the main account's copies (the kernels find data with globs on run_*/clean_sg.fits). Speedie cubes are NOT uploaded: run
speedie_prep on that account instead (CPU kernel, no GPU quota) and its kernel_sources then resolve inside the account.
Accounts and keys: see orchestrator.py / README.md. The main account is never modified by this script unless you name it.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrator as orch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # DENOISING_DIFFUSION
SG = os.path.join(ROOT, "self-gravitating cube and dirty cube")
DATASETS = {                                   # slug -> list of (local path, name inside the dataset)
    "line-emission-data": [(os.path.join(ROOT, "Line Emission Data"), "Line Emission Data")],
    "exxa-sg-synth-pairs": [(os.path.join(SG, "sg_synth"), "kaggle-sg-training-dataset")],
    "self-gravitating-v2": [(os.path.join(SG, "kinematic_data_v2"), "kinematic_data_v2")],
}
CODE_FILES = ["src/data/analytic_gi.py", "experiments/wiggle_scorecard.py", "experiments/make_analytic_gi_cubes.py",
              "experiments/train_wiggle_correction.py", "experiments/train_unified.py", "experiments/speedie_prep.py",
              "src/data/sph_cube.py", "experiments/validate_sph_speedie.py", "experiments/alma_unified.py", "experiments/pv_curve.py", "experiments/make_sph_cubes.py"]


def stage(items, tmp):
    for src, name in items:
        dst = os.path.join(tmp, name)
        r = subprocess.run(["cp", "-cR", src, dst], capture_output=True)       # APFS clone
        if r.returncode:
            shutil.copytree(src, dst)


def create_or_version(acct, slug, tmp, dry):
    meta = {"title": slug, "id": f"{acct['user']}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    json.dump(meta, open(os.path.join(tmp, "dataset-metadata.json"), "w"))
    n = sum(len(f) for _, _, f in os.walk(tmp))
    gb = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(tmp) for f in fs) / 1e9
    print(f"{meta['id']}: {n} files, {gb:.2f} GB", flush=True)
    if dry:
        return
    exists = "ref" in orch.kaggle(acct, "datasets", "list", "--mine", "-s", slug) or f"{acct['user']}/{slug}" in orch.kaggle(acct, "datasets", "list", "--mine", "-s", slug)
    cmd = ["datasets", "version", "-p", tmp, "-m", "sync", "--dir-mode", "zip"] if exists else ["datasets", "create", "-p", tmp, "--dir-mode", "zip"]
    t = time.time()
    print(orch.kaggle(acct, *cmd)[-300:], f"[{(time.time() - t) / 60:.1f} min]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "code", "status"])
    ap.add_argument("account")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    acct = orch.accounts()[a.account]
    if a.what == "status":
        for slug in list(DATASETS) + ["exxa-wiggle-code"]:
            print(slug, "->", orch.kaggle(acct, "datasets", "status", f"{acct['user']}/{slug}"))
        return
    if a.what == "data":
        for slug, items in DATASETS.items():
            if a.only and slug not in a.only.split(","):
                continue
            for src, _ in items:
                assert os.path.isdir(src), f"missing local source {src}"
            tmp = tempfile.mkdtemp(prefix=f"ds_{slug}_")
            stage(items, tmp)
            create_or_version(acct, slug, tmp, a.dry_run)
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp = tempfile.mkdtemp(prefix="ds_code_")
        man = {"files": {}, "local_head": subprocess.run(["git", "-C", ROOT, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip() + " (+ uncommitted files)"}
        for rel in CODE_FILES:
            flat = rel.replace("/", "__")
            shutil.copyfile(os.path.join(ROOT, rel), os.path.join(tmp, flat))
            man["files"][rel] = {"flat": flat, "sha256": hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()}
        json.dump(man, open(os.path.join(tmp, "MANIFEST.json"), "w"), indent=1)
        create_or_version(acct, "exxa-wiggle-code", tmp, a.dry_run)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
