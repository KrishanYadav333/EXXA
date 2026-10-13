"""
One place for every notebook and its results (RULES.md #12: keep every checkpoint; #10: the notebook as it ran).

For every project kernel on every configured Kaggle account (orchestrator.accounts()):
  archive/<account>/<slug>/code/         the kernel source exactly as pushed (`kaggle kernels pull -m`, with its metadata)
  archive/<account>/<slug>/output/       its output files: json, png, logs, csv, small fits ... (repo clones inside old outputs are skipped)
  archive/<account>/<slug>/checkpoints/  *.pth / *.ckpt, always kept
  archive/<account>/<slug>/meta.json     account, URL, last run, status, what was fetched, what was left on Kaggle (too large) with its URL
and writes archive/INDEX.md plus results/KERNELS.md (the tracked copy of the index: slug, account, link, status, last run, purpose from RUNS.md).
Idempotent: files already present with the same size are skipped, so it can run after every batch of jobs.

  python kaggle/accounts/archive_all.py [--out archive] [--since 2026-10-01] [--max-mb 100] [--only slug-substring] [--dry-run]

The archive/ folder is git-ignored (checkpoints are ~120 MB each); KERNELS.md is the tracked map to it.
"""
import argparse, datetime as dt, json, os, re, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orchestrator as orch

ROOT = os.path.dirname(os.path.dirname(HERE))                       # DENOISING_DIFFUSION
INCLUDE = re.compile(r"^(unified|eval|alma|sph|speedie|wiggle|ram-leak|14-|05-|13-|16-|12-|09-|08-|07-|06-)")
SKIP_PATH = re.compile(r"(^|/)(EXXA|\.git|__pycache__|wvenv|venv)(/|$)")
CKPT = (".pth", ".ckpt", ".pt")


def list_kernels(acct):
    out = orch.kaggle(acct, "kernels", "list", "--mine", "--csv", "--page-size", "200")
    rows = []
    for l in out.splitlines()[1:]:
        p = l.split(",")
        if len(p) >= 4 and "/" in p[0]:
            rows.append({"ref": p[0], "title": p[1], "last_run": p[3]})
    return rows


def session_files(acct, ref):
    """All output files of the latest session, following pagination: [(name, url)], plus the log text."""
    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    key = json.load(open(os.path.join(acct["cfg"], "kaggle.json")))
    files, log, token = [], "", None
    with KaggleClient(env=None, verbose=False, username=key["username"], password=key["key"]) as c:
        while True:
            r = ApiListKernelSessionOutputRequest(); r.user_name, r.kernel_slug = ref.split("/")
            if token:
                r.page_token = token
            resp = c.kernels.kernels_api_client.list_kernel_session_output(r)
            files += [(f.file_name, f.url) for f in resp.files]
            log = log or (resp.log or "")
            token = getattr(resp, "nextPageToken", None) or getattr(resp, "next_page_token", None)
            if not token:
                break
    return files, log


def fetch(url, dst, max_bytes, force=False):
    """Download unless larger than max_bytes (returns 'skipped-large' with the size) or already present with the same size."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EXXA archive)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        n = int(r.headers.get("Content-Length") or -1)
        if os.path.exists(dst) and n >= 0 and os.path.getsize(dst) == n:
            return "have", n
        if max_bytes is not None and n > max_bytes:
            return "large", n
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dst)
        return "got", os.path.getsize(dst)


def purposes():
    """slug -> first RUNS.md row that mentions it: the purpose column, shortened."""
    out, path = {}, os.path.join(ROOT, "results", "RUNS.md")
    if os.path.exists(path):
        for l in open(path).read().splitlines():
            if l.startswith("|"):
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                for m in re.findall(r"`([^`]+)`", cells[0]):
                    slug = m.split("/")[-1].replace("(deepsceneloc)", "").strip()
                    out.setdefault(slug, " ".join(cells[3:5])[:230] if len(cells) > 4 else cells[-1][:230])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "archive"))
    ap.add_argument("--since", default="2026-10-01")
    ap.add_argument("--max-mb", type=float, default=100.0, help="non-checkpoint files above this stay on Kaggle (listed with their URL)")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    purp, index = purposes(), []
    for name, acct in orch.accounts().items():
        for k in list_kernels(acct):
            slug = k["ref"].split("/")[1]
            if not INCLUDE.match(slug) or k["last_run"] < a.since or (a.only and a.only not in slug):
                continue
            d = os.path.join(a.out, name, slug)
            meta = {"account": name, "ref": k["ref"], "url": f"https://www.kaggle.com/code/{k['ref']}", "last_run": k["last_run"], "fetched": [], "left_on_kaggle": []}
            print(f"{name:8s} {k['ref']:55s} {k['last_run']}", flush=True)
            if a.dry_run:
                index.append((name, slug, meta, "?")); continue
            try:
                orch.kaggle(acct, "kernels", "pull", k["ref"], "-p", os.path.join(d, "code"), "-m")
            except Exception as e:
                meta["pull_error"] = str(e)[:120]
            try:
                st = orch.kaggle(acct, "kernels", "status", k["ref"])
                meta["status"] = st.split("KernelWorkerStatus.")[-1].split()[0].strip('"') if "KernelWorkerStatus." in st else st[:40]
                files, log = session_files(acct, k["ref"])
            except Exception as e:
                meta["status"] = meta.get("status", "?"); meta["error"] = f"{type(e).__name__}: {str(e)[:100]}"; files, log = [], ""
            if log:
                os.makedirs(os.path.join(d, "output"), exist_ok=True); open(os.path.join(d, "output", "kernel_log.json"), "w").write(log)
            for fname, url in files:
                if SKIP_PATH.search(fname):
                    continue
                is_ck = fname.lower().endswith(CKPT)
                dst = os.path.join(d, "checkpoints" if is_ck else "output", fname.replace("/", "__"))
                try:
                    res, n = fetch(url, dst, None if is_ck else a.max_mb * 1e6)
                except Exception as e:
                    meta["left_on_kaggle"].append({"file": fname, "reason": f"{type(e).__name__}"}); continue
                if res == "large":
                    meta["left_on_kaggle"].append({"file": fname, "mb": round(n / 1e6), "reason": "larger than --max-mb"})
                else:
                    meta["fetched"].append({"file": fname, "mb": round(n / 1e6, 1), "ckpt": is_ck})
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            index.append((name, slug, meta, purp.get(slug, "")))
    # ---- index
    L = ["# Every notebook and its results, in one place", "",
         f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `kaggle/accounts/archive_all.py` (kernels run since {a.since}, all configured accounts). Files live in `archive/<account>/<slug>/` "
         "(`code/` as pushed, `output/`, `checkpoints/`, `meta.json`; the folder is git-ignored). Run histories, numbers and what each run means: `results/RUNS.md`, `results/PROGRESS.md`.", "",
         "| account | notebook | status | last run | files | checkpoints | left on Kaggle (large) | purpose (RUNS.md) |", "|---|---|---|---|---|---|---|---|"]
    for name, slug, meta, pu in sorted(index, key=lambda x: (x[0], x[2]["last_run"]), reverse=False):
        nck = sum(1 for f in meta["fetched"] if f["ckpt"]); nf = len(meta["fetched"]) - nck
        big = ", ".join(f"{x['file'].split('/')[-1]} ({x.get('mb', '?')} MB)" for x in meta["left_on_kaggle"][:3]) + (" ..." if len(meta["left_on_kaggle"]) > 3 else "")
        L.append(f"| {name} | [{slug}]({meta['url']}) | {meta.get('status', '?')} | {meta['last_run'][:16]} | {nf} | {nck} | {big} | {pu} |")
    txt = "\n".join(L) + "\n"
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "INDEX.md"), "w").write(txt)
        open(os.path.join(ROOT, "results", "KERNELS.md"), "w").write(txt)
    print(f"\n{len(index)} kernels", "(dry run)" if a.dry_run else f"-> {a.out}/INDEX.md and results/KERNELS.md")


if __name__ == "__main__":
    main()
