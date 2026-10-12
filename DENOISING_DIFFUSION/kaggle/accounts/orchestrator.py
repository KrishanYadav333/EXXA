"""
Run kernels on whichever of several Kaggle accounts has a free GPU slot and quota, then collect logs and outputs.

Accounts are discovered from key files, never passed on a command line or printed:
  main    ~/.kaggle/kaggle.json
  <name>  ~/.kaggle-<name>/kaggle.json   (chmod 600)
State (no secrets) is kept in kaggle/accounts/state.json: which job went to which account, status, where outputs were collected.

  python orchestrator.py accounts                          who is configured (usernames only)
  python orchestrator.py submit DIR --slug my-run [--account NAME] [--dry-run]
                                                           DIR holds kernel_*.py + kernel-metadata.json (a normal kaggle/<job>/ folder);
                                                           id is rewritten to <account user>/<slug>; tries accounts in order until one accepts
  python orchestrator.py status                            polls every job this script submitted
  python orchestrator.py collect [--out results/...]       downloads output + log of finished jobs

Limits it respects: 2 concurrent GPU sessions per account ("Maximum batch GPU session count") and the weekly quota ("Maximum weekly GPU
quota"; that account is parked until the next Saturday 05:30 IST). Kernels that read another kernel's output (kernel_sources) are refused for
non-owner accounts: that input does not cross accounts, so the owner prefix of dataset_sources / kernel_sources is rewritten to the
target account (it must hold its own copy: sync_account.py; speedie-prep-fast run there).
"""
import argparse, datetime as dt, glob, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def accounts():
    out = {}
    home = os.path.expanduser("~")
    cands = [("main", os.path.join(home, ".kaggle", "kaggle.json"))] + \
            [(os.path.basename(os.path.dirname(p))[len(".kaggle-"):], p) for p in sorted(glob.glob(os.path.join(home, ".kaggle-*", "kaggle.json")))]
    for name, p in cands:
        if os.path.isfile(p):
            if oct(os.stat(p).st_mode & 0o777) != "0o600":
                print(f"skip {name}: {p} must be chmod 600", file=sys.stderr); continue
            out[name] = {"user": json.load(open(p))["username"], "cfg": os.path.dirname(p)}
    return out


def env_for(acct):
    key = json.load(open(os.path.join(acct["cfg"], "kaggle.json")))
    return dict(os.environ, KAGGLE_CONFIG_DIR=acct["cfg"], KAGGLE_USERNAME=key["username"], KAGGLE_KEY=key["key"])


def kaggle(acct, *args):
    r = subprocess.run(["kaggle", *args], env=dict(env_for(acct), PYTHONWARNINGS="ignore"), capture_output=True, text=True)
    out = r.stdout + "\n" + r.stderr
    return "\n".join(l for l in out.splitlines() if l.strip() and "warn" not in l.lower() and "urllib3" not in l.lower()).strip()


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"jobs": {}, "parked": {}}


def save(s):
    json.dump(s, open(STATE, "w"), indent=1)


def next_reset(now=None):
    now = now or dt.datetime.now(IST)
    d = now + dt.timedelta(days=(5 - now.weekday()) % 7)          # Saturday
    t = d.replace(hour=5, minute=30, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=7)


def cmd_submit(a):
    accs, st = accounts(), load()
    meta = json.load(open(os.path.join(a.dir, "kernel-metadata.json")))
    order = [a.account] if a.account else list(accs)
    now = dt.datetime.now(IST)
    for name in order:
        acct = accs[name]
        until = st["parked"].get(name)
        if until and dt.datetime.fromisoformat(until) > now:
            print(f"{name}: parked until {until}"); continue
        # each account has its OWN copy of the datasets (sync_account.py) and its own speedie-prep-fast run: point the sources at it
        own = lambda refs: [f"{acct['user']}/{r.split('/', 1)[1]}" for r in refs]
        # a kernel's output CAN be attached from another account when that kernel is public or shared with the target (Krishan's nb14 resume across
        # accounts worked this way): --shared-kernel-sources keeps kernel_sources as they are instead of pointing them at the account's own run
        ks = meta.get("kernel_sources", []) if a.shared_kernel_sources else own(meta.get("kernel_sources", []))
        ds = meta.get("dataset_sources", []) if a.shared_datasets else own(meta.get("dataset_sources", []))
        m = dict(meta, id=f"{acct['user']}/{a.slug}", title=a.slug, dataset_sources=ds, kernel_sources=ks)
        if a.public:
            m["is_private"] = "false"
        tmp = tempfile.mkdtemp(prefix="kjob_")
        for f in os.listdir(a.dir):
            if f != "kernel-metadata.json":
                shutil.copy(os.path.join(a.dir, f), tmp)
        json.dump(m, open(os.path.join(tmp, "kernel-metadata.json"), "w"), indent=2)
        if a.dry_run:
            print(f"[dry-run] would push {m['id']} from {a.dir} (datasets {m.get('dataset_sources')})"); shutil.rmtree(tmp); return
        out = kaggle(acct, "kernels", "push", "-p", tmp)
        shutil.rmtree(tmp)
        print(f"{name}: {out if 'successfully' not in out else out.splitlines()[-1]}")      # full reply on failure: the last line alone hid the cause
        if "successfully pushed" in out:
            st["jobs"][m["id"]] = {"account": name, "submitted": now.isoformat(timespec="seconds"), "status": "submitted", "collected": None}
            save(st); return
        if "weekly GPU quota" in out:
            st["parked"][name] = next_reset().isoformat(timespec="minutes"); save(st)
    print("no account accepted the job (slots busy or quota used); try again later")


SHARED = {"datasets": ["line-emission-data", "exxa-sg-synth-pairs", "self-gravitating-v2", "exxa-wiggle-code"], "kernels": ["speedie-prep-fast", "unified-wiener", "unified-wiener-real-k15", "unified-wiener-real-sgv2"]}


def cmd_check(a):
    """Can each non-main account SEE what the main account is supposed to have shared with it? (website sharing is manual; this verifies it)"""
    accs = accounts(); owner = accs["main"]["user"] if "main" in accs else "krishanyadav333"
    for name, acct in accs.items():
        if name == "main":
            continue
        for kind, slugs in SHARED.items():
            for slug in slugs:
                raw = kaggle(acct, kind, "files" if kind == "datasets" else "status", f"{owner}/{slug}")
                lines = [l for l in raw.splitlines() if l.strip() and "warn" not in l.lower() and "urllib3" not in l.lower()]
                out = "\n".join(lines)
                good = ("has status" in out) if kind == "kernels" else any(l.startswith("name") or l.startswith("----") for l in lines)   # success markers; error words also appear in file sizes
                bad = not good
                print(f"{name:10s} {kind[:-1]:7s} {owner}/{slug:24s} {'NOT visible: ' + (lines[-1][:80] if lines else '(no output)') if bad or not lines else 'visible'}")


def cmd_status(a):
    accs, st = accounts(), load()
    for ref, j in st["jobs"].items():
        if j["status"] in ("complete", "error", "cancelled"):
            print(f"{ref:60s} {j['account']:8s} {j['status']}"); continue
        out = kaggle(accs[j["account"]], "kernels", "status", ref)
        s = out.split("KernelWorkerStatus.")[-1].split()[0].strip('"').lower() if "KernelWorkerStatus." in out else out[:60]
        j["status"] = {"complete": "complete", "error": "error", "cancel_acknowledged": "cancelled", "cancelled": "cancelled"}.get(s, s)
        print(f"{ref:60s} {j['account']:8s} {j['status']}")
    save(st)
    for n, until in st["parked"].items():
        print(f"parked: {n} until {until}")


def cmd_collect(a):
    accs, st = accounts(), load()
    for ref, j in st["jobs"].items():
        if j["status"] not in ("complete", "error") or j["collected"]:
            continue
        dest = os.path.join(a.out, ref.replace("/", "__"))
        os.makedirs(dest, exist_ok=True)
        out = kaggle(accs[j["account"]], "kernels", "output", ref, "-p", dest)
        j["collected"] = dest
        print(f"{ref}: {j['status']} -> {dest}  ({len(os.listdir(dest))} files)")
    save(st)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("accounts")
    s = sp.add_parser("submit"); s.add_argument("dir"); s.add_argument("--slug", required=True); s.add_argument("--account"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--shared-datasets", action="store_true", help="keep dataset_sources: the datasets are privately shared with the target account on kaggle.com")
    s.add_argument("--shared-kernel-sources", action="store_true", help="keep kernel_sources (the source kernel is public or shared with the target account)")
    s.add_argument("--public", action="store_true", help="push the kernel public (e.g. speedie-prep-fast, derived from CC-BY data) so other accounts can attach its output")
    sp.add_parser("status")
    sp.add_parser("check")
    c = sp.add_parser("collect"); c.add_argument("--out", default=os.path.join(HERE, "collected"))
    a = ap.parse_args()
    if a.cmd == "accounts":
        for n, v in accounts().items():
            print(f"{n:10s} {v['user']}")
    else:
        {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
