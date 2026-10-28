#!/usr/bin/env python3
"""Check that relative links in the project's markdown documents point at files that exist.

    python tools/check_md_links.py            # print every broken link, exit 1 if any
    python tools/check_md_links.py --count    # print only the number (for before / after comparisons)

Scope: the repo-root README and the living documents of DENOISING_DIFFUSION (top-level *.md, results/{PLAN,PROGRESS,RUNS,README}.md,
results/final/*.md, models/README.md, references/ and history/ and midterm/ indexes). Archived run folders are not checked: they record what
existed on the day, and rewriting their links would rewrite history. External (http/mailto) links and in-page anchors are skipped.
"""
import glob, os, re, sys
from urllib.parse import unquote

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                       # DENOISING_DIFFUSION
ROOT = os.path.dirname(PROJECT)                       # repository root
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def documents():
    pats = [os.path.join(ROOT, "README.md"), os.path.join(PROJECT, "*.md"), os.path.join(PROJECT, "results", "PLAN.md"),
            os.path.join(PROJECT, "results", "PROGRESS.md"), os.path.join(PROJECT, "results", "RUNS.md"), os.path.join(PROJECT, "results", "README.md"),
            os.path.join(PROJECT, "results", "final", "*.md"), os.path.join(PROJECT, "models", "README.md"),
            os.path.join(PROJECT, "references", "*.md"), os.path.join(PROJECT, "history", "*.md"), os.path.join(PROJECT, "midterm", "*.md")]
    return sorted({p for pat in pats for p in glob.glob(pat)})


def broken(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for m in list(LINK.finditer(text)) + list(IMG.finditer(text)):
        target = m.group(1).strip("<>")
        if re.match(r"^(https?:|mailto:|#|data:)", target):
            continue
        rel = unquote(target.split("#")[0])
        if rel and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), rel))):
            out.append((text.count("\n", 0, m.start()) + 1, target))
    return out


if __name__ == "__main__":
    bad = [(p, ln, t) for p in documents() for ln, t in broken(p)]
    if "--count" in sys.argv:
        print(len(bad))
    else:
        for p, ln, t in bad:
            print(f"{os.path.relpath(p, ROOT)}:{ln}: {t}")
        print(f"{len(bad)} broken relative links in {len(documents())} documents")
    sys.exit(1 if bad else 0)
