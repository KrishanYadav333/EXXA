#!/usr/bin/env python
"""
src/evaluation/recover_version.py
=================================
Recover a Kaggle run's outputs from the commit its GitHub integration pushed.

Several notebook versions completed, wrote their CSVs to the kernel Output tab, and were
never downloaded — the artifacts are gone, but the *executed notebook* was pushed to git
with its outputs still embedded. That commit holds every figure as base64 PNG and the whole
stdout, which is enough to reconstruct most of what the run produced.

Figures are named from the ``savefig`` call in their own cell rather than by output order.
Ordering by position mislabelled three of notebook 05 v17's four figures — it writes
``sweepwinner_loss.png``, not ``unet_line_emission_loss.png``, and writes no
``moment_map_holdout_summary.png`` at all.

CSVs cannot be recovered this way; only what the notebook printed. Where a printed table
carries the same numbers, reconstruct it by hand and name the file ``.RECONSTRUCTED.csv``
so it is never mistaken for the artifact the run actually wrote.

Usage:
    python -m src.evaluation.recover_version <commit> <notebook.ipynb> <dest-dir>
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Dict


def recover(commit: str, notebook_path: str, dest: str, verbose: bool = True,
            overwrite: bool = False) -> Dict:
    """Extract every embedded figure and the full stdout of a pushed Kaggle run.

    Existing files are left alone unless ``overwrite=True``. A recovered figure is the
    notebook's *display* copy, which is not the same file as the ``savefig`` output that
    was downloaded from the Output tab -- lower fidelity and a different size. Filling a
    gap with it is useful; silently replacing a genuine downloaded artifact with it is a
    quiet loss, which is what happened to notebook 05 v16 before this guard existed.
    """
    raw = subprocess.run(["git", "show", f"{commit}:{notebook_path}"],
                         capture_output=True, text=True).stdout
    if not raw:
        raise FileNotFoundError(f"{notebook_path} not found in {commit}")
    nb = json.loads(raw)
    os.makedirs(dest, exist_ok=True)

    files, log, skipped = {}, [], []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        # the name a figure was saved under, taken from its own cell
        names = re.findall(
            r"savefig\(\s*(?:os\.path\.join\([^,]+,\s*)?['\"]([^'\"]+\.png)['\"]", src)
        imgs = []
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                log.append("".join(o.get("text", [])))
            p = o.get("data", {}).get("image/png")
            if p:
                imgs.append(p)
        for name, payload in zip(names, imgs):
            blob = base64.b64decode(payload)
            base = os.path.basename(name)
            out = os.path.join(dest, base)
            if os.path.exists(out) and not overwrite:
                skipped.append(base)
                continue
            with open(out, "wb") as f:
                f.write(blob)
            files[base] = {
                "bytes": len(blob),
                "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
                "provenance": f"extracted from {commit}; named from savefig('{name}')",
            }

    text = "\n".join(log)
    if os.path.exists(os.path.join(dest, "run_log.txt")) and not overwrite:
        skipped.append("run_log.txt")
        if verbose:
            print(f"kept {len(skipped)} existing file(s): {skipped}")
        return files
    with open(os.path.join(dest, "run_log.txt"), "w") as f:
        f.write(text)
    files["run_log.txt"] = {"bytes": len(text),
                            "provenance": f"stdout captured in {commit}"}
    if verbose:
        print(f"recovered {len(files) - 1} figure(s) + run log -> {dest}")
        if skipped:
            print(f"   kept existing (not overwritten): {skipped}")
        for n, v in sorted(files.items()):
            print(f"   {v['bytes']:>9,}  {n}")
    return files


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    recover(sys.argv[1], sys.argv[2], sys.argv[3])
