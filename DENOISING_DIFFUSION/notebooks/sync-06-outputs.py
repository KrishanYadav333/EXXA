#!/usr/bin/env python
"""Copy the executed 06-section outputs out of the merged root 05 notebook
into the standalone 06 notebook in this folder.

Workflow:
  1. Run 05-unet-line-emission.ipynb on Kaggle (it contains the appended
     "06 - Continuum" section), push from Kaggle.
  2. `git pull` here.
  3. python DENOISING_DIFFUSION/notebooks/sync-06-outputs.py

The root 05 is the Kaggle-linked source of truth. This script only writes the
folder copy of 06, matching cells 1:1 by source and copying outputs across.
"""
import json, sys, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_05 = os.path.join(REPO, "05-unet-line-emission.ipynb")
FOLDER_06 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "06-unet-line-emission-continuum.ipynb")
DIVIDER_MARK = "06 — Continuum"


def load(path):
    return json.loads(open(path, "rb").read().decode("utf-8"))


def src(cell):
    return "".join(cell["source"]).strip()


def main():
    merged = load(ROOT_05)
    six = load(FOLDER_06)
    cells = merged["cells"]

    div = next((i for i, c in enumerate(cells)
                if c["cell_type"] == "markdown" and DIVIDER_MARK in "".join(c["source"])), None)
    if div is None:
        sys.exit("ERROR: 06 divider not found in root 05 — is the 06 section still appended?")
    section = cells[div + 1:]

    if len(section) != len(six["cells"]):
        sys.exit(f"ERROR: 06 section has {len(section)} cells but folder 06 has "
                 f"{len(six['cells'])} — cells drifted, re-align before syncing.")

    copied = 0
    for k, (m, s) in enumerate(zip(section, six["cells"])):
        if src(m) != src(s):
            sys.exit(f"ERROR: cell {k} source mismatch — root 05 and folder 06 drifted. Aborting.")
        if s["cell_type"] == "code":
            s["outputs"] = m.get("outputs", [])
            s["execution_count"] = m.get("execution_count")
            copied += len(s["outputs"])

    with open(FOLDER_06, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(six, ensure_ascii=False, indent=1))

    print(f"OK: copied outputs from {len(section)} 06-section cells "
          f"({copied} output objects) into folder 06.")


if __name__ == "__main__":
    main()
