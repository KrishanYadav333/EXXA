# EXXA — working instructions

## Read the rules before touching a notebook

**[DENOISING_DIFFUSION/RULES.md](DENOISING_DIFFUSION/RULES.md) is mandatory reading before
creating or changing any notebook in this repo.** Read it first, not after a review finds
the same bug again. Each rule names the run that was lost to it, so a violation costs
hours of GPU time that have already been paid once.

The twelve rules, in short — the file has the incident behind each:

1. **Persist a trained model the instant it finishes training**, never in a cleanup cell.
   `CKPT_DIR` is inside the git clone, which the bootstrap wipes and which is not part of
   the notebook Output. Call `persist_ckpt(...)` after every `.train(...)`.
2. **The in-notebook git pull (cell 0b) updates `src/` only, never the cells.** Kaggle's
   own GitHub pull does update them — use it. The push is the danger: it is automatic and
   sends a stale Kaggle copy back over committed work. Cell 0b printing the newest SHA
   says nothing about the cells; verify a marker from the latest commit instead.
3. **Never upload a checkpoint as a Kaggle Dataset.** A torch checkpoint is a zip; Kaggle
   unpacks it into a directory `torch.load` rejects. Attach the producing notebook's
   Output instead.
4. **Compare losses only within one objective and one training view.** Rank objectives by
   PSNR. A raw `best_val_loss` minimum selects by scale, not quality.
5. **Attribute every result** through `collect_outputs()`; map run folders to Kaggle
   versions in `results/RUNS.md`. Do not list checkpoints there — rule 1 already saved
   them.
6. **Never quote a number whose metric you cannot name.** Moment improvements are
   signal-masked and noise-clipped; older whole-map numbers are not comparable. Say
   whether a spread is across seeds or across cubes.
7. **One notebook, one file — the Kaggle-linked slug name.** A second copy under a
   friendlier name stops receiving runs at the first push and then looks just as
   authoritative. 08 kept two that disagreed on every moment. Downloaded copies belong
   under `results/<notebook>/v<N>_.../`, never beside the live notebook.
8. **A zero from a diagnostic is a suspect, not a pass.** 08 v2 reported 0.0% invented
   blobs for all four arms; the detector could not fire. Same checkpoints, fixed
   threshold: 22.3–39.0%. Check the denominator before believing a clean result.
9. **Check which branch the kernel pulled — it decides the metric.** 05 bootstraps from
   `line-emission`, which has none of the moment fixes; 08 pulls `midterm-prep`, which has
   all three. Their numbers are not comparable. Work happens on `midterm-prep`.
10. **Archive the notebook itself with every run, failures included.** The run folder holds
    the notebook *as it ran, outputs intact*, not the stripped repo copy, plus its log,
    figures, and a README. `<N>` is the Kaggle version, which only the author or Kaggle's
    push commit knows. A failed run is archived the same way with the error and what
    survived. v19 and v20 were filed without their notebooks and those copies are gone.
11. **Log progress the moment a run ends, a notebook lands, or a bug appears.**
    `results/PROGRESS.md` is the chronological record; `RUNS.md` maps numbers to runs, this
    maps the project to its history. A bug entry names what was wrong, how it was caught,
    and **which published numbers it touches**. `winner_beam`'s M0 sat in RUNS.md and the
    blog as a finding before anyone noticed it was measured with a dead branch.
12. **Keep every checkpoint until GSoC finishes.** Refuted arms, worst seeds, passed-over
    configs, all of them. `models/` is the store, indexed in `models/README.md`. Correcting
    `winner_beam` meant re-running inference on that exact checkpoint; had it been cleaned
    up as a losing arm the published number would have been wrong permanently.

## Before every Kaggle run

- Import/pull the notebook, then confirm a symbol that changed in the latest commit.
- Check `results/RUNS.md` for what the previous version of that notebook actually produced.

## After every Kaggle run

- Check whether Kaggle's push reverted anything: `git log -p -1 -- <notebook>.ipynb`, or
  grep for a symbol you added.
- Archive the run under `DENOISING_DIFFUSION/results/<notebook>/` and add its row to
  `results/RUNS.md`, including the Kaggle version number — the kernel cannot see it, so it
  has to come from the author.

## Before every git push

- Update `results/PROGRESS.md` with whatever this push contains (RULES.md #11 already
  requires this the moment a run ends, a notebook lands, or a bug appears).
- Update `context.md` if the push changes anything it currently claims: the active branch,
  which notebooks exist, a phase's status, an open item that's now closed. A stale
  `context.md` reads as current to the next session that loads it.
- Both are part of the push, not a follow-up. Include them in the same commit or the one
  right before it, not "later."

## Repo orientation

- `DENOISING_DIFFUSION/src/` — the library the notebooks import. This *does* hot-reload on
  Kaggle via section 0b.
- `DENOISING_DIFFUSION/tests/` — plain scripts, run with `PYTHONPATH=. python3 tests/x.py`.
  `test_notebook_cell_order.py` checks every notebook cell's names resolve from earlier
  cells; run it after editing a notebook.
- `DENOISING_DIFFUSION/results/RUNS.md` — the run index. The record of what produced which
  number.
- `DENOISING_DIFFUSION/results/PROGRESS.md` — the chronological log: runs, arrivals,
  bugs, and which published numbers each bug touches.
- Work happens on `midterm-prep`. Leave `line-emission` untouched.
