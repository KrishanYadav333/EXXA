# EXXA — working instructions

## Read the rules before touching a notebook

**[DENOISING_DIFFUSION/RULES.md](DENOISING_DIFFUSION/RULES.md) is mandatory reading before
creating or changing any notebook in this repo.** Read it first, not after a review finds
the same bug again. Each rule names the run that was lost to it, so a violation costs
hours of GPU time that have already been paid once.

The nine rules, in short — the file has the incident behind each:

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

## Before every Kaggle run

- Import/pull the notebook, then confirm a symbol that changed in the latest commit.
- Check `results/RUNS.md` for what the previous version of that notebook actually produced.

## After every Kaggle run

- Check whether Kaggle's push reverted anything: `git log -p -1 -- <notebook>.ipynb`, or
  grep for a symbol you added.
- Archive the run under `DENOISING_DIFFUSION/results/<notebook>/` and add its row to
  `results/RUNS.md`, including the Kaggle version number — the kernel cannot see it, so it
  has to come from the author.

## Repo orientation

- `DENOISING_DIFFUSION/src/` — the library the notebooks import. This *does* hot-reload on
  Kaggle via section 0b.
- `DENOISING_DIFFUSION/tests/` — plain scripts, run with `PYTHONPATH=. python3 tests/x.py`.
  `test_notebook_cell_order.py` checks every notebook cell's names resolve from earlier
  cells; run it after editing a notebook.
- `DENOISING_DIFFUSION/results/RUNS.md` — the run index. The record of what produced which
  number.
- Work happens on `midterm-prep`. Leave `line-emission` untouched.
