# 05 v19 — 2026-08-14 — CRASHED, no moment scores

Batch run (Save & Run All). Died at **14413s (4.0 h)** with `DeadKernelError`, well inside
Kaggle's 12 h limit, so this was a crash and not a timeout.

**Section 6 never ran. This run produced no moment-map scores at all.** Nothing here
changes any published number; the moment figures in the blog still come from 08 v4.

## Reused, not retrained

`_import_nb08()` worked exactly as intended: 12 checkpoints imported from the
`exxa-nb08-checkpoints-v4` Dataset, and all four core arms skipped training.

    sweep_winner     42/43/44   SKIPPED (PSNR 37.2303 / 37.7237 / 38.9694)
    sweep_winner_aug 42/43/44   SKIPPED (PSNR 38.9389 / 39.8084 / 39.1412)
    sweep_winner_p10 42/43/44   SKIPPED (PSNR 38.9538 / 39.0344 / 39.8298)
    v12_cfg          42/43/44   SKIPPED (PSNR 38.6593 / 36.6707 / 37.4657)

The `.ckpt` rename survived the round trip: Kaggle did not unpack them, and `torch.load`
accepted all 12 (RULES.md #3).

## What this run actually trained

| arm | seed | result | recorded? |
|---|---|---|---|
| `sweep_winner` | 49 | PSNR **36.1624**, SSIM 0.9908, best ep 11, 20 run | yes, CSV row written |
| `winner_beam` | 42 | PSNR **38.7102**, SSIM 0.9933, best ep 24, 29 run | yes, CSV row written |
| `winner_patch` | 42 | best ep 21 (val 0.0046), early stop ep 26 | **no** — died before `val_metrics` |
| `winner_native600` | 42 | never started | no |

`winner_patch`'s checkpoint exists (written during training, RULES.md #1) but has no
PSNR/SSIM row: the kernel died between its last epoch and the metric computation.

**PSNR only.** No arm here has a moment score, so none of these numbers may be used to
rank arms — RULES.md #4 and #6. The 36.16 for seed 49 in particular does **not** settle
the sweep-reproduction question: the sweep ran at `N_SAMPLES=50` and this at 150.

## Cause

Not the session limit. The signature points at memory pressure:

- `winner_patch` epoch times drift upward over the run — 111, 112, 113, 115, 121, 123,
  128s — rather than staying flat as they do in every other arm (~90s, dead steady).
- The kernel then went silent for 6905s (1.9 h) after `early stop at epoch 26`, printing
  neither `winner_patch`'s PSNR line nor the `winner_native600` header.
- It died with `ConnectionResetError: [Errno 104] Connection reset by peer` inside
  `multiprocessing/resource_sharer.py` → `deliver_challenge`, which is DataLoader worker
  IPC, plus `WARNING: attempted to send message from fork`.

A worker pool that stops answering, after epochs that were already slowing, is the shape
of the OS reaping processes under memory pressure rather than a bug in the arm's logic.

`FlatPatchDataset.__getitem__` calls `self._inner.base[img_i]` once **per patch**, so an
8-patch image is decoded 8 times per epoch. That explains the 113s epochs (vs ~90s for
full 256px images despite 16× fewer pixels), and it means the patch arm holds far more
decoded array churn than any other. It is the most likely contributor, but it was not
proven — the run died without leaving a memory trace.

## If this is re-run

`winner_native600` never got to start and is the untested risk: 600px at batch 4, on a
session that had already shown memory pressure at 64px. Drop it, or run it alone.

The three checkpoints this run did train are in its Output. Scoring them needs no
retraining — attach that Output and section 6 will pick them up.

## Missing from this archive

**The notebook itself was not kept** (RULES.md #10, which this run is part of the reason
for). Only `run_log.txt`, extracted from the cell outputs before the local copy was
stripped, survives. Kaggle did not auto-push this version, so there is no copy in git
either. What is lost: the stored figures, the exact traceback formatting, and the papermill
per-cell timings that would pin down where the memory pressure began.

The error itself is recorded above in full: `DeadKernelError` at 14413s (4.0 h), preceded by
`ConnectionResetError: [Errno 104] Connection reset by peer` inside
`multiprocessing/resource_sharer.py`, after `winner_patch` early-stopped at epoch 26 with
its epoch times drifting 111 -> 128s.
