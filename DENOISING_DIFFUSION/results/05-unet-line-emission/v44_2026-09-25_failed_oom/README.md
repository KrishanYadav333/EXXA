# 05-unet-line-emission, Version 44, FAILED (CUDA OOM at scoring), 2026-09-25

Cells current: cell 0b pulled `1b1ee7f`, so per-arm scoring size, `STALE_MOMENT_ARMS` and both control arms were live.
Runtime not recorded in the log; about 1.5 h by the epoch times (2 arms x 30 epochs x 84 s, plus setup and scoring). **Training finished; scoring died.** Traceback in cell 18: `torch.OutOfMemoryError: Tried to allocate 3.96 GiB`
(T4, 14.56 GiB, 3.90 free) inside `UNet.forward` while scoring `winner_aug_res480` (480 px) with a sub-batch of `BS = 32`
channels. `winner_aug_res320` scored fine at 320 px (20 channels' worth fit), so the failure is specific to 480 and would
have hit `native600` too. Cause: the fix that scores each arm at its own size did not shrink the batch with it.

## What survived (persist_ckpt runs at the moment of training, rule 1)

| arm | init | epochs | PSNR | SSIM | best ep |
|---|---|---|---|---|---|
| `winner_hybrid_ft` | `sweep_winner_aug` | 30 (early stop) | 39.574 | | |
| `winner_hybrid_p10_ft` | `sweep_winner_p10` | 30 (early stop) | 40.2533 | 0.9945 | 17 |

Both persisted as `nb05_winner_hybrid{,_p10}_ft_seed42.pth` in `/kaggle/working`. **A failed version's Output cannot be attached
as an input**, so they must be downloaded from this version's Output tab (RULES #3: re-upload as `.ckpt`, never `.pth`).
Moment scores for both were computed before the crash and are in the log (below and in PROGRESS.md).

| arm (5 holdout cubes, one seed) | M0 | M1 | M2 |
|---|---|---|---|
| `winner_hybrid_ft` (control, from aug) | 39.7 | 80.9 | 69.9 |
| `winner_hybrid_p10_ft` (control, from p10) | 40.4 | 76.2 | 68.4 |
| `winner_aug_res320`, scored at 320 px | -114.0 / 34.8 / 38.3 / -113.5 / 65.4 per cube | | |

`winner_aug_res480` was the arm that crashed; `winner_aug_native600` had not started. Nothing else was lost: every earlier arm was
`SKIPPED` or scored from its stored checkpoint.

## Host RAM (the new breakdown was live)
`main` grows ~0.1 GB/epoch (2.4 to 5.2 GB over 28 epochs in arm 1, 8.6 GB by the end of arm 2); `workers` fixed at 12.4 then 39.7 GB
of RSS (shared pages, not additive); free RAM 27.7 to 21.4 GB. So the growth is in the main process, not the loader workers. Harmless at 256 px.

Fixed afterwards: `bs = max(1, int(BS * (TARGET_SIZE / size) ** 2))` in `denoise_cube` (32 at 256, 20 at 320, 9 at 480, 5 at 600).
