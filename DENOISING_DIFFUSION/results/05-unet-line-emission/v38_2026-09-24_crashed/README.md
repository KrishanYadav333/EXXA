# 05-unet-line-emission — Version 38, crashed, 2026-09-24

Pulled `9856f17` (confirmed by cell 4's `HEAD is now at`). Cells were current this time,
unlike v36: `band_mom.get` guard, `SEED_OVERRIDE` and the `winner_aug_res*` arms all
present. Died of host-RAM exhaustion at epoch 19 of the third new arm, no traceback
(the kernel is killed before Python can raise).

## What finished

| arm | seed | PSNR | SSIM | best epoch | ran |
|---|---|---|---|---|---|
| `winner_aug_res320` | 43 | 37.3597 | 0.9932 | 18 | 50 |
| `winner_aug_res480` | 43 | 40.1815 | 0.9969 | 41 | 50 |

First runs of winner_aug_seed43's recipe (WINNER hyperparameters, D4 augmentation, from
scratch) at higher resolution, the arms the mentor asked for. Both `persisted` to
`/kaggle/working`, but a failed version's Output cannot be attached, so they are only
recoverable by downloading them by hand (see below).

Read these with care. PSNR at 320, 480 and 256 is measured on different pixel grids (each
view has its own val set), so it does not rank resolutions. The fair comparison is the
600px moment maps in section 6, which this run never reached. `winner_aug_res480` landing
within 0.0002 dB of v33's old fine-tuned `winner_res480` (40.1817) is a coincidence, not a
duplicate: different training (30-epoch fine-tune, best epoch 25, against 50-epoch scratch,
best epoch 41) and SSIM differs in the fourth digit.

`winner_mae_p10_ft` was the third arm of the session and never finished (epoch 19, best
val 0.0061, nothing persisted).

## The RAM leak is NOT fixed, and the 09-20 diagnosis was wrong

`persistent_workers=True` was live in this run (HEAD contains `80a199f`). Free RAM at the
start of each arm, and the per-epoch slope within it:

| arm | free at ep 1 | free at last epoch | slope |
|---|---|---|---|
| `winner_aug_res320` (320px) | 27.7 GB | 17.1 | 0.216 GB/epoch |
| `winner_aug_res480` (480px) | 16.4 | 2.3 | 0.288 GB/epoch |
| `winner_mae_p10_ft` (256px) | 2.3 | 0.3 | 0.111 GB/epoch |

Three facts, all read off the log:
- The 480px slope, 0.29 GB/epoch, is what v33 measured **before** the fix (about 0.3). The
  fork-storm explanation predicted the leak would disappear. It did not move.
- The slope grows with image size (0.11, 0.22, 0.29 at 256, 320, 480px), so it is tied to
  the data, not to arm bookkeeping.
- Nothing is returned when an arm ends: 17.1 to 16.4 and 2.3 to 2.3 across the two arm
  boundaries. So arms consume RAM cumulatively, and the session cap is what keeps a run
  alive, not any fix.

The epoch loop itself only accumulates Python floats (`tr_hist`, `va_hist`), so the leak is
not in `train_unet`'s own bookkeeping. Which of main process, DataLoader workers or page
cache holds it is not known. `_host_ram_note()` now prints `main / workers / cache` on
every epoch line so the next run answers it.

## Recovering the two finished arms

If v38's Output tab still allows downloads: take `nb05_winner_aug_res320_seed43.pth`,
`nb05_winner_aug_res480_seed43.pth` and `nb05_seed_repeats.csv`, rename the checkpoints to
`.ckpt` (RULES.md #3), upload as a Dataset, attach it. Resume globs `nb05_*.ckpt` from any
input and restores the rows. Without them both arms retrain, about 2.2 h and 4.3 h.

## Next session

Everything left except `winner_aug_native600` (gated off) is a 256px arm, about 0.11
GB/epoch, so roughly 4 to 5 GB per arm at 30 to 45 epochs. Three of them fit in a fresh
31 GB. The res arms are what ate the budget here. If the two res arms are not recovered,
run that session with `MAX_NEW_ARMS_PER_SESSION = 1` so 320 and 480 do not share a session.
