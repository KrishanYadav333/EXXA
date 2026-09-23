# 05-unet-line-emission — Version 33, crashed, 2026-09-20

Pulled `fe52be5` (Version 32). Died at 27347.4s host RAM exhaustion -- "tried to allocate
more memory than is available", no traceback captured in the notebook (classic OOM-kill:
the kernel dies before Python can catch anything).

## What happened

All 8 loss-sweep arms (mae/wavelet/starlet/gradient x ft/fresh) and `winner_res320`/
`winner_res480` resumed correctly from Version 32's Output -- confirms the resume mechanism
is solid. New arms this session: `winner_res480` (fine-tune, 30 epochs, PSNR 40.1817),
`winner_res320_fresh` (50 epochs, PSNR 38.9608), then `winner_res480_fresh` started and
never finished -- log shows it reaching epoch 28 with RAM at 0.9/31.3 GB free, declining
~0.3 GB/epoch the whole way, then nothing. No `early stop`, no `persisted` line -- died
mid-epoch.

## Root cause found (from this run + v34's identical death)

`train_unet`'s `DataLoader`s never set `persistent_workers=True`. With `num_workers=4`
(Kaggle default) and the PyTorch default `persistent_workers=False`, every worker process
for BOTH loaders is torn down and respawned at the end of EVERY epoch's iteration -- a
fork-storm that leaks host RAM independent of image resolution, matching the observed
~0.3 GB/epoch decline across every arm and every notebook session since the leak was first
noticed (2026-09-16). Fixed in `src/training/sweep.py` and, since the same pattern existed
wherever a notebook builds its own DataLoader with `num_workers>0`, also in notebooks 06
and 07 and 13's DDPM/DDRM sections.

## Results that survived

| arm | PSNR | SSIM | notes |
|---|---|---|---|
| `winner_res480` (ft) | 40.1817 | 0.9968 | 30 epochs, best ep 25 |
| `winner_res320_fresh` | 38.9608 | 0.9942 | 50 epochs, best ep 31 |

`winner_res480_fresh`: never completed, no checkpoint, retrains next session.
