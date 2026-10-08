# 14-native600-loss-sweep, Kaggle Version 11: training OK, status ERROR (null cell ids), 2026-10-08

Code `af13d05` (watchdog, allocator tuning, pinned memory off). Restored the same 5 arms, trained `winner_patch_600`, and the RAM watchdog
stopped it at epoch 13 (3.0 GB free, losing 1.93 GB/epoch), kept epoch 13 (best), persisted `nb14_winner_patch_600.pth` (116 MB), scored
PSNR 27.3972 / SSIM 0.9861 on the 64 px patch view (under-trained; incomparable to the 600 px full-image arms' 35 to 37 dB), then deferred
every other arm at the cap of one. ~56 minutes of session, no kernel death.

The version ended **ERROR** only because Kaggle's post-run nbconvert rejected `cells[9]['id'] = None` (7 cells of the repo notebook had null
ids: 9 and 17 to 22). That is a notebook-format fault, fixed by assigning unique ids; it is unrelated to the RAM leak. Because the version is
ERROR, its Output is not what the next session restores from, so `winner_patch_600` must be retrained.

What this run settled: the leak is in the main process (3.9 -> 26.7 GB, workers flat at 8.1 GB) and glibc tuning plus pinned memory off
did not change the slope (1.93 vs 1.95 GB/epoch). Cause: `ram_leak_probe_2026-10-08/`.
