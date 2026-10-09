# 14-native600-loss-sweep, Kaggle Version 14 (2026-10-09)

Pushed by Kaggle's GitHub integration as commit `c8c634c` ("Kaggle Notebook | 14-native600-loss-sweep | Version 14"); the Kaggle version number comes from that
commit message (RULES.md #5, #10). Code at run time: `21902a8` (the bootstrap printed it). The notebook here is the executed copy, outputs intact; its cells are
identical, cell by cell, to the committed `21902a8` copy, so Kaggle's push reverted nothing (RULES.md #2).

**Result.** One new arm this session (cap `MAX_NEW_ARMS_PER_SESSION = 1`): `winner_k1_600`, PSNR **40.4042**, SSIM **0.9977**, 35 epochs (ran to max_epochs 35 (no early stop)), ~345s per epoch on 2 x T4
with DataParallel (700 line-emission items at 600 px, batch 4). PSNR is on 600 px images and does not compare with 256 px arms (RUNS.md).

**RAM (the DataParallel leak, measured).** Free host RAM 27.7 -> 18.1 GB over 35 epochs (0.27 GB/epoch); `main` process RSS 2.4 -> 12.0 GB while the dataloader
workers stayed at 4.8 GB. That is ~1.4 MB per iteration x ~175 iterations/epoch, the same DataParallel leak found on 2026-10-08, still present for full 600 px images (the fix only turned DataParallel off
below 256 px). Harmless for these 25-35 epoch arms; fixed for the kin_gamma0 / sg_k3 arms (single GPU) in the 2026-10-10 change.

Files: `14-native600-loss-sweep.ipynb` (as run), `run_log.txt` (its printed output).
