# 14-native600-loss-sweep, Kaggle Version 7: FAILED (host RAM), 2026-10-02

Pushed from `ba69ebe` (the zip-outputs commit). Restored 5 earlier arms from a prior Output (`sweep_winner_600`,
`sweep_winner_aug_600`, `sweep_winner_p10_600`, `v12_cfg_600`, `winner_beam_600`), then trained `winner_patch_600`
(the first new arm) and died.

**What happened.** Host RAM available fell by 1.9 to 2.0 GB every epoch (26.2 -> 24.3 -> 22.4 -> ... -> 1.0 GB over 14 epochs,
epoch time flat at ~205 s), the kernel was killed at epoch 15, and the session then sat dead until second 16249 (4.5 h)
before Kaggle ended it. Nothing from `winner_patch_600` was saved (no `persist_ckpt` had been reached; best val 0.0060 at
epoch 10 was in memory only). The 5 restored arms are intact in the prior Output.

**Cost.** ~4.8 h of a GPU session for zero new arms.

**Why it matters beyond this arm.** The same leak is in `train_unet`'s main process for every notebook (05, 13, 14); 05 v38
measured 0.11 GB/epoch at 256 px, 0.29 at 480; here 1.95 at 600 on the patch view. Earlier sessions only survived through the
per-session arm cap (`MAX_NEW_ARMS_PER_SESSION`). Fix and mechanism: `results/PROGRESS.md` 2026-10-08.

Files: `run_log.txt` (trimmed Kaggle log, nothing edited but the git-clone progress lines).
