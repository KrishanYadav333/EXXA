# 14-native600-loss-sweep, Kaggle Version 12: COMPLETE, 2026-10-08, code a0422fa (the DataParallel fix)

Restored the same 5 arms, trained `winner_patch_600` with DataParallel OFF (64 px tiles, one GPU), early-stopped at epoch 25 on its own
patience (best epoch 17, val 0.0026), persisted `nb14_winner_patch_600.pth` (116 MB), then deferred every other arm at the cap of one.
Session ~66 minutes. Status COMPLETE (the null-cell-id fix removed Version 11's ERROR).

**Result: `winner_patch_600` PSNR 30.1802, SSIM 0.9886**, scored on the full 600 px validation images (not the 64 px tiles it trained on),
so it is comparable with the other 600 px arms (35 to 37 dB): it is clearly the weakest, as the patch view was in 05.

**The fix held.** Free RAM 28.6 GB after epoch 1 and 28.5 GB after epoch 25; main-process RSS 1.6 GB throughout, workers 3.9 GB.
Version 11 (DataParallel on) went 26.3 -> 3.0 GB free over 13 epochs. Epoch time 147 s against 205 to 250 s with DataParallel.

**One thing to watch, not caused by this fix:** validation loss jumps from epoch 20 (0.0029 -> 0.040, 0.035, 0.060, 0.046, 0.055, 0.045)
while training loss keeps falling. The model is trained on 64 px tiles and validated on 600 px images, and drifted into a state that
generalises badly to the full field at lr 8.2e-4. Early stopping kept epoch 17, so the saved checkpoint is not affected, but the
patch arm's training is unstable at this learning rate.

Output also holds `14-native600-loss-sweep_outputs.zip` (567.8 MiB, 8 files) from the new `collect_outputs` zip step. Files: `run_log.txt`.
