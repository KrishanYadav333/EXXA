# 05-unet-line-emission, Version 40, completed, 2026-09-24

Kaggle push `90f2390`. Cells current (the KeyError guard, `SEED_OVERRIDE` and the `winner_aug_*` arms all present).
No traceback. Trained 3 new arms, which is the session cap `MAX_NEW_ARMS_PER_SESSION = 3` doing what it is
for: this session, the four `p10`-sourced loss arms begin: `winner_mae_p10_ft`, `winner_wavelet_p10_ft`, `winner_starlet_p10_ft`.

| arm | seed | PSNR | SSIM | best epoch | epochs run |
|---|---|---|---|---|---|
| `winner_mae_p10_ft` | 42 | 40.3443 | 0.9940 | 27 | 32 |
| `winner_wavelet_p10_ft` | 42 | 40.4146 | 0.9944 | 25 | 30 |
| `winner_starlet_p10_ft` | 42 | 40.2013 | 0.9939 | 14 | 30 |

Every other arm in the log is `SKIPPED, already done` or `scored from stored checkpoint`, restored from earlier
sessions' Output (and, for `winner_aug_res320`/`winner_aug_res480`, from the recovered v38 checkpoints).
The log and the notebook are as Kaggle pushed them. PSNR here is the 256px validation PSNR (`sweep.val_metrics`) and
does not rank arms; the moment-map table in cell 18 does, and is summarised in `results/PROGRESS.md` 2026-09-25.
`v39` is not in git, so it is not archived here.
