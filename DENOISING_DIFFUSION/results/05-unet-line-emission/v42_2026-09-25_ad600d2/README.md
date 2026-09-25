# 05-unet-line-emission, Version 42, completed, 2026-09-25

Kaggle push `ad600d2`. Cells current (the KeyError guard, `SEED_OVERRIDE` and the `winner_aug_*` arms all present).
No traceback. Trained 3 new arms, which is the session cap `MAX_NEW_ARMS_PER_SESSION = 3` doing what it is
for: this session, `winner_starlet_beam_ft`, `winner_gradient_beam_ft`, and `winner_k1` (one spectral neighbour), the last beam arms plus the first spectral-context arm.

| arm | seed | PSNR | SSIM | best epoch | epochs run |
|---|---|---|---|---|---|
| `winner_starlet_beam_ft` | 42 | 39.6210 | 0.9938 | 18 | 30 |
| `winner_gradient_beam_ft` | 42 | 40.2153 | 0.9942 | 30 | 35 |
| `winner_k1` | 42 | 42.5869 | 0.9959 | 52 | 57 |

Every other arm in the log is `SKIPPED, already done` or `scored from stored checkpoint`, restored from earlier
sessions' Output (and, for `winner_aug_res320`/`winner_aug_res480`, from the recovered v38 checkpoints).
The log and the notebook are as Kaggle pushed them. PSNR here is the 256px validation PSNR (`sweep.val_metrics`) and
does not rank arms; the moment-map table in cell 18 does, and is summarised in `results/PROGRESS.md` 2026-09-25.
`v39` is not in git, so it is not archived here.

## Read this before quoting v42's moment table

**The `winner_aug_res320` and `winner_aug_res480` rows are not valid.** Cell 18's `denoise_cube` resized every arm to
256 px, so those two models, trained at 320 and 480 px, were scored on inputs at a scale they never saw (M0 -12.3 / +9.7,
M1 +50.0 / +66.1, M2 -4.4 / +54.5). They say nothing about less downsampling. Their 256px validation PSNR (37.36, 40.18)
is unaffected: it comes from `train_unet` on the arm's own views. Fixed 2026-09-25 (per-arm size, stale rows dropped so
they re-score). Every other row in the table is scored at 256 px, which is what those arms were trained at. See
`results/PROGRESS.md` 2026-09-25.
