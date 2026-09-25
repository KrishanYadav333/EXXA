# 14-native600-loss-sweep, Version 4, 2026-09-24

Kaggle push `8779624`, branch `native600-loss-sweep`. No traceback. `MAX_NEW_ARMS_PER_SESSION = 1`, so one new arm per session.
Ran the code as of `926d671` (before the DDRM fix in `bf8d1e7`, which was not pushed then).

| arm trained this session | PSNR (600px validation) | SSIM |
|---|---|---|
| `sweep_winner_aug_600` | 35.1745 | 0.9948 |

Epochs run: 35, about 313 s each (~3.0 h of training); early stop at epoch n/a. Host RAM free 27.8 GB at the start of the arm to 18.2 GB at the end.
Already scored when the session started: 'sweep_winner_600', 'sweep_winner_aug_600'.

PSNR here is `train_unet`'s validation PSNR on the 600px view, not the moment-map score, and it does not compare with the 256px arms' PSNR or rank arms.
The notebook and log are as Kaggle pushed them.
