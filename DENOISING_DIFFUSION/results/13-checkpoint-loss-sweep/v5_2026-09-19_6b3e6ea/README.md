# 13-checkpoint-loss-sweep — Kaggle Version 5, 2026-09-19

Second clean finish. Pulled `d93803d`, auto-pushed as `6b3e6ea`, verified against `d93803d`
-- same 21 cells, all markers intact, nothing reverted.

## What happened

`kin_gamma0_mae_ft`/`_fresh` (from Version 4) resumed correctly via `_import_prior_nb13` --
CSV + checkpoint both matched, no retrain. `MAX_NEW_ARMS_PER_SESSION=2` trained
`kin_gamma0_wavelet_ft` and `kin_gamma0_wavelet_fresh`, deferred `starlet`/`gradient`
(both sources) and all of sg/ddpm/ddrm.

**RAM: 27.8 -> 3.3 GB free over 86 epochs.** Closer to the edge than Version 4's run (which
ended at 6.5 GB) -- the leak's slope is consistent (~285 MB/epoch here vs ~280 in v4), but
this session's two arms together ran longer (86 vs 82 epochs) and ate further into the
margin. `MAX_NEW_ARMS_PER_SESSION=2` is not a fixed safety margin, it's however many epochs
2 arms happen to take -- worth dropping to 1 if a future session's arms are epoch-heavier
than these, or actually tracing the leak before it matters again.

## Results

| arm | PSNR | SSIM |
|---|---|---|
| `kin_gamma0_wavelet_ft` | 42.8834 | 0.9957 |
| `kin_gamma0_wavelet_fresh` | 38.8636 | 0.9947 |

`kin_gamma0_wavelet_ft` is now the best PSNR of any kin_gamma0 arm (mae_ft was 42.20).
Pixel metric only, 31-channel val set -- not comparable to 05's PSNRs (RULES.md #4), not
wiggle-scored yet.

Still deferred: `kin_gamma0_starlet_ft/_fresh`, `kin_gamma0_gradient_ft/_fresh`, and all of
`sg_k3_fresh`, `ddpm_seed42`, `ddrm_prior` (24 arms total).
