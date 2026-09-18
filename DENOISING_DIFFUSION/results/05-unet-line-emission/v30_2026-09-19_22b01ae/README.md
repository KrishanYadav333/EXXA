# 05-unet-line-emission — Kaggle Version 30, 2026-09-19

First clean finish since the loss-fn sweep landed. Pulled `d93803d` (the `KeyError` fix for
deferred arms in section 6). Auto-pushed as `22b01ae`, verified against `d93803d` -- same 33
cells, all markers (`band_mom.get(name) or {}`, `MAX_NEW_ARMS_PER_SESSION`,
`FINETUNE_LR_SCALE`) intact, nothing reverted.

## What happened

The previous session's Output (the one that crashed on the `KeyError`, `v_pending_2026-09-17_1832c54`,
likely Kaggle Version 29 though never confirmed) was never attached -- Kaggle cannot attach a
failed version's Output, and it wasn't manually recovered. So `winner_mae_ft`,
`winner_wavelet_ft` and `winner_starlet_ft` were NOT resumed; `MAX_NEW_ARMS_PER_SESSION=3`
retrained all three from scratch (fine-tuned from `sweep_winner_aug`), consuming this
session's whole cap. `winner_gradient_ft` and everything else stayed deferred.

Section 6 completed this time -- the `KeyError` fix held. `collect_outputs` ran.

RAM: 28.2 -> 17.9 GB free over 98 epochs (~105 MB/epoch) -- same leak, contained by the cap,
didn't come close to exhausting this session.

## Results (retrained, numbers shift slightly from the crashed run's -- expected, different
seed-path through training, not a discrepancy)

All three still beat `sweep_winner_aug` (PSNR 39.30, M0 +29.2%, M1 +74.0%, M2 +55.0%) on
every metric:

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| `winner_mae_ft` | 39.78 | +26.3% | +77.4% | +70.5% |
| `winner_wavelet_ft` | 40.13 | +38.8% | +76.1% | +70.0% |
| `winner_starlet_ft` | 40.32 | +42.6% | +80.5% | +81.1% |

`moment_improvement`, clipped + signal-masked (RULES.md #6), 1 seed each -- not wiggle-scored.
`winner_starlet_ft` is now the best on every metric of any arm in this notebook's history.

Still deferred: `winner_gradient_ft` and all `_fresh`/`_p10_ft`/`_beam_ft`/res/spectral arms.
