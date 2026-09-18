# 05-unet-line-emission — crashed run, 2026-09-17

**Kaggle version number: unconfirmed.** Pulled `1832c54` (the host-RAM/lr-scale fix commit).
Rename this folder to `v<N>_2026-09-17_1832c54` once the author supplies `N` from Kaggle's
own `Kaggle Notebook | 05-unet-line-emission | Version N` push commit — the kernel cannot see
its own version number (RULES.md #10).

## What happened

`MAX_NEW_ARMS_PER_SESSION=3` worked as designed: trained `winner_mae_ft`, `winner_wavelet_ft`,
`winner_starlet_ft`, deferred everything else including `winner_gradient_ft`. Section 4 and
the PSNR band table (section 4's tail) completed. Section 6's moment-map table then crashed:

```
KeyError: 'M0'
  at band_mom[name][m][0] for m in moments
```

Root cause: `band_mom[name]` is built for every `name in CONFIGS` unconditionally, but its
VALUE is `{}` when the arm has no scored cube (here: `winner_gradient_ft`, deferred, no
checkpoint). The guard `if name not in band_mom: continue` never catches this — the KEY is
always present, only sometimes mapping to an empty dict. `winner_gradient_ft[M0]` then raised.

Cells 19-32 (including `collect_outputs`) never ran — execution stopped at the exception.
This notebook file preserves cell 18's outputs (the traceback, the PSNR table, and the
moment-map rows that DID compute before the crash) with the fix already applied to its
source, so source and output no longer match for that one cell — the evidence is what
matters here, not that correspondence.

**Fixed in `1c...` (see PROGRESS.md 2026-09-17 for the commit):** `band_mom.get(name) or {}`
plus `all(m in b for m in moments)`, in both section 6's table (cell 18) and section 6d's
headline figure (cell 24, same bug, same pattern, caught by inspection before it could crash
a second time).

## Results that survived (PSNR + moment-map, before the crash)

All three beat `sweep_winner_aug` (PSNR 39.30, M0 +29.2%, M1 +74.0%, M2 +55.0%) on every
metric, at the corrected `FINETUNE_LR_SCALE=0.1`:

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| `winner_mae_ft` | 39.93 | +40.1% | +77.1% | +70.0% |
| `winner_wavelet_ft` | 40.18 | +42.4% | +76.1% | +58.4% |
| `winner_starlet_ft` | 40.14 | +39.3% | +75.8% | +59.9% |

Metric: `moment_improvement`, clipped + signal-masked (RULES.md #6), 1 seed each. Not yet
wiggle-scored — this is the PSNR/moment ranking metric, not the kinematic diagnostic.

`winner_gradient_ft` and every fresh/p10/beam/res arm: deferred by the session cap, not
trained this run.
