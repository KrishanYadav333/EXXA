# 05-unet-line-emission — Version 36, crashed, 2026-09-24

Not a new bug. Kaggle ran cells frozen at commit `1832c54` (2026-09-16) -- three fix
generations behind HEAD at the time. Fingerprinted from the notebook's own source (not
git): had the fine-tune lr fix and `MAX_NEW_ARMS_PER_SESSION=3` from `1832c54`, but not
`d93803d`'s KeyError guard (2026-09-18), not the `persistent_workers` RAM fix
(2026-09-20), and not the winner_aug res-arm recipe restored the same day this crashed.
RULES.md #2: cell 0b only pulls `src/`, never the cells -- Kaggle's own GitHub
pull/import does, and was not done before this run.

## What happened

`winner_mae_ft`, `winner_wavelet_ft`, `winner_starlet_ft` scored (39.93 / 40.18 / 40.14
PSNR -- consistent with existing numbers, no reason to distrust them).
`winner_gradient_ft`, `winner_gradient_fresh`, `winner_k1`, `winner_k2` DEFERRED at the
session cap, never attempted. Old-recipe `winner_res320`/`winner_res480`/
`winner_res320_fresh`/`winner_res480_fresh` rows from v33/v34 carried into the resumed
CSV but are not this session's work.

Then cell 18 (moment-map comparison table) hit `KeyError: 'M0'` printing
`winner_gradient_ft`'s row -- the exact failure `d93803d` already fixed by skipping any
arm whose `band_mom` entry is missing a moment, because these stale cells never got that
guard.

## Fix

No code fix here -- `midterm-prep` HEAD already has all three missing fixes, confirmed
present in the notebook this session restored/wrote. The fix is procedural: before the
next 05 run, use Kaggle's own GitHub-pull/import in the notebook editor (not cell 0b),
then confirm a marker from the latest commit is actually in the pulled source
(`SEED_OVERRIDE` or `band_mom.get(name)`) before hitting Run.

## Numbers this touches

None invalidated. The three finished PSNR values match the existing record and are not
re-quoted from this run. Nothing else scored.
