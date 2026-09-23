# 05-unet-line-emission — Version 34, crashed, 2026-09-20

Second attempt straight after Version 33 (same day). Since Kaggle cannot attach a failed
version's Output, this session couldn't resume V33's checkpoints either -- retrained every
arm through `winner_res320_fresh` from scratch again (all persisted, PSNRs consistent with
V33's), then died on `winner_res480_fresh` again, this time at epoch ~25, RAM 27.9 -> 0.9 GB
over 109 total epoch-lines across the session. Same wall, same arm, same cause.

Root cause and fix: see V33's README, or `src/training/sweep.py` (`persistent_workers=True`
on both DataLoaders, fixing the per-epoch worker fork-storm this leak turned out to be).

No new results beyond what V33 already produced -- this run is archived for the RAM-leak
evidence (a second, independent confirmation of the exact same failure point), not for new
numbers.
