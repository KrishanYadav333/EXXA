# Preview of notebook 16's figures, from a LOCAL run (not a Kaggle result)

3 checkpoints (`winner_aug_seed43`, `winner_beam_seed42`, `sg_k3_fresh`) on ONE cube (`run_0002_00560_rt_00`, all 201
channels), CPU, 2026-09-25. It exists to show what the figure suite looks like and to catch plotting bugs. Do not quote
these scores as results: 3 checkpoints, 1 cube, and the real run covers ~40 checkpoints on 7 cubes.

Start with `nb16_sheet_run_0002_00560_rt_00_topk.png` (clean, dirty and every checkpoint, seven views each), then the
`*_M1`, `*_err_M1`, `*_wiggle`, `*_chan1`, `*_sharpness`, `*_invented`, `*_spectra` and `*_radial_power` sheets. With ~40
checkpoints each sheet becomes a contact sheet of all of them; here each has 3 tiles.

Caveat on `*_wiggle`: this cube is inclined 63.7 deg, so a flat Keplerian model leaves a large residual that clean and dirty
share, and every tile shows r ~ 1.00 whatever the model did. That is the limit logged in PROGRESS.md 2026-09-25, not a result.
