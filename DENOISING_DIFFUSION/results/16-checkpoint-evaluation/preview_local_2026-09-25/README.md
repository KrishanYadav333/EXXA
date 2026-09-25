# Preview of notebook 16's figures, from a LOCAL run (not a Kaggle result)

3 checkpoints (`winner_aug_seed43`, `winner_beam_seed42`, `sg_k3_fresh`) on ONE cube (`run_0002_00560_rt_00`, all 201
channels), CPU, 2026-09-25. It exists to show what the figure suite looks like and to catch plotting bugs. Do not quote
these scores as results: 3 checkpoints, 1 cube, and the real run covers ~40 checkpoints on 7 cubes.

Start with `nb16_sheet_run_0002_00560_rt_00_topk.png` (clean, dirty and every checkpoint, seven views each), then the
`*_M1`, `*_err_M1`, `*_wiggle`, `*_chan1`, `*_sharpness`, `*_invented`, `*_spectra` and `*_radial_power` sheets. With ~40
checkpoints each sheet becomes a contact sheet of all of them; here each has 3 tiles.

Caveat on `*_wiggle`: this cube is inclined 63.7 deg, so a flat Keplerian model leaves a large residual that clean and dirty
share, and every tile shows r ~ 1.00 whatever the model did. That is the limit logged in PROGRESS.md 2026-09-25, not a result.

## `sg_v2/`: the cube where the wiggle and the smoothing are visible

The same three checkpoints on the SG v2 cube (channels 240 to 360), with clean rescaled x330 to the input's units (its dirty is ~336x its
clean; every amplitude figure says so in its title). This is the cube the earlier `wiggle_all_methods.png` was made on, and
`nb16_sheet_sg_v2_wiggle_classic_p01.png` reproduces it: residual RMS clean 0.23 / dirty 0.21 / U-Net 0.22, with the concentric ring and
blotches of clean's residual gone from the models'. Also worth opening:

- `*_calibration.png`: dirty sits on the diagonal (slope 1.00); `aug` and `beam` have their peaks shrunk (slopes 0.74 and 0.73), which is the
  smoothing as a number on a picture; `sg_k3_fresh` overshoots with a wide cloud (slope 1.09).
- `*_integrated_spectrum.png`: `aug` and `beam` raise the baseline by roughly 2,300 to 2,900 flux units in the line-free wings where clean is about
  zero; `sg_k3_fresh` adds about 500. An observation from three checkpoints on one cube, not yet a finding.
- `*_ensemble.png`, `*_error_hist.png`, `*_sharpness.png`, `*_topk.png`.

Wiggle correlations here are 0.790 / 0.761 / 0.736 against dirty's 0.873 and the error ratio is 1.29 to 1.47 (all above 1), so every model
loses to dirty on the wiggle on this cube, as the project found before. Not a Kaggle result: 3 checkpoints, 1 cube.
