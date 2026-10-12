# GI wiggle scorecard, 2026-10-08

Produced by `experiments/wiggle_scorecard.py` (local CPU, about 1 minute, deterministic, no
training, no seeds). Analysis and the bug it exposed: `results/PROGRESS.md` 2026-10-08. Method
source: Terry et al. 2022 (`research.md` 2026-10-08).

Command: `python3 experiments/wiggle_scorecard.py` (defaults: all 6 cubes, methods
`beam_only,dirty,wiener3d_a2,wiener3d_a4`).

| file | content |
|---|---|
| `scorecard.json` | every metric, per cube and method, plus fit metadata (au/px, inclination, fitted M_*, R_in, R_out, track count) |
| `scorecard.md` | the same as a table |
| `run_log.txt` | stdout, including the spine self-test |
| `<cube>.png` | rows = clean / beam_only / dirty / Wiener a=2 / a=4; columns = M0, quadratic M1, M1 minus clean's rotation curve, systemic channel with the Terry spine overlaid, spine y_n(R) vs clean |

Headline metric is `rc_r` (correlation with clean of M1 minus clean's own fitted rotation
curve, R >= 2 beams). `resid_r` (Keplerian-subtracted) is kept for comparison with older
numbers only; it mostly scores the axisymmetric velocity field. `beam_only` uses the clean
cube and is an oracle (resolution loss without noise), not a method.
