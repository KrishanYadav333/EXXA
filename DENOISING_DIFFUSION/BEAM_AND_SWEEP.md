# Beam-Metadata Conditioning + Hyperparameter Sweep — Code Walkthrough

Line-anchored walkthrough of the changes from the 2026-07-20 mentor meeting
(implemented in commit `0bc1c82`).

## [fits_cube_dataset.py](src/data/fits_cube_dataset.py)

**Line 41 — `beam_features_of(header)`**
- *What*: 4-vector from FITS header: `sin(2·BPA), cos(2·BPA), BMAJ·3600, BMIN·3600`
- *Why sin/cos*: an angle is circular — feeding raw degrees makes 359° and 1° look far apart. sin/cos encodes it continuously. The `2×`: a beam ellipse at 10° is the same ellipse at 190°, doubling the angle makes both give the same value
- *Why deg→rad*: header stores degrees, numpy trig wants radians — Jason explicitly flagged this trap
- *Why ×3600*: FWHM in degrees is ~4e-5, awkwardly tiny; arcsec gives ~0.15 — same scale as the sin/cos terms, so no normalization layer needed
- *Why zeros fallback*: cube without beam keys → zero vector = clean "no info" null, doesn't crash

**Line 111 — `return_beam` flag, line 155 — per-cube precompute**
- *What*: beam read **once per cube** at init from the **dirty** header, cached; `__getitem__` just indexes
- *Why dirty header*: the beam belongs to the observation (the noisy data), not the ground truth
- *Why the warning (line 162)*: if every cube has the same beam, the feature is a constant → model can't learn anything from it → know that *before* interpreting A/B results
- *Why default False*: old notebooks/DDPM expect 2-tuples — nothing breaks

## [unet.py](src/models/unet.py)

**Line 197 — `beam_dim` param, line 218 — `beam_emb` MLP, line 297 — forward injection**
- *What*: `Linear(4→128) → SiLU → Linear(128→128)`, output **added to the time embedding**
- *Why this injection point*: every ResidualBlock already consumes `t_emb` (line 120 `h = h + time_emb_proj(t_emb)`). Adding beam there conditions **all** blocks — encoder, bottleneck, and the upsampling path Jason asked for — with zero changes to block code. Concatenating a constant 4-channel map at one decoder level would touch fewer layers and cost more code
- *Why it works despite t=0*: this U-Net always runs with timestep 0, so `t_emb` is a constant — the beam term becomes the only *varying* conditioning signal
- *Why `beam_dim=0` default*: no new parameters created → V12 checkpoint loads bit-for-bit, full backward compat

## [sweep.py](src/training/sweep.py) (new file)

**Line 74 — `train_unet`**
- *What*: one config end-to-end; early stopping at line 165: min 20 / max 100 / patience 5, best-epoch weights restored before eval
- *Why*: Jason's exact prescription — "train at least 20, at most 100, stop after ~3–5 epochs no improvement." No wasted compute, no hand-picked epoch count

**Line 48 — `val_metrics`**
- *Why separate*: sweep runs have *different* loss weights — comparing their losses is meaningless. Fixed PSNR/SSIM/MSE is the constant yardstick (his rule: never score on the swept loss)

**Line 200 — `SPACE`, line 231 — `run_sweep`**
- *What*: random draw per run — width {16,32,48,64}, depth, lr log-uniform 1e-4→3e-3, α∈[0.5,0.95], scheduler patience, beam on/off
- *Why lr log-uniform*: lr matters by order of magnitude, not linearly
- *Why α only (β=1−α)*: weights on 2 loss terms have 1 real degree of freedom — sweeping both wastes samples
- *Why random not Bayesian*: his method — random explores, Bayesian later seeded with these runs; starting Bayesian can lock onto the wrong region
- *Why CSV-per-run + flush (line 291)*: Kaggle sessions die — every finished run survives
- *Why OOM halving (line 286)*: a big-width draw at batch 32 may not fit; halve batch instead of losing the sweep slot

## Notebook §4 — the A/B control (the one to emphasize)

- *What*: trains **control (no beam)** and **beam** with the identical harness, only `use_beam` differs
- *Why not just compare beam vs V12*: V12 was a fixed 30-epoch run; new runs use early stopping. Compare beam directly to V12 and you can't tell whether the delta came from the beam or the schedule. The control isolates the single variable — that's the experiment design point worth saying out loud

## Tests — [test_beam_sweep.py](tests/test_beam_sweep.py)

- Feature math checked against **Jason's own example header** (BPA 16.93° → 0.5573/0.8303, beams 0.1535″/0.1185″)
- Asserts beam actually *changes* model output (wiring proof, not just shape check)
- `train_unet` + `run_sweep` run end-to-end tiny on CPU
