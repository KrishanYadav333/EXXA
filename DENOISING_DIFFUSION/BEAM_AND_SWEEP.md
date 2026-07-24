# Beam-Metadata Conditioning + Hyperparameter Sweep

Design documentation for the two work items from the 2026-07-20 mentor meeting
(Jason: *stick with the U-Net; briefly investigate beam metadata as a model
input; then run hyperparameter sweeps with early stopping*). Implemented in
commit `0bc1c82`.

Reference baseline throughout: **V12** (continuum-subtracted U-Net, epoch 28) —
PSNR 32.95 dB / SSIM 0.9857 / MSE 0.000681; 5-cube holdout M0 +69.8%±15.2%,
M1 +17.5%±7.8%, M2 +20.1%±14.3%.

---

## 1. Beam features — `src/data/fits_cube_dataset.py`

### `beam_features_of(header)`

Builds a 4-feature vector from a FITS header:

```
[sin(2·BPA_rad), cos(2·BPA_rad), BMAJ·3600, BMIN·3600]
```

| Decision | Why |
|---|---|
| sin/cos instead of the raw angle | An angle is circular: 359° and 1° are nearly the same orientation but numerically far apart. The sin/cos pair encodes it continuously. |
| The `2×` inside sin/cos | A beam ellipse at position angle θ is the same ellipse at θ+180°. Doubling the angle maps both to the same encoding (invariance to the beam's 180° ambiguity). |
| Degrees → radians first | The header stores BPA in **degrees**; numpy trig expects radians. Mentor explicitly flagged this conversion. |
| BMAJ/BMIN × 3600 (deg → arcsec) | FWHM in degrees is ~4e-5 (awkwardly tiny). Arcsec gives ~0.15 — the same order as the sin/cos terms, so the four features are naturally co-scaled and need no normalization layer. |
| Missing keys → zero vector | A cube without beam keywords gets a clean "no information" null instead of a crash. |

Example (the header the mentor shared): BPA=16.9333°, BMAJ=4.2629e-5°,
BMIN=3.2923e-5° → `[0.5573, 0.8303, 0.1535, 0.1185]`.

### `FITSChannelDataset(return_beam=True)`

- Each item becomes `(dirty, clean, beam)`; default `False` keeps the old
  2-tuple shape so existing notebooks and the DDPM pipeline are untouched.
- The beam is read **once per cube** at init from the **dirty** header (the
  beam belongs to the observation, not the ground truth) and cached;
  `__getitem__` only indexes.
- **Null-experiment tripwire**: at init, if the beam vector is identical
  across every cube in the split, a warning prints — a constant feature
  carries zero information, and the A/B result would be a null by
  construction. Check this before interpreting any beam numbers.

---

## 2. Beam conditioning — `src/models/unet.py`

`UNet(..., beam_dim=4)` adds a small MLP
(`Linear(4→128) → SiLU → Linear(128→128)`) whose output is **added to the
time embedding** in `forward(x, t, beam)`.

| Decision | Why |
|---|---|
| Inject via the time-embedding pathway | Every `ResidualBlock` already consumes `t_emb` (`h = h + time_emb_proj(t_emb)`). Adding the beam there conditions **all** blocks — encoder, bottleneck, and the entire upsampling path the mentor asked about — with zero changes to block code. Concatenating a constant 4-channel map at one decoder level would reach fewer layers and cost more code. |
| Works despite `t=0` | This U-Net always runs with timestep 0, so `t_emb` is constant; the beam term becomes the only *varying* conditioning signal. |
| `beam_dim=0` default | No new parameters are created, so the V12 checkpoint loads bit-for-bit. Full backward compatibility. |

---

## 3. Sweep harness — `src/training/sweep.py`

### `train_unet(...)` — one config, end to end

- Early stopping per mentor spec: `min_epochs` (default 20), `max_epochs`
  (default 100), `patience` 5 on val loss; best-epoch weights are restored
  (and checkpointed) before final evaluation.
- Multi-GPU via `DataParallel` when available; checkpoints always saved from
  the unwrapped model.

### `val_metrics(...)` — the fixed yardstick

Sweep runs use *different* loss weights, so their losses are not comparable.
Every run is scored on **fixed PSNR / SSIM / MSE** over the val split
(mentor's rule: never score a sweep on the loss it optimizes when that loss
is itself swept).

### `run_sweep(...)` — random search

Default space (`SPACE`):

| Parameter | Range | Sampling |
|---|---|---|
| `base_channels` | {16, 32, 48, 64} | choice |
| `channel_multipliers` | (1,2,4) / (1,2,2,4) / (1,2,4,8) | choice |
| `lr` | 1e-4 … 3e-3 | **log-uniform** (lr matters by order of magnitude) |
| `alpha` (HybridLoss; β=1−α) | 0.5 … 0.95 | uniform — two loss weights have one real degree of freedom, so only α is swept |
| `sched_patience` | {3, 5, 8} | choice |
| `use_beam` | {False, True} | choice — beam participates in the sweep as one more binary |

- **Random first, Bayesian later** (mentor's method): random exploration
  gathers unbiased statistics; a Bayesian sweep seeded with these runs can
  follow. Starting Bayesian risks locking onto the wrong region.
- **Crash-safe**: each run's CSV row is appended and flushed immediately
  (`results/sweep_results.csv`) — a dead Kaggle session keeps every
  completed run.
- **OOM handling**: a run that doesn't fit retries at half batch (up to 2
  halvings) instead of losing its sweep slot.

---

## 4. Notebook `05-unet-line-emission.ipynb` (root, Kaggle-linked)

| Section | Content |
|---|---|
| §3 | Datasets with `return_beam=True`; prints per-cube beam vectors (check the null-experiment tripwire here first) |
| §4–5 | **Beam A/B**: control (no beam) vs beam trained under the *identical* early-stopping harness; loss curves + verdict |
| §6 | All-5-holdout moment-map evaluation of the beam model (same contract as V12's numbers) |
| §7–8 | 12-run random sweep + top-5 table + per-parameter correlation with PSNR |
| §9 | Persist checkpoints/CSVs to `/kaggle/working` |

**Why an A/B control instead of comparing beam vs V12 directly**: V12 was a
fixed 30-epoch run; the new runs use early stopping. A direct beam-vs-V12
comparison would conflate the beam effect with the schedule change. The
control isolates the single variable — `use_beam` — which is the experiment.

---

## 5. Tests — `tests/test_beam_sweep.py`

1. Feature math verified against the mentor's example header values.
2. Beam-conditioned forward: asserts the beam vector actually *changes* the
   output (wiring proof, not just a shape check).
3. `train_unet` end-to-end on a tiny synthetic dataset (early stopping
   fields + fixed metrics present).
4. `run_sweep` completes runs and writes the CSV.

Run: `python -m tests.test_beam_sweep` from `DENOISING_DIFFUSION/`.

---

## Open items

- Self-gravitating cubes (moment-map test only, no training) — waiting on
  the mentor to send the dataset.
- Bayesian sweep seeded with the random runs — after the random sweep's
  statistics justify it.
