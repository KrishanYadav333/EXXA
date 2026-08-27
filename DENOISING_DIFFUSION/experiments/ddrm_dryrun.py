"""
Scaled-down dry run of notebook 07, on REAL data, to catch plumbing failures before a
multi-hour Kaggle GPU run hits them.

64px, 2 cubes, 3 epochs. The prior will be essentially untrained and the restoration numbers
will be meaningless -- that is not what this checks. It checks that every stage runs on real
cubes and produces finite, correctly-shaped output: data loading, unconditional training,
checkpointing, DDRM sampling, and the GI wiggle scoring.

Run: PYTHONPATH=.. python3 ddrm_dryrun.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import torch
import torch.nn.functional as Fn
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset

from src.data.cube_split import split_cubes
from src.data.fits_cube_dataset import FITSChannelDataset
from src.models.diffusion_unet import default_diffusion_config
from src.training.diffusion import DenoisingDiffusion, data_transform, inverse_data_transform
from src.training.ddrm import beam_transfer_function, ddrm_steps
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, fit_keplerian, wiggle_residual

SIZE, EPOCHS, N_SAMPLES = 64, 3, 20
SG = "self-gravitating cube and dirty cube/kinematic_data_v2"
CHANNELS = list(range(290, 310, 4))

print("=" * 70)
print(f"DDRM dry run -- {SIZE}px, {EPOCHS} epochs. Plumbing only, numbers are meaningless.")
print("=" * 70)

# --- stage 1: data --------------------------------------------------------------------
train_cubes, val_cubes, _ = split_cubes(data_dir="Line Emission Data", n_holdout=3,
                                        val_fraction=0.2, seed=42, verbose=False)
kw = dict(n_samples=N_SAMPLES, target_size=SIZE, seed=42,
          subtract_continuum=True, continuum_n=5, verbose=False)
train_pairs = FITSChannelDataset(train_cubes[:2], **kw)
val_pairs = FITSChannelDataset(val_cubes[:1], **kw)


class CleanOnly(Dataset):
    def __init__(self, p): self.p = p
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.p[i][1]


train_ds, val_ds = CleanOnly(train_pairs), CleanOnly(val_pairs)
x0 = train_ds[0]
print(f"\n[1] data: train {len(train_ds)} val {len(val_ds)} | sample {tuple(x0.shape)} "
      f"range [{x0.min():.3f}, {x0.max():.3f}]")
assert x0.shape[0] == 1, "prior expects single-channel clean images"

# --- stage 2: unconditional training --------------------------------------------------
cfg = default_diffusion_config(image_size=SIZE)
cfg.data.conditional = False
cfg.diffusion.prediction_type = "v"
cfg.diffusion.min_snr_gamma = 5.0
cfg.diffusion.beta_schedule = "cosine"
cfg.model.ch = 32                      # smaller than the notebook's 64, for CPU speed

ckpt = "/tmp/ddrm_dryrun_prior.pth"
runner = DenoisingDiffusion(config=cfg, device="cpu", lr=2e-5, checkpoint_path=ckpt)
print(f"[2] model: {sum(p.numel() for p in runner._core.parameters())/1e6:.1f}M params")

t0 = time.time()
hist = runner.train(DataLoader(train_ds, batch_size=4, shuffle=True),
                    DataLoader(val_ds, batch_size=4), n_epochs=EPOCHS, verbose=False)
print(f"    trained {EPOCHS} epochs in {time.time()-t0:.0f}s | "
      f"train {hist['train_losses'][0]:.1f} -> {hist['train_losses'][-1]:.1f}")
assert all(np.isfinite(hist["train_losses"])), "non-finite training loss"
assert os.path.exists(ckpt), "checkpoint not written"
print(f"    checkpoint: {os.path.getsize(ckpt)/1e6:.0f} MB")

# --- stage 3: DDRM restoration on the real SG cube ------------------------------------
with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
    hdr = h[0].header
    sg_clean = np.stack([np.asarray(h[0].data[c], np.float32) for c in CHANNELS])
with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
    sg_dirty = np.stack([np.asarray(h[0].data[c], np.float32) for c in CHANNELS])
beam = fits.getdata("results/self-gravitating/dirty_beam_recovered_v2.fits").astype(np.float64)
print(f"\n[3] SG cube: {sg_clean.shape} | beam {beam.shape}")

transfer = beam_transfer_function(beam, (SIZE, SIZE))
transfer = transfer / transfer.max()
print(f"    transfer: peak {transfer.max():.3f}, "
      f"{100*float((transfer > 0.01).float().mean()):.1f}% of modes above 1% gain")

restored = []
for i in range(len(CHANNELS)):
    d = sg_dirty[i]
    lo, hi = float(d.min()), float(d.max())
    d01 = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    t = torch.from_numpy(d01)[None, None].float()
    t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
    y = data_transform(t)
    sigma_y = float(hdr.get("RMS", 0.0013)) / max(hi - lo, 1e-12) * 2.0
    with torch.no_grad():
        xs, _ = ddrm_steps(y, list(range(0, runner.num_timesteps, runner.num_timesteps // 10)),
                           runner.model, runner.betas, transfer, sigma_y=sigma_y,
                           prediction_type="v")
    out = inverse_data_transform(xs[-1])
    out = Fn.interpolate(out, sg_clean.shape[-2:], mode="bilinear", align_corners=False)
    restored.append(out[0, 0].numpy() * (hi - lo) + lo)
restored = np.stack(restored)
print(f"    restored {restored.shape}, finite={np.isfinite(restored).all()}, "
      f"range [{restored.min():.4g}, {restored.max():.4g}]")
assert np.isfinite(restored).all(), "DDRM produced non-finite output"

# --- stage 4: scoring ------------------------------------------------------------------
velax = ((hdr["CRVAL3"] + (np.array(CHANNELS) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0)
au_per_px = abs(hdr["CDELT1"]) * 3600.0 * hdr.get("DIST_PC", 140.0)
m0, _, _ = generate_moment_maps("", data_velax=(sg_clean.astype(np.float64), velax))
mask = signal_mask(m0, frac=0.02)
print(f"\n[4] scoring: mask {mask.sum()} px")

rows = {}
for tag, cube in (("clean", sg_clean), ("dirty", sg_dirty), ("DDRM", restored)):
    v0, _ = quadratic_moment1(cube.astype(np.float64), velax)
    m1 = v0 / 1000.0
    init = None if tag == "clean" else {k: rows["clean"]["geom"][k]
                                        for k in ("cx", "cy", "pa_deg", "incl_deg")}
    g = fit_keplerian(m1, mask, au_per_px, init=init)
    rows[tag] = dict(m1=m1, geom=g, resid=wiggle_residual(m1, g))

for tag in ("dirty", "DDRM"):
    ok = np.isfinite(rows["clean"]["resid"][mask]) & np.isfinite(rows[tag]["resid"][mask])
    r = float(np.corrcoef(rows["clean"]["resid"][mask][ok], rows[tag]["resid"][mask][ok])[0, 1])
    print(f"    {tag:6s} residual r = {r:+.4f}  (mstar {rows[tag]['geom']['mstar_msun']:.3f})")

print("\n" + "=" * 70)
print("PLUMBING OK -- every stage ran on real data and produced finite output.")
print("The restoration numbers are meaningless at this scale; the notebook is safe to run.")
print("=" * 70)
