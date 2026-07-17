#!/usr/bin/env python
"""
Generator for notebooks/06-ddpm-line-emission.ipynb.

Mirrors 05-unet-line-emission's structure (same bootstrap, same cube split,
same continuum-subtracted data, same Section-11 all-5-holdout moment-map eval)
but swaps the single-pass DenoisingUNet for the conditional DDPM in
src/training/diffusion.py (DDIM sampling). Run this file to (re)emit the .ipynb.
"""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}

cells = []

cells.append(md(
"""# Line Emission Denoising — Conditional DDPM

Companion to `05-unet-line-emission` (single-pass U-Net). Same data, same split, same
continuum-subtraction, same Section-11 evaluation — but the denoiser is a **conditional DDPM**
(`src/training/diffusion.py`): it predicts the noise added to the *clean* channel, conditioned on
the *dirty* channel `[x_cond, x_t]`, and denoises via **DDIM** sampling.

**Why:** the U-Net is one forward pass; the DDPM is an iterative generative denoiser. This notebook
produces PSNR/SSIM/MSE and M0/M1/M2 improvement (mean±std over 5 held-out cubes) directly comparable
to the V12 U-Net numbers.

**Data (identical to 05):** line-emission FITS cubes (201, 600, 600), cube-level split (3 RunID groups
held out), continuum-subtracted (mean of first/last CONTINUUM_N line-free channels), each channel
256×256, shared dirty-scale [0,1] normalization.

### Kaggle setup
GPU on, Internet on, Add Input → your line-emission Dataset (FITS cubes). Bootstrap finds it under
/kaggle/input/. **Heads-up:** DDIM sampling a full 201-channel cube is much slower than a U-Net forward
pass, so Section 11 takes noticeably longer here.
"""))

cells.append(md("## 0. Bootstrap (clone repo for src/, locate data)"))
cells.append(code(
"""import os, sys, subprocess, glob

ON_KAGGLE = os.path.exists('/kaggle')
if ON_KAGGLE:
    REPO='/kaggle/working/EXXA'; PKG=os.path.join(REPO,'DENOISING_DIFFUSION')
    if not os.path.exists(REPO):
        subprocess.run(['git','clone','--branch','line-emission','--depth','1','https://github.com/KrishanYadav333/EXXA.git',REPO], check=True)
    else:
        subprocess.run(['git','-C',REPO,'fetch','origin','line-emission'], check=True)
        subprocess.run(['git','-C',REPO,'reset','--hard','origin/line-emission'], check=True)
    # pytorch-msssim for SSIM; bettermoments for moment-map evaluation (Sec. 9+)
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','pytorch-msssim','bettermoments'], check=True)
    os.chdir(os.path.join(PKG,'notebooks'));  sys.path.insert(0, PKG)
    hits = glob.glob('/kaggle/input/**/*_dirty.fits', recursive=True)
    DATA_DIR = os.path.dirname(os.path.dirname(hits[0])) if hits else None
else:
    if os.path.basename(os.getcwd()) != 'notebooks' and os.path.isdir('notebooks'): os.chdir('notebooks')
    sys.path.insert(0, os.path.abspath('..'))
    DATA_DIR = '../data/Line Emission Data'
print('cwd:', os.getcwd(), '| DATA_DIR:', DATA_DIR)"""))

cells.append(md(
"""## 0b. Pull latest code (re-run anytime — NO kernel restart needed)

After pushing to the `line-emission` branch, re-run this one cell to fetch + hot-reload `src/`."""))
cells.append(code(
"""import os, sys, subprocess
ON_KAGGLE = os.path.exists('/kaggle')
REPO = '/kaggle/working/EXXA'
if ON_KAGGLE and os.path.exists(REPO):
    subprocess.run(['git', '-C', REPO, 'fetch', 'origin', 'line-emission'], check=True)
    subprocess.run(['git', '-C', REPO, 'reset', '--hard', 'origin/line-emission'], check=True)
    print(subprocess.run(['git','-C',REPO,'log','--oneline','-1'],capture_output=True,text=True).stdout.strip())
elif ON_KAGGLE:
    print('repo not cloned yet — run the bootstrap cell (0.) first')
for _m in [m for m in list(sys.modules) if m == 'src' or m.startswith('src.')]:
    del sys.modules[_m]
print('src.* cleared from module cache — now re-run the imports cell below.')"""))

cells.append(md("## 1. Imports, device, config"))
cells.append(code(
"""import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.cube_split import split_cubes
from src.data.fits_cube_dataset import FITSChannelDataset, continuum_of
from src.data.stacked_pair import StackedPairDataset
from src.models.diffusion_unet import default_diffusion_config
from src.training.diffusion import DenoisingDiffusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU  = torch.cuda.device_count()
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print('device:', device, '| GPUs:', N_GPU,
      '->', [torch.cuda.get_device_name(i) for i in range(N_GPU)] if N_GPU else 'cpu')

TARGET_SIZE   = 256     # match the U-Net for an apples-to-apples comparison
N_SAMPLES     = 50      # channels per cube
EPOCHS        = 30
LR            = 2e-4    # DDPM eps-prediction; higher than the 2e-5 image-gen default since
                        # we have few steps/epoch (~350 items). Tune if loss stalls.
BATCH_SIZE    = 8       # 256px DDPM is heavy; probe shrinks this on OOM (never resolution)
SAMPLING_STEPS = 25     # DDIM steps for sampling/evaluation
SUBTRACT_CONTINUUM = True
CONTINUUM_N        = 5"""))

cells.append(md("## 2. Cube-level split (3 RunID groups held out for inference only)"))
cells.append(code(
"""train_cubes, val_cubes, holdout_cubes = split_cubes(data_dir=DATA_DIR, n_holdout=3,
                                                     val_fraction=0.2, seed=SEED)"""))

cells.append(md(
"""## 3. Datasets + DataLoaders — continuum-subtracted, stacked for the conditional DDPM

`FITSChannelDataset` yields `(dirty, clean)`; the DDPM trainer wants a single `(2,H,W)` stacked
tensor `[dirty, clean]`. `StackedPairDataset` is a thin wrapper that stacks the pair (all the
continuum-subtraction + shared-scale normalization still happen inside the wrapped dataset)."""))
cells.append(code(
"""base_train = FITSChannelDataset(train_cubes, n_samples=N_SAMPLES, target_size=TARGET_SIZE, seed=SEED,
                                subtract_continuum=SUBTRACT_CONTINUUM, continuum_n=CONTINUUM_N)
base_val   = FITSChannelDataset(val_cubes,   n_samples=N_SAMPLES, target_size=TARGET_SIZE, seed=SEED,
                                subtract_continuum=SUBTRACT_CONTINUUM, continuum_n=CONTINUUM_N)
train_ds = StackedPairDataset(base_train)   # -> ((2,H,W), 0)
val_ds   = StackedPairDataset(base_val)
print('train items:', len(train_ds), '| val items:', len(val_ds))

def make_loaders(bs):
    nw = 4 if ON_KAGGLE else 0
    return (DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True),
            DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True))
train_loader, val_loader = make_loaders(BATCH_SIZE)"""))

cells.append(md(
"""## 4. Model — conditional DDPM U-Net at 256px

`default_diffusion_config` targets 64px (old 4 GB card). For 256px we extend `ch_mult` to
`[1,2,2,2,4]` (5 levels: 256→128→64→32→16) so self-attention at 16×16 actually fires — with the
default `[1,2,2,4]` the network only reaches 32×32 and `attn_resolutions=[16]` would never trigger."""))
cells.append(code(
"""cfg = default_diffusion_config(image_size=TARGET_SIZE)
cfg.model.ch_mult = [1, 2, 2, 2, 4]     # 5 levels -> bottleneck at 16x16 (attn@16 fires)
cfg.model.ema_rate = 0.999
print('config:', 'ch', cfg.model.ch, '| ch_mult', cfg.model.ch_mult,
      '| attn', cfg.model.attn_resolutions, '| T', cfg.diffusion.num_diffusion_timesteps)

CKPT = '../results/checkpoints/ddpm_line_emission_continuum_best.pth.tar'
os.makedirs(os.path.dirname(CKPT), exist_ok=True)

# quick param count + forward-shape sanity (conditional input = 2 channels [cond, x_t])
_probe = DenoisingDiffusion(config=cfg, device=str(device), lr=LR, checkpoint_path=CKPT,
                            data_parallel=False)
print('DDPM params:', f'{sum(p.numel() for p in _probe._core.parameters()):,}')
with torch.no_grad():
    _x = torch.randn(2, 2, TARGET_SIZE, TARGET_SIZE, device=device)
    _t = torch.randint(0, cfg.diffusion.num_diffusion_timesteps, (2,), device=device).float()
    _o = _probe._core(_x, _t)
print('forward (2,2,%d,%d) -> %s' % (TARGET_SIZE, TARGET_SIZE, tuple(_o.shape)))
del _probe; torch.cuda.empty_cache()"""))

cells.append(md(
"""## 5. Train — 30 epochs, conditional eps-loss, DDIM eval

`DenoisingDiffusion` auto-wraps in `nn.DataParallel` on multi-GPU (Kaggle T4×2), keeps EMA + saves
the best (lowest val-loss) checkpoint from the unwrapped model. An OOM-safe probe shrinks the batch
(never the resolution) if 256px doesn't fit."""))
cells.append(code(
"""# OOM-safe batch probe (256px fixed; shrink batch only)
def probe_batch(bs0):
    bs = bs0
    while bs >= 1:
        try:
            dd = DenoisingDiffusion(config=cfg, device=str(device), lr=LR, checkpoint_path=CKPT)
            tl, _ = make_loaders(bs)
            x, _ = next(iter(tl)); x = x.to(device)
            from src.training.diffusion import data_transform, noise_estimation_loss
            x = data_transform(x)
            e = torch.randn_like(x[:, 1:, :, :])
            t = torch.randint(0, dd.num_timesteps, (x.size(0),), device=device)
            loss = noise_estimation_loss(dd.model, x, t, e, dd.betas)
            loss.backward()
            del dd, loss, x, e; torch.cuda.empty_cache()
            return bs
        except RuntimeError as ex:
            if 'out of memory' not in str(ex).lower(): raise
            torch.cuda.empty_cache(); bs //= 2
            print(f'[OOM] reducing batch -> {bs} (image stays {TARGET_SIZE})')
    raise RuntimeError('does not fit even at batch 1')

BATCH_USED = probe_batch(BATCH_SIZE)
gpu_note = f'DataParallel x{N_GPU} (~{max(1,BATCH_USED // max(N_GPU,1))}/GPU)' if N_GPU > 1 else 'single GPU'
print(f'BATCH SIZE USED: {BATCH_USED} at {TARGET_SIZE}x{TARGET_SIZE}  [{gpu_note}]')
train_loader, val_loader = make_loaders(BATCH_USED)"""))

cells.append(code(
"""diffusion = DenoisingDiffusion(config=cfg, device=str(device), lr=LR, checkpoint_path=CKPT)
t0 = time.time()
history = diffusion.train(train_loader, val_loader, n_epochs=EPOCHS, log_every=1)
print(f'\\ntrained {EPOCHS} epochs in {time.time()-t0:.0f}s')
print('best val loss:', round(diffusion.best_val_loss, 4), '-> checkpoint', CKPT)"""))

cells.append(md("## 6. Loss curve"))
cells.append(code(
"""tr_hist = history['train_losses']; va_hist = history['val_losses']
plt.figure(figsize=(8,5))
plt.plot(range(1,len(tr_hist)+1), tr_hist, marker='o', ms=3, label='train')
if va_hist:
    plt.plot(range(1,len(va_hist)+1), va_hist, marker='s', ms=3, label='val')
    be = int(np.argmin(va_hist)) + 1
    plt.axvline(be, color='gray', ls=':'); plt.scatter([be],[min(va_hist)], color='#E8715A', zorder=5)
plt.xlabel('epoch'); plt.ylabel('DDPM eps-loss')
plt.title(f'Line-emission DDPM ({TARGET_SIZE}px, batch {BATCH_USED})')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
os.makedirs('../results', exist_ok=True)
plt.savefig('../results/ddpm_line_emission_loss.png', dpi=140); plt.show()"""))

cells.append(md(
"""## 7. Validation metrics — PSNR / SSIM / MSE (DDIM-sampled)

`evaluate` DDIM-samples each validation channel from its dirty condition and scores against clean.
Uses EMA weights. Slower than a U-Net forward pass — this is the sampler running per channel."""))
cells.append(code(
"""val_metrics = diffusion.evaluate(val_loader, sampling_timesteps=SAMPLING_STEPS, use_ema=True)
print('Validation ({} channels):  PSNR {:.4f} dB | SSIM {:.4f} | MSE {:.6f}'.format(
    val_metrics['n'], val_metrics['psnr'], val_metrics['ssim'], val_metrics['mse']))"""))

cells.append(md("## 8. Visualize 5 random validation channels — dirty | DDPM denoised | clean"))
cells.append(code(
"""import random
idxs = random.Random(SEED).sample(range(len(base_val)), 5)
fig, ax = plt.subplots(5, 3, figsize=(10, 16))
cols = ['dirty', 'DDPM denoised', 'clean GT']
for k, t in enumerate(cols): ax[0,k].set_title(t, fontweight='bold')
for r, ix in enumerate(idxs):
    d, c = base_val[ix]                                   # each (1,H,W), [0,1] shared scale
    pred = diffusion.sample(d[None], sampling_timesteps=SAMPLING_STEPS, use_ema=True)[0,0].cpu().numpy()
    ci, ch = base_val.index[ix]
    for col, im in enumerate([d[0].numpy(), pred, c[0].numpy()]):
        ax[r,col].imshow(np.clip(im,0,1), cmap='inferno'); ax[r,col].axis('off')
    ax[r,0].set_ylabel(f'{base_val.cube_paths[ci][2]}\\nch {ch}', fontsize=8)
fig.suptitle('Line-emission DDPM — validation channels', fontweight='bold', y=0.995)
plt.tight_layout()
os.makedirs('../experiments', exist_ok=True)
plt.savefig('../experiments/line_emission_ddpm_comparison.png', dpi=140); plt.show()
print('saved -> experiments/line_emission_ddpm_comparison.png')"""))

cells.append(md(
"""## 9. All-5-Holdout Evaluation (Continuum-Subtracted, DDPM)

Same evaluation as `05` Section 11, but denoising uses DDIM sampling instead of a U-Net forward pass.
Load the best DDPM checkpoint, denoise every channel of each held-out cube, continuum-subtract the
clean/dirty references identically, and report M0/M1/M2 improvement (mean±std over the 5 cubes).

Output: `results/moment_map_holdout_summary_ddpm.csv` + summary bar chart.

**Note:** DDIM sampling per channel makes this the slowest cell by far (201 channels × 5 cubes ×
25 DDIM steps). `DENOISE_BS` is small to stay within memory at 256px."""))
cells.append(code(
"""import csv
import matplotlib.ticker as mticker
import bettermoments as bm
from astropy.io import fits
from src.evaluation.moment_maps import generate_moment_maps

OUT_DIR    = '../results'
DENOISE_BS = 8                 # channels per sampling batch (256px DDIM is memory-heavy)
os.makedirs(OUT_DIR, exist_ok=True)

# fresh trainer <- best checkpoint (robust across restarts); EMA weights used for sampling
eval_diff = DenoisingDiffusion(config=cfg, device=str(device), lr=LR, checkpoint_path=CKPT)
eval_diff.load_checkpoint(CKPT)
print('DDPM checkpoint loaded: epoch', eval_diff.start_epoch,
      '| best_val', round(float(eval_diff.best_val_loss), 4))
print('Evaluating', len(holdout_cubes), 'held-out cubes (DDIM sampling — slow)...')

def mdiff(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.nanmean(np.abs(a[mask] - b[mask])))

def denoise_cube_ddpm(ho_entry):
    with fits.open(ho_entry['dirty'], memmap=False) as hdul:
        dirty_raw = np.ascontiguousarray(hdul[0].data).astype(np.float32)
        hdr = hdul[0].header.copy()
    C, H, W = dirty_raw.shape
    dcont = continuum_of(dirty_raw, CONTINUUM_N)
    dirty_csub = dirty_raw - dcont[None, :, :]
    los  = dirty_csub.reshape(C, -1).min(axis=1)
    his  = dirty_csub.reshape(C, -1).max(axis=1)
    rngs = his - los
    norm = np.zeros_like(dirty_csub)
    nz = rngs > 0
    norm[nz] = (dirty_csub[nz] - los[nz, None, None]) / rngs[nz, None, None]

    denoised_csub = np.empty_like(dirty_csub)
    import torch.nn.functional as F
    for s in range(0, C, DENOISE_BS):
        t   = torch.from_numpy(norm[s:s+DENOISE_BS])[:, None].float()          # (b,1,600,600) [0,1]
        t256 = F.interpolate(t, (TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False)
        out  = eval_diff.sample(t256, sampling_timesteps=SAMPLING_STEPS, use_ema=True)  # (b,1,256,256) [0,1]
        out600 = F.interpolate(out.cpu(), (H, W), mode='bilinear', align_corners=False)[:, 0].numpy()
        for k in range(out600.shape[0]):
            ch = s + k
            denoised_csub[ch] = (out600[k] * rngs[ch] + los[ch]) if rngs[ch] > 0 else \\
                                np.full((H, W), los[ch], np.float32)
    out_path = os.path.join(OUT_DIR, 'denoised_ddpm_' + ho_entry['folder'] + '.fits')
    fits.writeto(out_path, denoised_csub.astype(np.float32), header=hdr, overwrite=True)
    return out_path, dirty_csub

rows_data = []
col_w = 30
hdr_line = '{:<{w}} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
    'cube', 'dirty_M0', 'imp_M0%', 'dirty_M1', 'imp_M1%', 'dirty_M2', 'imp_M2%', w=col_w)
print('\\n' + hdr_line); print('-' * len(hdr_line))

for ho in holdout_cubes:
    print('  denoising', ho['folder'], '...', end=' ', flush=True)
    out_fits_ho, dirty_csub = denoise_cube_ddpm(ho)
    print('done')
    with fits.open(ho['clean'], memmap=False) as h:
        clean_raw = np.ascontiguousarray(h[0].data).astype(np.float32)
    ccont = continuum_of(clean_raw, CONTINUUM_N)
    clean_csub = clean_raw - ccont[None]
    _, velax = bm.load_cube(ho['dirty'])
    c0, c1, c2 = generate_moment_maps(None, data_velax=(clean_csub, velax))
    d0, d1, d2 = generate_moment_maps(None, data_velax=(dirty_csub, velax))
    n0, n1, n2 = generate_moment_maps(out_fits_ho)
    row = {'cube': ho['folder']}
    for nm, cl, di, no in [('M0', c0, d0, n0), ('M1', c1, d1, n1), ('M2', c2, d2, n2)]:
        dd = mdiff(cl, di); nn = mdiff(cl, no)
        imp = 100.0 * (1 - nn / dd) if dd > 0 else float('nan')
        row['dirty_' + nm] = round(dd, 6); row['imp_' + nm] = round(imp, 2)
    rows_data.append(row)
    print('{:<{w}} {:>10.4g} {:>9.1f}% {:>10.4g} {:>9.1f}% {:>10.4g} {:>9.1f}%'.format(
        ho['folder'], row['dirty_M0'], row['imp_M0'], row['dirty_M1'], row['imp_M1'],
        row['dirty_M2'], row['imp_M2'], w=col_w))

moments = ['M0', 'M1', 'M2']
imps  = {m: [r['imp_'+m] for r in rows_data if r['imp_'+m] == r['imp_'+m]] for m in moments}
means = {m: float(np.mean(imps[m])) for m in moments}
stds  = {m: float(np.std(imps[m], ddof=1)) if len(imps[m]) > 1 else 0.0 for m in moments}
print('\\n' + '=' * len(hdr_line))
print('DDPM SUMMARY (n=' + str(len(rows_data)) + ' cubes):')
for m in moments:
    print('  ' + m + ': mean {:+.1f}%  std {:.1f}%  (n={})'.format(means[m], stds[m], len(imps[m])))

csv_path = os.path.join(OUT_DIR, 'moment_map_holdout_summary_ddpm.csv')
fieldnames = ['cube', 'dirty_M0', 'imp_M0', 'dirty_M1', 'imp_M1', 'dirty_M2', 'imp_M2']
with open(csv_path, 'w', newline='') as cf:
    w = csv.DictWriter(cf, fieldnames=fieldnames); w.writeheader()
    for r in rows_data: w.writerow(r)
    w.writerow({'cube': 'MEAN', **{'imp_'+m: round(means[m],2) for m in moments},
                **{'dirty_'+m: '' for m in moments}})
    w.writerow({'cube': 'STD',  **{'imp_'+m: round(stds[m],2) for m in moments},
                **{'dirty_'+m: '' for m in moments}})
print('\\nCSV saved ->', csv_path)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(moments))
bar_means = [means[m] for m in moments]; bar_stds = [stds[m] for m in moments]
colors = ['#4E91C7' if v >= 0 else '#E8715A' for v in bar_means]
bars = ax.bar(x, bar_means, yerr=bar_stds, capsize=6, color=colors, alpha=0.85,
              error_kw=dict(elinewidth=1.5, ecolor='#333333'))
ax.axhline(0, color='#333333', lw=0.8, ls='--')
ax.set_xticks(x); ax.set_xticklabels(['Moment 0\\n(intensity)','Moment 1\\n(velocity)','Moment 2\\n(dispersion)'])
ax.set_ylabel('Improvement over dirty (%)')
ax.set_title('DDPM: moment-map improvement across 5 held-out cubes\\n(positive = denoised closer to clean than dirty)')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
for bar, mv, sv in zip(bars, bar_means, bar_stds):
    ht = bar.get_height(); va = 'bottom' if ht >= 0 else 'top'
    ax.text(bar.get_x()+bar.get_width()/2, ht + (sv+2)*(1 if ht>=0 else -1),
            '{:+.1f}% +/-{:.1f}%'.format(mv, sv), ha='center', va=va, fontsize=9)
for i, m in enumerate(moments):
    ax.scatter([i]*len(imps[m]), imps[m], color='k', s=22, zorder=5, alpha=0.7)
ax.grid(axis='y', alpha=0.3); plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'moment_map_holdout_summary_ddpm.png')
plt.savefig(chart_path, dpi=140); plt.show()
print('Chart saved ->', chart_path)"""))

cells.append(md(
"""## 10. Persist checkpoint to /kaggle/working (survives session end)

Copies the best DDPM checkpoint to the **top level** of `/kaggle/working` (outside the git clone) so
`kaggle kernels output <slug>` retrieves it directly — no wading through the repo's `.git`. This is the
fix for the lost-V7-weights problem: after this run, save the notebook version and the `.pth.tar` is a
downloadable output artifact."""))
cells.append(code(
"""import shutil
if ON_KAGGLE and os.path.exists(CKPT):
    dst = '/kaggle/working/ddpm_line_emission_continuum_best.pth.tar'
    shutil.copy2(CKPT, dst)
    print('checkpoint persisted ->', dst, '({:.1f} MB)'.format(os.path.getsize(dst)/1e6))
    # also persist the CSV summary next to it for easy download
    src_csv = '../results/moment_map_holdout_summary_ddpm.csv'
    if os.path.exists(src_csv):
        shutil.copy2(src_csv, '/kaggle/working/moment_map_holdout_summary_ddpm.csv')
        print('summary CSV persisted -> /kaggle/working/moment_map_holdout_summary_ddpm.csv')
else:
    print('not on Kaggle (or checkpoint missing) — skip persist. CKPT exists:', os.path.exists(CKPT))"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "06-ddpm-line-emission.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
