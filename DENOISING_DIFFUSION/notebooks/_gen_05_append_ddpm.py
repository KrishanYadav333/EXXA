#!/usr/bin/env python
"""
Appends a "Part 2 — Conditional DDPM" section to the ROOT, Kaggle-linked
notebook (../../05-unet-line-emission.ipynb relative to this file), so the
DDPM comparison run happens inside the ONE notebook that supports Kaggle's
GitHub push/pull sync (root 05, not the standalone notebooks/06-ddpm-*.ipynb,
which the Kaggle kernel has no way to pull).

Reuses everything from Part 1 (U-Net) already in scope by the time Part 2
runs: device/SEED/ON_KAGGLE/DATA_DIR, train_cubes/val_cubes/holdout_cubes,
SUBTRACT_CONTINUUM/CONTINUUM_N, TARGET_SIZE/N_SAMPLES, and the already-built
continuum-subtracted train_ds/val_ds (just wraps them in StackedPairDataset
instead of rebuilding from FITS). Distinct variable/file names throughout
(*_DDPM, ddpm_*) so nothing Part 1 defined gets clobbered.

Idempotent: if a "Conditional DDPM" markdown cell already exists, re-running
this script strips outputs (for the <1MB Kaggle fetch limit) but does not
duplicate the append.
"""
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "05-unet-line-emission.ipynb"))


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


ddpm_cells = []

ddpm_cells.append(md(
"""---

# Part 2 — Conditional DDPM (comparison vs Part 1 U-Net)

Same data, same cube split, same continuum subtraction, same all-5-holdout evaluation as Part 1 —
but the denoiser is a **conditional DDPM** (`src/training/diffusion.py`): predicts the noise added
to the *clean* channel, conditioned on `[dirty, x_t]`, denoised via **DDIM** (25 steps).

Reuses `train_ds`/`val_ds`/`train_cubes`/`val_cubes`/`holdout_cubes`/`device`/`SEED`/`TARGET_SIZE`/
`SUBTRACT_CONTINUUM`/`CONTINUUM_N` from Part 1 above — run Part 1's cells (0 through 2) first.

**Heads-up:** DDIM sampling a full 201-channel cube is much slower than a U-Net forward pass —
Section 19's 5-cube holdout eval takes noticeably longer than Part 1's Section 11."""))

ddpm_cells.append(md("## 12. Imports + DDPM-specific config"))
ddpm_cells.append(code(
"""from src.data.stacked_pair import StackedPairDataset
from src.models.diffusion_unet import default_diffusion_config
from src.training.diffusion import DenoisingDiffusion

EPOCHS_DDPM      = 30
LR_DDPM          = 2e-4   # eps-prediction; higher than the 2e-5 image-gen default (few steps/epoch, ~350 items)
BATCH_SIZE_DDPM  = 8      # 256px DDPM is heavy; probe below shrinks this on OOM (never resolution)
SAMPLING_STEPS   = 25     # DDIM steps for sampling/evaluation
print('DDPM config: epochs', EPOCHS_DDPM, '| lr', LR_DDPM, '| batch', BATCH_SIZE_DDPM,
      '| DDIM steps', SAMPLING_STEPS)"""))

ddpm_cells.append(md(
"""## 13. Datasets — stack Part 1's continuum-subtracted pairs for the DDPM

`train_ds`/`val_ds` (Part 1, Section 3) already yield continuum-subtracted, shared-dirty-scale
`(dirty, clean)` pairs. `StackedPairDataset` just stacks each pair into the `(2,H,W)` tensor the
DDPM trainer expects — no FITS re-read, no re-normalization."""))
ddpm_cells.append(code(
"""ddpm_train_ds = StackedPairDataset(train_ds)
ddpm_val_ds   = StackedPairDataset(val_ds)
print('DDPM train items:', len(ddpm_train_ds), '| val items:', len(ddpm_val_ds))

def make_ddpm_loaders(bs):
    nw = 4 if ON_KAGGLE else 0
    return (DataLoader(ddpm_train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True),
            DataLoader(ddpm_val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True))
ddpm_train_loader, ddpm_val_loader = make_ddpm_loaders(BATCH_SIZE_DDPM)"""))

ddpm_cells.append(md(
"""## 14. Model — conditional DDPM U-Net at 256px

`default_diffusion_config` targets 64px by default. For 256px, extend `ch_mult` to `[1,2,2,2,4]`
(5 levels: 256->128->64->32->16) so self-attention at 16x16 actually fires — with the default
`[1,2,2,4]` the network only reaches 32x32 and `attn_resolutions=[16]` would never trigger."""))
ddpm_cells.append(code(
"""cfg = default_diffusion_config(image_size=TARGET_SIZE)
cfg.model.ch_mult = [1, 2, 2, 2, 4]     # 5 levels -> bottleneck at 16x16 (attn@16 fires)
cfg.model.ema_rate = 0.999
print('config:', 'ch', cfg.model.ch, '| ch_mult', cfg.model.ch_mult,
      '| attn', cfg.model.attn_resolutions, '| T', cfg.diffusion.num_diffusion_timesteps)

CKPT_DDPM = '../results/checkpoints/ddpm_line_emission_continuum_best.pth.tar'
os.makedirs(os.path.dirname(CKPT_DDPM), exist_ok=True)

# quick param count + forward-shape sanity (conditional input = 2 channels [cond, x_t])
_probe = DenoisingDiffusion(config=cfg, device=str(device), lr=LR_DDPM, checkpoint_path=CKPT_DDPM,
                            data_parallel=False)
print('DDPM params:', f'{sum(p.numel() for p in _probe._core.parameters()):,}')
with torch.no_grad():
    _x = torch.randn(2, 2, TARGET_SIZE, TARGET_SIZE, device=device)
    _t = torch.randint(0, cfg.diffusion.num_diffusion_timesteps, (2,), device=device).float()
    _o = _probe._core(_x, _t)
print('forward (2,2,%d,%d) -> %s' % (TARGET_SIZE, TARGET_SIZE, tuple(_o.shape)))
del _probe; torch.cuda.empty_cache()"""))

ddpm_cells.append(md(
"""## 15. Train — 30 epochs, conditional eps-loss, DDIM eval

`DenoisingDiffusion` auto-wraps in `nn.DataParallel` on multi-GPU (Kaggle T4x2), keeps EMA + saves
the best (lowest val-loss) checkpoint from the unwrapped model. An OOM-safe probe shrinks the batch
(never the resolution) if 256px doesn't fit."""))
ddpm_cells.append(code(
"""# OOM-safe batch probe (256px fixed; shrink batch only)
def probe_ddpm_batch(bs0):
    bs = bs0
    while bs >= 1:
        try:
            dd = DenoisingDiffusion(config=cfg, device=str(device), lr=LR_DDPM, checkpoint_path=CKPT_DDPM)
            tl, _ = make_ddpm_loaders(bs)
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

BATCH_DDPM_USED = probe_ddpm_batch(BATCH_SIZE_DDPM)
gpu_note = f'DataParallel x{N_GPU} (~{max(1,BATCH_DDPM_USED // max(N_GPU,1))}/GPU)' if N_GPU > 1 else 'single GPU'
print(f'BATCH SIZE USED: {BATCH_DDPM_USED} at {TARGET_SIZE}x{TARGET_SIZE}  [{gpu_note}]')
ddpm_train_loader, ddpm_val_loader = make_ddpm_loaders(BATCH_DDPM_USED)"""))

ddpm_cells.append(code(
"""diffusion = DenoisingDiffusion(config=cfg, device=str(device), lr=LR_DDPM, checkpoint_path=CKPT_DDPM)
t0 = time.time()
history = diffusion.train(ddpm_train_loader, ddpm_val_loader, n_epochs=EPOCHS_DDPM, log_every=1)
print(f'\\ntrained {EPOCHS_DDPM} epochs in {time.time()-t0:.0f}s')
print('best val loss:', round(diffusion.best_val_loss, 4), '-> checkpoint', CKPT_DDPM)"""))

ddpm_cells.append(md("## 16. Loss curve (DDPM)"))
ddpm_cells.append(code(
"""tr_hist = history['train_losses']; va_hist = history['val_losses']
plt.figure(figsize=(8,5))
plt.plot(range(1,len(tr_hist)+1), tr_hist, marker='o', ms=3, label='train')
if va_hist:
    plt.plot(range(1,len(va_hist)+1), va_hist, marker='s', ms=3, label='val')
    be = int(np.argmin(va_hist)) + 1
    plt.axvline(be, color='gray', ls=':'); plt.scatter([be],[min(va_hist)], color='#E8715A', zorder=5)
plt.xlabel('epoch'); plt.ylabel('DDPM eps-loss')
plt.title(f'Line-emission DDPM ({TARGET_SIZE}px, batch {BATCH_DDPM_USED})')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
os.makedirs('../results', exist_ok=True)
plt.savefig('../results/ddpm_line_emission_loss.png', dpi=140); plt.show()"""))

ddpm_cells.append(md(
"""## 17. Validation metrics — PSNR / SSIM / MSE (DDIM-sampled)

`evaluate` DDIM-samples each validation channel from its dirty condition and scores against clean.
Uses EMA weights. Slower than a U-Net forward pass — this is the sampler running per channel."""))
ddpm_cells.append(code(
"""val_metrics_ddpm = diffusion.evaluate(ddpm_val_loader, sampling_timesteps=SAMPLING_STEPS, use_ema=True)
print('Validation ({} channels):  PSNR {:.4f} dB | SSIM {:.4f} | MSE {:.6f}'.format(
    val_metrics_ddpm['n'], val_metrics_ddpm['psnr'], val_metrics_ddpm['ssim'], val_metrics_ddpm['mse']))"""))

ddpm_cells.append(md("## 18. Visualize 5 random validation channels — dirty | DDPM denoised | clean"))
ddpm_cells.append(code(
"""import random
idxs = random.Random(SEED).sample(range(len(val_ds)), 5)
fig, ax = plt.subplots(5, 3, figsize=(10, 16))
cols = ['dirty', 'DDPM denoised', 'clean GT']
for k, t in enumerate(cols): ax[0,k].set_title(t, fontweight='bold')
for r, ix in enumerate(idxs):
    d, c = val_ds[ix]                                     # each (1,H,W), [0,1] shared scale
    pred = diffusion.sample(d[None], sampling_timesteps=SAMPLING_STEPS, use_ema=True)[0,0].cpu().numpy()
    ci, ch = val_ds.index[ix]
    for col, im in enumerate([d[0].numpy(), pred, c[0].numpy()]):
        ax[r,col].imshow(np.clip(im,0,1), cmap='inferno'); ax[r,col].axis('off')
    ax[r,0].set_ylabel(f'{val_ds.cube_paths[ci][2]}\\nch {ch}', fontsize=8)
fig.suptitle('Line-emission DDPM -- validation channels', fontweight='bold', y=0.995)
plt.tight_layout()
os.makedirs('../experiments', exist_ok=True)
plt.savefig('../experiments/line_emission_ddpm_comparison.png', dpi=140); plt.show()
print('saved -> experiments/line_emission_ddpm_comparison.png')"""))

ddpm_cells.append(md(
"""## 19. All-5-Holdout Evaluation (Continuum-Subtracted, DDPM)

Same evaluation as Part 1's Section 11, denoising uses DDIM sampling instead of a U-Net forward
pass. Directly comparable to Part 1's V12 numbers (M0 +69.8%+/-15.2% / M1 +17.5%+/-7.8% /
M2 +20.1%+/-14.3%). Output: `results/moment_map_holdout_summary_ddpm.csv` + bar chart.

**Note:** DDIM sampling per channel makes this the slowest cell by far (201 channels x 5 cubes x
25 DDIM steps). `DENOISE_BS_DDPM` is small to stay within memory at 256px."""))
ddpm_cells.append(code(
"""import csv
import matplotlib.ticker as mticker
import bettermoments as bm
from astropy.io import fits
from src.evaluation.moment_maps import generate_moment_maps
from src.data.fits_cube_dataset import continuum_of

OUT_DIR         = '../results'
DENOISE_BS_DDPM = 8            # channels per sampling batch (256px DDIM is memory-heavy)
os.makedirs(OUT_DIR, exist_ok=True)

# fresh trainer <- best checkpoint (robust across restarts); EMA weights used for sampling
eval_diff = DenoisingDiffusion(config=cfg, device=str(device), lr=LR_DDPM, checkpoint_path=CKPT_DDPM)
eval_diff.load_checkpoint(CKPT_DDPM)
print('DDPM checkpoint loaded: epoch', eval_diff.start_epoch,
      '| best_val', round(float(eval_diff.best_val_loss), 4))
print('Evaluating', len(holdout_cubes), 'held-out cubes (DDIM sampling -- slow)...')

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
    for s in range(0, C, DENOISE_BS_DDPM):
        t   = torch.from_numpy(norm[s:s+DENOISE_BS_DDPM])[:, None].float()          # (b,1,600,600) [0,1]
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

ddpm_cells.append(md(
"""## 20. Persist DDPM checkpoint to /kaggle/working (survives session end)

Copies the best DDPM checkpoint to the **top level** of `/kaggle/working` (outside the git clone)
so `kaggle kernels output <slug>` retrieves it directly. Save the notebook version afterward so the
`.pth.tar` is a downloadable output artifact."""))
ddpm_cells.append(code(
"""import shutil
if ON_KAGGLE and os.path.exists(CKPT_DDPM):
    dst = '/kaggle/working/ddpm_line_emission_continuum_best.pth.tar'
    shutil.copy2(CKPT_DDPM, dst)
    print('checkpoint persisted ->', dst, '({:.1f} MB)'.format(os.path.getsize(dst)/1e6))
    src_csv = '../results/moment_map_holdout_summary_ddpm.csv'
    if os.path.exists(src_csv):
        shutil.copy2(src_csv, '/kaggle/working/moment_map_holdout_summary_ddpm.csv')
        print('summary CSV persisted -> /kaggle/working/moment_map_holdout_summary_ddpm.csv')
else:
    print('not on Kaggle (or checkpoint missing) -- skip persist. CKPT exists:', os.path.exists(CKPT_DDPM))"""))


def strip_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return nb


def main():
    with open(ROOT, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb = strip_outputs(nb)

    already = any(
        c["cell_type"] == "markdown" and "Conditional DDPM" in "".join(c["source"])
        for c in nb["cells"]
    )
    if already:
        print("DDPM section already present -- outputs stripped, no cells appended.")
    else:
        nb["cells"].extend(ddpm_cells)
        print(f"appended {len(ddpm_cells)} cells (Part 2 -- Conditional DDPM).")

    with open(ROOT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    size_kb = os.path.getsize(ROOT) / 1024
    print(f"wrote {ROOT} -- {len(nb['cells'])} cells, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
