# Local checkpoint store

Every trained model this project has produced, one copy each, 20 files. Gitignored
(`*.pth` is ignored repo-wide) and never committed, so this is the only place they exist
outside Kaggle.

**Retention: keep every checkpoint until GSoC finishes.** Nothing here gets deleted before
then, including arms that were refuted, arms that were superseded, and the seed repeats. A
result can need re-measuring at any point, and the beam arm is the standing example: its
moment scores were wrong for three days and only a surviving checkpoint made the correction
possible.

Files here are **hardlinks**, not copies. Building this folder cost no extra disk, and the
older upload folders were collapsed onto the same inodes rather than deleted, so
`05-upload/`, `08-upload/`, `06/` and `results/checkpoints/` all still resolve and now cost
nothing extra. A checkpoint is only really gone once every path to it is removed, so removing
one path is safe and removing the last one is not.

```
models/
  05-unet/    3 models trained inside notebook 05
  06-ddpm/    5 diffusion models from notebook 06
  07-ddrm/    1 unconditional diffusion prior from notebook 07
  08-seeds/  12 U-Net seed repeats from notebook 08, reused by 05 and the moment tables
  10-sg/      2 U-Nets trained on self-gravitating data, notebook 10
```

All 20 are single-root torch archives, verified. RULES.md #3 exists because a checkpoint
that loses its single top-level directory stops loading, so that property is the thing
worth re-checking after any move:

```bash
python3 - <<'PY'
import glob, zipfile
for f in sorted(glob.glob("DENOISING_DIFFUSION/models/*/*.pth")):
    z = zipfile.ZipFile(f)
    assert len({n.split('/')[0] for n in z.namelist()}) == 1, f
    assert z.testzip() is None, f
print("all ok")
PY
```

## 08-seeds — the 12 core U-Net checkpoints

Trained once in notebook 08 and reused everywhere since. Numbers from that run's
`seed_repeats.csv` (kept beside them here), 08 at commit `1ca611f`, archived under
`results/08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/`.

| config | seed | best epoch | val loss | PSNR | SSIM |
|---|---|---|---|---|---|
| v12 | 42 | 59 | 0.001492 | 38.659 | 0.99343 |
| v12 | 43 | 17 | 0.002018 | 36.671 | 0.99121 |
| v12 | 44 | 21 | 0.001848 | 37.466 | 0.99196 |
| winner | 42 | 21 | 0.001125 | 37.230 | 0.99228 |
| winner | 43 | 23 | 0.001060 | 37.724 | 0.99264 |
| winner | 44 | 23 | 0.001023 | 38.969 | 0.99292 |
| winner_aug | 42 | 26 | 0.001077 | 38.939 | 0.99240 |
| winner_aug | 43 | 46 | 0.000914 | 39.808 | 0.99346 |
| winner_aug | 44 | 29 | 0.001020 | 39.141 | 0.99277 |
| winner_p10 | 42 | 30 | 0.000920 | 38.954 | 0.99352 |
| winner_p10 | 43 | 35 | 0.000927 | 39.034 | 0.99355 |
| winner_p10 | 44 | 54 | 0.000809 | 39.830 | 0.99427 |

`winner_p10` came back only as `.ckpt` from the Kaggle Dataset, never as `.pth`. It is the
same file, renamed on the way out and renamed back here.

## 05-unet — 3 models trained in notebook 05 directly

From `results/05-unet-line-emission/v22_2026-08-17_6f5c798/` (Kaggle Version 22). The `nb05_`
filename prefix these carried on Kaggle is dropped here; the folder already says which
notebook.

| config | seed | best epoch | PSNR |
|---|---|---|---|
| sweep_winner | 49 | 11 | 36.162 |
| winner_beam | 42 | 24 | 38.710 |
| winner_patch | 42 | (not retrained) | 33.956 |

`winner_patch` has no epoch of its own in that run: v19 died before writing its metric row,
so v20 and v22 both scored the stored weights without retraining. Its PSNR is also measured
on the 64px patch view, not the full-image view the other two use, so it belongs in a
different column than the numbers above it (RULES.md #4).

`winner_beam`'s checkpoint is fine. The moment score printed in the v22 notebook is not:
that run scored the arm without feeding the trained beam vector back at inference, and
`UNet.forward` ignores `beam=None` silently, so the conditioning branch was dead. Re-measured
in Kaggle Version 24, M0 goes from −95.7% to +9.6% and M2 changes sign. Corrected figures are
in [`../results/RUNS.md`](../results/RUNS.md), run folder
`results/05-unet-line-emission/v24_2026-08-17_beamfix/`.

**Architecture** for both folders: `base_channels=48, channel_multipliers=(1,2,4,8)` for
winner / winner_aug / winner_p10 / winner_beam / winner_patch; `base_channels=32, (1,2,4)`
for v12. `beam_dim=4` only for `winner_beam`, `0` otherwise. Definition in
[`../src/models/unet.py`](../src/models/unet.py).

## 06-ddpm — 5 conditional diffusion models

From 06 Kaggle Version 13, `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/`.
`ddpm_seed42.pth` is the production one: 60 epochs, v-prediction, cosine schedule, min-SNR
weighting, PSNR 38.180 / SSIM 0.9933. The four `ddpm_sweep_*` are the objective sweep that
picked it. Definition in [`../src/training/diffusion.py`](../src/training/diffusion.py).

## Putting one back on Kaggle

Attach the producing notebook's Output. If it has to travel between the two accounts
instead, it goes as a Dataset and the extension has to change to `.ckpt` first, because
Kaggle unpacks a file it recognises as a zip and `torch.load` will not read the directory
that comes out. That is RULES.md #3, and it is the whole reason half these files arrived
here named `.ckpt`. Existing datasets: `exxa-nb08-checkpoints-v4`,
`exxa-nb05-checkpoints-v19`.

Notebook 05's `_import_nb08` and `_import_prior_nb05` accept `.pth`, `.ckpt`, and
`.pth.tar`, so a restored file does not need renaming back.

## 10-sg — trained on self-gravitating data

Notebook 10 Kaggle Version 1 (push `cc194ef`, code `be616fd`), 2026-09-04. Trained on the
pairs synthesized in `experiments/synthesize_sg_pairs.py`, NOT on the ones Jason shipped,
which differ clean-to-dirty by 0.4-7% RMS and would have taught the identity. Run archived at
[`../results/10-sg-training/v1_2026-09-04_cc194ef/`](../results/10-sg-training/v1_2026-09-04_cc194ef/).

| file | init | best epoch | val loss | PSNR | SSIM | holdout M0 / M1 / M2 |
|---|---|---|---|---|---|---|
| `sg_finetune.pth` | `winner_aug_seed43`, 0.1x LR | 24 | 0.002733 | 30.859 | 0.98343 | -6.5 / **+36.5** / +15.6 |
| `sg_fresh.pth` | random | 25 | 0.003339 | 30.024 | 0.98055 | **+5.0** / +21.8 / **+26.0** |

Both beat the frozen `winner_aug` baseline (-10.3 / -0.6 / -43.6) on M1 and M2, which is the
domain gap closing. `fresh` beats `finetune` on M0 and M2 despite the WORSE validation loss --
one more case of pixel metrics not tracking moment reliability. On one holdout cube and one
seed, so directional only.

Architecture is `winner_aug`'s: `base_channels=48, channel_multipliers=(1,2,4,8)`, `beam_dim=0`.
Both verified single-root and `strict=True`-loadable against that architecture.

These arrived from the Kaggle Output named `.zip`, because a torch checkpoint IS a zip and the
browser labelled it accordingly. They were **renamed, not unpacked** -- unpacking gives the
directory `torch.load` rejects, which is RULES.md #3 in the other direction.
