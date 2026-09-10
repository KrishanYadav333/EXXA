# Local checkpoint store

Every trained model this project has produced, one copy each, 40 files. Gitignored
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
  05-unet/       3 models trained inside notebook 05
  06-ddpm/       5 diffusion models from notebook 06
  07-ddrm/       1 unconditional diffusion prior from notebook 07
  08-seeds/     12 U-Net seed repeats from notebook 08, reused by 05 and the moment tables
  08-kinematic/  4 U-Nets, kinematic_gamma sweep on line emission, notebook 08
  10-sg/         2 U-Nets trained on self-gravitating data, notebook 10
  11-loo/       10 U-Nets, leave-one-out over the five SG disks, notebook 11
  12-spectral/   4 U-Nets, spectral-context sweep on SG training, notebook 12
  best_models/   3 wiggle-confirmed picks + DDRM (kept as a negative result) + untested/
                 (2 best-PSNR checkpoints never scored on the wiggle), all hardlinks
```

All 40 are single-root torch archives, verified. RULES.md #3 exists because a checkpoint
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

## 11-loo — leave-one-out over the five SG disks

Notebook 11 Kaggle Version 2 (author-reported, no push commit to verify against), 2026-09-10.
10 checkpoints: 5 folds x 2 arms (`finetune` from `winner_aug`, `fresh` random init), each
fold holding out one disk entirely. Run archived at
[`../results/11-sg-loo/v2_2026-09-10_b7b140d/`](../results/11-sg-loo/v2_2026-09-10_b7b140d/).

| fold | holdout | finetune epoch/val_loss | fresh epoch/val_loss |
|---|---|---|---|
| 0 | run_9015 | 1 / 0.000956 | 26 / 0.004635 |
| 1 | run_9019 | 24 / 0.002818 | 26 / 0.003176 |
| 2 | run_9025 | 2 / 0.005938 | 12 / 0.007684 |
| 3 | run_9032 | 2 / 0.003101 | 24 / 0.003302 |
| 4 | run_9074 | 7 / 0.002585 | 9 / 0.004713 |

All 10 verified single-root, uncorrupted, and strict-`load_state_dict`-compatible against
`winner_aug`'s architecture; every checkpoint's stored epoch matches the run log's printed
best epoch exactly. Arrived from the Kaggle Output named `.zip` (a torch checkpoint is a zip;
the browser labelled it as such), renamed not unpacked, per RULES.md #3.

Notebook 10's `sg_finetune.pth` / `sg_fresh.pth` (V1) are a DIFFERENT run, single holdout
(`run_9074`) rather than leave-one-out, and are not superseded by these -- both are kept.

## 12-spectral — spectral context on SG training

Notebook 12, run 2026-09-11 (code `07f047f`, no push commit, downloaded manually). Three
`fresh` (random-init) arms, `n_neighbors=k`, differing only in `in_channels` (2k+1). Same
split as notebook 10: train `{9015, 9019, 9032}`, val `9025`, holdout `9074`. Run archived at
`../results/PROGRESS.md` 2026-09-11 (no dedicated results/12-*/ folder yet, see open item).

| k | in_channels | epoch | val_loss | PSNR | holdout M0/M1/M2 | wiggle resid_r |
|---|---|---|---|---|---|---|
| 0 (control) | 1 | 46 | 0.003133 | 30.053 | -26.5 / +4.2 / -44.4 | 0.366 |
| 1 | 3 | 33 | 0.002004 | 32.828 | +29.2 / +31.0 / +62.5 | 0.590 |
| 2 | 5 | 29 | 0.001779 | 33.576 | +61.4 / +31.4 / +67.6 | 0.506 |
| 3 | 7 | 60 | 0.001366 | 35.114 | **+53.0 / +38.8 / +70.1** | **0.681** |

`k=3` scores 0.681 against dirty's own 0.594 on this holdout -- the first SG-trained arm in
this project to EXCEED doing nothing on the wiggle, not just approach it -- while also posting
the best PSNR, M1 and M2 in the SG thread. Not monotonic (`k=2` dips below `k=1`), but the
trend is clearly upward. This is Block 1's recipe: spectral context, `k=3` as the strongest
candidate, `k=1` as the cheaper alternative (22 min vs 34 at a real but smaller wiggle cost).

All three verified single-root, strict-`load_state_dict`-compatible at their respective
`in_channels`, epochs matching the run log exactly. Arrived as `.zip` (a torch checkpoint is
one), renamed not unpacked, RULES.md #3.

## 08-kinematic — kinematic_gamma sweep on line emission, complete at all 4 arms

Notebook 08. `gamma=0/0.1/1` from the run 2026-09-11 (code before the memory-clear fix;
`kin_gamma10` died mid-run that session, a platform kill with no Python traceback in the
training code, not a failure of these three arms). `gamma=10` reran alone after
`f398484` freed GPU memory between arms, completed clean, Kaggle Version 5. `n_neighbors=15,
out_channels=31` (the channel-stack architecture) throughout, on the line-emission dataset,
not SG data.

| gamma | epoch | val_loss | in/out channels |
|---|---|---|---|
| 0.0 | 27 | 0.000721 | 31 / 31 |
| 0.1 | 17 | 0.001721 | 31 / 31 |
| 1.0 | 28 | 0.016901 | 31 / 31 |
| 10.0 | 10 | 0.089597 | 31 / 31 |

`val_loss` is not comparable across gamma: the loss function itself changes weight
(`kinematic_gamma` scales the velocity term added to it), so a higher number at higher gamma
does not mean worse pixel performance, RULES.md #4. Not yet scored on moments or the wiggle.

All four verified single-root, strict-`load_state_dict`-compatible at `in_channels=31,
out_channels=31`, and each checkpoint's own `kinematic_gamma` field matches its arm name
exactly. Arrived as `.zip`, renamed not unpacked, RULES.md #3.
