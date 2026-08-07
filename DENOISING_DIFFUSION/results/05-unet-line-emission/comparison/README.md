# comparison/ — notebook 05, versions 7, 9 and 12 as executed

The Kaggle-executed notebooks themselves, pulled from the push history with their outputs
intact. Unlike the sibling run folders — which hold extracted figures and logs — these are
the full notebooks, so each version's code, configuration and printed output can be read
together and diffed against the others.

| File | Ver | push | code | Figures | PSNR printed |
|---|---:|---|---|---:|---|
| `..._v7_95d9034.ipynb` | 7 | `95d9034` | `662fef5` | 10 | 31.02 / 33.18 |
| `..._v9_d0adb14.ipynb` | 9 | `d0adb14` | `7fa5fce` | 10 | 32.31 / 33.04 / 33.02 |
| `..._v12_82abd23.ipynb` | **12** | `82abd23` | `c61d112` | 6 | **32.95** ← the reference |

v7 and v9 each contain **two runs in one notebook** — a line-emission section and an appended
continuum section — which is why they carry ten figures and two or three PSNR values. v12 is
a single run.

## These numbers are on the old metric

All three predate the noise clip (`bab16d0`) and the signal mask (`8f4da16`). Every moment
figure inside them is on the original unmasked metric and **cannot be compared** to anything
scored after those commits — the clip alone shifts M1 by roughly 4×. See
[`../../RUNS.md`](../../RUNS.md).

## v12's weights still exist

v12 wrote `unet_line_emission_continuum_best.pth` (epoch 28, val_loss 0.0034) into
`/kaggle/working`, so the checkpoint is in Kaggle's Output for that version. Attach that
Output as an Input and notebook 05 section 4 restores it — that is how V12 gets re-scored
under the current metric instead of being quoted from a table.
