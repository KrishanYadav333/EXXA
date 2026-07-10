# Kaggle Notebook Snapshots — V7 and V9

Frozen copies of the Kaggle-executed notebooks for the **continuum-subtracted line-emission U-Net** runs.
These are read-only historical records. Do **not** edit them; use the live notebooks in `../` for new runs.

---

## Files

| File | Kaggle Version | Git Commit | Description |
|---|---|---|---|
| `V7_05-unet-line-emission.ipynb` | Version 7 | `95d9034` | Full merged Kaggle notebook (root `05`) as executed for V7 |
| `V7_06-unet-line-emission-continuum.ipynb` | Version 7 | `d4ff643` | Standalone `06` notebook synced with V7 outputs |
| `V9_05-unet-line-emission.ipynb` | Version 9 | `d0adb14` | Full merged Kaggle notebook (root `05`) as executed for V9 (includes ablation cells) |
| `V9_06-unet-line-emission-continuum.ipynb` | Version 9 | `5933292` | Standalone `06` notebook synced with V9 outputs |

---

## What V7 and V9 Are

Both are Kaggle T4×2 training runs of notebook `06-unet-line-emission-continuum.ipynb`
(continuum-subtracted line-emission U-Net, 30 epochs, 256×256 full images, `HybridLoss`).

| | V7 | V9 |
|---|---|---|
| **continuum_n** | 5 (avg first+last 5 channels) | Ablation: n=1 AND n=5 side-by-side |
| **Best epoch** | 28 | 25 |
| **Val SSIM** | 0.9868 | 0.9867 (n=1) / 0.9793 (n=5) |
| **M0 improvement** | +84.9% | +61.9% |
| **M1 improvement** | +20.9% | +15.8% |
| **M2 improvement** | +18.4% | +2.5% |

Key finding: `continuum_n=1` ≈ `continuum_n=5` on pixel metrics. M2 variance is due to small
training set (350 channels), not config differences.

Result images: `../../../results/for_jason/V7_*.png` and `V9_*.png`.

---

## How These Were Created

```powershell
# From repo root
git show 95d9034:"05-unet-line-emission.ipynb"   > kaggle_versions/V7_05-unet-line-emission.ipynb
git show d4ff643:"DENOISING_DIFFUSION/notebooks/06-unet-line-emission-continuum.ipynb" > kaggle_versions/V7_06-unet-line-emission-continuum.ipynb
git show d0adb14:"05-unet-line-emission.ipynb"   > kaggle_versions/V9_05-unet-line-emission.ipynb
git show 5933292:"DENOISING_DIFFUSION/notebooks/06-unet-line-emission-continuum.ipynb" > kaggle_versions/V9_06-unet-line-emission-continuum.ipynb
```
