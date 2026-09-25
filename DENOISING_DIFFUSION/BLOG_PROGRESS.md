# Progress update: which loss, how many pixels, and what the wiggle really says

*Krishan Yadav, Google Summer of Code, ML4Sci / EXXA. Mentors: Jason Terry, Gaurav S.*
*Code: https://github.com/KrishanYadav333/EXXA (branch `midterm-prep`). Draft of 2026-09-25.*

Since the midterm post I followed the plan from the 12 September meeting: MAE first, less downsampling, other losses, and start on real ALMA data.
Every number below is a mean over the 5 held-out line-emission cubes, from **one seed per arm**, using the clipped, signal-masked moment improvement
(fraction of the clean-vs-dirty error removed, so higher is better). The baselines carry a spread across 3 seeds. Differences smaller than that spread are not established.

## 1. Loss functions (asks 1 and 3)

All arms start from the same best U-Net recipe (base 48, augmented views). "Fine-tuned" arms continue a trained model with a new loss; "fresh" arms train from scratch.

| arm | PSNR (dB) | M0 | M1 | M2 |
|---|---|---|---|---|
| baseline, 3 seeds | 39.30 | 29.2 ± 7.2 | 74.0 ± 2.0 | 55.0 ± 13.9 |
| MAE / wavelet / starlet / gradient, fine-tuned | 39.78 / 40.13 / 40.32 / 40.14 | 26.3 / 38.8 / 42.6 / 32.1 | 77.4 / 76.1 / 80.5 / 78.7 | 70.5 / 70.0 / 81.1 / 77.3 |
| **control: same continuation, original loss, from aug / from p10** | 39.57 / 40.25 | 39.7 / 40.4 | 80.9 / 76.2 | 69.9 / 68.4 |
| same four, fresh (control exists) | 39.96 / 40.03 / 39.90 / 39.73 | 36.0 / 20.4 / 38.1 / 33.9 | 75.2 / 66.1 / 77.8 / 75.6 | 69.1 / 60.9 / 75.6 / 56.4 |

- **The control arms change the story.** Every fine-tuned arm also gets 30+ extra epochs, so I ran the same continuation with the ORIGINAL loss.
  From aug it scores M0 39.7 / M1 80.9 / M2 69.9, from p10 40.4 / 76.2 / 68.4 (one seed). That is +7 to +37 points over the plain baselines with no new loss at all.
  **Most of the fine-tuned gain was extra training, not the loss.**
- Against those controls: from aug, MAE is worse on M0 (26.3 vs 39.7) and only starlet is clearly ahead, on M2 (81.1 vs 69.9). From p10, the losses sit 0 to 4 points
  above the control on M0 and M1 and 2 to 15 on M2 (MAE and starlet the largest). One seed per arm, so this is suggestive, not established.
- **Fresh arms are unaffected** (they have a clean control): from scratch they still beat the plain baseline on M1 by 10 to 22 points and on M2 by 50 to 70.
- **PSNR does not rank.** The arm fed one neighbouring spectral channel has the best PSNR (42.59) and the worst M2 (40.5); two neighbours (42.81) are no better.

## 2. Less downsampling (ask 2)

Every resolution model scored at its own size (notebooks 05 and 16), M0 / M1 / M2, means over 2 to 5 holdout cubes, one seed per resolution:

| model | M0 | M1 | M2 |
|---|---|---|---|
| 256 px, 3 seeds (2 cubes) | 30.4 | 80.0 | 51.6 |
| 320 px, two different models | 44.8 / -39.6 | 80.9 / 24.2 | 67.3 / -56.9 |
| 480 px | 44.5 | 81.0 | 68.9 |
| 600 px (three models) | -13 to 5 | 61 to 68 | -23 to 45 |

- **No established gain from less downsampling.** 480 px is level with or slightly above 256 (inside the seed spread). 600 px is worse, and costs 3.5 times the compute per cube.
- **320 px is inconclusive, not bad.** One 320 px model matches 480; another fails badly on one cube. That is a training-run spread, so a second seed per resolution is needed.

## 3. A correction to earlier wiggle numbers

Building a per-checkpoint comparison, I found the kinematic fit for the wiggle metric started from the mask's shape, the degenerate end of the mass-inclination valley.
On 2 of the 5 line-emission cubes it put the disk centre at the image edge (mass 18.7 Msun against a true 1.0). Refit properly, the 3 valid cubes give the kinematic
loss 0.694 against dirty 0.200 (a larger gap than before), but the earlier headline means over all 5 cubes (0.8155 vs 0.4284) are **no longer supported**. Self-gravitating
results were unaffected: on that cube every model still loses to the dirty image on the wiggle (error ratio 1.29 to 1.47, above 1).

## 4. One comparison for every checkpoint, as images

Notebook 16 scores every checkpoint with one protocol and produces, for each, the moment and error maps, channel maps, the classic wiggle residual figure,
radial profile, power spectrum, calibration, integrated spectrum and sharpness. First Kaggle run: 34 checkpoints on 2 line-emission cubes and the SG v2 cube
(102 rows, 49 minutes, 84 figures); it reproduces the published number for the reference model. The best line-emission checkpoint on those two cubes is a kinematic-input
model fine-tuned with the starlet loss (M0 +50, M1 +84, M2 +73), but two cubes is a shortlist, not a result. On the self-gravitating cube every line-emission model is worse than the dirty
image on the velocity field, and even the model trained on self-gravitating data scores poorly there, which I am treating as suspect until the per-cube figures are checked.

## 5. Real ALMA (started)

Following the meeting: exoALMA fiducial line images, 13CO first, MWC 758, no PSF or mask. The inference pipeline (crop, continuum subtraction, per-channel scaling, resample to the
training pixel scale, denoise, moments, DS9-ready FITS) is written and waiting for its first Kaggle run. Real cubes differ from the training data (real CLEAN images, circular
0.15" beam, about 100 m/s channels), so the first output is a sanity check, not a result.

## What is next
480 and 600 px scoring (05), full checkpoint comparison (16), MWC 758 13CO then 12CO with the best 2 to 3 models, then the upstream PR and the final blog.
