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

Each resolution model scored at its own size, one seed each, against the 256 px baseline (3 seeds; M0 29.2 ± 7.2, M1 74.0 ± 2.0, M2 55.0 ± 13.9):

| model | PSNR (dB) | M0 | M1 | M2 |
|---|---|---|---|---|
| 320 px | 37.36 | -17.8 | 48.0 | -3.9 |
| 480 px | 40.18 | 39.7 | 73.5 | 67.3 |

320 px is worse, and fails on 2 of 5 cubes. 480 px matches 256 (M1 the same, M0 and M2 higher by less than one seed spread), so **less downsampling did not give an established gain.**
The 320 result being worse than 256 while 480 is not is not monotonic, so a second seed is needed before concluding anything about resolution itself. Full 600 px models exist (notebook 14) and are
being scored next.

## 3. A correction to earlier wiggle numbers

Building a per-checkpoint comparison, I found the kinematic fit for the wiggle metric started from the mask's shape, the degenerate end of the mass-inclination valley.
On 2 of the 5 line-emission cubes it put the disk centre at the image edge (mass 18.7 Msun against a true 1.0). Refit properly, the 3 valid cubes give the kinematic
loss 0.694 against dirty 0.200 (a larger gap than before), but the earlier headline means over all 5 cubes (0.8155 vs 0.4284) are **no longer supported**. Self-gravitating
results were unaffected: on that cube every model still loses to the dirty image on the wiggle (error ratio 1.29 to 1.47, above 1).

## 4. One comparison for every checkpoint, as images

Notebook 16 scores every checkpoint with one protocol and produces, for each, the moment and error maps, channel maps, the classic wiggle residual figure,
radial profile, power spectrum, calibration, integrated spectrum and sharpness. On the SG v2 cube, 3 checkpoints so far: the wiggle figure reproduces the earlier one
(residual RMS 0.23 / 0.21 / 0.22), the calibration plot puts the smoothing at peak amplitude 0.73 to 0.74 of the truth, and the integrated spectrum shows two models
raising the line-free baseline. That is 3 models on 1 cube, an observation not a finding; the full run is next.

## 5. Real ALMA (started)

Following the meeting: exoALMA fiducial line images, 13CO first, MWC 758, no PSF or mask. The inference pipeline (crop, continuum subtraction, per-channel scaling, resample to the
training pixel scale, denoise, moments, DS9-ready FITS) is written and waiting for its first Kaggle run. Real cubes differ from the training data (real CLEAN images, circular
0.15" beam, about 100 m/s channels), so the first output is a sanity check, not a result.

## What is next
480 and 600 px scoring (05), full checkpoint comparison (16), MWC 758 13CO then 12CO with the best 2 to 3 models, then the upstream PR and the final blog.
