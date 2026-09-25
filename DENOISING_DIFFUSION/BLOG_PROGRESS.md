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
| same four, fresh (control exists) | 39.96 / 40.03 / 39.90 / 39.73 | 36.0 / 20.4 / 38.1 / 33.9 | 75.2 / 66.1 / 77.8 / 75.6 | 69.1 / 60.9 / 75.6 / 56.4 |

- **MAE did not lift M0** (26.3 against 29.2, inside the spread). It helped the velocity moments: M1 +3 to +9 points, M2 +15 to +50.
- **No loss clearly beats the others.** The M0 range across the four (20 to 45) is inside single-seed noise; starlet and MAE are best and indistinguishable.
- For the fresh arms the comparison is clean (same recipe, only the loss differs): they beat the plain baseline on M1 by 10 to 22 points and on M2 by 50 to 70.
- **Caveat I have not yet closed.** Fine-tuned arms also get 30+ extra epochs, so part of their gain may be the training, not the loss. Two control arms
  (same source, same budget, original loss) are queued. Until they run, "the loss helped" is established for the fresh arms and only suggested for the fine-tuned ones.
- **PSNR does not rank.** The arm fed one neighbouring spectral channel has the best PSNR (42.59) and the worst M2 (40.5). Pixel accuracy and moment accuracy disagree.

## 2. Less downsampling (ask 2): trained, not yet answered

Models at 320 and 480 px are trained. Their moment rows are **invalid**: my scoring resized every model to 256 px, a scale those two never saw, so I am not quoting them.
The fix is in and they re-score in the next Kaggle session.

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
Control arms and resolution re-score (05), full checkpoint comparison (16), MWC 758 13CO then 12CO with the best 2 to 3 models, then the upstream PR and the final blog.
