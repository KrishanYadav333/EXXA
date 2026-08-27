# DDRM: sampler built, and a hard number on what it would have to do

`src/training/ddrm.py`, with `tests/test_ddrm.py` validating the operator machinery against
known answers (no trained model needed for those checks).

## Why DDRM, now

DDRM was refuted for the line-emission training data on 2026-08-20: Phase 0 measured `A = I`
there, and DDRM has nothing to invert when the operator is the identity. That still stands.

The 2026-08-27 beam ablation changed the picture for the self-gravitating cube. Convolving
the CLEAN cube with the recovered beam, with no noise and no model, reproduced the entire
loss of the GI wiggle (residual RMS 1.394 -> 0.174, correlation 0.116 against the real dirty
cube's 0.111). The beam ERASES the signal. No denoiser can undo that, because removing noise
cannot restore what a convolution destroyed. A measurement-consistency prior can, in
principle, because it reconstructs what the instrument could not measure rather than cleaning
what it did.

## The specialisation

DDRM works in the SVD basis of the operator. For a convolution that basis is the Fourier
basis, so the SVD is free: `A` is a multiply by |FFT(beam)|, `A^T` is the same operation for
a symmetric kernel, and the singular values are the transfer function itself. Each reverse
step costs two FFTs, no decomposition.

Per Fourier mode, the algorithm splits by how well the instrument measured it:
modes the beam passed strongly are pinned to the data, modes it suppressed below the noise
are left entirely to the prior, and the rest are blended.

## Validated (no model required)

| check | result |
|---|---|
| `apply_operator` vs the reference numpy convolution | max relative error 2.1e-07 |
| DC gain equals the beam's sum | exact |
| inversion on the modes the beam DID pass | median relative error 5.2e-05 over 6309 modes |
| measured/unmeasured split responds correctly to noise level | monotonic, correct limits |

The last two matter most. The inversion is exact where it is possible, so the machinery is
right. Image-space correlation on the synthetic test is only 0.62, and that is not an error:
the test beam nulls 61% of the spectrum, and the information in those modes is gone.

## The hard number

On the 600x600 grid, the REAL recovered beam passes:

| threshold | fraction of Fourier modes above it |
|---|---|
| 1e-4 of peak gain | 23.7% |
| 1e-3 | 4.0% |
| **1e-2** | **1.3%** |
| 0.1 | 0.7% |

**Only about 1.3% of modes survive above 1% of peak gain.** That is a quantitative statement
of what the ablation showed qualitatively: the instrument response destroyed the overwhelming
majority of spatial-frequency content on this cube.

## What that means for expectations, stated before training anything

DDRM here would reconstruct roughly 99% of the spectrum from the prior alone, with the data
constraining only the lowest frequencies. That is a genuinely hard regime, well beyond typical
DDRM demonstrations (deblurring, 4x superresolution, inpainting), where the operator usually
retains a much larger measured subspace.

Two honest possibilities, and it is worth committing to them in advance rather than
rationalising afterwards:

1. **The prior recovers real structure.** The GI wiggle residual correlation rises meaningfully
   above the beam-only floor of 0.116. That would be a strong result and a direct answer to
   "can measurement-consistent generation recover kinematics a denoiser cannot".
2. **The prior hallucinates plausible structure that is not the truth.** Correlation stays near
   0.116 while the images look convincing. That is the known failure mode of generative
   restoration at extreme compression, it is exactly the "invented structure" this project has
   measured before (22-39% of channels in the artifact diagnostics), and it would be a
   legitimate negative result worth reporting.

The GI wiggle residual correlation is the metric that separates these, and it already exists
and is validated. That is what makes the experiment worth running: it cannot be fudged by
producing a nice-looking image.

## Not yet done

The diffusion prior. Training data is available and unblocked -- 14 line-emission cubes x ~201
channels of clean Jy/beam disk images, roughly 2800 images, the same domain as `clean_sg.fits`
(also Jy/beam). `src/training/diffusion.py` already exists from the DDPM work.
