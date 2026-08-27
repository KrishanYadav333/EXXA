# Out-of-distribution test: winner_aug on the self-gravitating pair

The trained U-Net had never actually been run on this cube before 2026-08-27, three days
after its beam was recovered. This closes that gap.

`experiments/eval_self_gravitating.py`, `winner_aug` seed 43 (PSNR 39.808, the best of its
three seeds), no retraining or fine-tuning.

## Why this is a real generalisation test, not another holdout cube

Every line-emission training cube has clean AND dirty already in Jy/beam, `A = I` per Phase 0
(`results/PROGRESS.md` 2026-08-20/21): the model learned to remove additive noise with no
mechanism for undoing a beam, because there was never a beam between its input and its
target. This pair is different: `lines.fits` is Jy/pixel, an unconvolved sky model, and
`dirty_cube.fits` has a real dirty beam recovered separately (peak 0.911, FWHM 5 px, -2.8%
sidelobe ring, `results/self-gravitating/dirty_beam_recovered.fits`). Nothing about the
training regime predicts what happens here.

## Result: the model makes it worse

| moment | improvement over dirty (signal-masked) |
|---|---|
| M0 | **-86.5%** |
| M1 | -10.8% |
| M2 | **-168.3%** |

Every moment is negative. The denoised cube is FURTHER from the truth than the raw dirty
cube was; doing nothing would have scored better. See
`results/self-gravitating/ood_moment_comparison.png`.

This is not a subtle degradation. A model trained only to suppress additive noise has no
mechanism to invert a beam it never saw, and applying it to one does not fail gracefully, it
actively damages the signal. M2 (velocity dispersion) is hit hardest, which fits: it is the
moment most sensitive to structure the model is inventing or destroying pixel-by-pixel,
compounding over the collapse.

## Preprocessing note: the cube's edges are padding, not a line-free baseline

Channels 0-29 and 571-600 of `lines.fits` are byte-identical repeats within each block, not
real data -- confirmed with `np.array_equal`. That breaks two things the training pipeline
and `bettermoments` both assume: continuum subtraction (mean of edge channels) would subtract
repeated disk emission rather than a line-free baseline, and `bettermoments.estimate_RMS`
reads `data[:N]`/`data[-N:]` literally, so it would estimate noise from padding.

The eval therefore trims to channels [30, 571) before doing anything -- 541 channels -- and
skips continuum subtraction entirely rather than apply it to data it would corrupt. This is
itself worth knowing: the mentor's own channel-sampling note (`src/data/channel_sampler.py`)
assumes line-free ends exist on every cube, and on this one they do not.

## What this changes

Nothing here is optional-nice-to-know: it is the first concrete evidence that this project's
line-emission model does not transfer to real dirty-beam data, which is precisely the
regime real ALMA observations sit in. It is a strong argument for the DDRM/VIREO direction
being worth building once more data or the antenna config arrives (email sent 2026-08-27,
see `results/PROGRESS.md`), and it is a caution for ALMA validation: a model trained the way
this one was should not be expected to denoise real interferometric images out of the box.
