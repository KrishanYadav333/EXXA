# The corrected self-gravitating pair (2026-08-27)

Jason's reply to the "Doubt" email thread: the original self-gravitating cube "was made with
a completely different script" -- his words, his mistake. Sent a replacement via the same
Drive folder, `clean_sg.fits` / `dirty_sg.fits`. Checksummed against the originals to confirm
this is genuinely different data, not a resend (different SHA-256, same 601x600x600 shape).

## A different setup from the original pair

| | original (`lines.fits`/`dirty_cube.fits`) | corrected (`clean_sg.fits`/`dirty_sg.fits`) |
|---|---|---|
| clean BUNIT | JY/PIXEL (unconvolved sky) | **JY/BEAM** |
| dirty BUNIT | Jy/beam | Jy/beam |
| beam in header | dirty only, none on clean | **both**, BMAJ/BMIN/BPA identical |
| negative pixels (dirty) | 51.55% | 19.25% |
| simulation-recipe keys | DISTPC, HACNTR, TRKLEN, NTIME, DECDEG | gone; replaced by RMS, PBCOR, SEED |

Both cubes here carry the SAME beam in their header, unlike the original where only dirty
did. That alone does not mean `A = I` the way it did for the line-emission training set,
because the header identity does not guarantee the two arrays are actually related by nothing
more than noise -- Phase 0 exists precisely because header claims are not measurements.

## A numerical bug found and fixed along the way

First Phase 0 run on this cube: `fitted amplitude A = 0.000`, Gaussian match RMS in the
billions, best-fit beam sigma 0.81 px against a header claiming 5.63 px. Looked like the
cube did not fit any sane model. Before believing that (RULES.md #8), checked the raw
power spectra directly: `P_dirty` is ~120,000x `P_clean` at the lowest frequency, falling to
below `P_clean` by mid-band -- clean and dirty differ by roughly two orders of magnitude in
amplitude, confirmed visually (`kinematic_data_v2_amplitude_check.png`): both panels show the
same physical structure, dirty is ~110x brighter at peak and visibly noisier.

The forward-model fit's least-squares solve was not built for that dynamic range and returned
garbage. Fixed by column-scaling the problem before solving (standard Jacobi/Ruiz
preconditioning) -- `src/evaluation/forward_operator.py`, regression case 11 in
`tests/test_forward_operator.py` verifies a real ~1e10 power-scale factor no longer breaks
it. This was a real bug in the pipeline, exposed by data at a scale nothing had tested before,
not a property of this cube.

## Phase 0, after the fix

```
verdict: NON_GAUSSIAN_CONVOLUTION
  best-fit beam sigma  = 6.27 px
  fitted amplitude A   = 81381.319
  header beam sigma    = 5.63 px
  measured / header    = 1.11  <- consistent
```

Figure: `phase0_v2_cube.png`. A real dirty beam is present, and the fitted sigma agrees with
the header's declared value to 11% -- a genuine convolution, unlike the original line-emission
training set's `A = I`. This is a real deconvolution problem, same as the first
self-gravitating pair was (before it turned out to be the wrong script).

## Beam recovery: works, less cleanly than before

`estimate_beam_from_pair` on 12 line-bright channels: peak 0.991 at the exact centre, sum
355, a shallow negative sidelobe (-1.6% of peak). Held out 4 channels not used in the fit:
**correlation 0.80, residual 60% of the dirty cube's own rms** -- markedly weaker than the
original pair's 0.994.

Not glossing over this: the earlier pair's beam recovery was near-perfect; this one is not.
The likely cause is that this cube's ~100x amplitude scale factor may not be perfectly
constant channel to channel, and a beam averaged across 12 fit channels does not correct for
that when applied elsewhere. Not yet confirmed; worth checking per-channel amplitude before
trusting this recovered beam for anything downstream.

Figure + data: `dirty_beam_recovered_v2.png`, `dirty_beam_recovered_v2.fits`.

## Not yet done

The GI wiggle analysis (Keplerian fit, quadratic estimator, the OOD-style denoising test)
that was run on the original pair has not been repeated on this corrected one. That is the
natural next step, now that Phase 0 and beam recovery are working on the right data.
