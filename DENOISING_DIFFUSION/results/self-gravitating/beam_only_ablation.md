# The beam alone erases the wiggle -- confirmed by ablation

The v2-cube test left an open question: residual correlation with truth was low for BOTH
dirty (0.111) and denoised (0.108), despite the bulk rotation field recovering nearly
perfectly (raw M1 r=0.98-0.99 for both). Hypothesis recorded at the time: the beam
convolution itself, not noise and not the denoiser, may be smoothing away the fine structure
the wiggle consists of.

## The ablation

Take the CLEAN cube. Convolve it with the recovered beam. Add **no noise**. Run **no model**.
Then run the identical Keplerian-fit-and-residual pipeline.

| | residual RMS (km/s) | residual r vs clean |
|---|---|---|
| clean (reference) | 1.394 | -- |
| **beam-only smoothed, no noise, no model** | **0.174** | **0.116** |
| real dirty | 0.176 | 0.111 |
| real denoised | 0.170 | 0.108 |

**The beam alone reproduces the full effect.** Smoothing the clean cube collapses the
residual from RMS 1.39 to 0.174 and drops correlation to 0.116 -- statistically
indistinguishable from what the real dirty cube (0.111) and the denoised cube (0.108) give.
Raw M1 correlation stays high at 0.988, matching the real cubes' 0.98-0.99, so the bulk
rotation survives smoothing exactly as it does in the real data.

Figure: `gi_wiggle_beam_only_ablation.png`.

## What this settles

**The wiggle is destroyed by the instrument response, before noise or denoising enter the
picture.** On this cube, the earlier finding was never a model failure -- there was nothing
left in the dirty cube for the model to preserve.

Three consequences:

1. **A denoiser cannot fix this, in principle.** Removing noise cannot restore structure a
   convolution erased. No amount of training improves this specific number.
2. **The earlier "denoising destroys the kinematics" framing does not apply to this cube.**
   It applied to the ORIGINAL (wrong-script) cube, where denoised raw-M1 correlation fell to
   0.25 against dirty's 0.77 -- a real degradation. Here both sit at 0.98 and the residual is
   already gone from the dirty data.
3. **This is the case DDRM/VIREO are built for.** A measurement-consistency prior
   reconstructs structure the instrument did not measure, which is categorically different
   from denoising. That moves DDRM from "worth trying eventually" to the indicated approach
   for this data specifically -- and the forward operator it needs is already recovered and
   saved (`dirty_beam_recovered_v2.fits`).

## Caveat

This tests the RECOVERED beam, whose held-out validation on this cube was 0.80, not the
near-perfect 0.994 achieved on the original pair. A more accurate beam could in principle
smooth slightly differently. But the agreement with the real dirty cube (0.116 vs 0.111) is
close enough that the recovered beam is clearly capturing the dominant effect.
