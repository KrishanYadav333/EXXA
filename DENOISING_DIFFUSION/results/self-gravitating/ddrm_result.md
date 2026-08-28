# DDRM result: no recovery beyond the beam-only floor

Prior trained on Kaggle (60 epochs, 115 min, T4 x2, unconditional, v-prediction, 256px,
loss 8106.7 -> 16.5, best val 22.95). Restoration and scoring run locally on 31 channels
(240-360), covering the full 260-348 range over which the line centre varies.

## Result

| | mstar (Msun) | incl (deg) | raw M1 r | residual r |
|---|---|---|---|---|
| clean | 0.564 | 31.6 | -- | -- |
| dirty | 0.618 | 29.4 | 0.9939 | **0.9970** |
| DDRM | **8.205** | **8.0** | 0.9572 | **0.1277** |

Beam-only floor: **0.116**. DDRM: **0.1277**. No recovery.

This is outcome 2 of the two written down in `ddrm_feasibility.md` before the prior was
trained: *"the prior hallucinates plausible structure that is not the truth -- correlation
stays near 0.116 while the images look convincing."*

## Why this is a real negative, not a failed run

Three things support it:

1. **The prior trained properly.** Loss fell three orders of magnitude and was still improving
   at epoch 60. This is not an undertrained model.
2. **The signal was there to recover.** With a proper channel window the dirty cube's own
   residual correlates at **0.997** with the truth. (The earlier 0.111 came from a 10-channel
   window that collapsed the fit; see the 2026-08-28 entry.) So DDRM was not asked to
   reconstruct something that had already been destroyed beyond reach.
3. **The failure is physically diagnostic, not numerical.** DDRM's fitted geometry is badly
   wrong -- 8.2 Msun against the true 0.56, inclination 8 degrees against 32 -- while its raw
   M1 correlation stays high at 0.957. It produced a disk-shaped velocity field whose implied
   physics is incoherent. That is what hallucination looks like when it is measured rather
   than eyeballed.

The figure (`ddrm_restoration.png`) shows the same thing: clean and dirty residuals are nearly
identical two-lobe patterns; DDRM's is washed out and structureless, and its M1 map carries
visible banding the real cubes do not have.

## Interpretation

The constraint was known in advance and is the likely explanation: **the beam passes only 1.3%
of Fourier modes above 1% of peak gain**, so DDRM had to invent ~99% of the spectrum from the
prior. That is far outside the regime where DDRM is normally demonstrated (deblurring,
4x superresolution), where the measurement anchors most of the reconstruction.

A measurement-consistency prior does exactly what it promises: it produces an image consistent
with the data wherever the instrument measured anything. When the instrument measured almost
nothing, consistency is a weak constraint, and the prior's own biases fill the rest. The
result looks like a protoplanetary disk because the prior was trained on protoplanetary disks,
not because the information was recovered.

## What this settles

- **DDRM does not solve the beam-erased-wiggle problem on this data.** Measured, not assumed.
- **It does not follow that DDRM is useless here in general.** A cube with a less punishing
  beam, or a prior trained on many disks rather than 7 line-emission cubes, could land
  differently. What is established is that this operator, at 1.3% mode survival, is too
  destructive for this approach.
- **The honest framing for the write-up:** four physics-informed approaches have now been
  tested against measurements on this project's data and none improved the science metric.
  That is a legitimate and useful result, particularly since each was refuted for a specific,
  identified reason rather than by a vague failure.
