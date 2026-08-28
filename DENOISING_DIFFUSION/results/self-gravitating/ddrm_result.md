# DDRM result: actively destroys signal that survives the beam

> **This file was corrected on 2026-08-28 after publication.** The first version
> compared DDRM's score against a floor measured at a DIFFERENT channel sampling and
> concluded "no recovery". Measured at DDRM's own configuration the floor is 0.938,
> not 0.116, so DDRM scores far BELOW it. The conclusion is stronger and different in
> kind. The error and its cause are documented below.

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


---

# CORRECTION (2026-08-28, after publication)

## The comparison above was invalid

DDRM was scored on 31 channels sampled every 4th (240-360, step 4). The 0.116 floor came from
the 2026-08-27 ablation, which used 481 channels at step 1. **Different configurations, not
comparable**, and I published the comparison anyway.

## Why sampling matters this much

The GI wiggle residual is extremely sensitive to velocity sampling, because `quadratic_moment1`
fits a parabola to the peak channel and its two neighbours:

| step | dv (km/s) | clean residual RMS | dirty residual RMS | resid r |
|---|---|---|---|---|
| 1 | 0.033 | 0.182 | 1.341 | 0.115 |
| 2 | 0.067 | 0.185 | 1.355 | 0.117 |
| 4 | 0.133 | **1.436** | 1.418 | **0.997** |
| 8 | 0.267 | 1.741 | 1.689 | 0.997 |

At step 4 the CLEAN cube's own residual jumps 8x, from 0.18 to 1.44. That is not signal, it is
parabola-fit error from coarse sampling, and because it is a deterministic artifact of the
sampling rather than noise, clean and dirty share it -- which is why they correlate at 0.997
there. The high correlation measures a shared artifact, not a recovered wiggle.

## The corrected result

Measured at DDRM's own configuration (31 channels, step 4):

| | residual r vs clean |
|---|---|
| beam-only (no model) | **0.938** |
| DDRM | **0.128** |

**DDRM scores far below the beam-only floor.** It does not merely fail to recover the wiggle;
it destroys structure that survives beam convolution untouched. That is consistent with its
fitted geometry being physically incoherent (8.2 Msun against the true 0.56, inclination 8
degrees against 32) while raw M1 correlation stayed high at 0.957 -- the restoration produced a
disk-shaped velocity field with the fine structure replaced by the prior's own invention.

## What still stands, and what does not

**Stands:** the prior trained properly; DDRM's fitted geometry is physically wrong; the beam
passes only 1.3% of Fourier modes above 1% gain; the qualitative conclusion that DDRM is not
usable for this problem on this data.

**Does not stand:** the specific claim "0.128 against a 0.116 floor, no recovery beyond the
floor". The floor at that configuration is 0.938 and the result is a clear degradation, not a
null.

**Also does not stand:** the correction I made earlier the same day, attributing the
2026-08-27 value of 0.111 to "a 10-channel window collapsing the fit". That value came from a
481-channel window at step 1 and is a sound measurement of its own configuration. The real
cause of the discrepancy is the sampling step, not the window length.

## The methodological lesson

Every GI wiggle number in this project is only comparable to another computed at the SAME
channel range AND step. Numbers from different configurations must not be compared, and three
separate errors today came from doing exactly that. Future comparisons should record the
configuration alongside the value.
