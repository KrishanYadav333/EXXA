# Retraction: the wiggle comparisons used a broken methodology

Several findings from 2026-08-27 and 2026-08-28 are **withdrawn**. They shared one flaw:
each cube was given its OWN Keplerian fit before its residual was compared to the truth's.

## The flaw

The GI wiggle residual is `M1 - Keplerian_model`. If every cube fits its own model, the
residual means something different for each one, and the comparison is dominated by how the
fits disagree rather than by the wiggle. The correct comparison subtracts **one** reference
model (fitted on the clean cube) from every method.

Measured, on beam-convolved clean data versus the truth, across five channel ranges:

| range (step 1) | own-fit r | shared-model r |
|---|---|---|
| 240-360 | **0.117** | 0.920 |
| 230-370 | **0.998** | 0.999 |
| 250-350 | **0.153** | 0.998 |
| 200-400 | **0.949** | 0.998 |
| 60-540 | **0.116** | 0.997 |

Same data, same question. The own-fit column is noise. The shared-model column is stable.

## What is withdrawn

**1. "The beam alone erases the GI wiggle" (2026-08-27).** False. Under a shared model the
beam-convolved cube correlates with the truth at **0.92 to 0.999**, and its residual RMS
(0.168) is within 10% of the truth's (0.182). The beam preserves the wiggle. This finding
motivated the entire DDRM effort.

**2. "Denoising makes the kinematics worse than doing nothing" (2026-08-27).** The direction
survives but the magnitude does not; see the corrected table below.

**3. Every DDRM verdict published on 2026-08-28.** Three were issued in one day, all invalid:
"no recovery (0.128 vs 0.116)", then "destroys signal (0.128 vs 0.938)", then an unpublished
"recovers (0.505 vs 0.117)". All compared per-method fits, and two also compared across
different channel samplings.

## The corrected result

Channels 240-360, step 1 (121 channels), one shared Keplerian model from the clean cube:

| method | residual r vs clean | residual RMS (km/s) |
|---|---|---|
| clean (reference) | -- | 0.182 |
| **beam-only** (no noise, no model) | **0.920** | 0.168 |
| **dirty** (beam + noise) | **0.891** | 0.170 |
| **U-Net** (winner_aug seed 43) | **0.804** | 0.169 |
| **DDRM** | **0.583** | 0.303 |

Read plainly:

- The beam preserves the wiggle almost entirely (0.920).
- Noise costs a little (0.891).
- **The U-Net degrades it** (0.804), consistent with the earlier direction.
- **DDRM degrades it most** (0.583), and inflates the residual amplitude by 1.7x.

So the ordering "do nothing > U-Net > DDRM" holds, and every model makes the kinematics worse
than leaving the dirty cube alone. What is new is that the beam was never the culprit, and
that DDRM's damage is larger than the U-Net's rather than a partial recovery.

## Fixed in code

`compare_wiggles()` in `src/evaluation/gi_wiggle.py` fits the reference once and subtracts that
same model from every method. Regression case 7 in `tests/test_gi_wiggle.py` covers it.

Also fixed a real bug found while adding it: `fit_keplerian`'s default initial guess could fall
outside its own bounds (a uniform mask gives inclination ~0 against a lower bound of 1),
raising "x0 is infeasible". Only reachable without an explicit `init`, which is why every
earlier test missed it.

## The lesson

Three conclusions were published in one day from an unstable statistic, each contradicting the
last. The instability was visible the whole time in the swing across channel ranges, and it was
not checked until the fourth attempt. A comparison metric should be validated for stability
against its own free parameters before any result is read from it.
