# Testing the corrected cube: a different failure mode, not a repeat

`winner_aug` seed 43, no retraining, on the corrected `clean_sg.fits`/`dirty_sg.fits` pair.
Trimmed range [60, 541), 481 channels (this cube's padding blocks are wider than the
original's: 0-59 and 541-600, checked directly). One 599s denoising pass fed both tests below.

## Test 1: standard moment-improvement metric

| moment | improvement over dirty (signal-masked) |
|---|---|
| M0 | **+7.5%** |
| M1 | -62.3% |
| M2 | -79.5% |

Genuinely different from the original cube's result (M0 -86.5%, M1 -10.8%, M2 -168.3%). Here
M0 is actually positive -- the model improves total-intensity reliability on this cube -- while
M1 and M2 are still clearly negative. Not a repeat of "the model makes everything worse";
mixed, and worse on the moments that matter for kinematics specifically.

## Test 2: GI wiggle, quadratic estimator

| | mstar (Msun) | incl (deg) | PA (deg) | vsys (km/s) | resid RMS (km/s) |
|---|---|---|---|---|---|
| clean | 0.572 | 32.0 | 178.3 | 0.089 | 1.394 |
| dirty | 0.579 | 30.7 | 0.0 | 0.085 | **0.176** |
| denoised | 0.500 | 32.3 | 0.0 | 0.093 | **0.170** |

(PA 0.0 for dirty/denoised is not a discrepancy: PA is reported mod 180, and 0.0 is 1.7 deg
from clean's 178.3 -- the same orientation, wrapped.)

**All three fitted geometries agree closely this time.** Unlike the original cube's denoised
fit (inclination off by ~40 deg, PA a genuinely different orientation, vsys off by nearly
2 km/s), here mstar/incl/PA/vsys for dirty AND denoised both land within a few percent of
clean's. No unphysical local optimum this time.

| | raw M1 r | residual r |
|---|---|---|
| dirty | **0.986** | **0.111** |
| denoised | **0.979** | **0.108** |

This is the part worth being careful about. Raw M1 correlates extremely well with truth for
BOTH dirty and denoised (0.99 / 0.98) -- the bulk rotation field is easy to recover either
way, and matches the close geometry agreement above. But the RESIDUAL correlation is low for
BOTH (0.11 / 0.11), and nearly identical between them.

**That is not "denoising destroys the wiggle."** It is "the wiggle is not clearly recoverable
from the dirty cube in the first place, and denoising does not meaningfully change that
either way." The figure (`v2_cube_test.png`) shows why: clean's residual retains a large,
structured pattern (RMS 1.39, matching the 2026-08-25 clean-cube finding); dirty and
denoised's residuals are both nearly flat (RMS 0.18/0.17, roughly 8x smaller) -- their own
Keplerian fits absorb almost all of the variance in their M1 fields, leaving little behind to
correlate with anything.

One physical reading, not yet confirmed: on this cube the beam convolution itself (not noise,
not denoising) may be smoothing out the fine non-Keplerian structure the wiggle consists of.
If so, that is an information-loss problem rather than a noise problem, and denoising --
however good -- cannot recover what the beam genuinely erased. That is exactly the situation
DDRM and VIREO are designed for: a measurement-consistency prior can fill in structure the
instrument did not measure, where a plain denoiser has nothing to work from. This is a
hypothesis, not yet tested.

## Net read

Neither "the model is fine" nor "the model destroys everything" describes this cube. M0
improves, M1/M2 don't, the bulk rotation field is recoverable from dirty either way, and the
specific wiggle signal is faint-to-absent in the dirty cube itself before denoising ever
enters the picture. Different cube, different failure mode -- both are real, and the answer
to "is it better than the old cube" is: better on some axes, and the actual kinematic
question (does the wiggle survive) is unresolved here rather than clearly answered no.
