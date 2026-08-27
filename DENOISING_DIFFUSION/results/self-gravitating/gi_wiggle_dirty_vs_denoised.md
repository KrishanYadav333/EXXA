# Does the GI wiggle survive noise, and does the U-Net destroy it?

Extends `gi_wiggle_check.md` (clean cube only) to `dirty_cube.fits` and the `winner_aug`
seed-43 denoised output from the 2026-08-27 OOD eval. Reuses that eval's saved moment maps
rather than re-running inference. Same mask throughout, defined once from the clean cube's M0
(`moment_improvement`'s own convention), so all three are compared on identical pixels.

## The clean, unconfounded result: denoising makes it worse

| | correlation of raw M1 with the true clean M1 |
|---|---|
| dirty (no processing) | **0.72** |
| denoised (winner_aug seed 43) | **0.53** |

No Keplerian fit involved, no convergence caveats. **The raw dirty cube's velocity field is
closer to the truth than what the trained model produces.** Running this U-Net on real
dirty-beam data does not recover the kinematics, it degrades them below doing nothing. This
lines up exactly with the 2026-08-27 OOD moment result (M0 -86.5%, M2 -168.3%, signal-masked)
-- same conclusion, independent metric.

## The Keplerian-fit numbers, and why they need more caveats

| | mstar (Msun) | incl (deg) | typical rotation (km/s) | fit converged? |
|---|---|---|---|---|
| clean | 0.639 | 26.7 | 0.635 | yes, clean |
| dirty | 49.971 (pinned at bound) | 1.8 (near-degenerate) | 0.422 | **no** |
| denoised | 0.136 | 14.3 | 0.034 | yes, but unphysical |

**Dirty's fit does not converge to physical values, even starting from the clean cube's own
geometry as the initial guess** (the physically correct thing to try, since it is the same
disk). Mass runs to its 50 Msun upper bound; inclination collapses toward face-on. That is
itself informative: a naive per-pixel Keplerian fit to the raw dirty cube's moment-1 map does
not work. Some form of imaging or deconvolution is a genuine prerequisite for this kind of
kinematic analysis on real interferometric data, not a nice-to-have -- which is a real
argument for why this project's task matters, even though the current model does not yet
deliver it (see above).

**Denoised's fit converges, but to a disk with roughly a fifth of the true mass and almost no
real rotation** (0.034 km/s median deviation from systemic, against 0.635 km/s in the truth).
Its residual correlates with the clean cube's residual at 0.82, which at first glance reads as
"the wiggle survives". It should not be read that way: with almost no fitted rotation, the
subtracted Keplerian model is nearly flat, so the "residual" is close to the raw denoised M1
map itself rather than a genuine wiggle isolated from a correctly recovered rotation curve.
The raw-M1 correlation above (0.53) is the number that is not confounded this way, and it
already answers the question in the other direction.

## What this settles, and what it does not

**Settled:** the current `winner_aug` checkpoint should not be used to prepare a cube for GI
wiggle analysis. It removes recoverable kinematic information rather than preserving it, on
both the raw-field and Keplerian-fit view.

**Not settled:** whether a moment-1-based Keplerian fit is even the right tool for the raw
dirty cube. The literature (Hall+2021 in particular) works from individual channel maps, not
the collapsed moment-1, specifically because M1 is more noise-sensitive; this project's fit
failing to converge on the dirty cube is consistent with that being the actual reason, not
proof that no signal is recoverable from the dirty data by a better method.

Figure: `gi_wiggle_clean_vs_dirty_vs_denoised.png`.
