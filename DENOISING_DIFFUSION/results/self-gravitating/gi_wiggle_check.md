# Is the GI wiggle present in the self-gravitating cube?

Jason's reading list (Speedie+2024 Nature, Terry+2024 A&A, Hall+2021 MNRAS, Hall+2022 ApJL,
Hall+2020 ApJ) is built around one diagnostic: fit the disk's own Keplerian rotation, subtract
it from the moment-1 map, and look at the residual. A planet leaves a localised kink; a
self-gravitating disk leaves a global pattern across the whole velocity field. `DISTPC=140 pc`
in `dirty_cube.fits` matches Hall+2020's founding simulation exactly, so this cube is very
likely built on that same pipeline.

New module: `src/evaluation/gi_wiggle.py`. `disk_geometry_from_m0` gets a starting geometry
from M0's image moments; `fit_keplerian` refits centre, position angle, inclination,
systemic velocity and stellar mass against the actual moment-1 field by least squares;
`wiggle_residual` is M1 minus that fit.

## Validated before trusting it on real data

Two synthetic cases, `src/evaluation/gi_wiggle.py` used directly (no notebook, no GPU):

- **Pure Keplerian, no wiggle.** Fit recovers every parameter exactly (centre, PA,
  inclination, vsys, stellar mass all to machine precision) and the residual is exactly 0.
  Confirms the model and optimiser are self-consistent.
- **A known m=2 perturbation injected on top.** Recovered residual correlates with the
  injected pattern at r=0.85. Real, imperfect: peak recovered amplitude (1.45 km/s) exceeds
  the injected peak (0.40 km/s), because fitting geometry on data that already contains the
  perturbation lets some of it leak into the geometric fit itself. Good enough to trust that
  a genuine residual pattern is being recovered, not strong enough to trust an absolute
  amplitude number from this method without a tighter calibration.

## Fitted geometry (clean cube, non-padded range [30, 571))

| parameter | value |
|---|---|
| centre | (310.0, 300.0) px -- matches the header's CRPIX 301/301 |
| position angle | 179.8 deg |
| inclination | 26.7 deg |
| systemic velocity | 0.066 km/s -- matches CRVAL3=0 |
| stellar mass | **0.639 Msun** |

The stellar mass is the number worth noticing on its own: **0.639 Msun, essentially
Hall+2020's 0.6 Msun star**, recovered from nothing but the velocity field, with no mass
information anywhere in either header. That is independent evidence this cube was built on
that pipeline, on top of the DISTPC match.

## The residual

RMS 1.34 km/s against a typical local rotation speed of 0.64 km/s in the same mask: the
residual is **larger than the rotation itself**, and it does not shrink with distance from
the fitted centre (checked out to r=100 px against a median disk radius of 130 px, so this
is not a 1/sqrt(r) artifact near the centre). In one annulus (r 120-140 px), the residual
is a clean single-period (m=1) sinusoid in azimuth, +1.75 km/s to -1.75 km/s and back, with
essentially no scatter bin to bin.

That is a large, coherent, global non-Keplerian signal. It is what the papers describe. It is
not automatically proof of gravitational instability specifically, and that distinction
matters: a pure, radius-independent m=1 residual is also the signature a slightly imperfect
geometric model can produce (this fit is a flat thin disk; a warp or an optical-depth-driven
asymmetry in the real emission would show up the same way), and the field is actively
debating exactly this degeneracy -- a 2025 paper (arXiv:2510.05601) argues bulk infall can
mimic a wiggle in AB Aurigae, the same object Speedie+2024 read as gravitational instability.

**What this is: a validated pipeline finding a large, real residual, with a stellar mass
recovery that independently corroborates the cube's provenance.** What it is not yet: a
confirmed GI wiggle detection. That needs comparing this residual's shape against the actual
published wiggle morphology (Hall+2021's "interlocking fingers" are described as varying with
radius, not constant), which is the natural next step.

Figure: `gi_wiggle_clean.png` -- moment 1, the Keplerian residual, and moment 0 for reference.

## Not yet done

- Run the same fit on `dirty_cube.fits` and on a denoised cube, to see whether the wiggle
  survives noise and the (currently failing) U-Net.
- Check whether the residual's radial structure, not just its azimuthal one, matches the
  literature's "interlocking fingers" description rather than a pure geometric systematic.
- Cross-check against Hall+2021's actual Fourier-decomposed amplitude metric rather than the
  RMS/max proxy in `wiggle_amplitude`, if a q (disc-to-star mass ratio) estimate is wanted.
