# The self-gravitating pair, and its measured forward operator

Jason shared these cubes on 2026-08-07 (Google Drive, folder `kinematic_data`). They were
downloaded on 08-08 and first opened on 08-21. The cubes themselves are gitignored and live
in `DENOISING_DIFFUSION/self-gravitating cube and dirty cube/` (1.5 GB zip, two
601 x 600 x 600 cubes).

## They are a different kind of data from the training set

| | line-emission training cubes | this pair |
|---|---|---|
| clean | `BUNIT=JY/BEAM` | **`BUNIT=JY/PIXEL`** |
| dirty | `BUNIT=JY/BEAM` | `BUNIT=Jy/beam` |
| negative pixels in dirty | few | **51.55%** |
| operator between them | none, `A = I` | a real dirty beam |

`lines.fits` is an unconvolved sky model: no negative pixels at all, minimum +41.55.
`dirty_cube.fits` is a genuine dirty image: half its pixels negative, minimum at -0.34 of its
maximum, total flux ~0. That combination, deep negatives with zero net flux, is what an
interferometer produces before deconvolution. There is no zero-spacing baseline to carry the
total, and the sidelobes push flux below zero.

This is why the DDRM and VIREO refutations recorded elsewhere are specific to the
line-emission data. Those cubes have the beam already inside both sides of the training pair.
This one does not.

## `dirty_beam_recovered.fits`

The forward operator, measured from the pair rather than requested from the mentor. With both
a clean and a dirty cube in hand, `B = <D conj(C)> / <|C|^2>` recovers it directly
(`src/evaluation/forward_operator.py::estimate_beam_from_pair`).

| | |
|---|---|
| peak | 0.9109, at the exact centre pixel |
| FWHM | 5 px |
| deepest sidelobe | -2.77% of peak, a ring from r ~ 26 px |
| total flux, full field | ~0 |
| fitted on | 12 line-bright channels, 250 to 360 |

Peak near 1 with zero net flux is the definition of a dirty beam. The negative ring is why a
Gaussian `A` would be the wrong operator here, and it is what Phase 0's `non_gaussian_convolution`
verdict was detecting. As an independent cross-check, Phase 0's Gaussian fit gave sigma
2.07 px, i.e. FWHM 4.9 px, against the 5 px measured directly.

**Validated on held-out channels.** Convolving `lines.fits` with this beam reproduces
`dirty_cube.fits` at correlation **0.9939** on channels 150 to 450, none of which were used to
fit it. The residual is 16.6% of the dirty cube's own rms, which is the noise term.

**Caveat on the crop.** The file is the central 129 x 129 px. The full field sums to ~0; this
crop sums to 26.5 because it cuts off the outer negative bowl. Truncating changes the
operator, so a use that needs flux conservation should re-derive at full size rather than
zero-pad this.

## What it unlocks

DDRM's operator for this data is now known, so "ask Jason for the PSF image" is off the
critical path. What still blocks a DDRM arm is data volume: one pair cannot train a model, and
Jason's "I'll give you more later" is the ask that matters.

## The dirty cube's header is a simulation recipe, not CASA metadata

`dirty_cube.fits` carries keywords no CASA task writes: `DISTPC`, `PXAS`, `HACNTR`, `TRKLEN`,
`NTIME`, `DECDEG`, `RMSJYB`, `PBAPP`. This was made by a custom uv-sampling simulator, and its
header happens to record the physical parameters that simulator used.

```
DISTPC  140.0        distance, pc
PXAS    0.009524     pixel scale, arcsec  (matches lines.fits CDELT1 exactly, 0.009524)
HACNTR  0.31327       hour angle at centre, rad (~17.9 deg)
TRKLEN  2.04958       tracking length, hours
NTIME   92            number of time samples
DECDEG  -56.72212     source declination
RMSJYB  0.00818       noise RMS, Jy/beam
PBAPP   False         primary-beam correction NOT applied
```

`RESTFREQ` (220398700000.0 Hz, 13CO(2-1)) is present on both cubes and matches, contrary to an
earlier check of mine that looked for `RESTFRQ` and reported it missing -- the header spells
it `RESTFREQ`.

**What this fixes and what it does not.** Declination, hour-angle centre and track length
determine the uv-coverage GEOMETRY for a `simobserve`-style reproduction. They do not
determine the ARRAY: the same geometry observed with a compact configuration and an extended
one gives very different beams, and the recovered beam's 0.048" FWHM at 220 GHz is consistent
with an extended configuration but does not pin one down. Reproducing this properly needs the
antenna configuration file (or the actual antenna positions) Jason used, not a guess from the
beam size.

**The ask, updated again:** not the PSF (recovered), not more pairs (declined) -- the ALMA
configuration / array file used to make `dirty_cube.fits`. With DISTPC/HACNTR/TRKLEN/NTIME/
DECDEG already known, that one file is what closes the gap to a `simobserve` reproduction and,
from there, a synthesised multi-configuration dataset for testing whether DDRM generalises
across beams where a supervised U-Net does not.
