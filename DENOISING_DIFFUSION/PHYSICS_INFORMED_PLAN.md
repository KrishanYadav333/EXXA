# Physics-informed denoising — DDRM and VIREO

Plan for the two measurement-consistency approaches, deferred to after the Aug 10–14
midterm. Both replace *hinting* the network about the instrument with *constraining* the
output to be consistent with it.

## Why this direction at all

The measured failure is invented structure: 22–39% of validation channels carry a
hallucinated blob, ~7× worse below the median SNR (notebook 08, floor-relative diagnostics).
A conditional DDPM is free to generate anything that looks plausible — nothing ties its
output to what the telescope actually recorded. That is the gap these methods close, and it
is the failure mode that makes a result unusable for science: a false source detection.

**This is not the same as beam conditioning, which was already tried and did not help.**
That fed four scalars — `[sin(2·BPA), cos(2·BPA), BMAJ·3600, BMIN·3600]` — as extra network
input. `use_beam` correlated **−0.33** with PSNR across the 12-run sweep and every top-5
config had it off. A hint the network may ignore is not a constraint; plausibly it only
added input channels and more capacity to overfit six disks.

| | beam conditioning (done) | DDRM / VIREO (planned) |
|---|---|---|
| what the model gets | 4 scalars describing beam shape | the forward operator itself |
| mechanism | soft hint | hard consistency at every reverse step |
| can the model ignore it | yes — and it did | no |

---

## Phase 0 — GATE: what actually produced the dirty cubes?

**Everything below depends on this and it is not yet known.** Do this first; it is an
afternoon, not a project.

DDRM constrains the reverse chain with a known forward operator `A` such that
`dirty ≈ A(clean) + noise`. The whole method is only meaningful if `A` is a non-trivial
operator we can write down.

Two possibilities, with opposite conclusions:

1. **`dirty = clean ⊛ beam + noise`** — a real convolution. DDRM applies, build it.
2. **`dirty = clean + noise`** — no convolution. Then `A = I`, DDRM degenerates to the
   conditional DDPM already implemented, and **there is nothing to gain**. Stop.

The presence of `BMAJ`/`BMIN`/`BPA` in the headers and the MCFOST provenance both point at
(1), but that is an inference, not a check.

**How to settle it** (needs the FITS data, so run on Kaggle or after downloading a cube):

```python
# Compare the azimuthally-averaged power spectra of a clean and dirty channel.
# A convolution suppresses high spatial frequencies in a specific, beam-shaped way;
# additive noise raises the high-frequency floor instead. The two are easy to tell apart.
# Also: fit a Gaussian to a point-like feature in dirty vs clean and compare widths
# against BMAJ/BMIN.
```

Record the answer here before writing any DDRM code.

---

## Phase 1 — DDRM (feasible, conditional on Phase 0)

### Prerequisite: build the beam kernel in pixel units

Header gives the beam in angular units; the kernel needs pixels.

Worked from the recorded example header (`BPA=16.9333°, BMAJ=4.2629e-5°, BMIN=3.2923e-5°`):

```
beam FWHM = 0.1535 × 0.1185 arcsec, PA 16.93°

  CDELT = 0.005"/px  ->  30.7 × 23.7 px FWHM  (sigma 13.0 × 10.1 px)
  CDELT = 0.010"/px  ->  15.3 × 11.9 px FWHM  (sigma  6.5 ×  5.0 px)
  CDELT = 0.020"/px  ->   7.7 ×  5.9 px FWHM  (sigma  3.3 ×  2.5 px)
```

Well resolved at every plausible pixel scale, so an elliptical-Gaussian kernel is
meaningful rather than sub-pixel.

**`CDELT1`/`CDELT2` are not currently read anywhere in `src/`.** `beam_features_of` takes
only BPA/BMAJ/BMIN. First code change: extend it (or add `beam_kernel_of(header, shape)`)
to read CDELT and return an elliptical Gaussian, rotated by BPA, normalised to unit sum.

### The honest limitation, stated up front

A Gaussian is the **restoring** beam, not the **dirty** beam. A real interferometric dirty
beam has sidelobes from incomplete uv coverage, and those sidelobes are exactly what
produces the striping artefacts CLEAN exists to remove. Using a Gaussian `A`:

- is correct *if* the synthetic dirty cubes were made by Gaussian convolution (Phase 0
  answers this),
- is an approximation otherwise, and DDRM's consistency projection will enforce the wrong
  constraint — potentially worse than no constraint.

If Phase 0 shows a true dirty beam was used, ask Jason for the PSF image. He generated the
cubes and will have it.

### Implementation sketch

DDRM works in the spectral domain of `A`. For a convolution with a Gaussian kernel, `A` is
diagonal in the Fourier basis, so the SVD is free — no explicit decomposition needed, which
is what makes this tractable here.

```
A       = FFT -> multiply by beam transfer function -> IFFT
A^T     = same (a real symmetric kernel is its own transpose)
singular values = |FFT(beam)|, per frequency
```

At each reverse step, project the current estimate so that its beam-convolved version stays
consistent with the observed dirty image, with the tolerance set by the per-frequency
singular value and the noise level. Frequencies the beam suppresses to ~0 are unconstrained
and left to the diffusion prior — which is the entire point: the prior fills in only what
the instrument genuinely could not measure.

New code: `src/training/ddrm.py` with a `ddrm_steps(...)` sampler mirroring the existing
`generalized_steps` signature, so `DenoisingDiffusion.sample` can dispatch to it.

### Effort and risk

- beam kernel + CDELT plumbing + tests: **half a day**
- `ddrm_steps` + dispatch + a synthetic round-trip test (known blur, known noise, recover):
  **1–2 days**
- evaluation on the holdout: one run, same cost as the current holdout (~2 h)

Risk: moderate. The maths is standard for a diagonal operator, and it is testable on
synthetic data with a known kernel before touching real cubes.

---

## Phase 2 — VIREO (blocked on data we do not have)

VIREO feeds the PSF **and raw interferometric uv-visibilities** into the network.

**Blocker: there are no visibilities in this project.** A search of `src/` finds no
visibility, uv-plane or measurement-set handling anywhere — the data is image-plane FITS
cubes (clean/dirty pairs) only. The uv data either was never exported from the simulation
or is not shared.

Consequences:

- VIREO as published **cannot be implemented** with the current dataset.
- What *can* be done is a reduced version: condition on the beam **kernel image** rather
  than four scalars — a spatial map the network can convolve against, instead of a hint.
  That is weaker than VIREO and stronger than what was tried.
- The real unlock is asking Jason for the visibilities or at minimum the PSF image. Worth
  raising after midterm; he generated the cubes with PHANTOM + MCFOST and will have both.

**Recommendation: do not plan VIREO until the data question is answered.** Phase 1 delivers
most of the measurement-consistency benefit and needs nothing new from the mentor beyond
the PSF (and possibly not even that).

---

## Ordering

1. **Phase 0** — determine the forward operator. Cheap, and it can cancel Phase 1 outright.
2. **Phase 1** — DDRM with a Gaussian beam, if Phase 0 supports it.
3. **Ask Jason** — PSF image, and whether uv-visibilities exist.
4. **Phase 2** — only if the visibilities arrive.

Both sit behind the higher-value post-midterm items already recorded in `context.md`:
measuring the patch and native-600 arms that are implemented but unmeasured, cross-validation
over all 11 RunIDs, and folding in the new self-gravitating cubes — which takes training from
6 to 7 independent disks and is worth more than any architectural change here.
