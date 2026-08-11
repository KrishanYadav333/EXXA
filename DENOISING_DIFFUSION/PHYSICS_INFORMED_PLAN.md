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

## Phase 2 — VIREO, in two parts

VIREO as published feeds the PSF **and** raw uv-visibilities into the network. The
visibilities are not available here, but most of the mechanism does not need them, and that
part is implementable today.

Splitting it accordingly:

### Phase 2a — VIREO-lite: PSF map + data-consistency loss (NO new data needed)

Two changes, both of which work for the **U-Net as well as the DDPM** — which matters,
because the U-Net is the primary model and DDRM only helps the DDPM.

**(i) Condition on the beam as a MAP, not four scalars.**
Beam conditioning fed `[sin(2·BPA), cos(2·BPA), BMAJ·3600, BMIN·3600]` — four numbers, no
spatial structure, trivially ignorable, and measured at r=−0.33. Instead append the beam
kernel itself as a second input channel, at the same resolution as the image. The network
can then convolve features against the actual beam shape rather than being told about it in
the abstract. Input becomes `(2, H, W)` = `[dirty, beam_kernel]`.

**(ii) Add a data-consistency term to the loss.** This is the important half.

```
L = alpha * L_recon(pred, clean)  +  lambda_dc * || A(pred) - dirty ||^2
```

where `A` is the beam convolution from Phase 1. The second term asks: *if this prediction
were observed by the same telescope, would it reproduce the dirty image we actually got?*
An invented blob has no counterpart in `dirty`, so it is penalised directly. That is a loss
aimed precisely at the measured failure — 22–39% of channels carrying hallucinated
structure — rather than at pixel error in general.

Unlike DDRM this needs no change to the sampler, applies to any architecture, and costs one
extra convolution per training step.

**Effort:** beam-map channel ~half a day (reuses the Phase 1 kernel); data-consistency loss
~half a day including a test that an invented blob raises the term while a faithful
reconstruction does not. **Both depend on the same Phase 0 answer** — if `A = I` the
consistency term reduces to `||pred − dirty||`, which would actively push the model *toward*
the noisy input and must not be shipped.

**Risk:** `lambda_dc` needs tuning on validation. Too large and the model reproduces the
dirty image, undoing the denoising; too small and it changes nothing. Sweep it as its own
arm in notebook 09 rather than picking a value.

### Phase 2b — VIREO-full: uv-visibilities (blocked on data)

**Blocker: no visibilities exist in this project.** A search of `src/` finds no visibility,
uv-plane or measurement-set handling; the data is image-plane FITS cubes only. Either they
were never exported from the simulation or they were not shared.

What full VIREO adds over 2a is a consistency constraint in the **uv plane** rather than the
image plane. That is strictly better for interferometry, because the instrument samples the
Fourier plane sparsely and *unsampled* baselines carry no information at all. An image-plane
constraint cannot distinguish "the telescope measured this and it was zero" from "the
telescope never measured this" — a uv-plane constraint can, and only lets the prior fill in
the genuinely unmeasured baselines.

**Ask Jason for, in priority order:**
1. the **PSF / dirty-beam image** — unlocks the correct operator for Phase 1 and 2a even
   without visibilities;
2. the **uv-visibilities** (measurement sets, or the uv sampling function) — unlocks 2b;
3. whether the dirty cubes were made by Gaussian convolution or a real dirty beam — this
   answers Phase 0 directly and costs him one sentence.

**Effort once data arrives:** substantially larger than 2a — a uv-plane forward model,
gridding/degridding, and the sampling mask, plus handling that the FFT of a 600x600 image at
every training step is not free. Treat as a multi-week item, not a follow-up.

## Ordering

1. **Phase 0** — determine the forward operator. Cheap, and it gates everything below.
2. **Ask Jason** — PSF image, visibilities, and how the dirty cubes were made. Send this at
   the same time as Phase 0; his answer to (3) may settle Phase 0 outright.
3. **Phase 2a — VIREO-lite** *before* DDRM. The data-consistency loss is cheaper to build,
   needs no sampler changes, applies to the **U-Net** as well as the DDPM, and attacks the
   measured hallucination failure directly. Best value per day of work in this document.
4. **Phase 1 — DDRM**, which reuses 2a's beam operator and helps the DDPM only.
5. **Phase 2b — VIREO-full**, only if visibilities arrive; multi-week.

Both sit behind the higher-value post-midterm items already recorded in `context.md`:
measuring the patch and native-600 arms that are implemented but unmeasured, cross-validation
over all 11 RunIDs, and folding in the new self-gravitating cubes — which takes training from
6 to 7 independent disks and is worth more than any architectural change here.
