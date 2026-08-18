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

**The check is written**: `src/evaluation/forward_operator.py`, with
`tests/test_forward_operator.py` covering it against three synthetic cases where the true
operator is known by construction. On a cube pair:

```python
from src.evaluation.forward_operator import phase0_from_fits, format_report
print(format_report(phase0_from_fits(clean_path, dirty_path)))
```

or `python -m src.evaluation.forward_operator clean.fits dirty.fits`.

It returns one of three verdicts:

| verdict | meaning | what to do |
|---|---|---|
| `no_convolution` | `A = I` | stop, DDRM has nothing to constrain |
| `gaussian_convolution` | the header's beam is the operator | build DDRM as in Phase 1 |
| `non_gaussian_convolution` | a real dirty beam, sidelobes and all | DDRM still applies, but ask Jason for the PSF image first, because a Gaussian `A` would enforce the wrong constraint |

The discriminator is the azimuthally averaged power spectrum. Writing `P_d`, `P_c` for the
dirty and clean spectra, a convolution gives `P_d = |B|^2 P_c + N` and additive noise gives
`P_d = P_c + N`. The raw ratio `P_d / P_c` therefore dips below 1 in the first case and
never can in the second, whatever the noise level, since noise only ever adds power. The
verdict is read off that raw ratio. A noise-subtracted version is used only to fit the
beam's shape, and deliberately not for the verdict: the noise estimate is biased high
wherever the clean field still has power at high k, and subtracting too much manufactures a
dip on its own. Both mistakes are in the test as regression cases, because both were made
while writing it.

Two more things the report gives, and both matter for Phase 1:

- the beam sigma **measured from the data**, cross-checked against the one the header
  claims. They should agree. If they do not, `beam_kernel_of` is building the wrong `A`.
- `pixel_scale_arcsec`, reading `CDELT`, which nothing in `src/` did before. `beam_features_of`
  takes only BPA/BMAJ/BMIN, enough to describe a beam in angular units but not to build a
  kernel in pixels.

**Verdict on the project's own cubes: not yet run.** It needs the FITS data, which is on
Kaggle. Record the answer here before writing any DDRM code.

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

---

## Phase 3 — Physics-informed constraints beyond the beam

The beam is one piece of physics; it is not the only one the reconstruction violates. Three
further constraints are known to hold for this data and are currently unenforced. Each maps
onto a specific measured failure rather than being added for elegance.

| Constraint | Physical statement | Failure it targets | Currently |
|---|---|---|---|
| non-negativity | sky brightness cannot be negative | floor leak | unenforced (linear head) |
| flux conservation | denoising must not create or destroy total flux | M0 error | unenforced |
| spectral continuity | a line profile is smooth along velocity | **M1 / M2 weakness** | unenforced |

### 3a — Spectral continuity (highest value in this document)

**The one that addresses the diagnosed root cause.** M0 is a spectral *sum* and scores
~+70%. M1 and M2 are spectral *shape* statistics and lag badly. The models denoise each
channel independently, so nothing constrains consistency along the velocity axis — which is
exactly the axis M1 and M2 are computed over. No amount of per-channel improvement fixes
this, because the information is not being used.

Two options, cheapest first:

**(i) A spectral-smoothness penalty.** Add `mu * || d^2 pred / dv^2 ||^2` — a second-derivative
penalty along the channel axis — to the loss. Requires training on channel *triples* (or any
contiguous run) rather than independent channels, so the dataset must yield neighbours. Cheap,
architecture-agnostic, no model change.

**(ii) 2.5D input: feed adjacent channels.** Input becomes `(2k+1, H, W)` — channel *i* plus
*k* neighbours each side — output stays 1 channel. `in_channels = 2k+1` instead of 1;
negligible extra parameters; roughly unchanged training time. Standard in video and medical
denoising. This gives the model spectral context rather than merely penalising its absence,
and is the stronger of the two.

**Effort:** (i) ~half a day plus a dataset change to emit neighbours; (ii) ~1 day.
**Risk:** low. Both are additive and can be swept as arms in notebook 09.
**Expected effect:** should move M1 and M2 specifically. If it does not, the
per-channel-independence explanation for their weakness is wrong and worth revisiting.

### 3b — Non-negativity

Sky brightness is non-negative. The model has a linear output head (deliberately — a sigmoid
could not represent shared-dirty-scale clean values that exceed 1, which was a real bug) and
so may emit negatives. Floor leak was measured, though on the pre-fix artifact code, so the
magnitude needs re-establishing after the current metric lands.

Options: a softplus head, a clamp at inference, or a penalty `nu * ||min(pred, 0)||^2`. The
penalty is preferable — a hard clamp hides the problem rather than training it away, and
would make the floor-leak diagnostic read zero regardless of what the model learned.

**Caveat that must be checked first:** the data is *continuum-subtracted*, and a subtracted
map legitimately contains negative pixels where the continuum estimate over-subtracts. So
non-negativity applies to the physical sky, not necessarily to the arrays being trained on.
Verify against the clean cubes before enforcing it — this is exactly the kind of constraint
that looks obviously correct and quietly is not.

**Effort:** ~half a day. **Risk:** low, but gated on the caveat above.

### 3c — Flux conservation

Total flux in a channel should survive denoising. Add `kappa * (sum(pred) - sum(clean))^2`,
or the softer `| sum(pred) / sum(dirty) - 1 |`, which needs no ground truth and so would also
work on real observations later.

M0 is integrated intensity, so this is close to optimising the reported metric directly.
That is a reason for care rather than enthusiasm: it would improve M0 partly by construction,
and any gain must be reported as such rather than as better reconstruction. Worth trying,
worth labelling honestly.

**Effort:** ~2 hours. **Risk:** low technically; moderate in interpretation, per the above.

### 3d — SNR-aware training (the reference's own strongest suggestion)

The briefing says it twice and it is right: *"models optimized with a known noise
distribution profile dramatically outperform general blind models"*, citing SNRAware for MRI
denoising. This is the best fit to this project's measured failure of anything in that text,
and it was missed on the first pass.

**Why it fits here specifically.** Hallucination is not uniform — it is **~7x worse below
the median SNR** (1.580 vs 0.213 blobs/channel, notebook 08). The model currently has no
idea which regime it is in. It treats a bright line-core channel and an empty band-edge
channel identically, then invents structure in the latter. Telling it the noise level is the
minimum information needed to behave differently where behaving differently matters.

**And the noise level is already known, per channel, for free.** `bettermoments.estimate_RMS`
computes it from the line-free edge channels and is already called on every cube in
`generate_moment_maps`. No new data, no new estimator — the quantity exists and is discarded.

Two routes, usable together:

**(i) Condition on it.** Feed per-channel `rms` (or `peak/rms`) as an extra input, either as
a scalar broadcast to a plane or via FiLM-style modulation. Unlike the four beam scalars —
which described the *instrument*, were constant across channels, and were duly ignored
(r=-0.33) — this varies per sample and carries information the model provably lacks.

**(ii) Weight the loss by it.** Down-weight channels whose SNR makes the target
unrecoverable. A channel that is pure noise has no learnable signal, and forcing the model
to fit it teaches it to invent plausible structure from nothing, which is exactly the
observed failure. This is the same reasoning as Min-SNR weighting on the diffusion timestep
axis, applied instead across channels.

**Effort:** (i) ~half a day; (ii) ~2 hours. **Risk:** low, and it is architecture-agnostic —
it applies to the U-Net, not just the DDPM.
**Falsifiable prediction:** the low-SNR blob rate should fall relative to the high-SNR rate.
If the ~7x gap does not narrow, the SNR-blindness explanation is wrong.

---

## Phase 4 — Transformer architectures (Restormer / SwinIR)

The only *architecture* in the briefing not already tried. Notebook 09 exists precisely for
this: it gives every architecture an equal tuning budget, so adding a fourth arm to the
existing `ARCHITECTURES` registry is a contained change rather than a new pipeline.

**Restormer** is the right one to try first. It applies self-attention across the feature
*channel* dimension rather than spatial pixels, so cost does not blow up quadratically with
image size — the property that makes ViTs impractical at 256px, let alone 600px.

**Expected to lose, and worth running anyway.** The briefing's own table lists the data
requirement as *"large synthetic/paired datasets"*. This project has **6 independent training
disks**. Attention carries weaker inductive bias than convolution, so it needs more data, not
less; the same argument that predicts the DDPM underperforms predicts a transformer will too.
"We gave a transformer an equal budget and the U-Net still won at this data scale" is a
legitimate result, and 09 is built to make exactly that claim fairly.

**Effort:** ~1 day for the architecture entry plus its search space; then one 09 sweep arm.
**Risk:** low technically. The honest risk is spending a day to confirm an expected negative
— acceptable, since the negative is itself reportable.

**A caveat on the SNR figures in that briefing.** Its 30–45 dB ranges come from *fluorescence
microscopy and MRI*. This project's U-Net sits at 37–39 dB, which looks competitive against
that table and means nothing: different noise statistics, different dynamic range, different
data. PSNR is not portable across domains. Cite the **classical baselines in notebook 07**,
which are measured on this data, not a microscopy benchmark.

## Ordering

1. **Phase 0** — determine the forward operator. Cheap, and it gates everything below.
2. **Ask Jason** — PSF image, visibilities, and how the dirty cubes were made. Send this at
   the same time as Phase 0; his answer to (3) may settle Phase 0 outright.
3. **Phase 3a — spectral continuity.** Ungated by Phase 0 (it needs no beam operator), so
   it can start immediately, and it targets the diagnosed cause of the M1/M2 weakness rather
   than a symptom. Cheapest route to the weakest numbers in the project.
4. **Phase 2a — VIREO-lite.** Data-consistency loss plus beam-as-map: no sampler change,
   applies to the **U-Net** as well as the DDPM, and attacks the measured hallucination
   failure directly.
5. **Phase 3d — SNR-aware training.** Also unblocked and also cheap. The noise level is
   already computed per channel and thrown away, and hallucination is 7x worse at low SNR --
   the model is blind to precisely the variable that predicts its worst failure.
6. **Phase 3b / 3c** — non-negativity and flux conservation. Hours each, but check 3b's
   continuum-subtraction caveat before enforcing, and label 3c's M0 gain honestly.
7. **Phase 1 — DDRM**, which reuses 2a's beam operator and helps the DDPM only.
8. **Phase 4 — Restormer** as a notebook 09 arm. Expected to lose at 6 disks; the negative
   is reportable and 09 is built to make the comparison fair.
9. **Phase 2b — VIREO-full**, only if visibilities arrive; multi-week.

Note the reordering: **3a needs no beam and no mentor input**, so it is not blocked by
Phase 0 or by anything Jason has to send. It is the only item here that can start today.

Both sit behind the higher-value post-midterm items already recorded in `context.md`:
measuring the patch and native-600 arms that are implemented but unmeasured, cross-validation
over all 11 RunIDs, and folding in the new self-gravitating cubes — which takes training from
6 to 7 independent disks and is worth more than any architectural change here.
