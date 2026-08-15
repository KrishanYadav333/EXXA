# Denoising Astronomical Observations of Protoplanetary Disks

*Krishan Yadav, Google Summer of Code, ML4Sci / EXXA*
*Mentors: Jason Terry, PhD · Gaurav S.*
*Code: https://github.com/KrishanYadav333/EXXA (branch `midterm-prep`)*

> **Status.** This is the midterm update blog.

---

## 1. Why denoise a protoplanetary disk?

Planets form in protoplanetary disks, and a forming planet usually shows up not as a dot but
as a **disturbance**: it carves gaps, launches spirals, and perturbs the velocity field of
the gas. In a rotating disk the gas follows Keplerian motion, so the line-of-sight velocity is

```
v_obs(r, φ) = sqrt(G M* / r) · sin(i) · cos(φ) + v_sys
```

A planet breaks that clean pattern locally, producing a "kink" in the velocity map. Detecting
those kinks is how embedded protoplanets have actually been found in ALMA data (Terry et al.
2022, 2023). The catch: a kink is a small deviation on top of a large smooth pattern, and
interferometric data is noisy in a way that specifically attacks small deviations.

An interferometer samples the sky's Fourier transform on a sparse set of points. Inverting
that incomplete measurement gives the "dirty" image:

```
I_dirty = F⁻¹[ S(u,v) · V(u,v) ] = I_true ⊛ B_dirty
```

The corruption is **not** independent per-pixel Gaussian noise, which is what most denoising
literature assumes. It is a structured, spatially correlated sidelobe field set by the array
geometry, different for every observation. That single fact shapes every decision below.

---

## 2. The data and the metric

14 FITS cube pairs across 11 simulation RunIDs, each `(201, 600, 600)`: 201 velocity channels
of a 600×600 sky image, from hydrodynamic simulation plus radiative transfer. Each pair is a
dirty cube and its matching clean cube.

The pipeline is identical for every model tested, which is what makes the comparisons
meaningful: continuum subtraction, Gaussian channel sampling centred on channel 100,
downsample 600→256, **shared dirty-scale normalisation** (both dirty and clean normalised by
the *dirty* channel's min and max, the only statistics available at inference), model,
invert, reassemble, then moment maps.

Two commitments constrain everything. Training is **channel by channel**, because with 14
cubes there is not enough data for a 3D model. Evaluation is always on **genuinely held-out
cubes**, five of them, split by RunID so that radiative-transfer variants of one simulation
cannot straddle train and test.

### 2.1 Moment maps, the metric that actually matters

PSNR is not the deliverable. The science lives in moment maps, collapsing the cube along
velocity:

```
M0 = ∫ I(v) dv                                    integrated intensity
M1 = ∫ v · I(v) dv / ∫ I(v) dv                    intensity-weighted velocity
M2 = sqrt( ∫ I(v)·(v − M1)² dv / ∫ I(v) dv )      velocity dispersion
```

M0 says where the gas is. **M1 is the rotation map, and it is where planet-induced kinks
appear.** M2 traces turbulence. Scores are quoted as percentage improvement over the dirty
input, computed under a signal mask and a 3σ noise clip unless stated otherwise.

That caveat matters more than it sounds. This project has three metric generations (raw,
clipped, clipped + masked) and **numbers from different generations are not comparable**.
Every table below names its metric.

> **[FIGURE]** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_maps_holdout.png`
>
> *One held-out cube: clean truth against the dirty input. The intensity map survives the noise
> reasonably well; the velocity and dispersion maps are visibly destroyed. That gap is the
> thing to close.*

---

## 3. How the project got here

The project began on **continuum** images, single-channel dust maps. Four learned models
against four classical filters gave a result we did not expect:

```
Method                     PSNR      SSIM
Median 3x3                22.88     0.359
Noisy input (no filter)   21.57     0.192
AE MSE-only               20.25     0.616
AE HybridLoss             19.92     0.761
```

**Every classical filter beat every learned model on PSNR, and the untouched input scored
21.57 dB while removing no noise at all.** A blur wins on pixel error by refusing to commit.
SSIM ranks them the other way round. That is the first sign in this project that the metric
decides the answer. The U-Net was chosen for what it preserves: skip connections carry
spatial detail a bottleneck destroys. Training uses `L = α·MSE + (1−α)·(1−SSIM)`, α = 0.8.

**The mentors then redirected the project to line emission**, velocity cubes rather
than dust maps, because M1 is what planets perturb. The first line-emission run scored PSNR
26.46 dB and **M0 = −6395%**. Respectable pixels, useless science.

**Continuum subtraction was the fix.** The static dust disk contaminates the line, and because
M0 integrates over ~201 channels, a constant pedestal accumulates linearly:

```
                             PSNR       SSIM        M0
No continuum subtraction     31.02     0.9832   −1671.7%
Continuum-subtracted         33.18     0.9868     +84.9%
```

**PSNR moved 2.2 dB. M0 moved 1756 percentage points.** The same intervention barely registers
on the pixel metric and decides the scientific one.

---

## 4. The U-Net, and a bug worth more than an architecture

The 05 notebook is the main U-Net track. Its most useful lesson was not a model change.

An early version normalised each channel by **its own** min and max, then inverted with the
*clean* channel's statistics, which do not exist at inference. Worse, dirty backgrounds sit
near 0.35 after normalisation while clean backgrounds sit near 0, so a model trained against
one scale and decoded with the other places the entire empty sky at a raised pedestal. Fixing
it to a **shared dirty-scale** normalisation, invertible from information available at
inference, is what made the moment scores meaningful at all.

**Transferable lesson: any normalisation must be invertible from information available at
inference time.** A closely related failure appears in the diffusion model in Section 7.

> **[FIGURE]** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell20.png`
>
> *Five validation channels: dirty, U-Net denoised, clean truth (05 v20, `sweep_winner_p10`
> seed 42, selected on M0 rather than PSNR). Rows 1, 2 and 5 are near-perfect recoveries. Row 3
> is the failure mode this project keeps meeting: on a channel with almost no signal, the model
> invents smooth structure that is not in the truth.*

And the same checkpoint scored where it counts, on the moment maps of a held-out cube:

> **[FIGURE]** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell18_1.png`
>
> *Moment maps for one held-out cube: dirty input, U-Net denoised, clean truth, across all
> three moments. M1 and M2 are shown over the scored region only. The dirty velocity map is
> speckled and its dispersion map is pure noise; the denoised row recovers a clean rotation
> dipole. Per-cube improvements here are M0 +29.2%, M1 +64.0%, M2 +8.0%.*

---

## 5. Architectures, and classical baselines

Four architectures on identical line-emission data, same split, same schedule:

```
Model            PSNR      M0        M1        M2
U-Net           33.18   +84.9%   +20.9%    −46.1%
Autoencoder     28.94   +51.2%   +12.4%    −98.3%
VAE             29.76   −220.4%  +45.7%   −152.8%
DDPM            18.02   −812.3%   −8.1%    −74.6%
```

**Every architecture loses badly on M2**, and the U-Net is the worst of the three
non-diffusion models on dispersion while winning everything else. Per cube its M2 is negative
on all five, so it is not one bad cube dragging a mean. This is treated as an open scientific
problem rather than something to patch away.

> **[FIGURE]** `results/09-architecture-comparison/v7_2026-08-02T1721_ee491fc/architecture_moment_maps.png`
>
> *The same held-out cube through each architecture. The ordering here is the moment ordering,
> not the PSNR ordering, which is the whole point.*

Classical filters are the honest floor, and on line emission the picture inverts from the
continuum result: the best classical filter reaches **M0 +11.7%** against V12's **+69.8%** on
the same metric, a gap of 58 percentage points. A median filter is excellent at not being
wrong and useless at recovering a velocity field.

> **[FIGURE]** `results/07-classical-baselines/v2_2026-07-31T1904_eb03589/classical_vs_learned_moments.png`
>
> *Classical filters against the learned models on moment recovery.*

---

## 6. Seeds, sweeps, and augmentation

A 12-run hyperparameter sweep found a 37.11 dB configuration that we then could not
reproduce: the retrain landed at 30.28 dB. The cause was embarrassing and important.
`run_sweep` trains run *i* at seed `base + i`, so the winning row was trained at seed 49 while
the retrain used 42. **The "+4.16 dB improvement" was an order statistic over 12 lucky draws,
not a measured effect.**

The rule adopted: **no configuration is called an improvement until it is checked across
multiple seeds.** Four configurations × 3 seeds, on the raw metric that V12's published
figures use:

```
Configuration          PSNR (3 seeds)        M0              M1              M2
V12 reference           37.60 ± 1.00   +82.0 ±  8.5   +28.2 ± 17.5   +26.4 ± 16.3
Sweep winner            37.97 ± 0.90   +85.5 ±  9.3   +29.1 ± 14.5   +27.0 ± 17.0
Winner + D4 aug         39.30 ± 0.46   +87.5 ±  4.2   +25.4 ± 26.4   +26.7 ± 19.7
Winner, patience 10     39.27 ± 0.48   +85.8 ±  6.6   +35.9 ± 10.9   +24.1 ± 17.3
V12 as published        32.95 (n=1)    +69.8 ± 15.2   +17.5 ±  7.8   +20.1 ± 14.3
```

The V12 and "winner" bands overlap almost completely: the sweep result was seed noise.
**Augmentation gives the best M0 anywhere in this project, +87.5% ±4.2, at the lowest
variance.** D4 augmentation applies the 8 lossless orientations of the dihedral group
identically to dirty and clean, which on a 14-cube corpus is the cheapest regularisation
available.

**And the notebook's own promotion check refuses to call it an improvement:**

```
winner_aug vs V12 config    M0  +5.6 pp  (spread  9.5 pp)  within noise
                            M1  −2.9 pp  (spread 31.7 pp)  within noise
                            M2  +0.3 pp  (spread 25.5 pp)  within noise
```

So augmentation gives the best numbers and the tightest spread, and "augmentation beats the
reference on the science" is still **not demonstrated at n=3**. It would have been easy to
quote +87.5% against +69.8% and move on. This is the same discipline that killed the 37.11 dB
sweep result, applied to our own best number.

> **[FIGURE]** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell22.png`
>
> *All six U-Net arms on the full metric, five held-out cubes. Coloured bars span seeds, dots
> are individual cubes. The dots matter more than the bars: M0 runs from +100% to below −300%
> on the same model, which is why every claim here carries its spread.*

### 6.1 Invented structure, and where it lives

The most scientifically serious failure mode is not blur, it is **hallucination**: plausible
structure in channels that contain none. A per-channel diagnostic counts invented regions
against each channel's own noise floor:

```
Configuration          Channels with a blob   Peak overshoot
V12 config                     39.0%              2.741
Sweep winner                   37.7%              2.349
Winner + augmentation          29.7%              1.736
Winner, patience 10            22.3%              1.889
```

**Augmentation reduces hallucination** (37.7% → 29.7% at otherwise identical settings), which
is exactly the hypothesis the experiment was built to test. And **hallucination concentrates
in faint channels**: 1.580 invented blobs per channel below the median SNR against 0.213
above it, roughly 7× at a split of SNR 3.9.

An earlier version of this diagnostic reported **0.0% invented structure for every
configuration**. That was not a clean result, it was a broken detector: the background mask
selected zero pixels, so nothing could fire. A zero from a diagnostic is a suspect, not a pass.

---

## 7. The DDPM, and why it fails

A conditional DDPM is the natural thing to try: it models a distribution rather than a mean,
so it should not blur. Rebuilt properly on line emission, with v-prediction, a cosine
schedule and min-SNR weighting, it reached **PSNR 36.03 dB**, genuinely competitive with the
U-Net. Then the moment maps:

```
Model (all rows: mask + clip)        M0             M1             M2
U-Net, winner + D4 aug          +29.2% ±  7.2  +74.0% ± 2.0  +55.0% ± 13.9
U-Net, winner + patience 10     +33.5% ±  9.6  +70.7% ± 6.7  +31.8% ± 11.1
U-Net, V12 config                −4.4% ± 33.9  +58.0% ±20.0  +15.5% ± 31.2
DDPM                            −56.1% ±152.2  +13.7% ±90.0   +5.2% ± 83.6
```

The best U-Net arm beats the DDPM by **85 percentage points on M0** with a spread twenty times
tighter. (These four rows share one metric. Run 05 v20 re-scored fifteen U-Net checkpoints
under the DDPM's exact metric, without retraining any of them, specifically so this comparison
could be made honestly.)

**The diagnosis is a pedestal, not lost structure.** The sampler returns values in
`[0.348, 0.701]` when it should span roughly `[0, 1]`: it never learns to output true black,
so empty sky is decoded to a constant offset. Because M0 is a **sum** over ~201 channels, a
constant bias accumulates linearly while random noise grows only as √N. A bias 14× smaller
than the per-channel noise still dominates M0. That is why the DDPM can look competitive on
PSNR and fail catastrophically on integrated intensity.

Five re-scores tested the diagnosis without retraining anything:

```
arm                        M0                M1                M2
baseline (K=4)        −56.9 ±149.9       13.7 ±90.0         5.2 ±83.6
kavg1 (K=1)           −51.4 ±148.7       15.1 ±90.3         6.7 ±84.2
rescaled (K=4)       −264.5 ±158.7       38.3 ±24.6        42.4 ±24.4
kavg1 + rescaled     −190.5 ±107.2       51.6 ±10.8        61.3 ±10.8
```

**Averaging is not the cause**: dropping from 4 posterior draws to 1 moves M0 by 4.7
percentage points against a 56% deficit, and is 4× faster. **Rescaling is not the fix for
M0** either: matching the output's mean to the dirty input re-imposes the very pedestal that
needs removing, and a synthetic test confirmed it structurally, taking a *perfect* denoiser
from +100% to −0.0% on M0.

But the same rescale **fixes M1 and M2**, which are ratios and therefore insensitive to a
constant offset while sensitive to range. At **M2 +61.3% ±10.8** the rescaled DDPM beats every
U-Net arm on dispersion, the one moment on which every architecture had been negative. It is
the project's first DDPM result that is better than the U-Net at anything.

There is also a structural reason a diffusion model should lose on pixel error. A regression
model trained on MSE learns the posterior **mean**; a single diffusion draw is a **sample**
from the posterior. For the same posterior, E‖x₀−μ‖² = tr(Σ) while E‖x₀−x̃‖² = 2·tr(Σ), a
3.01 dB penalty by construction. The measured PSNR gap is ~1 dB, so sampling explains the
pixel gap and **nothing about the moment failure**, which is a bias problem.

---

## 8. What is next

1. **Kill the diffusion pedestal.** One test: rescale std only, leaving the mean untouched.
   If M2's +61.3% survives while M0 stops collapsing, the DDPM becomes useful for dispersion.
2. **Fix M2 for the U-Net.** Negative dispersion recovery is the most consistent failure in
   the project and is not yet explained.
3. **Suppress hallucination in faint channels**, where it is concentrated 7:1.
4. **Score on ALMA-like data**, applying realistic beam convolution and noise, then real
   archival observations.
5. **Seed-validate the exploratory arms.** Beam conditioning and 64px patches were each run
   at one seed and both failed badly (M0 −95.7% and −40.8%); one seed is enough to deprioritise
   them, not enough to call them settled.

---

## 9. Closing

The through-line of this midterm is a single lesson, learned repeatedly and expensively:
**the metric decides the answer, and pixel metrics are the wrong metric here.**

It showed up as classical filters beating every learned model on PSNR while destroying
structure. As an untouched noisy input scoring 21.57 dB. As continuum subtraction moving PSNR
by 2.2 dB and M0 by 1756 points. As a diffusion model reaching 36.03 dB and −56% M0. As beam
conditioning taking the second-best PSNR in a six-arm table and the worst M0 by a factor of
two. Every one of those would have been read as a success on pixel error alone.

What exists now: a reproducible pipeline over five self-bootstrapping notebooks, a
seed-validated U-Net, moment-map evaluation on genuinely held-out cubes, classical baselines,
an architecture comparison, a diagnosed diffusion failure with a measured mechanism, and a
run index mapping every number in this post to the notebook version that produced it,
including the numbers later found to be wrong, which are kept and marked rather than quietly
replaced.

What does not exist yet: a model that recovers velocity dispersion, a diffusion model that
beats the U-Net on intensity, and any result at all on real ALMA observations. Those are the
second half.

### Reproducing this

Everything is in [github.com/KrishanYadav333/EXXA](https://github.com/KrishanYadav333/EXXA)
on branch `midterm-prep`: `DENOISING_DIFFUSION/src/` for the library, the numbered notebooks
for the runs, `results/RUNS.md` for the run index, and `tests/` for the checks that guard the
evaluation code.

### Acknowledgements

Thank you to my mentors **Jason Terry** and **Gaurav S.** for direction that repeatedly turned
out to be right, in particular the pivot to line emission and the insistence on moment maps as
the deliverable rather than pixel metrics. Thanks also to the wider ML4Sci / EXXA organisation.

*Tags: Machine Learning, Astrophysics, Diffusion Models, Exoplanets, Google Summer of Code*
