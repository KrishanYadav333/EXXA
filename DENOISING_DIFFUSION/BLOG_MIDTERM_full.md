# Denoising Astronomical Observations of Protoplanetary Disks

*Krishan Yadav, Google Summer of Code 2026, ML4Sci / EXXA*
*Mentors: Jason Terry, PhD · Gaurav S.*
*Code: https://github.com/KrishanYadav333/EXXA (branch `midterm-prep`)*

> **Status.** This is the midterm update blog.

---

## 1. Background: why denoise a protoplanetary disk?

Planets form in protoplanetary disks: rotating envelopes of gas and dust around young stars.
When a planet forms inside one, it does not usually show up as a dot. It shows up as a
*disturbance*. The planet's gravity carves gaps, launches spiral arms, and, most usefully,
perturbs the velocity field of the gas around it. In a rotating disk the gas follows
Keplerian motion, so the line-of-sight velocity at radius r and azimuth φ is

```
v_obs(r, φ) = sqrt(G M* / r) · sin(i) · cos(φ) + v_sys
```

where i is the disk inclination and v_sys the systemic velocity. A planet breaks that clean
pattern locally, producing a "kink" in the velocity map. Detecting those kinks is how
embedded protoplanets have actually been found in ALMA data (Terry et al. 2022, 2023).

The catch: the kink is a small deviation on top of a large smooth pattern, and the data we
get from a radio interferometer is noisy in a way that specifically attacks small deviations.

**This is not generic image denoising.** An interferometer like ALMA does not photograph the
sky. It measures the sky's Fourier transform, sampled only at the spatial frequencies its
antenna baselines happen to cover. By the van Cittert–Zernike theorem, each baseline measures
one complex visibility:

```
V(u, v) = ∫∫ I(l, m) · exp[ -2πi (u·l + v·m) ] dl dm
```

The array samples V only on a sparse set of (u, v) points, described by a sampling function
S(u, v). Inverting that incomplete measurement gives the "dirty" image:

```
I_dirty = F⁻¹[ S(u,v) · V(u,v) ] = I_true ⊛ B_dirty ,   B_dirty = F⁻¹[S]
```

The dirty image is the true sky **convolved with the dirty beam**. The corruption is not
independent per-pixel Gaussian noise, which is what most denoising literature assumes. It is
a structured, spatially correlated sidelobe field set by the array geometry: long-range
correlations, anisotropic, and different for every observation.

That single fact is why an off-the-shelf denoiser is not obviously the right tool, and it
shapes every decision below.

---

## 2. System architecture

The pipeline is the same for every model tested, which is what makes the comparisons in this
post meaningful.

```
FITS cube pair (dirty, clean)
        │
        ├─ continuum subtraction        mean of first/last N line-free channels
        │
        ├─ channel sampling             Gaussian centred on channel 100
        │
        ├─ downsample 600 → 256         memory; reversed at evaluation
        │
        ├─ shared dirty-scale norm      both dirty and clean use the DIRTY channel's
        │                               (min, max), the only stats available at inference
        │
        ├─ MODEL                        U-Net / autoencoder / VAE / conditional DDPM
        │
        ├─ invert normalisation         per channel, using the stored dirty (min, max)
        │
        ├─ reassemble the full cube     201 channels, back to 600×600
        │
        └─ moment maps (bettermoments)  M0 / M1 / M2, compared clean vs dirty vs denoised
```

Two design commitments worth stating because they constrain everything:

**Channel-by-channel, not full-cube.** Each of the ~201 velocity channels is an independent
training example. A 3D model over the whole cube would see the velocity structure directly,
but with only 14 cubes there is not enough data to train one.

**Evaluation is always on genuinely held-out cubes.** Five cubes are reserved, never seen in
training or validation, denoised channel by channel, reassembled, and scored on moment maps.
Every headline number in this post comes from that protocol.

The repository is five self-bootstrapping Kaggle notebooks over a shared `src/` library, a
test suite, and `results/RUNS.md`, a run index mapping every number to the notebook version
that produced it, including the numbers later found to be wrong.

---

## 3. The data, and what it looks like before anything is done to it

14 FITS cube pairs spanning 11 distinct simulation RunIDs, each of shape `(201, 600, 600)`:
201 velocity channels of a 600×600 sky image. Synthetic observations built from hydrodynamic
simulation plus radiative transfer (PHANTOM and MCFOST lineage). Each pair is a dirty cube
and its matching clean cube. Roughly 7.6 GB, hosted as a Kaggle Dataset.

> **[FIGURE LAYOUT]** the next three figures render as one block: `sample_grid.png` fills the
> left column, with `noise_difference.png` top-right and `pixel_distribution.png` beneath it.

> **[FIGURE]** `results/stats/sample_grid.png`
>
> *Sample channels across the dataset. The variety in disk morphology, inclination and
> brightness across simulations is what the model has to generalise over from six independent
> training disks.*

> **[FIGURE]** `results/stats/noise_difference.png`
>
> *Dirty minus clean: the corruption itself, isolated. This is the structured sidelobe field
> from Section 1, not white noise, note the spatial correlation and the way it concentrates
> around bright emission.*

> **[FIGURE]** `results/stats/pixel_distribution.png`
>
> *Pixel intensity distributions, dirty against clean. Clean puts almost all its mass in a
> single spike at zero: in the truth, empty sky is exactly empty. Dirty smears that background
> into a broad hump past 0.4. The two backgrounds do not sit at the same value, which is what
> breaks when a model trained against one scale is decoded with the other (Section 8.1).*

**Cube-level splitting, grouped by RunID.** Radiative-transfer variants of the same
simulation (`run_0002_00560_rt_00`, `_rt_01`, `_rt_04`) are near duplicates. Splitting them
across train and test would leak. Grouping by RunID gives:

```
TRAIN     7 cubes   RunIDs 0006, 0010, 0020, 0022, 0030, 0035
VAL       2 cubes   RunIDs 0016, 0036
HOLDOUT   5 cubes   RunIDs 0002, 0025, 0026    inference only, never trained or validated
```

Six independent disks for training. That is a small number, and it shapes several results
below.

**Channel sampling.** Channels are drawn per cube from a Gaussian centred on channel 100,
avoiding the line-free extremes where there is nothing to denoise. Calibrated to 73.8% of
sampling mass inside channels [50, 150], against the mentor's ~75% guidance.

### 3.1 The metric that actually matters: moment maps

A denoised cube that looks nice is worth nothing if the science extracted from it is wrong.
Astronomers do not analyse raw channels; they collapse the cube along velocity into **moment
maps**. Using `bettermoments`, for a cube I(v) at each sky pixel:

```
M0 = ∫ I(v) dv                                    integrated intensity
M1 = ∫ v · I(v) dv  /  ∫ I(v) dv                  intensity-weighted velocity
M2 = sqrt( ∫ I(v) · (v − M1)² dv  /  ∫ I(v) dv )  velocity dispersion
```

M0 tells you where the gas is. M1 is the rotation map, and it is where planet-induced kinks
appear. M2 traces turbulence and line broadening.

> **[FIGURE]** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_maps_holdout.png`
>
> *Moment maps of one held-out cube: clean truth against the dirty input. The intensity map
> survives the noise reasonably well; the velocity and dispersion maps are visibly destroyed.
> That gap is the thing to close.*

Improvement over the dirty cube, per moment, averaged over the 5 held-out cubes:

```
Improvement(M) = 100 × ( 1 −  mean |M_denoised − M_clean|
                            ─────────────────────────────  )
                              mean |M_dirty    − M_clean|
```

100% means perfect recovery, 0% means no better than the dirty input, and negative means the
denoiser made the science *worse* than doing nothing. Later runs average this over a **signal
mask**, the pixels where the clean M0 exceeds a fixed fraction of its peak, because scoring
over the whole map lets empty sky dominate, and the dispersion of a pixel containing no line
is not a quantity anyone reports.

---

## 4. First architecture comparison, on continuum images

The project did not start on line emission. The first weeks worked on **continuum** images:
single-channel dust maps rather than velocity cubes. Four learned models against four
classical filters, 100 validation samples.

```
Method                     PSNR      SSIM        MSE
Median 3×3                22.88     0.359     0.006317
Gaussian σ=2              22.78     0.423     0.006380
Gaussian σ=1              22.73     0.389     0.006462
Wiener                    22.57     0.340     0.006702
Noisy input (no filter)   21.57     0.192     0.008391
AE MSE-only               20.25     0.616     0.012804
VAE (MSE+SSIM+KL)         20.03     0.612     0.013077
AE HybridLoss             19.92     0.761     0.013920
```

> **[FIGURE]** `results/stats/baseline_metrics_chart.png`
>
> *The same comparison as a chart. The ordering flips completely depending on which bar you
> read.*

Read the first two columns against each other. **Every classical filter beats every learned
model on PSNR, and loses to all of them on SSIM.** Median 3×3 scores 22.88 dB against the
autoencoder's 19.92, while its SSIM is 0.359 against 0.761. Worse, the *unfiltered noisy
input* scores 21.57 dB, beating every learned model on PSNR while having, by construction,
done nothing at all.

> **[FIGURE PAIR]** the next two figures render side by side.

> **[FIGURE]** `experiments/unet_vs_all_comparison.png`
>
> *U-Net against every alternative on continuum data. The visual ordering is the SSIM
> ordering, not the PSNR ordering, which is the whole point.*

> **[FIGURE]** `experiments/hybrid_vs_mse_comparison.png`
>
> *MSE-only against the hybrid MSE+SSIM loss on the same architecture. MSE alone produces the
> smoother, lower-error, less correct image.*

**Why we chose the U-Net.** Three reasons, in order of weight:

1. **Skip connections preserve high-frequency detail.** Recovering a faint ring means keeping
   spatial detail that a bottleneck destroys. Both bottlenecked architectures, autoencoder
   and VAE, sit below the U-Net on structural metrics in every comparison run since.
2. **A VAE optimises for plausible, not correct.** Its latent bottleneck plus KL
   regularisation is designed to produce samples that look like the data distribution. For a
   measurement instrument that is the wrong objective: we need the disk that is there, not a
   disk that could be there.
3. **PSNR was already demonstrably misleading.** A denoiser that blurs everything wins on mean
   squared error and destroys the science. That observation, from this table, is why moment
   maps became the deliverable when the project moved to line emission.

The hybrid loss used everywhere after this comes from the same comparison:

```
L = α · MSE(ŷ, y) + (1 − α) · (1 − SSIM(ŷ, y)),    α = 0.8
```

MSE anchors numerical fidelity; the SSIM term protects structure, which per-pixel error alone
will happily smooth away.

---

## 5. The first DDPM run, and why it failed

A conditional DDPM was built alongside the continuum autoencoder work, because diffusion is
named in the project description and is the natural competitor to a regression model.

It did not work. On the continuum data it trailed every other model, and on the early
line-emission runs it scored **PSNR ~18–19 dB** against the U-Net's 32–33 dB on identical
data, a gap of more than 14 dB.

> **[FIGURE]** `results/_archive-continuum-era/superseded_2026-07-25T1305_bf7a819/diffusion_loss.png`
>
> *DDPM training loss from the continuum era. It converges; it simply converges to something
> that is not competitive.*

At the time this was attributed to the **data regime**: a diffusion model learns a full
distribution p(x₀ | y), which needs far more independent samples than a regressor needs to
learn a conditional mean, and six independent training disks is not much. That explanation
was reasonable, widely believed, and, as Section 12 shows, **wrong**.

The DDPM was kept alive as a documented comparison rather than dropped, which turned out to
matter.

---

## 6. The shift to line emission

The mentors redirected the project from continuum images to **line-emission
velocity cubes**. This was the optional stretch goal in my accepted proposal and became the
primary track.

The change is larger than it sounds. Continuum work is single-image denoising with a scalar
quality metric. Line emission is a 201-channel velocity cube where the deliverable is a
*kinematic* map, the thing planets actually perturb. Everything downstream changed with it:
the evaluation protocol became moment maps on held-out cubes, and the pixel metrics that had
ranked the continuum models became a secondary check rather than the score.

The first line-emission run scored **PSNR 26.46 dB, SSIM 0.810** on validation
channels. Respectable. It also scored **M0 = −6395%** on the held-out cube, the denoiser had
made integrated intensity sixty-four times worse than doing nothing.

Two separate causes, each worth thousands of percent, are the subject of the next two
sections.

---

## 7. Continuum subtraction

The static dust disk contaminates the line emission. Because M0 is an *integral over
channels*, a constant pedestal in every channel accumulates linearly into the intensity map, 
so a contaminant that barely moves per-channel PSNR can dominate the moment map completely.

The fix: estimate the continuum from the line-free channels at each end of the cube and
subtract it from every channel before normalisation.

This was tested as a direct side-by-side in the same notebook, same seed, same split:

```
                              PSNR      SSIM        M0        M1       M2
No continuum subtraction     31.02     0.9832   −1671.7%   +13.6%   +18.6%
Continuum-subtracted         33.18     0.9868     +84.9%   +20.9%   +18.4%
```

**PSNR moved 2.2 dB. M0 moved 1756 percentage points**, from catastrophically wrong to
usefully right. That ratio is the clearest single illustration in this project of why pixel
metrics cannot be trusted as the score here.

> **[FIGURE]** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_maps_continuum_comparison.png`
>
> *Moment maps with and without continuum subtraction. The contaminated version is not subtly
> worse; the intensity map is dominated by the dust pedestal.*

A follow-up ablation tested how many line-free channels to average for the estimate: n=1
scored at least as well as n=5 on every pixel metric (SSIM 0.9867 against 0.9793). The
default stayed at n=5 as the more conservative estimate, and the question is flagged for the
mentors rather than settled unilaterally.

---

## 8. The U-Net on line emission: what each run changed

The 05 notebook is the main U-Net track. Its version history is the project's spine, so it is
worth reading as a sequence rather than a single result.

```
Run    PSNR    SSIM    M0 on holdout   What changed
V-0    26.46   0.810        −6395.4%   first end-to-end line-emission run
V5     28.16   0.843        −6395.4%   normalisation bug still present
V6     31.73   0.985        −1736.8%   normalisation FIXED (see 8.1)
V7     33.18   0.987          +84.9%   continuum subtraction added (see 7)
V9     33.04   0.986          +61.9%   continuum-window ablation, n=1 vs n=5
V12    32.95   0.986   +69.8% ± 15.2   5-cube holdout protocol, REFERENCE
V15    34.94   0.987   +59.8% ± 33.0   beam conditioning + 12-run sweep
V16    30.28   n/a     +64.4% ± 14.6   sweep-winner retrain, did NOT reproduce
V17    29.92   n/a     +48.2% ± 12.6   second retrain, M0 fell further
V18    35.94   n/a     +81.2% ± 12.6   best to date, 5/5 cubes positive
V19    n/a     n/a               n/a   crashed at 4.0 h, no moment scores
```

### 8.1 The normalisation bug, and why it was worth more than any result

The first runs produced M0 in the −6000% range while per-channel SSIM sat at a healthy 0.81.
That contradiction is the most instructive thing that happened all summer.

The cause: training normalised the clean target by *clean's own* per-channel min and max,

```
ỹ = (y − min y) / (max y − min y)
```

so the network learned to emit values in clean-normalised space. But at inference, clean's min
and max are unavailable **by definition**, they are what we are predicting. Decoding
therefore used dirty's. Clean's background floor sits near zero while dirty's minimum is
negative (see the pixel distribution figure in Section 3), so every background pixel decoded
to dirty's negative floor instead of zero, imposing a DC offset on each channel. Summed over
201 channels, M0 went sharply negative.

The fix is one line of principle: **normalise both dirty and clean by the dirty channel's
statistics**, the only ones available at inference time.

```
x̃ = (x − min D) / (max D − min D)      for x ∈ {dirty, clean},  D = dirty channel
```

Clean is deliberately not clipped, since its peak can exceed 1 on this shared scale, which in
turn required replacing the sigmoid output with a linear head. Verified by channel diagnostic:
the denoised minimum moved from −0.0047 (tracking dirty, wrong) to +0.0001 (tracking clean,
correct). V6 shows the result: PSNR jumps 3.6 dB and M0 improves by 4659 percentage points
while still being wrong, because the continuum problem of Section 7 remained.

**Transferable lesson: any normalisation must be invertible from information available at
inference time.** Keep it in mind for Section 12, where a closely related failure appears in
the diffusion model.

> **[FIGURE]** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell20.png`
>
> *Five validation channels, dirty, U-Net denoised, clean truth, from 05 v20 (`sweep_winner_p10` seed 42, the checkpoint selected on M0 rather than PSNR). The same five channels the DDPM figure in Section 12 shows, so the two are directly comparable. Rows 1, 2 and 5 are near-perfect recoveries. Row 3 is the failure mode this project keeps returning to: a channel with almost no signal, where the model invents a smooth structure that is not in the truth.*

### 8.2 V12, the reference checkpoint

V12 consolidated the working configuration and added the 5-cube holdout protocol, which gives
error bars for the first time:

```
V12, 5 held-out cubes         PSNR 32.95   SSIM 0.9857   MSE 0.000681
  M0   +69.8% ± 15.2
  M1   +17.5% ±  7.8
  M2   +20.1% ± 14.3
```

> **[FIGURE]** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_map_holdout_summary.png`
>
> *V12 across all five held-out cubes. Positive on all three moments on every cube, the
> result that made V12 the reference every later experiment is measured against.*

### 8.3 The current best checkpoint

The most recent 05 run (v18) is the strongest to date:

```
v18, 5 held-out cubes         PSNR 35.94  (+2.99 dB over V12)
  M0   +81.2% ± 12.6
  M1   +31.5% ±  9.2
  M2   +19.9% ± 18.8
```

5/5 cubes positive on all three moments, and the first run to beat V12 on M0 and M1 together.
Caveat stated plainly: v18 was a **single seed**, so it is the strongest single run to date
rather than a seed-validated configuration, a distinction Section 11 shows to be essential.

> **[FIGURE]** `results/05-unet-line-emission/v18_2026-08-11_b24c84d/figure_cell16.png`
>
> *v18 against the V12 reference, five held-out cubes, dots are individual cubes. M0 and M1 both
> clear V12 while M2 lands level. This is the run that superseded V12 as reference checkpoint.*

---

## 9. Architecture comparison on line-emission data

The continuum-era comparison of Section 4 chose the U-Net, but on different data with a
different metric. Notebook 09 re-ran the question properly: same data, same schedule, same
5-cube holdout, and, rather than one configuration per architecture, a **24-run sweep, 8
configurations each**.

```
Architecture   runs   best PSNR   median   worst   best SSIM
U-Net             8      39.235   37.679  37.081      0.9946
Autoencoder       8      33.633   31.648  19.659      0.9904
VAE               8      30.498   21.152  16.899      0.9819
```

The U-Net's best configuration, `base_channels=16`, multipliers `1×2×4×8`, lr 0.00124,
α 0.745, reaches **39.235 dB at SSIM 0.9946**, the best pixel-metric result anywhere in this
project and 6.3 dB above V12.

The sweep earns its cost by showing something a single run per architecture would have hidden:
**the spread within an architecture rivals the gap between architectures.** The VAE ranges
16.9 to 30.5 dB across its own 8 configurations, a 13.6 dB range that swallows its entire
deficit to the U-Net. An architecture comparison built on one run each would have been
measuring configuration luck.

On the moment maps, with parameter counts:

```
Architecture      params      PSNR          M0                M1                M2
U-Net          3,477,665    37.414    −33.2% ±  71.8    +68.7% ± 22.5   −152.8% ± 81.6
Autoencoder    1,734,305    33.155   −152.0% ± 267.8    +58.2% ± 30.2    −46.1% ± 44.1
VAE              467,793    29.760   −220.4% ± 295.1    +45.7% ± 29.1    −75.9% ± 70.0
───────────────────────────────────────────────────────────────────────────────────
U-Net V12      3,400,000    32.950    +69.8% ±  15.2    +17.5% ±  7.8    +20.1% ± 14.3
                                      (V12 row is the OLDER metric, continuity only)
```

The notebook prints its own verdict on the U-Net's lead:

```
best M0: unet at −33.2%
  vs autoencoder   +118.8 pp   (pooled spread 277.2)  -> within cube-to-cube spread
  vs vae           +187.2 pp   (pooled spread 303.7)  -> within cube-to-cube spread
```

Under the clipped metric **every architecture scores negative on M0**, and the U-Net leads not
by being good but by being least bad, with the margin inside the cube-to-cube spread on both
comparisons. Each architecture here is a single retrained run, so there is no seed spread at
all. The honest statement is that the U-Net leads and the lead is not proven at n=1. What is
solid is M1: +68.7% ± 22.5 is the best velocity recovery measured anywhere in this project,
and the velocity field is the moment planets actually perturb.

The VAE's collapse is the clearest architectural signal in the table: −220.4% on M0 with 7×
fewer parameters than the U-Net, so the deficit is architectural, not a capacity artefact.

**Every architecture loses badly on M2**, from −46.1% to −152.8%, and the U-Net is the worst
of the three. Per cube its dispersion score is negative on all five, from −11.9% to −206.2%,
so it is not one bad cube dragging a mean. This recurs in every experiment below and is
treated as an open scientific problem rather than something to patch away.

> **[FIGURE]** `results/09-architecture-comparison/v7_2026-08-02T1721_ee491fc/architecture_moment_maps.png` (09 v7, the run the numbers above come from)
>
> *Moment maps by architecture on one held-out cube. The U-Net row is visibly closest to the
> clean truth on M0 and M1. The autoencoder and VAE rows show the boxy, over-smoothed
> reconstruction their bottlenecks impose.*

### 9.1 The same checkpoints, scored twice, disagree

The table above is the clipped metric. Earlier in the project the same three checkpoints were
scored *without* the 3σ noise clip, which was added to stop the dispersion map reading the
noise floor as real signal. Nothing about the models changed between these two columns:

```
Moment    no clip     3σ clip     change
M0         +77.0%      −33.2%    −110.2 pp
M1         +22.2%      +68.7%     +46.6 pp
M2          +8.5%     −152.8%    −161.3 pp
```

Nothing about the models changed. The metric did, and it did not shift everything one way:
**M1 more than tripled while M0 and M2 collapsed.** Under the clip the U-Net's M1 becomes the
best velocity recovery measured anywhere in this project, and its M0 goes negative.

The ranking survives, U-Net still leads both alternatives on M0 under the clip (−33.2%
against −152.0% and −220.4%), so "U-Net is the right architecture" holds. Any *number*
attached to that claim depends on which metric version produced it. Re-scoring every archived
checkpoint on one final metric is the largest piece of unfinished bookkeeping in this project.

---

## 10. Classical baselines on line emission

Before this project, no classical filter had been scored on this line-emission dataset, so
"the learned model beats traditional processing" was an assumption. Gaussian, median and
Wiener filters were tuned on the validation split and run through the identical 5-cube
protocol.

Channel level, 300 validation channels at 256×256:

```
Method                     PSNR      SSIM         MSE
Dirty (unfiltered)        20.43     0.310     0.010952
Median (window 9)         22.80     0.530     0.006658
Wiener (window 9)         22.83     0.538     0.006596
Gaussian (σ = 4)          24.96     0.701     0.004022
U-Net (V12)               32.95     0.986     0.000681
```

Moment improvements, 5 held-out cubes:

```
Method                                   M0            M1           M2
Dirty (unfiltered)                  0.0 ± 0.0     0.0 ± 0.0    0.0 ± 0.0
Median 9                           +3.9 ± 0.9    +0.3 ± 0.3   +0.2 ± 0.1
Wiener 9                           +5.0 ± 1.0    +0.3 ± 0.5   +0.2 ± 0.2
Gaussian σ=4, native 600           +11.7 ± 2.1   +1.1 ± 0.9   +0.5 ± 0.4
Gaussian σ=4, at 256 round trip    +36.5 ± 3.5   +4.3 ± 3.7   +2.1 ± 1.8
U-Net V12                          +69.8 ± 15.2  +17.5 ± 7.8  +20.1 ± 14.3
```

The honest comparison is the last two rows, both evaluated at the resolution the network
operates at: a **+33.3 percentage point** advantage on integrated intensity. Classical filters
do help, and they are nearly free, but they plateau early. Smoothing removes noise and signal
together; there is no σ that separates a faint ring from a sidelobe.

Note also how the gap *widens* across moments, +33.3 pp on M0, but classical filters score
+1.1% and +0.5% on M1 and M2 against the U-Net's +17.5% and +20.1%. The kinematic moments,
the ones carrying planet signatures, are where learned denoising earns its place.

> **[FIGURE]** `results/07-classical-baselines/v2_2026-07-31T1904_eb03589/classical_vs_learned_moments.png`
>
> *Learned against classical across all three moments, identical protocol.*

**This run also caught its own methodology bug.** Filters were tuned at 256 px but applied at
the cubes' native 600 px, and filter parameters are in pixels, so σ = 4 tuned at 256 smooths
2.3× too little at 600. Compounding it, the chosen σ = 4.0 sat on the *edge* of the search
grid, so the filter was never given its best setting. Both handicaps were on the classical
side, the opposite of the notebook's stated "generous to classical" framing. The grid was
widened and an explicit warning now fires on grid-edge hits.

---

## 11. Seeds, sweeps, augmentation

**Beam conditioning: a useful negative.** The observation beam is exposed to the network as a
4-vector `[sin 2·BPA, cos 2·BPA, BMAJ, BMIN]` and injected through the time-embedding path so
it reaches every block. It improved PSNR by **+1.24 dB**. It also dropped M0 from +69.8% to
+59.8% and doubled the variance:

```
                    PSNR         M0             M1            M2
U-Net V12          32.95   +69.8 ± 15.2   +17.5 ± 7.8   +20.1 ± 14.3
U-Net + beam       34.94   +59.8 ± 33.0   +19.2 ± 8.4   +23.2 ± 20.2
```

A project measuring only pixel metrics would have shipped this as an improvement. It shipped
as a documented "do not".

**A 12-run hyperparameter sweep found a 37.11 dB configuration**, and then we could not
reproduce it: the retrain landed at 30.28 dB, worse than the reference. The cause turned out
to be embarrassing and important. `run_sweep` trains run *i* at seed `base + i`, so the winning
row had been trained at seed 49 while the retrain used seed 42, and early stopping fired on a
different, noisier validation trajectory. **The "+4.16 dB improvement" was an order statistic
over 12 lucky draws, not a measured effect.**

The rule adopted: **no configuration is called an improvement until it has been checked across
multiple seeds.** Four configurations × 3 seeds = 12 training runs:

```
Configuration          PSNR (3 seeds)        M0              M1              M2
V12 reference           37.60 ± 1.00   +82.0 ±  8.5   +28.2 ± 17.5   +26.4 ± 16.3
Sweep winner            37.97 ± 0.90   +85.5 ±  9.3   +29.1 ± 14.5   +27.0 ± 17.0
Winner + D4 aug         39.30 ± 0.46   +87.5 ±  4.2   +25.4 ± 26.4   +26.7 ± 19.7
Winner, patience 10     39.27 ± 0.48   +85.8 ±  6.6   +35.9 ± 10.9   +24.1 ± 17.3
V12 as published        32.95 (n=1)    +69.8 ± 15.2   +17.5 ±  7.8   +20.1 ± 14.3
```

The V12 and "winner" bands overlap almost completely: the sweep result was indeed seed noise.
**Augmentation produces a real gain and halves the variance.** D4 augmentation applies the 8
lossless orientations of the dihedral group (4 rotations × optional flip) identically to
dirty and clean. On a 14-cube corpus that is the cheapest regularisation available.

**M0 +87.5% ±4.2 over three seeds**, the highest integrated-intensity recovery measured
anywhere in this project, at the lowest variance, and directly comparable to V12's published
+69.8% because both use the same metric. That last clause is the reason this table is the
one quoted here: the whole project is measured against V12, and this is the only
seed-validated result that can be placed beside it without changing the ruler.

The same twelve checkpoints were later re-scored with a 3σ noise clip added. Not a retrain,
not new weights, only a different metric, and it moves the numbers far enough that the two
tables must never be read against each other:

```
Configuration          PSNR (3 seeds)        M0              M1              M2
V12 reference           37.60 ± 1.00   +27.7 ± 17.6   +71.9 ± 10.3   −10.5 ± 39.7
Sweep winner            37.97 ± 0.90   +18.1 ± 45.9   +71.3 ± 16.3    −2.7 ± 63.1
Winner + D4 aug         39.30 ± 0.46   +38.7 ± 47.6   +75.4 ± 13.9   +40.3 ± 30.7
Winner, patience 10     39.27 ± 0.48   +11.6 ± 58.7   +71.1 ± 18.8   −28.3 ± 83.6
```

The clip alone moves M1 by roughly 4× and flips M2's sign. Same models, same cubes, same
seeds. It is a useful reminder that "M0 improved by X%" is not a fact about a model until
the metric is named, and it is why the DDPM comparison later in this post is careful to say
which of these rulers it is using.

**A third ruler, and the one the DDPM is measured on.** Run 05 v20 scored fifteen U-Net
checkpoints under the full metric, signal mask *and* 3σ clip, reusing every existing
checkpoint and training nothing. It adds two exploratory arms the earlier runs never scored:

```
Configuration          PSNR       M0             M1            M2
Winner + D4 aug        39.30  +29.2 ±  7.2  +74.0 ±  2.0  +55.0 ± 13.9
Winner, patience 10    39.27  +33.5 ±  9.6  +70.7 ±  6.7  +31.8 ± 11.1
Sweep winner           37.52  +11.4 ± 27.4  +55.6 ±  9.0   +6.0 ± 35.1
V12 config             37.60   −4.4 ± 33.9  +58.0 ± 20.0  +15.5 ± 31.2
Winner + beam          38.71  −95.7          +14.1         −27.3        (1 seed)
Winner, 64px patches   33.96  −40.8          +43.9         +18.5        (1 seed)
```

Augmentation and patience remain the two arms worth having, and augmentation still carries
the tightest spread on every moment. The two exploratory arms both fail, and the beam arm
fails hard: **M0 −95.7%**, worse than doing nothing at all. That is the third independent
time beam conditioning has looked good on a pixel metric and bad on the science, and at
38.71 dB it is the *second-highest PSNR in the table*. The patch arm is the honest negative
result of the two, and it is also the cheapest to explain: 64×64 crops cannot see a disk.

> **[FIGURE]** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell22.png`
>
> *All six arms on the full metric, five held-out cubes. Coloured bars span seeds, the hatched
> bar is V12's published figure on the old unmasked metric and is not comparable, dots are
> individual cubes. The dots matter more than the bars: M0 spreads from +100% to below −300%
> on the same model, which is why every claim in this section is stated with its spread.*

**And on the headline table, the notebook's own promotion check refuses to call that +87.5%
an improvement.** It compares each arm to the V12 configuration on a matched schedule and
weighs the gap against the cube-to-cube spread:

```
winner_aug vs V12 config    M0  +5.6 pp  (spread  9.5 pp)  within noise
                            M1  −2.9 pp  (spread 31.7 pp)  within noise
                            M2  +0.3 pp  (spread 25.5 pp)  within noise
```

So augmentation gives the best numbers and the tightest spread, and "augmentation beats the
reference on the science" is still **not demonstrated at n=3**. The PSNR gain is on firmer
ground and not settled either: +1.322 dB against a combined spread of 1.005, which the
notebook grades *suggestive, not established*.

This is the same discipline that killed the 37.11 dB sweep result, applied to our own best
number. It would have been easy to quote +87.5% against +69.8% and move on.

**A correction to the story in the paragraph above.** That run also re-tested the V16
shortfall on a matched schedule, and the answer is mostly *early stopping*, not seed luck:

```
patience  5   mean 27.3 epochs   PSNR 37.974 ±0.896    113% of the shortfall closed
patience 10   mean 48.3 epochs   PSNR 39.273 ±0.484    132% of the shortfall closed
```

Both clear the >50% threshold for "early-stopping artifact", and patience 10 overshoots the
sweep's 37.11 dB outright while *halving* the seed spread. The seed-49 explanation still
holds, 37.11 sits comfortably inside the winner's 3-seed band, but the dominant term in
V16's 30.28 dB was stopping at 27 epochs when the configuration wanted 48.

> **[FIGURE]** `results/08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/seed_spread.png`
>
> *Per-seed spread for four configurations. Dots are individual seeds. The overlap between the
> first two groups is the entire "reproduction failure" story.*

### 11.1 Hallucination, and where it lives

The most scientifically serious failure mode is not blur, it is **invented structure**. In a
disk map, a blob that is not in the truth reads as a false detection.

A per-channel diagnostic counts invented regions relative to each channel's own noise floor:

```
Configuration          Channels with a blob   Blobs/channel   Peak overshoot
V12 config                     39.0%              0.897           2.741
Sweep winner                   37.7%              0.940           2.349
Winner + augmentation          29.7%              0.743           1.736
Winner, patience 10            22.3%              0.857           1.889
```

**Augmentation reduces hallucination** (37.7% → 29.7% at otherwise identical settings), which
is exactly the hypothesis the experiment was built to test. And **hallucination concentrates
in faint channels**: 1.580 invented blobs per channel below the median SNR against 0.213
above it, roughly a 7× difference at a split of SNR 3.9. That answers a question the mentors
raised directly, and tells us where to aim a fix.

**A caution attached to those percentages.** An earlier version of this diagnostic reported **0%
invented structure for every configuration**. That was not a clean result, it was a broken
detector: the background mask was defined as `clean < 10% of peak`, which under shared
dirty-scale normalisation selects **zero pixels**, and `invented` is counted inside that mask, so
it came back empty by construction. The numbers in the table above are from notebook 08, which
bootstraps `midterm-prep` and has the floor-relative fix. Notebook 05 still bootstraps
`line-emission` and does not, so its own artifact panel continues to print zeros and is not
quotable. Same question, two branches, and only one of them can answer it.

A zero from a diagnostic is a suspect, not a pass. Always check the denominator.

---

## 12. The DDPM rerun

Section 5 left the diffusion model at ~18 dB with the gap attributed to the data regime. That
attribution was wrong, and finding out cost one objective sweep.

**Forward process.** Gaussian noise is added to the clean channel x₀ over T steps, with a
variance schedule β_t and ᾱ_t = ∏(1 − β_s):

```
q(x_t | x_0) = N( x_t ; sqrt(ᾱ_t) · x_0 , (1 − ᾱ_t) · I )

equivalently   x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε ,   ε ~ N(0, I)
```

The network sees the dirty channel concatenated with the noisy state, and sampling uses DDIM
for speed. Every earlier run used epsilon-prediction on a linear schedule. Sweeping the
objective on identical data:

```
Objective     Prediction   Schedule   Min-SNR γ    PSNR     SSIM
A             epsilon      linear         0       17.91    0.504
C             v            cosine         0       37.82    0.989
D             v            cosine        5.0      36.33    0.992
Patch view    v            cosine         0       35.97    0.993
```

**A 19.9 dB spread from the parameterisation alone**, and configuration A is what every
previous run had used. This has now been measured twice, on two independent runs, with spreads
of 19.9 and 19.2 dB.

Why v-prediction helps here specifically. The v-target (Salimans & Ho 2022) is

```
v = sqrt(ᾱ_t) · ε − sqrt(1 − ᾱ_t) · x_0
```

It behaves like epsilon-prediction at high noise and like x₀-prediction at low noise, so
neither end of the schedule is trivially predictable. Plain epsilon-prediction degenerates as
SNR → 0: when the input is essentially all noise, predicting the noise means echoing the
input, and the model learns nothing in that regime. **A faint spectral line in an otherwise
empty field is exactly that regime**, which is why this dataset punishes the default
parameterisation harder than natural-image benchmarks do.

The cosine schedule (Nichol & Dhariwal 2021) reinforces the same fix, α_bar at the schedule
midpoint is 0.078 for linear against 0.492 for cosine, so half of linear-schedule training
sits at timesteps carrying no usable signal. Min-SNR-γ weighting (Hang et al. 2023) did not
help (36.33 against 37.82), reported as measured rather than explained away.

Retrained at the winning configuration for 60 epochs: **PSNR 38.18 dB, SSIM 0.9933**. On pixel
metrics, diffusion and regression are now nearly tied.

> **[FIGURE]** `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/line_emission_ddpm_comparison.png`
>
> *DDPM v13 on validation channels: dirty, denoised, clean truth. Per channel this is a good
> reconstruction. Row 3 is the whole problem in one frame: the background is a flat mid-scale wash
> rather than black. Integrated over 201 channels that wash becomes the pedestal below.*

### 12.1 On the moment maps, they are not tied

Earlier drafts of this post had to hedge here, because the DDPM was scored with the signal
mask and the 3σ clip while no U-Net ever had been, so the two sat on different rulers. Run
05 v20 closed that gap: it re-scored fifteen U-Net checkpoints under the DDPM's exact
metric, without retraining any of them. The comparison below is finally like for like.

```
Model (all rows: mask + clip)        M0             M1             M2
U-Net, winner + D4 aug          +29.2% ±  7.2  +74.0% ± 2.0  +55.0% ± 13.9
U-Net, winner + patience 10     +33.5% ±  9.6  +70.7% ± 6.7  +31.8% ± 11.1
U-Net, V12 config                −4.4% ± 33.9  +58.0% ±20.0  +15.5% ± 31.2
DDPM                            −56.1% ±152.2  +13.7% ±90.0   +5.2% ± 83.6
```

The ranking the earlier hedge guessed at holds, and the gap is wider than the hedge allowed.
The best U-Net arm beats the DDPM by **85 percentage points on M0**, and does it with a
spread twenty times tighter. Note also that the U-Net's own V12 configuration goes *negative*
on M0 under this metric: the mask is strict, and being on the right side of zero here is not
automatic for anything.

That standard deviation is not a typo, and the mean hides the real structure, which is bimodal:

```
Cube                    M0        M1        M2
run_0002_..._rt_00    −84.8%   −126.5%   −130.8%
run_0002_..._rt_01    +41.9%    +61.8%    +47.8%
run_0002_..._rt_04    +18.5%    +73.5%    +51.2%
run_0025_01000_rt_04  −310.2%   −13.2%    +15.5%
run_0026_00005_rt_04  +54.1%    +79.2%    +70.7%
```

Three of five cubes behave like the U-Net; two collapse.

**The diagnosis is a pedestal, not lost structure.** The moment figure shows the denoised
velocity map recovering the rotation dipole cleanly and the intensity map recovering the
disk, while the sky, black in the truth, sits at a raised floor, and the dispersion map is
saturated across the entire field. The smoke test recorded the cause directly:

```
sampler OK: (8,1,256,256) -> (8,1,256,256), range [0.348, 0.701]
```

The sampler emits a narrow band around 0.5 instead of spanning [0, 1]. Inverting the
per-channel normalisation then places the entire field at mid-scale. **This is the Section 8.1
lesson recurring in a new costume**: the decode step assumes an output range the model does
not actually produce.

> **[FIGURE]** `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/ddpm_moment_maps.png`
>
> *DDPM moment maps. The velocity map recovers the rotation dipole cleanly. The intensity map
> shows the problem: the sky, black in the truth, sits at a raised floor. The dispersion map is
> saturated across the entire field.*

> **[FIGURE]** `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/moment_map_holdout_summary_ddpm.png`
>
> *The same result per cube. This is not a model that is uniformly bad; it is a model that is
> fine on three cubes and catastrophic on two.*

### 12.2 Testing the diagnosis: five re-scores, no retraining

Both suspects were testable on the checkpoint we already had. Because every model was
persisted the moment it finished training, none of this required a single training step, 
five arms re-scoring the same weights on the same five cubes, so any difference is the
sampling or the rescale and nothing else.

```
arm                              M0                M1                M2
baseline (K=4)          -56.9 ±149.9       13.7 ±90.0         5.2 ±83.6
kavg1 (K=1)             -51.4 ±148.7       15.1 ±90.3         6.7 ±84.2
rescaled (K=4)         -264.5 ±158.7       38.3 ±24.6        42.4 ±24.4
kavg1 + rescaled       -190.5 ±107.2       51.6 ±10.8        61.3 ±10.8
patch_model            -166.3 ±87.7       -68.1 ±29.0       -28.0 ±37.4
-----------------------------------------------------------------------
U-Net V12 (clip only)     27.7 ±17.6       71.9 ±10.3       -10.5 ±39.7
```

**Averaging is not the cause.** Dropping from 4 posterior draws to 1 moves M0 by 4.7
percentage points against a 56% deficit. That closes the hypothesis in Section 13.2 as an
explanation for the *moment* failure, it explains the ~1 dB PSNR gap and nothing more. K=1
is also 4× faster, so the averaging was buying nothing here.

**The rescale was wrong for M0, and provably so.** Matching each denoised channel's mean and
standard deviation to the *dirty* channel's makes M0 dramatically worse, −56% to −264%. The
reason is structural rather than empirical: M0 is a **sum**, so forcing the output's mean to
dirty's re-imposes exactly the pedestal that denoising is supposed to remove. On synthetic
data where the denoiser is *perfect*:

```
perfect denoiser                    M0 improvement  +100.0%
same perfect output, rescaled       M0 improvement    -0.0%
```

The rescale caps M0 at "no better than dirty" **by construction**, no matter how good the
model is.

**But it transforms the other two moments.** M1 goes +15 → +52 and M2 +11 → +61, and the
cube-to-cube spread collapses by a factor of eight, from ±90 and ±84 down to ±10.8 on both.
M1 and M2 are *ratios* normalised by the intensity sum: insensitive to a constant offset,
highly sensitive to the compressed dynamic range the sampler produces. Fixing the range is
precisely what they needed.

So the pedestal diagnosis was right about the symptom and wrong about the cure. The output
range genuinely is wrong; the dirty channel is simply the wrong reference to correct an
*absolute* quantity against. The obvious next test is to rescale the standard deviation only
and leave the mean alone, the part M1 and M2 needed, without the part that caps M0.

**And this produces the first result where the diffusion model wins.** On M2, the moment every
architecture in this project has struggled with, `kavg1 + rescaled` scores **+61.3% ±10.8**
against the U-Net's **−10.5% ±39.7**, a 72 percentage point gap, at a quarter of the
variance. It still loses on M0 and M1.

**Patch training does not help either.** Retraining the DDPM on 8400 64-pixel patches instead
of 1050 full channels, the strongest remaining lever for the small-data hypothesis from
Section 5, scores negative on all three moments (−166 / −68 / −28). More, smaller samples of
the same six disks is not what this model was missing.

A sixth arm, native-600 tiled inference, was cut off by Kaggle's 12-hour session limit: it
samples 9 overlapping tiles per channel, which needs roughly 14 hours on its own. Every
completed arm wrote its results per cube as it went, so the timeout cost only that arm.

---

## 13. DDPM against U-Net

Two independent reasons the diffusion model loses here, and both are quantitative.

### 13.1 Why moment maps punish bias so much harder than PSNR

Write the error in one channel as a bias δ plus zero-mean noise of scale σ. Moment 0 sums over
N = 201 channels. A **random** per-channel error accumulates as a random walk, growing like
sqrt(N); a **constant** per-channel bias accumulates linearly:

```
error_random(M0)  ~  σ · Δv · sqrt(N)
error_bias(M0)    =  δ · Δv · N
```

The bias term beats the noise term whenever δ > σ / sqrt(N), which for N = 201 means a bias
**14× smaller than the per-channel noise still dominates the integrated map**. PSNR is
computed per channel and is dominated by σ. M0 is an integral and is dominated by δ. A model
can therefore be nearly PSNR-optimal and still destroy the science, precisely what was
measured.

### 13.2 The structural disadvantage, and a number that matches

For squared error, the optimal estimator is the posterior mean μ = E[x₀ | y], with conditional
covariance Σ. A regression model like the U-Net is trained to output exactly that mean. A
diffusion model draws a *sample* from p(x₀ | y), which carries the posterior variance twice
over, once from the truth's spread, once from the sample's:

```
E‖x₀ − μ‖²   =  tr(Σ)              regression, the posterior mean
E‖x₀ − x̃‖²   =  2 · tr(Σ)          one diffusion sample  → 3.01 dB penalty
E‖x₀ − x̄_K‖² =  tr(Σ) · (1 + 1/K)  average of K draws
```

At K = 4 that predicts a penalty of 10·log₁₀(1.25) = **0.97 dB**. The measured gap between the
best U-Net (39.30 dB) and the best DDPM (38.36 dB) is **0.94 dB**. Not claimed as proof, the
two models differ in training run, seed and capacity, but strikingly consistent with the DDPM
being a well-calibrated posterior sampler paying exactly the theoretical price for sampling
instead of averaging.

**The practical reading:** on this task a diffusion model pays K× the sampling cost to
asymptotically approach what the regression model outputs directly. Its value is as a
documented comparison and as a route to uncertainty estimates, not as a candidate for the
final pipeline. That is consistent with the mentors' steer to stay with the U-Net, and it is
now a measured statement rather than an assumption.

A caveat this prediction now carries: Section 12.2 tested K = 1 directly, and the moment
scores barely moved. So this argument explains the **PSNR** gap and only that. It is not an
explanation for the moment-map failure, which is a bias problem by Section 13.1 and is
therefore invisible to a variance argument. Two separate deficits, two separate causes, and
they should not be conflated, as an earlier draft of this post did.


### 13.3 The two current best checkpoints, side by side

The latest run of each track, 05 v18 for the U-Net and 06 v13 for the diffusion model:

```
                       U-Net 05 v18      DDPM 06 v13       comparable?
PSNR (dB)                 35.94             38.18             yes
SSIM                      0.986             0.9933            yes
───────────────────────────────────────────────────────────────────────
M0                 +81.2% ± 12.6     −56.1% ± 152.2            NO
M1                 +31.5% ±  9.2     +15.0% ±  87.4            NO
M2                 +19.9% ± 18.8     +10.9% ±  81.7            NO
moment metric        raw, unmasked     signal mask + 3σ clip
branch bootstrapped   line-emission      midterm-prep
```

**The pixel metrics are comparable and the moment rows are not.** Notebook 05 bootstraps from
`line-emission`, which has none of the moment fixes; notebook 06 pulls `midterm-prep`, which has
all three. Section 9.1 measured what that difference is worth on identical checkpoints: 110
percentage points on M0, larger than most gaps anyone would want to read off this table.

What survives the caveat is still the result. **On PSNR the diffusion model is ahead by 2.24 dB**,
the first time in this project it has led the U-Net on any metric. On the science it is behind by
a margin far too large for the metric difference to explain: ±152 on M0 against ±12.6 means the
DDPM is unreliable per cube in a way v18 is not, and no choice of metric turns a bimodal
three-good-two-catastrophic result into a usable one.

The cleanest way to close this is bookkeeping, not modelling: move notebook 05's bootstrap to
`midterm-prep` and re-score v18's stored checkpoint under the mask and clip. No retraining, one
run, and the comparison above becomes a real one.

---

## 14. What is next

1. **Kill the diffusion pedestal.** Section 12.2 narrowed it to one test: rescale the
   standard deviation only and leave the mean alone. Matching the mean to the dirty channel
   caps M0 at "no better than dirty" by construction; matching the spread is what took M1 and
   M2 from +15/+11 to +52/+61. One re-score, no retraining.
2. **Re-score every archived checkpoint on one final metric.** Three metric generations exist
   in this project's history, no clip/no mask, clip only, clip plus signal mask, and results
   computed either side of a revision are not comparable, which is why several tables above are
   compared only within their own group. This is the largest piece of unfinished bookkeeping.
3. **Attack M2.** Every architecture tested, including every classical filter, is weakest on
   velocity dispersion. Established across five models now, this looks like a property of the
   problem rather than of any one model.
4. **Attack hallucination in low-SNR channels**, now that we know that is where it lives.
   Candidates: a penalty on asserted structure in low-signal regions, and testing whether
   posterior averaging suppresses it.
5. **DDRM**, projecting each reverse diffusion step onto the subspace consistent with the
   actual measurement, using the beam operator this project already extracts from the FITS
   headers. It targets invented structure directly: a freely-generating diffusion model has no
   constraint tying its output to what the telescope saw.
6. **More training disks.** Six independent disks is the binding constraint on everything
   here. A self-gravitating cube recently shared by Jason takes that to seven, a bigger
   relative gain than any hyperparameter change in this project.
7. **Real ALMA data**, validated against DSHARP. Deferred to the final blog by agreement.

---

## 15. Closing

Against the seven "Definite Goals" in my accepted proposal, all seven have working code and
real results behind them:

1. **FITS preprocessing pipeline**, done.
2. **Augmentation and normalisation strategies**, done: D4 augmentation plus the shared-scale
   normalisation of Section 8.1.
3. **Denoising models including autoencoder and diffusion**, exceeded: autoencoder, VAE,
   U-Net and conditional DDPM, where the proposal committed to two of the four.
4. **Train on hydro-sim + radiative-transfer synthetic data**, done, 14 cube pairs.
5. **Evaluation with PSNR / SSIM / MSE**, exceeded: added moment-map evaluation and
   hallucination diagnostics, because pixel metrics alone repeatedly gave the wrong answer.
6. **Compare against classical denoising**, done, Section 10, including median which the
   proposal did not name.
7. **Reproducible end-to-end pipeline**, done: self-bootstrapping notebooks, a test suite,
   and a run index tracing every number to its notebook version.

What moved was the calendar, not the scope. The pivot to line-emission velocity cubes was the
proposal's own optional stretch goal and became the primary track, so U-Net and diffusion work
ran in parallel rather than in sequence.

**If this project has one transferable lesson, it is that the metric is a design decision, not
a detail.** Continuum subtraction moved PSNR by 2.2 dB and M0 by 1756 percentage points. Beam
conditioning improved PSNR and made the science worse. A 12-run sweep produced a "+4.16 dB
improvement" that was pure seed luck. A diffusion model reached within 1 dB of the U-Net on
PSNR while scoring −56% on integrated intensity. In every one of those cases the pixel metric
said one thing and the science said another, and the science was right.

### Reproducing this

```bash
git clone -b midterm-prep https://github.com/KrishanYadav333/EXXA.git
cd EXXA/DENOISING_DIFFUSION
pip install -r ../requirements.txt
python -m pytest tests/ -q
```

Every notebook bootstraps itself from a blank Kaggle kernel plus the FITS dataset:

- `05-unet-line-emission.ipynb`, U-Net training and moment-map evaluation
- `06-ddpm-line-emission.ipynb`, conditional DDPM, objective sweep, DDIM sampling
- `07-classical-baselines.ipynb`, tuned Gaussian, median and Wiener baselines
- `08-seeds-and-augmentation.ipynb`, multi-seed and augmentation study
- `09-architecture-comparison.ipynb`, U-Net against autoencoder and VAE

`DENOISING_DIFFUSION/results/RUNS.md` is the run index: every number in this post traces to
the notebook version that produced it, including the ones later corrected.

Conventions: seed 42 throughout, cube-level splitting is mandatory, holdout cubes are
inference-only, moment results are always averaged over all 5 holdout cubes with a standard
deviation, and no result is recorded without real execution and artifacts on disk.

### Acknowledgements

Thank you to my mentors **Jason Terry** and **Gaurav S.** for direction that repeatedly turned
out to be right, in particular the pivot to line emission and the insistence on moment maps
as the deliverable rather than pixel metrics. Thanks also to the wider ML4Sci / EXXA
organisation.

*Tags: Machine Learning, Astrophysics, Diffusion Models, Exoplanets, Google Summer of Code*
