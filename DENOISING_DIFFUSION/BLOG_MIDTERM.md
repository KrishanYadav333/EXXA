# Denoising Protoplanetary Disk Line Emission: GSoC 2026 @ Machine Learning for Science (EXXA)

*Krishan Yadav, Google Summer of Code 2026, ML4Sci / EXXA*
*Mentors: Jason Terry, PhD · Gaurav S.*
*Code: https://github.com/KrishanYadav333/EXXA (branch `midterm-prep`)*

> **Status note.** This is the midterm write-up, covering community bonding through
> 12 August 2026. One experiment is still running on Kaggle as this is published (the
> diffusion rescaling test in Section 11); its numbers will be folded in as an update.
> Everything else comes from a completed run with artifacts on disk, and every number is
> traceable to the notebook version that produced it.

---

## 1. Why denoise a protoplanetary disk?

Planets form in protoplanetary disks: rotating envelopes of gas and dust around young
stars. When a planet forms inside one, it does not usually show up as a dot. It shows up
as a *disturbance*. The planet's gravity carves gaps, launches spiral arms, and, most
usefully, perturbs the velocity field of the gas around it. In a rotating disk the gas
follows Keplerian motion, so the line-of-sight velocity at radius r and azimuth φ is

```
v_obs(r, φ) = sqrt(G M* / r) · sin(i) · cos(φ) + v_sys
```

where i is the disk inclination and v_sys the systemic velocity. A planet breaks that
clean pattern locally, producing a "kink" in the velocity map. Detecting those kinks is
how embedded protoplanets have actually been found in ALMA data (Terry et al. 2022, 2023).

The catch: the kink is a small deviation on top of a large smooth pattern, and the data we
get from a radio interferometer is noisy in a way that specifically attacks small
deviations. That is the problem this project attacks.

---

## 2. Why this is not generic image denoising

An interferometer like ALMA does not photograph the sky. It measures the sky's Fourier
transform, sampled only at the spatial frequencies its antenna baselines happen to cover.
By the van Cittert–Zernike theorem, each baseline measures one complex visibility:

```
V(u, v) = ∫∫ I(l, m) · exp[ -2πi (u·l + v·m) ] dl dm
```

The array samples V only on a sparse set of (u, v) points, described by a sampling
function S(u, v). Inverting that incomplete measurement gives the "dirty" image:

```
I_dirty = F⁻¹[ S(u,v) · V(u,v) ] = I_true ⊛ B_dirty ,   B_dirty = F⁻¹[S]
```

The dirty image is the true sky **convolved with the dirty beam**. So the corruption is
not independent per-pixel Gaussian noise, which is what most denoising literature assumes.
It is a structured, spatially correlated sidelobe field set by the array geometry. It has
long-range correlations, it is anisotropic, and it changes with the observation.

This is the single most important fact about the problem, and it is why an off-the-shelf
denoiser is not obviously the right tool.

---

## 3. The data

14 FITS cube pairs spanning 11 distinct simulation RunIDs, each of shape
`(201, 600, 600)`: 201 velocity channels of a 600×600 sky image. These are synthetic
observations built from hydrodynamic simulation plus radiative transfer (PHANTOM and
MCFOST lineage). Each pair is a dirty cube and its matching clean cube. Roughly 7.6 GB
total, hosted as a Kaggle Dataset, never committed to git.

Three preprocessing decisions matter enough to state.

**Cube-level splitting, grouped by RunID.** Radiative-transfer variants of the same
simulation (`run_0002_00560_rt_00`, `_rt_01`, `_rt_04`) are near duplicates. Splitting them
across train and test would leak. Grouping by RunID gives 5 cubes held out entirely,
inference-only, never seen in training or validation. Six independent disks remain for
training, which is a small number and shapes several results below.

**Channel sampling.** Channels are drawn per cube from a Gaussian centred on channel 100,
avoiding the line-free extremes where there is nothing to denoise. Calibrated to 73.8% of
sampling mass inside channels [50, 150], against the mentor's ~75% guidance.

**Continuum subtraction.** The static dust disk contaminates the line emission. We estimate
it from the line-free channels at each end of the cube and subtract it from every channel
before normalisation. The effect was large, particularly on integrated intensity:

```
Metric                    No subtraction    Continuum-subtracted
PSNR                          31.02 dB              33.18 dB
SSIM                            0.9832                0.9868
Moment 0 (single cube)         −1672%                +84.9%
```

Continuum contamination was not a detail. It was the difference between failure and a
usable result.

---

## 4. The thing we actually have to get right: moment maps

A denoised cube that looks nice is worth nothing if the science extracted from it is wrong.
Astronomers do not analyse raw channels; they collapse the cube along velocity into
**moment maps**. Using `bettermoments`, for a cube I(v) at each sky pixel:

```
M0 = ∫ I(v) dv                                    integrated intensity
M1 = ∫ v · I(v) dv  /  ∫ I(v) dv                  intensity-weighted velocity
M2 = sqrt( ∫ I(v) · (v − M1)² dv  /  ∫ I(v) dv )  velocity dispersion
```

M0 tells you where the gas is. M1 is the rotation map, and it is where planet-induced kinks
appear. M2 traces turbulence and line broadening.

We score improvement over the dirty cube, per moment, on the 5 held-out cubes:

```
Improvement(M) = 100 × ( 1 −  mean |M_denoised − M_clean|
                            ─────────────────────────────  )
                              mean |M_dirty    − M_clean|
```

averaged over a **signal mask**: the pixels where the clean M0 exceeds a fixed fraction of
its peak. Scoring over the whole map lets empty sky dominate, and the dispersion of a pixel
containing no line is not a quantity anyone reports. 100% means perfect recovery, 0% means
no better than the dirty input, and negative means the denoiser made the science *worse*
than doing nothing.

**The methodological thread of this whole project:** several separate times, a change
improved PSNR/SSIM and made the moment maps worse or no better. Pixel metrics and science
metrics disagree here, and every result below is reported on both.

---

## 5. The continuum era, and the first sign that PSNR lies

The project did not start on line emission. The first weeks worked on **continuum** images:
single-channel dust maps rather than velocity cubes. That phase produced no headline result,
but it produced the question the rest of the project is organised around.

Eight methods on the continuum data, ranked here by SSIM:

```
Method                     PSNR      SSIM        MSE
AE HybridLoss             19.92     0.761     0.013920
U-Net HybridLoss          20.83     0.706     0.011091
VAE (MSE+SSIM+KL)         20.00     0.706     0.013336
AE MSE-only               20.25     0.616     0.012804
Gaussian σ=2              22.78     0.423     0.006380
Median 3×3                22.88     0.359     0.006317
Wiener                    22.57     0.340     0.006702
Noisy input (no filter)   21.57     0.192     0.008391
```

Read the first two columns against each other. **Every classical filter beats every learned
model on PSNR, and loses to all of them on SSIM** — median 3×3 scores 22.88 dB against the
U-Net's 20.83, while its SSIM is 0.359 against 0.706. Worse, the *unfiltered noisy input*
scores 21.57 dB, beating three of the four learned models on PSNR alone.

Nothing in that table is a good denoiser. But it is a clean demonstration that a single
scalar can rank methods in an order no one looking at the images would accept. Blurring
everything is an excellent way to lower mean squared error and a terrible way to preserve a
disk. That observation is why moment maps, not PSNR, became the deliverable when the project
moved to line emission.

**The pivot.** On 18 June the mentors redirected the project from continuum to
line-emission velocity cubes. This was the optional stretch goal in my accepted proposal and
became the primary track. Everything from Section 6 onward is on line emission; the
continuum numbers above are not comparable to anything that follows, and are reported only
for this one lesson.

---

## 6. Model 1: the U-Net

The workhorse is a U-Net (Ronneberger et al. 2015) mapping a dirty channel to its clean
counterpart, 3.4M parameters in the reference configuration. Skip connections matter here:
recovering a faint ring means preserving high-frequency spatial detail that a plain
encoder/decoder bottleneck destroys — which is visible in the continuum table above, where
both bottlenecked architectures sit below the U-Net.

Training uses a hybrid loss, the same MSE-plus-SSIM idea used elsewhere in EXXA:

```
L = α · MSE(ŷ, y) + (1 − α) · (1 − SSIM(ŷ, y)),    α = 0.8
```

MSE anchors numerical fidelity; the SSIM term protects structure, which per-pixel error
alone will happily smooth away.

### 6.1 A normalisation bug worth more than any result

The first end-to-end run produced **M0 = −6402%**. The denoiser had made the intensity map
sixty-four times worse than the raw input, while per-channel SSIM sat at a healthy 0.81.
That contradiction is the most instructive thing that happened all summer.

The cause: training normalised the clean target by *clean's own* per-channel min and max,

```
ỹ = (y − min y) / (max y − min y)
```

so the network learned to emit values in clean-normalised space. But at inference, clean's
min and max are unavailable by definition. Decoding therefore used dirty's. Clean's
background floor sits near zero while dirty's minimum is negative, so every background pixel
decoded to dirty's negative floor instead of zero, imposing a DC offset on each channel.
Summed over 201 channels, M0 went sharply negative.

The fix is one line of principle: **normalise both dirty and clean by the dirty channel's
statistics**, which are the only ones available at inference time.

```
x̃ = (x − min D) / (max D − min D)      for x ∈ {dirty, clean},  D = dirty channel
```

Clean is deliberately not clipped, since its peak can exceed 1 on this shared scale, which
in turn required replacing the sigmoid output with a linear head. Verified by channel
diagnostic: the denoised minimum moved from −0.0047 (tracking dirty, wrong) to +0.0001
(tracking clean, correct).

**Transferable lesson: any normalisation must be invertible from information available at
inference time.** Keep this in mind for Section 11, where a closely related failure shows up
in the diffusion model.

---

## 7. How much is the network actually buying us?

Before this project, no classical filter had ever been scored on this line-emission dataset,
so "the learned model beats traditional processing" was an assumption. We tuned Gaussian,
median, and Wiener filters on a validation split and ran them through the identical 5-cube
protocol.

Channel level, 300 validation channels at 256×256:

```
Method                     PSNR      SSIM
Dirty (unfiltered)        20.43     0.310
Median (window 9)         22.80     0.530
Wiener (window 9)         22.83     0.538
Gaussian (σ = 4)          24.96     0.701
U-Net (V12)               32.95     0.986
```

Moment 0 improvement, 5 held-out cubes, mean ± std:

```
Method                                    M0 improvement
Median                                      +3.9%  ± 0.9
Wiener                                      +5.0%  ± 1.0
Gaussian, native resolution                +11.7%  ± 2.1
Gaussian, at the network's resolution      +36.5%  ± 3.5
U-Net (V12)                                +69.8%  ± 15.2
```

The honest comparison is the last two rows, both evaluated at the resolution the network
operates at: a **+33.3 percentage point** advantage on integrated intensity. Classical
filters do help, and they are nearly free, but they plateau early. Smoothing removes noise
and signal together; there is no σ that separates a faint ring from a sidelobe.

This run also caught its own methodology bug: filters were tuned at 256 px but applied at
the cubes' native 600 px, and filter parameters are in pixels, so σ = 4 tuned at 256 smooths
2.3× too little at 600. Both handicaps were on the classical side, the opposite of the
intended "generous to classical" framing. The grid was widened and the numbers above are the
corrected ones.

---

## 8. Which architecture?

An equal-training-budget comparison: same data, same schedule, same 5-cube holdout. Rather
than pick one configuration per architecture and risk comparing lucky draws, we ran a
**24-run sweep — 8 configurations per architecture** — and took each one's best.

```
Architecture   runs   best PSNR   median   worst   best SSIM
U-Net             8      39.235   37.679  37.081      0.9946
Autoencoder       8      33.633   31.648  19.659      0.9904
VAE               8      30.498   21.152  16.899      0.9819
```

The U-Net's best configuration is `base_channels=16`, multipliers `1×2×4×8`, lr 0.00124,
α 0.745 — **39.235 dB at SSIM 0.9946**, the best pixel-metric result anywhere in this
project, and 6.3 dB above V12's published 32.95.

The sweep also earns its cost by showing something a single run per architecture would have
hidden: **the spread within an architecture rivals the gap between architectures.** The VAE
ranges 16.9 to 30.5 dB across its own 8 configurations, a 13.6 dB range that swallows its
entire deficit to the U-Net. An architecture comparison built on one run each would have been
measuring configuration luck.

On the moment maps, with parameter counts, this is the full cross-model comparison:

```
Architecture      params      PSNR         M0              M1              M2
U-Net          3,477,665    37.414   +77.0% ± 13.6   +22.2% ± 15.2   +8.5% ± 18.1
Autoencoder    1,734,305    33.155   +54.1% ± 36.3   +25.8% ± 13.6  +21.0% ± 14.8
VAE              467,793    29.760   +46.1% ± 22.3    +1.0% ± 18.1  +12.7% ± 12.8
U-Net V12      3,400,000    32.950   +69.8% ± 15.2   +17.5% ±  7.8  +20.1% ± 14.3
```

U-Net wins decisively on intensity and has the lowest cube-to-cube variance. The VAE is
clearly the wrong tool: its latent bottleneck plus KL regularisation is built to produce
plausible samples, and plausible is not the same as correct. Its M1 score of +1.0% means it
essentially failed to recover the velocity field at all — and it does that with 7× fewer
parameters than the U-Net, so the deficit is architectural, not a capacity artefact.

But the U-Net **loses on M2**, behind both other architectures. This is not a one-off; it
recurs in every experiment below, and we are treating it as an open scientific problem rather
than something to patch away.

### 8.1 The same checkpoints, scored twice, disagree

That table is not the end of the story, and the part that follows is the most uncomfortable
result in this write-up.

Midway through the project a 3σ noise clip was added to the moment computation, to stop the
dispersion map reading the noise floor. Re-scoring the **identical checkpoints** with it:

```
Moment    no clip     3σ clip     change
M0         +77.0%      −33.2%    −110.2 pp
M1         +22.2%      +68.7%     +46.6 pp
M2          +8.5%     −152.8%    −161.3 pp
```

Nothing about the models changed. The metric did, and it did not shift everything one way:
**M1 more than tripled while M0 and M2 collapsed.** Under the clip the U-Net's M1 rises to
the best velocity recovery measured anywhere in this project, and its M0 goes negative.

Two honest consequences.

First, **the ranking survives but the magnitudes do not.** U-Net still leads the autoencoder
and VAE on M0 under the clip (−33.2% against −152.0% and −220.4%), so "U-Net is the right
architecture" holds. Any *number* attached to that claim depends on which metric version
produced it.

Second, and this is the trap this project keeps walking into: the comparison table above puts
V12's published +69.8% in the same column as numbers computed under a different metric
version. That row was measured before either revision. It should not be read as directly
comparable to the three above it, and the notebook that printed the table did not say so.

Neither of these runs used the signal mask that Section 11's diffusion figures use, so there
are effectively **three metric generations** in this project's history. Re-scoring every
archived checkpoint on one final metric is item 2 of what is next, and it is the largest
single piece of unfinished bookkeeping here. Until that is done, comparisons are only valid
within a section, which is why this post never places a Section 8 number beside a Section 11
number.

---

## 9. Sweeps, seeds, and a result that was not real

**Beam conditioning: a useful negative.** The observation beam is exposed to the network as a
4-vector `[sin 2·BPA, cos 2·BPA, BMAJ, BMIN]` and injected through the time-embedding path so
it reaches every block. It improved PSNR by **+1.24 dB**. It also dropped M0 from +69.8% to
+59.8% and more than doubled the variance. A project measuring only pixel metrics would have
shipped this as an improvement. It shipped as a documented "do not".

**A 12-run hyperparameter sweep found a 37.11 dB configuration**, and then we could not
reproduce it: the retrain landed at 30.28 dB, worse than the reference. The cause turned out
to be embarrassing and important. `run_sweep` trains run *i* at seed `base + i`, so the
winning row had been trained at a different seed from the retrain, and early stopping fired
on a different, noisier validation trajectory. The "+4.16 dB improvement" was an order
statistic over 12 lucky draws, not a measured effect.

The rule adopted from this: **no configuration is called an improvement until it has been
checked across multiple seeds.** Rerunning the same configurations at 3 seeds each:

```
Configuration                       PSNR (3 seeds)
V12 reference config                 37.60 ± 1.00
Sweep "winner", no augmentation      37.97 ± 0.90
Sweep winner + D4 augmentation       39.30 ± 0.46
Sweep winner, patience 10            39.27 ± 0.48
```

The V12 and "winner" bands overlap almost completely: the sweep result was indeed seed noise.
Augmentation, by contrast, produces a real gain and *halves the variance*. D4 augmentation
applies the 8 lossless orientations of the dihedral group (4 rotations × optional flip)
identically to dirty and clean. On a 14-cube corpus that is the cheapest regularisation
available.

### 9.1 Hallucination, and where it lives

The most scientifically serious failure mode is not blur, it is **invented structure**. In a
disk map, a blob that is not in the truth reads as a false detection.

We built a per-channel diagnostic that counts invented regions, measured relative to each
channel's own noise floor:

```
Configuration            Channels with a blob   Blobs/channel
V12 config                       39.0%              0.897
Sweep winner                     37.7%              0.940
Winner + augmentation            29.7%              0.743
Winner, patience 10              22.3%              0.857
```

Two things worth reporting. **Augmentation reduces hallucination** (37.7% → 29.7% at
otherwise identical settings), which is exactly the hypothesis the experiment was built to
test. And **hallucination is concentrated in faint channels**: 1.580 invented blobs per
channel below the median SNR against 0.213 above it, roughly a 7× difference at a split of
SNR 3.9. That answers a question the mentors raised directly, and it tells us where to aim a
fix.

A methodological caution attached to this: an earlier version of this diagnostic reported
"0% invented structure" for every configuration. That was not a clean result, it was a broken
detector — the background mask was defined as `clean < 10% of peak`, which under shared
dirty-scale normalisation selected **zero pixels**. A zero from a diagnostic is a suspect,
not a pass. Always check the denominator.

**Current best U-Net checkpoint** (v18, 11 August): M0 **+81.2% ± 12.6**, M1 +31.5% ± 9.2,
M2 +19.9% ± 18.8 — the first run with every moment positive on every one of the 5 held-out
cubes. Caveat stated plainly: v18 was a single seed, so it is the strongest single run to
date rather than a seed-validated configuration.

---

## 10. Model 2: a conditional diffusion model

Diffusion is named in the project description, and it is the natural competitor, so we
maintain a conditional DDPM throughout.

**Forward process.** Gaussian noise is added to the clean channel x₀ over T steps, with a
variance schedule β_t and ᾱ_t = ∏(1 − β_s):

```
q(x_t | x_0) = N( x_t ; sqrt(ᾱ_t) · x_0 , (1 − ᾱ_t) · I )

equivalently   x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε ,   ε ~ N(0, I)
```

**Reverse process, conditioned on the dirty channel.** The network sees the dirty channel y
concatenated with the noisy state, and we sample with DDIM for speed.

**The parameterisation turned out to be the whole ballgame.** Every earlier run used
epsilon-prediction on a linear schedule and scored badly, and we had attributed the gap to
the data regime: 6 training disks is not much to learn a distribution from. That explanation
was wrong. An objective sweep on identical data:

```
Objective     Prediction   Schedule   Min-SNR γ    PSNR     SSIM
A             epsilon      linear         0       17.91    0.504
C             v            cosine         0       37.82    0.989
D             v            cosine        5.0      36.33    0.992
Patch view    v            cosine         0       35.97    0.993
```

A **19.9 dB spread from the parameterisation alone**, and configuration A is what every
previous run had used. This has now been measured twice, on two independent runs, with
spreads of 19.9 and 19.2 dB.

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

The cosine schedule (Nichol & Dhariwal 2021) reinforces the same fix by spending less of the
trajectory in the saturated all-noise regime. Min-SNR-γ weighting (Hang et al. 2023) did not
help here (36.33 against 37.82), which we report as measured rather than explain away.

Retrained at the winning configuration for 60 epochs, the DDPM reaches **PSNR 38.18 dB, SSIM
0.993**, within about 1 dB of the best U-Net at 39.30 dB. On pixel metrics, diffusion and
regression are nearly tied.

---

## 11. On the moment maps, they are not tied

```
Model                                  M0 (signal-masked metric)
U-Net (V12 configuration, this metric)      +27.7% ± 17.6
DDPM                                        −56.1% ± 152.2
```

That standard deviation is not a typo, and the mean hides the real structure, which is
bimodal. Three of five cubes behave like the U-Net (+18% to +54%); two collapse (−85%,
−310%).

**The diagnosis is a pedestal, not lost structure.** The moment figure shows the denoised
velocity map recovering the rotation dipole cleanly and the intensity map recovering the
disk — while the sky, which is black in the truth, sits at a raised floor, and the dispersion
map is saturated across the entire field. The smoke test recorded the cause directly:

```
sampler OK: (8,1,256,256) -> (8,1,256,256), range [0.348, 0.701]
```

The sampler emits a narrow band around 0.5 instead of spanning [0, 1]. Inverting the
per-channel normalisation then places the entire field at mid-scale. This is the Section 6.1
lesson recurring in a new costume: the decode step assumes an output range the model does not
actually produce.

### 11.1 Why moment maps punish this so much harder than PSNR

Write the error in one channel as a bias δ plus zero-mean noise of scale σ. Moment 0 sums
over N = 201 channels. A **random** per-channel error accumulates as a random walk, growing
like sqrt(N); a **constant** per-channel bias accumulates linearly:

```
error_random(M0)  ~  σ · Δv · sqrt(N)
error_bias(M0)    =  δ · Δv · N
```

So the bias term beats the noise term whenever δ > σ / sqrt(N), which for N = 201 means a
bias **14× smaller than the per-channel noise still dominates the integrated map**. PSNR is
computed per channel and is dominated by σ. M0 is an integral and is dominated by δ. A model
can therefore be nearly PSNR-optimal and still destroy the science, which is precisely what
we measured.

### 11.2 Why a diffusion model is at a structural disadvantage here

There is also a cleaner, more fundamental reason, and it predicts the gap we see.

For squared error, the optimal estimator is the posterior mean μ = E[x₀ | y], with
conditional covariance Σ. A regression model like the U-Net is trained to output exactly that
mean, so its error is tr(Σ). A diffusion model draws a *sample* from p(x₀ | y). A single
independent draw carries the full posterior variance twice over, once from the truth's spread
and once from the sample's:

```
E‖x₀ − μ‖²   =  tr(Σ)              regression, the posterior mean
E‖x₀ − x̃‖²   =  2 · tr(Σ)          one diffusion sample  → 3.01 dB penalty
E‖x₀ − x̄_K‖² =  tr(Σ) · (1 + 1/K)  average of K draws
```

At K = 4 that predicts a penalty of 10·log₁₀(1.25) = **0.97 dB**. Our measured gap between
the best U-Net (39.30 dB) and the best DDPM (38.36 dB) is **0.94 dB**. We are not claiming
that agreement as proof, since the two models differ in training run, seed, and capacity, but
it is strikingly consistent with the DDPM being a well-calibrated posterior sampler paying
exactly the theoretical price for sampling instead of averaging.

The practical reading: on this task, **a diffusion model pays K× the sampling cost to
asymptotically approach what the regression model outputs directly**. Its value is as a
documented comparison and as a route to uncertainty estimates, not as a candidate for the
final pipeline. That is consistent with the mentors' steer to stay with the U-Net, and it is
now a measured statement rather than an assumption.

**Both fixes are testable without retraining**, and that run is in flight as this is
published: score the same checkpoint at K = 1 to isolate the averaging effect, and rescale the
denoised channel to the dirty channel's own mean and standard deviation before inverting the
normalisation, which removes the pedestal directly.

---

## 12. Where the project stands against its plan

My accepted proposal listed seven "Definite Goals" for the 350-hour period. All seven have
working code and real results behind them:

1. **FITS preprocessing pipeline** — done.
2. **Augmentation and normalisation strategies** — done: D4 augmentation plus the shared-scale
   normalisation of Section 6.1.
3. **Denoising models including autoencoder and diffusion** — exceeded: autoencoder, VAE,
   U-Net and conditional DDPM, where the proposal committed to two of the four.
4. **Train on hydro-sim + radiative-transfer synthetic data** — done, 14 cube pairs.
5. **Evaluation with PSNR / SSIM / MSE** — exceeded: added moment-map evaluation and
   hallucination diagnostics, because pixel metrics alone repeatedly gave the wrong answer.
6. **Compare against classical denoising** — done, Section 7, including median which was not
   named in the proposal.
7. **Reproducible end-to-end pipeline** — done: self-bootstrapping notebooks, a test suite,
   and a run index tracing every number to its notebook version.

What moved was the calendar, not the scope. The pivot to line-emission velocity cubes
(mentor-directed, 18 June) was the proposal's own optional stretch goal and became the primary
track, so U-Net and diffusion work ran in parallel rather than in sequence.

Still open and deliberately deferred: real ALMA validation against DSHARP, which moves to the
final blog, and a self-gravitating cube recently shared by Jason, earmarked as a kinematic
test set rather than training data.

---

## 13. What is next

1. **Kill the diffusion pedestal**, using the two no-retrain tests in Section 11.2.
2. **Re-score the earlier runs on the corrected signal-masked metric**, so every number sits
   on one comparable scale. Two metric revisions happened mid-project (a signal mask and a 3σ
   noise clip), and results computed either side of them are not directly comparable, which is
   why some tables above are compared only within their own group.
3. **Attack M2.** Every architecture tested, including every classical filter, is weakest on
   velocity dispersion. Established across five models now, this looks like a property of the
   problem rather than of any one model.
4. **Attack hallucination in low-SNR channels**, now that we know that is where it lives.
   Candidates: a penalty on asserted structure in low-signal regions, and testing whether
   posterior averaging suppresses it.
5. **DDRM** — projecting each reverse diffusion step onto the subspace consistent with the
   actual measurement, using the known beam operator this project already extracts from the
   FITS headers. It targets invented structure directly: a freely-generating diffusion model
   has no constraint tying its output to what the telescope saw.
6. **The self-gravitating cube** as a substructure-recovery test, and **real ALMA data**.

---

## 14. Reproducing this

```bash
git clone -b midterm-prep https://github.com/KrishanYadav333/EXXA.git
cd EXXA/DENOISING_DIFFUSION
pip install -r ../requirements.txt
python -m pytest tests/ -q
```

Every notebook bootstraps itself from a blank Kaggle kernel plus the FITS dataset:

- `05-unet-line-emission.ipynb` — U-Net training and moment-map evaluation
- `06-ddpm-line-emission.ipynb` — conditional DDPM, objective sweep, DDIM sampling
- `07-classical-baselines.ipynb` — tuned Gaussian, median and Wiener baselines
- `08-seeds-and-augmentation.ipynb` — multi-seed and augmentation study
- `09-architecture-comparison.ipynb` — U-Net against autoencoder and VAE

`DENOISING_DIFFUSION/results/RUNS.md` is the run index: every number in this post traces to
the notebook version that produced it, including the ones later corrected.

Conventions: seed 42 throughout, cube-level splitting is mandatory, holdout cubes are
inference-only, moment results are always averaged over all 5 holdout cubes with a standard
deviation, and no result is recorded without real execution and artifacts on disk.

---

## 15. Acknowledgements and references

Thank you to my mentors **Jason Terry** and **Gaurav S.** for direction that repeatedly
turned out to be right, in particular the pivot to line emission and the insistence on moment
maps as the deliverable rather than pixel metrics. Thanks also to the wider ML4Sci / EXXA
organisation.

**References**

1. Andrews, S. M. et al. (2018). *The Disk Substructures at High Angular Resolution Project
   (DSHARP)*. ApJL. https://doi.org/10.3847/2041-8213/aaf741
2. Terry, J. P., Hall, C., Abreau, S., & Gleyzer, S. (2022). *Locating Hidden Exoplanets in
   ALMA Data Using Machine Learning*. ApJ 941(2). https://doi.org/10.3847/1538-4357/aca477
3. Terry, J. P., Hall, C., Abreau, S., & Gleyzer, S. (2023). *Kinematic Evidence of an
   Embedded Protoplanet in HD 142666 Identified by Machine Learning*. ApJ.
   https://doi.org/10.3847/1538-4357/acc737
4. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for
   Biomedical Image Segmentation*. MICCAI. https://arxiv.org/abs/1505.04597
5. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
   https://arxiv.org/abs/2006.11239
6. Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*.
   https://arxiv.org/abs/2010.02502
7. Salimans, T., & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion
   Models* (v-prediction). https://arxiv.org/abs/2202.00512
8. Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*
   (cosine schedule). https://arxiv.org/abs/2102.09672
9. Hang, T. et al. (2023). *Efficient Diffusion Training via Min-SNR Weighting Strategy*.
   https://arxiv.org/abs/2303.09556
10. Kawar, B., Elad, M., Ermon, S., & Song, J. (2022). *Denoising Diffusion Restoration
    Models*. NeurIPS. https://arxiv.org/abs/2201.11793
11. Teague, R. (2019). *bettermoments: A robust method to measure line centroids*.
    https://doi.org/10.5281/zenodo.3403130

*Tags: Machine Learning, Astrophysics, Diffusion Models, Exoplanets, Google Summer of Code*
