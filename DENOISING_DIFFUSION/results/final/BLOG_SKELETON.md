# Final blog: skeleton, figures and number sources

Working skeleton for the GSoC 2026 final blog (PLAN.md 0h). The midterm post is on Medium ("Denoising Astronomical Observations of
Protoplanetary Disks"); this one continues it. Draft to Jason on Thu Oct 22, published by Oct 31. Every number in the post comes from a file named
next to it and is re-checked against `results/RUNS.md` (RULES.md 6). Status marks: [have] the number / figure exists; [run] being produced; [todo].

Writing rules: plain language for an ML + astronomy reader; say what each metric means before using it; no em dashes; every claim with its
number and where it comes from; limits and negative results in full.

## 1. Where the midterm left off, and what changed

- Midterm: a U-Net on line-emission cubes, judged on moment maps; this term: one model for every image size and for the GI "wiggle",
  then real ALMA data. [have: midterm post, PROGRESS.md history]
- The question astronomers care about: does denoising keep the velocity structure (the GI wiggle) that a planet or a massive disc leaves?

## 2. Data

- Line-emission cubes (mentor-provided, private), hydro self-gravitating simulations (sg_synth: 4 training + 9074 held out; SG v2 held out),
  analytic GI discs (Longarini model, an01-an06 train, an07 / an08 held out), Young et al. 2026 GI simulations rendered by our projector
  (16 train, 6 + 8 held out), Speedie et al. 2024 GI disc and its Keplerian twin (test only). [have: context.md, research.md 2026-10-10]
- Held out means whole simulations never trained on. Training disc 9025 appears only as a labelled diagnostic.
- Noise: each cube's own measured noise spectrum, channel correlation, SNR 3 to 150 in training.

## 3. The model and the recipe

- Wiener-fed U-Net: the network starts as a blind 3D Wiener filter and learns a correction (zero-initialised).
- Recipe B: average of four checkpoints (R4n-long seeds 3 / 11 / 23 + R8-from-R6), do-no-harm gate (blend toward Wiener above blind SNR 40),
  two-pass Wiener, size fix (network run where the beam spans about 10 px) on noise-only cubes. [have: results/final/RECIPES.md, frozen 2026-10-18]
- Figure: the pipeline as a diagram. [todo]

## 4. Lead result: position-velocity curves along the minor axis

- What a PV curve is (Speedie 2024, Longarini 2024): M1 along the minor axis, where rotation adds nothing, so what is left is the wiggle.
  Our extraction: quadratic line centre, 1 px slice and 5 px slit, quadratic detrend, magnitude as in Speedie. [have: research.md 2026-10-17, pv_curve.py]
- Figure 1 (lead): Speedie GI disc, clean / dirty / Wiener / model PV curves. Model 0.87 vs Wiener 0.81 vs dirty 0.23; magnitude 54 vs 71 m/s
  (clean) vs 20 (Wiener). [have: eval-final-b, figs_pv/speedie_GI_r0.5_1024_pv.png]
- Figure 2: Young M1 R100 and an07 PV curves (0.80 / 0.90). [have]
- Table: PV r and magnitude for every held-out GI disc. [have for eval-final-b cases; run: the 8 new Young discs]

## 5. Five image sizes

- Table: PSNR 51.2 / 51.8 / 51.7 / 51.7 / 51.7 dB, M0 / M1 / M2 about 62 / 89 / 91 % at 256 / 300 / 320 / 480 / 600 px; Wiener 30.5 to 30.6 dB.
  The previous model fell with size; recipe B does not. [have: eval-final-b; PROGRESS.md 2026-10-17]

## 6. Wiggle against signal to noise, with its limit

- Main result from SNR 12 upward (Jason: real ALMA discs are above about 15 to 20; SNR 6 is a footnote).
- Figure: Speedie ladder, model vs Wiener vs the best possible linear filter (oracle), SNR 3 / 6 / 12 / 24. [have: r0.5 selection data;
  run: the r1.5 ONE-TIME final test, which is the reported curve]
- The 0.85 target at SNR 12 is above the linear limit (0.45): reported as a measured limit.

## 7. A hard case: hydro disc 9074

- Native SNR 9.5 after the beam. The model gets the wiggle's wavelength (spine 167 to 178 au vs 180) but not its pattern (agreement at the
  wiggle's scale 0.25 to 0.29), while the beam-only oracle reaches 0.92: noise, not resolution. [have: PROGRESS.md 2026-10-17 / 18]
- Does it recover at ALMA-like SNR? [run: eval-ladder-sg, 9074 at SNR 20 / 40 / 80; 9025 as a training-disc diagnostic]

## 8. Do no harm

- Gate: at SNR about 135 the plain model damaged M0 / M2 by minus 76 / minus 257 %; gated minus 4 / plus 50. [have]
- Invented wiggle on the no-wiggle twin, measured the PV way: about 20 m/s at SNR 12 (clean 11, model 32, Wiener 65). [have: eval-final-b]

## 9. Real ALMA: exoALMA

- MWC 758 13CO: flux kept (1.02), no invented structure, injection-recovery line-centre error 22 / 16 / 13 m/s vs 67 / 37 / 26 (dirty).
  MWC 758 is "a whole different thing" (Jason), not presented as a GI disc. [have: results/alma/mwc758-13co_base_v1_2026-10-17]
- Field of view and zoom: crop-only loses flux (1.02 / 0.98 / 0.96 at 512 / 256 / 128 px); zoom (beam to 10 px) [run].
- Continuum: exoALMA is already subtracted; the automatic check found none and subtracting made no difference. [have]
- 14 discs with PV curves, S1 to S6, published kinematics. [todo: Oct 20 to 21]

## 10. Real ALMA: DSHARP

- 12CO at 0.35 km/s: the model must run on channels interpolated to 0.1 km/s (native fails, interpolated passes: S5 12 / 8 / 6 m/s).
  [have: results/alma/dsharp-hd142666_spectral-test_v2_2026-10-17]
- HD 163296, IM Lup, AS 209, HD 143006 (also in exoALMA), HD 142666 (Terry et al. 2023's planet kink). [todo: Oct 20 to 21]

## 11. What did not work

- R7 (training SNR 3 to 40), R8 loss weights 0.003 / 0.03, R9 (training at the beam scale; seed spread 0.32 vs 0.61 at SNR 12), Young
  simulations in training, SG v2 in training, flip averaging, the size fix on deconvolution cubes, crop-only fields on exoALMA. [have: PROGRESS.md]

## 12. Limits and what is next

- Low SNR (below about 6) not recoverable; 9074 noise-limited; the projector's small-scale noise floor (about 50 m/s); magnitude underestimated
  (about 75 %); single-disc evidence where it applies. Next: whatever Jason's last meeting adds.

## 13. Links

- Final submission branch (`DENOISING_DIFFUSION/Final Submission/` on the final branch), midterm submission, the midterm post.
