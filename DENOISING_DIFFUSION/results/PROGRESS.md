# Progress log

Chronological record of runs, arrivals and bugs, newest first. Written at the time, per
RULES.md #11. `RUNS.md` maps a number to the run that produced it; this file records what
state the project is in and how it got here.

Entry format: **date | trigger | notebook** then what happened, the evidence, the
consequence. Triggers are `run`, `added` (a notebook downloaded into the repo), `bug`.

---

## 2026-10-19 | decision | one recipe: B (Krishan)

Decision 4 of PLAN.md 0h closed: recipe B is shipped alone; A stored, R6 documented, as written in `results/final/RECIPES.md` (DECISION 2026-10-19). Open with Jason only as a confirmation on Fri Oct 23.
Numbers touched: none.

## 2026-10-19 | run | S3 and S4: the answer on real exoALMA data barely depends on the recipe (A) or the zoom (+-25 percent); R6 differs more

Both kernels finished (`chiragbhayal/alma-s3-recipes`, `chiragbhayal17/alma-s4-pixscale`; archives `results/alma/alma-s3-recipes_v1_2026-10-18/`, `alma-s4-pixscale_v1_2026-10-18/`, with `agreement_table.md`). Four discs (MWC 758, LkCa 15, HD 135344B, J1615; i 19 / 50 / 16 / 47 deg, blind SNR 11 to 15), final setting, compared with recipe B's final runs by `experiments/alma_agree.py` (light: 2D maps and a PV profile). Only MWC 758 carries an injection in these runs, so S5 is compared there only.

| variant vs B | flux, variant / B | median abs dM1 on the bright half of the mask (m/s) | PV 5 px agreement with B (r) | S5 GI rc_r MWC 758, SNR 12 / 24 / 36 (B 0.85 / 0.93 / 0.95) |
|---|---|---|---|---|
| **S3** recipe A (3 R4n seeds) | 1.006 to 1.016 | 3 to 4 | 0.94 to 1.00 | 0.85 / 0.93 / 0.95 |
| **S3** R6 (seed 31) | 0.961 to 1.017 | 10 to 11 | 0.81 to 0.98 | 0.74 / 0.89 / 0.93 |
| **S4** beam 7.5 px (zoom minus 25 percent) | 0.988 to 1.001 | 7 to 9 | 0.84 to 0.99 | 0.85 / 0.93 / 0.95 |
| **S4** beam 12.5 px (plus 25 percent) | 0.993 to 0.998 | 4 to 5 | 0.81 to 0.99 | 0.85 / 0.93 / 0.95 |

**Reading.** (1) **S3:** recipe A and B agree to 3 to 4 m/s and 1.6 percent in flux; the fourth network (R8) changes real-disc results very little. R6 differs by 10 to 11 m/s (about a quarter of the 40 to 70 m/s that denoising itself moves M1 by, `m1_shift_bright_median_ms`), with a lower injected-wiggle recovery on MWC 758 (0.74 vs 0.85 at SNR 12): consistent with R6 not being adopted. (2) **S4:** moving the beam by plus or minus 25 percent around the 10 px setting changes M1 by 4 to 9 m/s, flux by at most 1.2 percent, and the MWC 758 injection recovery not at all (identical to two decimals): the zoom setting is not a fine-tuned choice. (3) Where PV agreement with B drops (0.81 to 0.84: LkCa 15 and J1615 at the two beam settings, R6 on LkCa 15), those discs have PV profiles that are noise-dominated on real data (their PV against the raw cube is 0.54 to 0.65 for B); the PV-against-raw numbers of the variants on the same discs fall to 0.31 to 0.47 in three cases, so a PV curve on these discs should be read with a spread of that size. (4) Pass marks (ALMA_PLAN, provisional): none of the four discs' flux changes by more than 5 percent under any variant; M1 differences are inside the 20 m/s mark.
Numbers touched: none published.

## 2026-10-19 | run | DSHARP S5 rerun with correct framing: the model loses to Wiener at SNR 8 to 12 on DSHARP; flux bias much larger there

`aditisah11/alma-dsharp-s5fix` (archive `results/alma/dsharp-s5fix_v1_2026-10-19/`), truth framed as in exoALMA, noise from clean planes (HD 143006: 22 of 40; AS 209: 6 of 12, synthetic). GI rc_r at SNR 6 / 8 / 12 / 24, model / Wiener / dirty: **HD 143006** 0.00 / 0.10 / 0.21 / 0.85 vs Wiener -0.04 / 0.01 / 0.12 / 0.56; **AS 209** 0.01 / 0.16 / 0.41 / 0.93 vs **Wiener 0.05 / 0.19 / 0.65 / 0.89**. Line-centre error model 45 / 31 / 20 / 12 and 28 / 21 / 15 / 9 m/s (dirty 248 / 72 / 34 / 15 and 236 / 81 / 39 / 17). Flux / truth, model: HD 143006 1.41 / 1.20 / 1.02 / 0.99, AS 209 1.35 / 1.19 / 1.06 / 1.02 (dirty 0.86 to 0.99).
**Reading.** The earlier AS 209 S5 (0.86 at SNR 12) used the old framing (truth larger than the field) and is **withdrawn**; with the exoALMA framing, on DSHARP (beam 9.5 to 16.5 px at 10 / 4 mas, 0.35 km/s channels interpolated) the model beats Wiener only at SNR 24 and loses to it at SNR 12 on AS 209 (0.41 vs 0.65). The line-centre error is still 2 to 5 times lower than dirty. The low-SNR flux bias is much larger than on exoALMA (+35 to +41 percent at SNR 6, +19 to +25 at SNR 8). DSHARP is reported as a weaker regime for the model (coarser relative beam, interpolated channels); exoALMA remains the supported case.
Numbers touched: AS 209 S5 (withdrawn; logged 2026-10-18, never published); ANALYSIS.md section 7 needs this row.

## 2026-10-18 | launch + check | S3 (recipe agreement) and S4 (pixel-scale robustness) launched; the S5 fix verified on HD 143006

**S5 fix, local check** (`results/alma/dsharp-hd143006-s5fix-smoke_local_2026-10-18/`, one checkpoint, CPU, SNR 12): with `--inject-field-frac 0.625` and the 18 contaminated planes dropped, the HD 143006 injection shows the whole disc as in exoALMA (R 129 to 1026 au, 3.9 au/px) and no central knot. GI rc_r model 0.27, Wiener 0.12, dirty minus 0.01; line-centre error 17 vs 34 m/s (dirty). Modest because HD 143006's DSHARP beam (16.5 px, 64 au here) is 2.8 times coarser relative to the disc than exoALMA's after zoom; the recipe-B numbers come from `aditisah11/alma-dsharp-s5fix` (running).
**S3 / S4 launched** (ALMA_PLAN, planned for Oct 20 to 21): new `kaggle/alma/kernel_alma_variants.py` runs several recipes or zooms per disc on one download; four discs spanning inclination 16 to 50 deg and blind SNR 11 to 15 (MWC 758 with S5, LkCa 15, HD 135344B, J1615). `chiragbhayal/alma-s3-recipes`: recipe A (R4n seeds 3 / 11 / 23) and R6 (seed 31; the account that owns it; seeds 32 / 33b live on accounts whose outputs cannot be attached there). `chiragbhayal17/alma-s4-pixscale`: recipe B with the beam at 7.5 and 12.5 px (+-25 percent around 10; grid cap raised to 1100 so 12.5 is not clipped). Comparison against B's final runs with the new `experiments/alma_agree.py` (flux ratio, median |dM1| on the bright half of the mask, PV agreement on B's geometry, S6 and PV against raw for each, S5 rc_r on MWC 758).
Numbers touched: none.

## 2026-10-18 | check | V4046 Sgr's continuum subtraction was justified (a weak point source at the star, about 1 sigma per channel); HD 34282 refit lands on exoALMA I's PA

**V4046 Sgr 13CO** (raw cube from Dataverse; script and output `results/alma/exoalma-final-p0_v1_2026-10-18/v4046_continuum_check.*`): the mean of the far blue line-free planes (28 to 36) and of the far red ones (25 to 30) both peak at the star (pixels (257, 255) and (259, 254), centre 256), at 5.7 and 5.5 sigma of the mean; each peak reappears at 3.4 and 4.1 sigma on the other side, and the two maps correlate at 0.50 within 3 beams of the star (0.06 over the whole field). Same compact structure on both sides of the line = continuum, so the automatic decision to subtract it was right. Size: about 5.5 sigma of a 30-plane mean, i.e. about 1 sigma per channel, at the star only: the residual of a uv-plane continuum subtraction on the brightest continuum source of the sample (exoALMA I Table 2: 668 mJy). Its effect on the line flux and moments is negligible (one beam, about 1 sigma per channel), and V4046 Sgr's numbers stand with "continuum: a 1 sigma residual point source at the star removed". One open point: its flux ratio (1.075 at blind SNR 18; HD 135344B 1.045 at 15, MWC 758 1.037 at 11) sits above the truth-based bias (about 1.02 at SNR 12 and above); blind SNR is not the injection's peak SNR, so the comparison is approximate, and the subtraction cannot cause it (it is applied before both the raw and the denoised cube are measured). Jason's rule (exoALMA is already subtracted, do not subtract again) holds for the 14 other cubes, where the check found nothing; the check subtracts only where both far sides agree.
**HD 34282** (`alma_refit.py`, raw cube from Dataverse, denoised cube from the p0 Output; archive `.../HD_34282_13CO/refit/`): sky PA 117.1 vs exoALMA I's 117.4 (0.3 deg; was 50 deg off with M* 7.2), M* 1.68 vs 1.61; S6 rc_r 0.85, PV 5 px r 0.51 against the raw curve.
**Geometry, all 15 exoALMA discs:** 14 with a sound fit within 0.0 to 2.6 deg of exoALMA I (three after the refit: HD 143006, PDS 66, HD 34282); J1604 (6 deg) untestable.
Numbers touched: none published.

## 2026-10-18 | run | R6 now has three seeds: it does not beat recipe B anywhere that matters; recommendation: ship one checkpoint (B), R6 documented

`jensipaneliya/unified-r6-s33b` v2 finished (80 rounds + evaluation; checkpoint `models/unified/unified_r6_k15_s33b.pth`; archive `results/unified/r6-k15-s33b_v2_2026-10-18/`). The three R6 seeds against recipe B (same evaluation cases; B from eval-final-b):

| | 9074 @600 rc_r (deconv.) | 9074_ai @600 rc_r (noise-only) | line @600 PSNR | Speedie r0.5 SNR 12 / 24 rc_r | an07 @600 |
|---|---|---|---|---|---|
| R6 seed 31 (crop256) | 0.43 | 0.37 | 45.8 | 0.61 / 0.81 | 0.85 |
| R6 seed 32 | 0.38 | 0.38 | 46.9 | 0.57 / 0.78 | 0.81 |
| R6 seed 33b | 0.41 | 0.40 | 48.1 | 0.63 / 0.80 | 0.83 |
| R6 mean +- spread (3 seeds) | 0.40 +- 0.03 | 0.38 +- 0.02 | 46.9 +- 1.2 | 0.60 / 0.80 | 0.83 |
| **recipe B** | 0.35 | **0.45** | **51.7** | 0.60 / **0.83** | **0.87** |

**Reading.** R6's only edge is 9074 in the deconvolution regime (0.40 vs 0.35, about two seed spreads); in the noise-only regime (the ALMA case) B is better (0.45 vs 0.38), B is 4.8 dB higher in PSNR, equal or better on the Speedie ladder and an07. The adoption rule (9074 0.43 reproduced across seeds) is not met (0.43 / 0.38 / 0.41). **Recommendation for decision 4 (one checkpoint or two): one, recipe B; R6 is reported as the documented kinematics alternative with these three-seed numbers.** Krishan decides.
Numbers touched: none published.

## 2026-10-18 | run + bug fix | the faint-disc flux excess IS the model (truth-based); exoALMA 15 / 15 done; DSHARP HD 143006 S5 was framed 10x too tight and fed the disc's own emission as noise

**Flux, decided against the truth** (`chiragbhayal17/alma-aatau-s5flux`, archive `results/alma/aatau-s5flux_v1_2026-10-18/`): GI disc injected into AA Tau's own noise, unclipped flux over the truth's disc divided by the true flux:

| peak SNR | 6 | 8 | 12 | 24 |
|---|---|---|---|---|
| model / truth, GI disc | 1.085 | 1.039 | 1.016 | 1.017 |
| model / truth, no-wiggle twin | 1.103 | 1.060 | 1.025 | 1.023 |
| dirty / truth | 0.98 to 0.99 | 0.99 | 0.99 | 1.00 |

**Reading:** the model adds flux, about 2 percent at SNR 12 and above, 4 to 6 percent at SNR 8, 9 to 10 percent at SNR 6: a positive amplitude bias that grows as the SNR falls. It explains most of the real-disc excess (AA Tau +9.9 percent unclipped at blind SNR 8; the fainter discs 1.06 to 1.15, PDS 66 1.29 at blind SNR 7; blind and peak SNR are not the same scale, so the match is qualitative). The JvM hypothesis (research.md 2026-10-18) is not needed. The S1-style clipped ratio model / dirty on the same injections reads 0.64 / 0.82 / 0.90 / 0.97 (the 3 sigma clip on the noisy dirty and the denoised cube behave differently on a large faint disc), so S1's ratio is not a flux measurement and is not quoted as one. For the write-up: flux from the denoised cube is biased high by about 2 percent at SNR 12 and above and up to about 10 percent at SNR 6; on the GI injections in this run the wiggle recovery is rc_r 0.16 / 0.30 / 0.72 / 0.87 (Wiener 0.12 / 0.13 / 0.19 / 0.51) with line-centre error 36 / 31 / 25 / 18 m/s (dirty 170 / 105 / 66 / 37).
**exoALMA p0** (`chiragbhayal17/alma-exoalma-final-p0`, archive `results/alma/exoalma-final-p0_v1_2026-10-18/`); exoALMA is now complete, 15 / 15 discs. MWC 758 sky PA 62.0 vs 240.3 (1.7 deg), HD 135344B 1.2 deg, V4046 Sgr 0.0 deg; HD 135344B S5 GI rc_r 0.86 / 0.93 / 0.95, line-centre error 19 / 14 / 12 m/s (dirty 65 / 37 / 26); MWC 758 S5 reproduces the zoom-512 run. HD 34282 (i 58, the most distant disc, 309 pc): fit failed (M* 7.2, PA 50 deg off), refit running locally. **V4046 Sgr: the automatic check found continuum on both far sides and subtracted it** (line-free mean peak 15.3 sigma_mean): the one exoALMA cube where it fired. exoALMA is delivered continuum-subtracted (Jason), so either a real residual is left in that cube or the check fired on something else; to check before V4046 Sgr's flux (1.075) or moments are quoted.
**Bug (mine): the DSHARP HD 143006 S5 was invalid twice over** (Krishan spotted the over-zoomed figure). (1) Framing: S5 keeps the injected truth's angular size (Speedie cube 1024 px at 20 mas = 20.5 arcsec); HD 143006's DSHARP field is 512 px at 4 mas = 2.05 arcsec, so the field held the central tenth of the injected disc (cavity and inner ring, R 21 to 168 au) and every method scored negative. exoALMA fields hold 62 percent of the truth's width (12.8 arcsec), AS 209's 38 percent. (2) Contaminated noise: 18 of HD 143006's 40 "line-free" planes (+8.2 to +14 km/s, red side only) carry a compact 7 to 11 sigma central source (fast inner-disc gas); S5 drew its noise from those planes, so a knot of the real disc sat in every injected cube (the bright centre in the dirty, Wiener and model M0 panels) and the model brightened it. The same planes enter S1's noise, which is why HD 143006's noise reduction read 2x. Fixes in `alma_unified.py`: `clean_noise_planes` drops line-free planes whose beam-smoothed peak exceeds 5.5 sigma (on HD 143006 exactly planes 35 to 52; synthetic noise from the clean 22 when too few remain; test in `tests/test_alma_fov.py`); `--inject-field-frac` scales the truth so the field shows that fraction of it (0.625 = the exoALMA framing; the au scale and the beam in au follow, so for HD 143006 the injection has the exoALMA 3.9 au/px). The other S5 runs are unaffected: their dirty M0 panels show no central knot (exoALMA noise comes from the far blue-side planes, which are clean; AS 209's was synthetic). Rerun: `aditisah11/alma-dsharp-s5fix` (HD 143006 and AS 209, SNR 6 / 8 / 12 / 24). The earlier HD 143006 DSHARP S5 numbers are void; AS 209's stand until the rerun replaces them. Not changed: S1's noise for HD 143006 (2x) still includes the contaminated planes, noted as a caveat.
Numbers touched: HD 143006 DSHARP S5 (void, never published); the faint-disc flux ratios (now explained as model bias; never published).

## 2026-10-18 | push | Oct 16 to 18 work pushed to midterm-prep; a truncated notebook found in the repo

Pushed (Krishan's go, 15:36 IST, commit e089dae): code (`pv_curve.py`, `alma_table.py`, `alma_refit.py`, `alma_flux_check.py`, `planet_check.py`, the `alma_unified.py` / `wiggle_scorecard.py` / `train_unified.py` changes), tests (`test_pv_curve.py`, `test_alma_fov.py`), kernels (`eval_final_b`, `eval_final_test`, `eval_ladder_sg`, `eval_young_test2`, `eval_planet`, `kernel_alma_dsharp.py`), every archive of Oct 16 to 18 under `results/alma/` and `results/unified/`, RESULTS.md (recipe B), RECIPES.md, BLOG_SKELETON.md, PLAN / PROGRESS / RUNS / research.md / context.md / models/README.md. Kept out on purpose: the eval-final and R7 figure folders (155 + 24 MB, local and in the kernel Outputs, as their READMEs say), the root-level downloaded notebook copies (`05-unet-line-emission V47.ipynb`, `13-checkpoint-loss-sweep (1).ipynb`, `13-checkpoint-loss-sweep V17.ipynb`: RULES.md 7, they belong under `results/<notebook>/v<N>/` once their versions are known), the uncommitted change to `16-checkpoint-evaluation.ipynb` (origin not checked), `2504.18688v1.pdf` (exoALMA I, public on arXiv), `speaker_notes*.pdf`, `nb14_loss_sweep.csv`, `.DS_Store`. Tests: pv_curve, alma_fov, gi_wiggle, line_free pass; `test_notebook_cell_order.py` crashes because the tracked `16-checkpoint-evaluation v6.ipynb` is truncated (exactly 52,428,801 bytes, 50 MiB + 1: a cut download, committed earlier, not today); it also sits beside the live notebook against RULES.md 7. To fix: re-download v6 into `results/16-checkpoint-evaluation/v6_.../` and remove the root copy.
Numbers touched: none.

## 2026-10-18 | run + review | 9074 PV with the beam-matched truth: the model is NOT better than dirty or Wiener on 9074's minor axis; visual review of every final ALMA figure

**9074 PV, beam-matched (`aditisah11/eval-ladder-sg`, archive `results/unified/eval-ladder-sg_v2_2026-10-18/`), 600 px, 5 px slit.** Deconvolution regime, against the beam-convolved truth: native model 0.45, dirty 0.51, Wiener 0.12; at SNR 11 / 20 / 40 / 80 model 0.47 / 0.37 / 0.33 / 0.21, dirty 0.53 / 0.46 / 0.45 / 0.40. Noise-only (`_ai`, against the true clean): native model 0.27, Wiener 0.77, dirty minus 0.29; SNR 11 / 20 / 40 / 80 model 0.36 / 0.43 / 0.53 / 0.80, Wiener 0.34 / 0.62 / 0.51 / 0.78. (The beam oracle scores only 0.19 against clean at 600 px with a 5 px slit, against 0.76 at the native 301 px in the local diagnostic: at 600 px the 5 px slit is half as wide on the sky and catches the unresolvable jumps.) **Reading:** on the minor axis of the held-out hydro disc the model is no better than the noisy cube (deconvolution) and no better than the Wiener filter (noise-only, equal or worse at every SNR), although it beats both on the 2D wiggle scores (rc_r, PROGRESS 2026-10-18 ladder). Reported as a negative result: the PV advantage measured on the GI discs (Speedie, Young, an07) does not carry over to 9074. The earlier sentence "the noise-only PV rises cleanly (0.36 to 0.80)" was true of the model but omitted that Wiener does as well.

**Visual review (Krishan asked for every figure: all 13CO, all S5, IM Lup).** Checked the moments, scorecard and PV figures of the 14 finished exoALMA discs, the 5 DSHARP discs and all S5 injections (MWC 758, DM Tau, LkCa 15, AS 209, HD 143006 DSHARP). Findings:
(1) **S5 injections look right where valid:** on MWC 758, DM Tau, LkCa 15 and AS 209 the model's PV curve tracks the injected GI curve (green on black) at SNR 12 and up where Wiener flattens it and dirty is dominated by noise spikes; at SNR 6 (AS 209) the model's M0 shows noise clumps and nothing recovers the wiggle. On the no-wiggle twin the model leaves a ripple of about plus or minus 20 to 30 m/s (dirty plus or minus 100), as on the final test. The twin's PV r is meaningless (its clean curve is flat: correlations of minus 0.9 to plus 0.9 are trend artefacts); the magnitude is the twin's measure.
(2) **HD 143006 DSHARP S5 is invalid by construction:** at 4 mas pixels the injected Speedie disc (out to about 1000 au, as in the exoALMA injections) is several times larger than the 512 px field, which holds only its cavity and inner edge (R 21 to 168 au); every method scores negative. Not reported; a valid version needs the truth scaled to the cube's field, not to its pixel size.
(3) **The central dot in the denoised "line-free plane" panels is line emission, not invention:** the panel shows the line-free plane next to the line, where the inner disc's high-velocity gas still emits; the continuum check found no matching peak on the far side of the line (at most 3 sigma) on any disc, and AA Tau's far line-free planes peak at 0.05 sigma after denoising.
(4) **Real discs: the denoised PV curve is systematically flatter than the raw one.** Raw / denoised 5 px PV magnitude: IM Lup 87 / 49 m/s, DM Tau 14 / 6, J1615 26 / 10, SY Cha 50 / 18, LkCa 15 21 / 12, HD 163296 47 / 41. Part is noise removal (expected below 1), but some removed features look coherent in the raw 5 px curve: IM Lup's + half sits at about +250 m/s from 40 to 75 au in raw and about +130 denoised (SNR 13); DM Tau's minus half has a 50 m/s step at 100 au that the denoised curve lacks; AA Tau's + half plateau at 125 m/s (50 to 95 au) is flattened. Consistent with the hydro ladder's 30 to 40 percent amplitude loss. Without truth this cannot be split into noise and lost signal; it is reported as a limit: on real data the model's PV curve is a lower bound on the minor-axis structure. HD 163296 (SNR 28) is the exception: raw and denoised agree closely (r 0.97).
(5) PDS 66's PV covers only 29 to 58 au (a compact disc; about 8 samples): its PV r is not meaningful. CQ Tau's 1 px + half raw curve is noise-dominated (jumps of 200 m/s between neighbouring samples).
Numbers touched: the 9074 PV statements of 2026-10-18 (ladder entry) now carry the Wiener comparison; nothing published.

## 2026-10-18 | run + analysis | planet check (Jason's two spots): Young M1 R100 has a compact feature at 7.6 o'clock in clean that the model does NOT recover; run_0002_00560 shows no clear isolated kink

`aditisah11/eval-planet` (45 min; archive `results/unified/eval-planet_v1_2026-10-18/`), recipe B at 600 px, then `experiments/planet_check.py` locally (rotation curve removed, azimuthal Fourier series to m = 3 removed per ring, beam-smoothed; candidate = clean's strongest localised feature beyond 3 beams).
Standard scores on the three run_0002_00560 views (all held out): rt_00 (i 63.7 deg) PSNR model 46.0 (Wiener 32.5, dirty 22.6), rc_r 0.88 (0.69 / 0.76); rt_01 (21.7 deg) 54.1 dB, rc_r 0.75 (0.08 / 0.04); rt_04 (11.7 deg) 59.5 dB, rc_r 0.58 (0.07 / 0.03).

| case | clean candidate | model: r near / amp / own peak | dirty | Wiener |
|---|---|---|---|---|
| run_0002_00560_rt_00 (63.7 deg) | 66 au, 4.0 o'clock, 566 m/s (ring rms 122) | 0.94 / 0.82 / 0.5 beam | 0.82 / 0.27 / 2.3 | minus 0.35 / minus 0.01 / 4.6 |
| run_0002_00560_rt_01 (21.7 deg) | 159 au, 8.9 o'clock, minus 20 m/s (rms 5) | minus 0.22 / 0.35 / 6.5 | 0.55 / 47 / 3.0 | 0.03 / 1.8 / 9.0 |
| run_0002_00560_rt_04 (11.7 deg) | 151 au, 11.5 o'clock, 20 m/s (rms 4) | 0.06 / minus 0.69 / 15.7 | 0.41 / 103 / 16.3 | minus 0.16 / minus 0.45 / 5.3 |
| Young M1 R100, whole disc | 45 au, 2.7 o'clock, 387 m/s (rms 44) | 0.99 / 0.78 / 0.3 | 0.99 / 0.86 / 0.1 | 0.94 / 0.61 / 2.1 |
| **Young M1 R100, 6.5 to 8.5 o'clock** | **69 au, 7.6 o'clock, 171 m/s (rms 37)** | **0.16 / 0.03 / 4.4** | 0.06 / 0.03 / 3.0 | minus 0.08 / minus 0.02 / 10.3 |

**Reading.** (1) **Young M1 R100, Jason's 7 to 8 o'clock:** clean has a compact localised feature there, 171 m/s on the M1 residual (4.6 times the ring scatter) with a bright compact knot in the clean channel at its velocity (figure: `planet_check/run_young_M1_R100_Mu0.4_600_sector6.5-8.5_planet.png`): a clump or planet-like kink about one beam across. The model does not recover it (amplitude 0.03, its own strongest feature 4.4 beams away; the knot is smoothed into the surrounding emission in its channel), and neither does the dirty cube or Wiener. Reported as a negative result: a beam-scale localised kinematic feature of about 170 m/s is below what any method here recovers at this cube's noise. (2) **run_0002_00560:** the simulation was rendered with planets' emission switched off but their gravity kept, so a kink is possible, but this analysis finds no clear isolated one. On rt_00 the strongest localised feature (566 m/s at 66 au) sits just outside the inner cut inside a symmetric multi-lobe pattern around the centre, the signature of the inner disc of a 64 deg disc (beam smearing and the emission surface) rather than a planet; the model reproduces it (r 0.94, amplitude 0.82) where dirty keeps a quarter of it and Wiener none. The two low-inclination views carry only about 20 m/s of localised structure (line-of-sight projection is small). Without the planets' positions (not in the delivered files) a kink cannot be confirmed or ruled out; the honest statement is "no isolated kink stands out in clean at the beam scale". (3) What the check shows for the model: beam-scale localised features that clean has are kept where they are large (rt_00, the Young inner feature: amplitude 0.78 to 0.82) and lost where they are about 170 m/s and one beam wide (Young 7.6 o'clock).
Numbers touched: none published.

## 2026-10-18 | rescore | HD 143006 and PDS 66 refits: PA 1.5 and 0.5 deg from exoALMA I, M* plausible; PV against raw 0.88 / 0.75

`experiments/alma_refit.py` (raw cubes from Dataverse, denoised cubes from the p1 / p3 Outputs, fit started also from Table 2's PA, lowest cost kept; archives `results/alma/exoalma-final-{p1,p3}_v1_2026-10-18/alma/<disc>_13CO/refit/`).
HD 143006 13CO (i 16.9, blind SNR 7): sky PA 167.4 vs 168.9 (1.5 deg; was 17.5 with M* at the 50 Msun bound), M* 1.31 vs Table 2's 1.56 (at 17 deg mass and inclination are nearly degenerate), S6 rc_r 0.43, PV 5 px r 0.88 against the raw curve. PDS 66 13CO (i 31.9, SNR 7): sky PA 9.5 vs 189.0 (0.5 deg modulo 180; was 68.7), M* 1.21 vs 1.28, S6 0.31, PV5 r 0.75. The kernel-side S6 / PV of both (0.98 / 0.98 and 0.97 / nan, on the failed geometry) are void.
**Geometry, all exoALMA discs in so far (14 of 15):** 12 with a sound fit land within 0.4 to 2.6 deg of exoALMA I's PA (HD 143006 and PDS 66 after the refit); J1604 (6 deg, face-on) is untestable. The pipeline places the minor axis correctly on real data.
Numbers touched: HD 143006 / PDS 66 S6 and PV (logged earlier today on the failed geometry; never published).

## 2026-10-18 | run | exoALMA p1 + p3: injection-recovery strong on DM Tau and LkCa 15; PA right on five more discs; two more failed fits (HD 143006, PDS 66); the flux excess tracks SNR

`chiragbhayal17/alma-exoalma-final-p1` and `chiragbhayal/alma-exoalma-final-p3` (archives `results/alma/exoalma-final-p1_v1_2026-10-18/`, `-p3_...`), recipe B, 512 px + zoom, no continuum found on any.

| disc (i deg) | blind SNR | flux | noise red. | S6 rc_r | PV5 r vs raw | sky PA fit / Table 2 (diff) | M* fit / Table 2 |
|---|---|---|---|---|---|---|---|
| LkCa 15 (50.3) | 11 | 1.018 | 124x | 0.77 | 0.65 | 61.2 / 62.1 (0.9) | 1.15 / 1.14 |
| J1615 (46.5) | 12 | 1.022 | 80x | 0.79 | 0.54 | 145.8 / 325.4 (0.4) | 1.14 / 1.14 |
| DM Tau (38.7) | 9 | 1.059 | 51x | 0.49 | minus 0.17 | 156.1 / 335.7 (0.4) | 0.48 / 0.45 |
| CQ Tau (36.3) | 9 | 1.141 | 100x | 0.24 | 0.58 | 52.7 / 235.1 (2.4) | 1.30 / 1.40 |
| J1842 (39.4) | 7 | 1.147 | 76x | 0.64 | 0.69 | 26.8 / 205.9 (0.9) | 1.04 / 1.07 |
| HD 143006 (16.9) | 7 | 1.142 | 41x | 0.98 | 0.98 | **6.4 / 168.9 (17.5)** | **50 (bound)** / 1.56 |
| PDS 66 (31.9) | 7 | 1.291 | 35x | 0.97 | nan | **77.7 / 189.0 (68.7)** | **50 (bound)** / 1.28 |

S5 (injection into the disc's own noise, GI disc rc_r at SNR 12 / 24 / 36): DM Tau model 0.68 / 0.84 / 0.90 (Wiener 0.16 / 0.48 / 0.61, dirty 0.04 / 0.18 / 0.48), line-centre error 27 / 19 / 16 m/s (dirty 67 / 37 / 26); LkCa 15 model 0.75 / 0.88 / 0.92 (Wiener 0.35 / 0.60 / 0.73, dirty 0.04 / 0.17 / 0.45), error 25 / 18 / 15 (dirty 66 / 37 / 26); no-wiggle twin rc_r 0.27 to 0.74, error 7 to 14 m/s (dirty 20 to 53).
**Reading.** (1) Where the fit converged, it is right: sky PA within 0.4 to 2.4 deg of exoALMA I's discminer values and M* within 0.01 to 0.10 Msun of Table 2 on all five (with the inclination fixed at Table 2's), the strongest check so far that our geometry and PV axis are placed correctly. (2) Two fits failed with M* pinned at the 50 Msun bound (HD 143006 nearly face-on, PDS 66 faint): their S6 / PV numbers are void until the local refit (alma_refit.py, started from the Table 2 PA). (3) Injection-recovery on two more real noise fields matches MWC 758 (0.85 / 0.93 / 0.95 there): the model recovers the injected wiggle well above Wiener and dirty at SNR 12 and up in every exoALMA noise field tried. (4) **Flux:** the excess grows as the blind SNR falls: 1.02 at SNR 11 to 12 (LkCa 15, J1615), 1.06 at 9 (DM Tau), 1.14 to 1.15 at 7 to 9 (CQ Tau, J1842, HD 143006), 1.29 at 7 (PDS 66); with p2, 12 discs follow the same trend. The AA Tau truth-based flux test (`chiragbhayal17/alma-aatau-s5flux`, running) decides whether this is the model.
Numbers touched: none published.

## 2026-10-18 | analysis + bug | the faint-disc flux excess is real (+10 percent on AA Tau, in the emission, not the clip, not a sky offset); exoALMA images are JvM-corrected (epsilon 0.23); a truth-based flux test added to S5. A duplicate eval-planet push (module shadowing)

**Flux check on AA Tau** (`experiments/alma_flux_check.py`, raw cube re-downloaded from Dataverse, denoised cube from the p2 Output). The clipped S1 ratio is reproduced (1.090). Hypothesis "the 3 sigma clip biases the raw reference" is **refuted**: the unclipped sum over the 103 line channels inside the same mask gives 1.099 +- 0.009 (the clip loses about 20 percent of both cubes' flux, equally). Wider areas give larger ratios (mask + 2 beams 1.124, + 5 beams 1.157, whole field 1.37), but there is no offset on empty sky (more than 8 beams from the disc, line channels: raw mean minus 0.004 sigma, model minus 0.001); the whole-field number is inflated by the raw image's slightly negative sky. Inside the mask the excess sits in the emission: where the model has emission, its mean is 3.76 sigma against raw 3.45 (about +9 percent); where it has none, both are near zero. So on this disc the model returns about 10 percent more line flux than the delivered image.
Two candidate causes, not yet separated: (a) a positive amplitude bias of the model at low SNR (blind SNR 8); (b) the reference: the exoALMA fiducial images are JvM-corrected (header: `exoALMA: Epsilon = 0.233`, CLEAN threshold 4 mJy at rms 3.46 mJy, multiscale, robust 1.2). Read the same day (research.md 2026-10-18, Czekala et al. 2021): the JvM-corrected image is the flux-accurate one (the residual is rescaled from dirty-beam to CLEAN-beam units), so the delivered image is a valid flux reference; what remains of (b) is that faint emission left in the residual has a dirty-beam shape scaled by epsilon, which a denoiser trained on clean-beam emission could read at the wrong amplitude. **Discriminating test (truth-based):** `inject()` now also records, for each S5 case, the unclipped flux of dirty / Wiener / model over the truth's disc divided by the TRUE flux, and S1's clipped model / dirty ratio as computed on real data, so the injection tells whether a ratio above 1 means the model adds flux. Smoke-tested on CPU (MWC 758 12CO, 128 px) before any push; then run on AA Tau's own noise at SNR 6 / 8 / 12 / 24.
**Bug (mine): a scratch script named `queue.py` shadowed Python's `queue` module.** A local analysis script run from the scratchpad imported torch, torch imported `queue`, and Python found my `queue.py` (the push queue) first: it ran, saw a free slot on chirag and pushed a second `eval-planet` (`chiragbhayal/eval-planet`, 14:24) next to the intended `aditisah11/eval-planet` (14:06). Same code, so the duplicate is a reproducibility check (as with the Young duplicate); the Kaggle CLI cannot cancel a running kernel. Renamed to `kq_push_queue.py`; no scratch file shares a standard-library name now.
Numbers touched: the flux ratios of the faint exoALMA discs (none published); no flux number is quoted until the injection test separates (a) from (b).

## 2026-10-18 | run | exoALMA p2 (AA Tau, SY Cha, J1604, J1852): PA right on the inclined discs; flux ratio above the 5 percent mark on the three faint ones (open)

`chiragbhayal/alma-exoalma-final-p2` (archive `results/alma/exoalma-final-p2_v1_2026-10-18/`), recipe B, 512 px + zoom to a 10 px beam, no continuum found on any.

| disc (i deg) | blind SNR | flux | noise red. | S6 rc_r | PV5 r vs raw | PV5 mag den / raw | sky PA fit / Table 2 | M* fit |
|---|---|---|---|---|---|---|---|---|
| AA Tau (58.7) | 8 | 1.090 | 82x | 0.80 | 0.38 | 26 / 27 m/s | 93.2 / 272.7 (0.5) | 0.74 |
| SY Cha (52.4) | 6 | 1.073 | 166x | 0.73 | 0.24 | 18 / 50 | 163.6 / 346.2 (2.6) | 0.66 |
| J1852 (32.7) | 13 | 1.065 | 143x | 0.81 | minus 0.32 | 9 / 17 | 118.1 / 117.1 (1.0) | 0.96 |
| J1604 (6.0) | 18 | 0.992 | 224x | 0.08 | 0.64 | 9 / 28 | 89.8 / 258.1 (11.7) | 0.43 |

(PA differences modulo 180: Table 2's discminer PAs run 0 to 360 with the redshifted side, the fit's angle is axial.)
**Reading.** (1) Geometry: on the three inclined discs the fit on the raw M1 lands within 0.5 to 2.6 deg of exoALMA I's discminer PA (M* 0.66 to 0.96 vs Table 2's 0.77 to 1.03; AA Tau 0.74 vs 0.79). J1604 is seen almost face-on (6 deg): rotation barely projects onto the line of sight, so its PA (11.7 deg off), M* (0.43 vs 1.29) and S6 (0.08) are unconstrained and it is reported as untestable here, not refitted. (2) PV against the raw curve is weak on these discs (0.38, 0.24, minus 0.32): at blind SNR 6 to 13 the raw curve carries most of the scatter (raw magnitude 17 to 50 m/s vs denoised 9 to 26), so agreement with it is a noise-limited comparison, as on HD 142666; the injection tests (S5, other parts) are the truth-based check. (3) **Flux: open problem.** The three faint discs give 1.065 to 1.090, above the provisional 5 percent mark (MWC 758 1.037, the brighter J1604 0.992, DSHARP 0.98 to 1.03), and the excess grows as the SNR falls. Untested hypothesis: the reference is biased, not the model; the raw M0 is a 3 sigma clipped sum over noisy channels that drops faint line emission, while the denoised cube's faint emission survives the same clip. To check before any flux number is quoted: unclipped sums over the line channels inside the same mask, raw vs denoised, on AA Tau (cubes from the Output and Dataverse).
Numbers touched: none published.

## 2026-10-18 | fix + rescore | HD 142666 refit: PA now 0.04 deg from Huang 2018, M* 1.47; with the right geometry its PV does not follow the raw curve (SNR 7: the raw curve is mostly noise)

The default start of the Keplerian fit (a PA guessed from the M0 shape) converged to a wrong minimum on HD 142666 (faint, inclined 62 deg). Changes: `wiggle_scorecard.score_cube` takes optional starting PAs (`pa_inits`) and keeps the lowest-cost fit, the default start included (so a published PA can only win by fitting the raw M1 better, never by being imposed); the ALMA S6 block is now a function, `alma_unified.s6`, shared by the kernel path and the new `experiments/alma_refit.py`, which re-reads the raw cube exactly as the run did (same crop, same continuum decision, asserts the raw rms matches the report) and the run's own denoised cube from the kernel Output, and rescores S6 / PV with the published PA as an extra start (both redshifted sides: fit angle = sky PA + 90 and - 90). The fit cost is now stored in the geometry (`fit_cost`).
HD 142666 (raw cube re-downloaded from NRAO, 540 MB; archive `results/alma/dsharp-final-p0_v1_2026-10-18/dsharp/HD_142666_interp01/refit/`): refit sky PA 162.07 vs Huang 162.11 (0.04 deg), M* 1.47 Msun (was 10.3), lower cost than the default start's fit. With this geometry: S6 rc_r 0.69 (was 0.94 on the wrong geometry), PV 5 px r against the raw curve minus 0.06 (was 0.62). **Reading:** the earlier 0.94 / 0.62 were artefacts of a wrong geometry and are withdrawn. At blind SNR 7 the raw minor-axis curve is dominated by noise, so agreement with it is not a test of the denoised curve; HD 142666 is reported as too faint for a PV comparison against the raw data (consistent with Jason: SNR 6 to 8 is not the regime that matters). Any exoALMA disc whose fitted PA misses the published one by more than about 10 deg gets the same refit.
Numbers touched: HD 142666 S6 / PV (logged earlier today as 0.94 / 0.62; never published) replaced by 0.69 / minus 0.06.

## 2026-10-18 | run | HD 163296 (after the multi-beam fix): the cleanest real-data result so far; all five DSHARP discs done

`aditisah11/alma-dsharp-final-hd163296` (archive `results/alma/dsharp-final-hd163296_v1_2026-10-18/`). The `beam_of` fix read the beam from the BEAMS table (105 channels, median 0.104 x 0.095 arcsec = 8.0 x 7.3 px at 13 mas; no zoom, the 1024 px grid cap). Blind SNR 28 (the brightest DSHARP disc here), 105 channels of 0.32 km/s, continuum check found none. Flux ratio 0.991, noise reduction 21x, S2 2e-5, S6 rc_r 0.98, PV 5 px r 0.97 against the raw curve (magnitude 41 m/s denoised vs 47 raw: at this SNR the raw curve carries little noise and the model keeps nearly all of it), sky PA 132.7 vs Huang's 133.33 (0.6 deg), M* 1.71 Msun (inclination fixed 46.7).
**DSHARP (5 / 5 run):** on the three discs with a good geometry fit and SNR 12 to 28 (HD 163296, IM Lup, AS 209) the fitted PA lands within 0.1 to 0.9 deg of Huang et al. 2018, the denoised PV curve follows the raw one at 0.63 to 0.97 and the flux is kept within 3 percent; HD 143006 (face-on, SNR 8): PA 3.9 deg off, PV 0.88; HD 142666 (SNR 7, 62 deg): geometry fit failed, refit pending.
**Process slip (mine):** a logging command used an unquoted shell heredoc again and the shell ran the backticks; it stopped at its own assert before touching RUNS / PROGRESS and only the archive README was mangled, rewritten from a script file. Rule kept: edit scripts go through a file, never an unquoted heredoc.
Numbers touched: none published.

## 2026-10-18 | code + docs | planet check set up (Jason: run_0002_00560, Young M1 R100 at 7 to 8 o'clock); RESULTS.md regenerated for recipe B

**Planet check.** run_0002_00560 (all three views held out) is a Phantom SPH simulation that mcfost rendered with `-turn-off_planets` (its `.para` command line): the planets' own emission is switched off, the kinks they cause in the gas are not, so a planet signature there is plausible. The model cubes are needed (600 x 600 x 201, beam about 19 px: too slow on this Mac's CPU), so: (1) `train_unified.py --save-pred DIR --save-pred-cases a,b` saves, for named cases, every method's quadratic M1 and the central +-4 km/s of channels of clean / dirty / model (float16, npz); checked on CPU at 128 px. (2) Kernel `kaggle/eval_planet/` (recipe B on the three run_0002_00560 views and the Young test sims at 600 px, saving run_0002_00560_rt_00 / 01 / 04 and run_young_M1_R100_Mu0.4) is queued for the first free slot. (3) `experiments/planet_check.py` (local): fit the geometry on clean M1, subtract clean's measured rotation curve (`rotation_curve_model`; a forced Keplerian left a rotation-shaped dipole where the beam smears the inner disc), remove per ring the azimuthal Fourier series up to m = 3 (ring mean, rotation error, the m = 2 pattern an elevated emission surface leaves on an inclined disc, smooth spirals; a planet kink is narrow in azimuth and survives), smooth by the beam, take clean's strongest feature beyond 3 beams as the candidate, and score each method there (correlation within 2 beams, amplitude ratio, whether its own strongest feature lies within a beam); figure with the residual maps and the clean / dirty / model channel at the candidate's velocity. Clock positions as our figures show them (origin lower: 12 up, 3 right). The CPU smoke at 128 px (beam 4 px) is not a result: its leftovers sit near the centre and the minor axis of a 64 deg disc, where a thin-disc rotation model is weakest.
**RESULTS.md.** `results/final/RESULTS.md` was the 2026-10-14 report of the pre-recipe-B model (`final+gate`, from eval-final). Regenerated with `experiments/final_report.py` from eval-final-b (recipe B, five sizes), with a header naming the source, the command, and that its SNR ladder is the r0.5 selection disc (the reported curve is the r1.5 final test). Line cubes at 600 px: PSNR 51.74 dB, M0 / M1 / M2 62 / 89 / 91 percent, rc_r 0.76 (Wiener 30.58 dB, 17 / 59 / 51, 0.39). The old file is in git (a2b49b6).
Numbers touched: RESULTS.md now shows recipe B (the deliverable) instead of the pre-B model; nothing published used the old file.

## 2026-10-18 | run | DSHARP p0: IM Lup clean (PA 0.1 deg from Huang 2018, PV r 0.94); HD 142666's geometry fit failed; HD 143006's injection test is invalid

`aditisah11/alma-dsharp-final-p0` (archive `results/alma/dsharp-final-p0_v1_2026-10-18/`), recipe B, 12CO, channels interpolated to 0.1 km/s; continuum check found none on all three (kept as delivered).

| disc | pixels / beam | blind SNR | flux | noise red. | S6 rc_r | PV5 r vs raw | PV5 mag den / raw | sky PA fit / Huang | M* fit |
|---|---|---|---|---|---|---|---|---|---|
| IM Lup | 10 mas, 12.2 px (no zoom) | 13 | 0.977 | 45x | 0.93 | 0.94 | 49 / 87 m/s | 144.4 / 144.5 | 1.02 |
| HD 143006 | 4 mas, 16.5 px | 8 | 1.006 | 2x | 0.84 | 0.88 | 12 / 44 m/s | 165.1 / 169.0 | 1.56 |
| HD 142666 | 10 mas, 7.7 px (zoom 1.29) | 7 | 1.030 | 10x | 0.94 | 0.62 | 15 / 27 m/s | **47.6 / 162.1** | **10.3** |

**Reading.** (1) IM Lup, the brightest of the three: the geometry fitted on the raw M1 lands on Huang's continuum PA to 0.1 deg, the denoised PV curve follows the raw one at 0.94 with half its scatter (49 vs 87 m/s: the raw curve's extra scatter is its noise), flux within 2.3 percent. (2) HD 143006 (face-on, 18.6 deg): PA 3.9 deg from Huang; noise reduction only 2x (blind SNR 8 and a 16.5 px beam, so the size fix runs the network on a coarser grid). Its S5 is not usable: rc_r is negative for EVERY method, dirty and Wiener included, at every SNR, even SNR 24 where the line-centre error is 8 m/s; that is the scoring geometry failing on the injected disc, not the denoising (the most likely cause: HD 143006's 4 mas pixels make the rescaled Speedie disc larger than the 512 px field; to check). The line-centre errors, which need no geometry, still hold: model 24 / 19 / 13 / 8 m/s vs dirty 36 / 26 / 17 / 8 at SNR 6 / 8 / 12 / 24. (3) HD 142666 (faint, SNR 7, inclined 62 deg): the Keplerian fit on the raw M1 failed (M* 10.3 Msun, PA 66 deg from Huang), so its S6 and PV numbers are computed on a wrong geometry and mean nothing yet. Fix planned locally (no GPU needed: the denoised cubes are in the Output): refit with the fit started from the published PA (both redshifted-side choices, lowest cost kept), then rescore S6 / PV; the same for any exoALMA disc whose PA misses by more than about 10 deg.
Numbers touched: none published.

## 2026-10-18 | analysis + fix | 9074's PV problem is the 1 px slice against a truth no beam can keep, not the detrend; PV now also scored against the beam-matched truth

Question (plan 0h, "9074 PV detrend fix"): on 9074 even the beam-only oracle (clean (*) beam, no noise) scored PV r -0.35, so the PV number for this disc meant nothing. Local diagnostic (`results/unified/pv-9074-diagnostic_local_2026-10-18/`), 9074 at native size (301 px, 1.99 au/px, beam 27.6 au, profile R 55 to 260 au):

| PV r against clean | quadratic (current) | degree 4 | degree 6 | high-pass 100 au | 150 au | 250 au |
|---|---|---|---|---|---|---|
| beam oracle, 1 px slice | minus 0.13 | 0.02 | 0.35 | 0.34 | 0.44 | 0.51 |
| beam oracle, 5 px slit | **0.76** | 0.51 | 0.26 | 0.74 | 0.77 | 0.80 |
| dirty, 1 px / 5 px | 0.32 / 0.39 | 0.12 / 0.31 | 0.06 / 0.14 | 0.37 / 0.32 | 0.38 / 0.33 | 0.39 / 0.32 |

**Reading.** (1) The detrend is not the problem: with the 5 px slit the quadratic already gives the oracle 0.76, and no alternative changes the picture by more than a few hundredths; the quadratic (Speedie 2024's choice) stays. (2) The 1 px slice is: on the minus half, clean's M1 drops by about 400 m/s between 65 and 105 au with near-vertical edges (about 40 au wide, 1.5 beams), and the beam-convolved clean does not show it at all (figure). M1 there jumps between two velocity components inside a spiral; a beam blends them and the quadratic line centre follows the brighter one. No beam-limited observation can keep that, so scoring a 1 px PV against clean asks for something impossible. (3) The profile is also short: 55 to 260 au (the minus half stops at 167 au) against a wiggle wavelength of about 180 au, so it holds one or two features, not a pattern; on 9074 PV is weak evidence whatever the reference, and the spine / rc_r numbers stay the main 9074 measures.
**Fix.** `wiggle_scorecard.score_cube` now also scores PV against the beam oracle when one exists (the hydro cases): `pv_r_vs_beam`, `pv5_r_vs_beam` (the beam-matched truth: what a noise-free observation at this resolution would show), next to the clean-referenced keys. Checked locally on 9074: dirty 0.22 / 0.53 (1 / 5 px) against the beam truth; oracle 1.0 by construction. The model's values need the cubes on Kaggle: `eval-ladder-sg` v2 (same kernel, stale-overlay check for the new key) is queued for the first free GPU slot on chirag2 / chirag / aditi. For the write-up: 9074 PV is quoted with the 5 px slit and against the beam-matched truth, with the one-wavelength caveat.
Numbers touched: the 9074 PV numbers in PROGRESS.md 2026-10-17 / 18 (unreliable, never published) get the beam-matched column when v2 lands.

## 2026-10-18 | mentor | email to Jason sent (13:25 IST)

Krishan sent the post-meeting email (draft: `Desktop/presentation_2026-10-16/email_to_jason_draft_v2.txt`; PV figures in `8_email_attachments/`). Content: the PV-curve method and the one-time final test's PV numbers (model 0.35 / 0.71 vs Wiener 0.01 / 0.27 at SNR 12 / 24, about 20 m/s residual ripple on the twin); the exoALMA zoom result (512 px field, beam to 10 px) and the 15 + 5 disc final runs; no second continuum subtraction; the beam keeps its angular size when rescaling; the limits (9074 noise-limited plus a generalisation gap and about 65 percent amplitude kept; weak Young wiggles not recovered). Two questions: one checkpoint or two (leaning one), and whether the checkpoints may be published. No data requested. Blog draft promised for Thursday, before the last meeting on Friday.

## 2026-10-18 | run + bug | first final DSHARP disc (AS 209): wiggle injected into its noise recovered at SNR 12 and up, PA matches Huang 2018 to 0.9 deg; HD 163296 failed on a multi-beam header

`aditisah11/alma-dsharp-final-p1` (archive `results/alma/dsharp-final-p1_v1_2026-10-18/`). **AS 209** 12CO, 768 px, beam 9.5 x 7.2 px (zoom 1.05), blind SNR 12, continuum check: none on both far sides, not subtracted. Flux ratio 1.030, noise reduction 11x, S2 1e-4, S6 rc_r 0.85 against the raw residual, PV 5 px r 0.63 against the raw curve (magnitude 30 m/s denoised vs 45 raw: the raw includes its noise). S5 (GI disc injected into AS 209's own noise; only 12 line-free planes, so noise drawn from their measured spectrum, lag-1 channel correlation 0.66): GI rc_r model 0.26 / 0.55 / 0.86 / 0.95 at SNR 6 / 8 / 12 / 24 (Wiener 0.12 / 0.15 / 0.23 / 0.39, dirty 0.02 / 0.04 / 0.08 / 0.40), line-centre error 23 / 17 / 12 / 7 m/s (dirty 73 / 43 / 25 / 12); no-wiggle twin rc_r 0.18 / 0.39 / 0.81 / 0.92, error 18 / 13 / 9 / 6 m/s. On 0.35 km/s DSHARP channels with interpolation this is as strong as on exoALMA.
**Position angle.** The scorecard's `pa_deg` is the major axis' angle from the image +x axis toward +y; with RA increasing to the left (no spatial flips in `read_cube`), the sky PA is `pa_deg - 90` (mod 180). AS 209: fit 174.9, sky 84.9 deg, against Huang et al. 2018's 85.76 (continuum ellipse): 0.9 deg apart. `experiments/alma_table.py` (new: one table from many reports) applies the conversion; it is checked on every disc as they arrive. The fitted M* is 1.24 Msun (inclination fixed at 34.97), above the usual 0.83 for AS 209: a flat-disc fit to the elevated 12CO surface; not used for anything, noted.
**Bug: HD 163296 failed at read** (`KeyError: 'BMAJ'`). Its cube is a CASA per-channel beam cube (`CASAMBM = T`, beams in a `BEAMS` table, no BMAJ keyword). New `beam_of` in `alma_unified.py`: the header keywords, or the median of the BEAMS table (test in `tests/test_alma_fov.py`). Rerun alone as `aditisah11/alma-dsharp-final-hd163296`. IM Lup (in `-p0`) may have the same header; it will show there.
Numbers touched: none published.

## 2026-10-18 | run | R6 s33b stopped itself at round 65 of 80 and was resumed

`jensipaneliya/unified-r6-s33b` v1 stopped at round 65 / 80 after 560.7 min (the RULES.md 14 deadline logic: 97 min left, a round 8.6 min, 1.5 h kept for evaluation) and wrote RESUME_NEEDED; the checkpoint was persisted locally at once (`models/unified/unified_r6_k15_s33b_round65.pth`) and v2 was pushed about 12:45 with its own output attached (`orchestrator.py resume`). Expected: 15 rounds (about 2.2 h) plus evaluation. Its purpose is unchanged: a third R6 seed for the one-or-two-checkpoints decision; it cannot change recipe B. Archive `results/unified/r6-k15-s33b_v1_2026-10-18_round65/`.
Numbers touched: none.

## 2026-10-18 | decision + launch | ALMA field = 512 px + zoom (beam to 10 px); the final exoALMA (15 discs) and DSHARP (5 discs) runs launched; PA now reported

**Zoom test, second half.** `chiragbhayal17/alma-mwc758-zoom512-v2` (archive `results/alma/mwc758-13co_zoom512_v2_2026-10-18/`). MWC 758 13CO, recipe B, S5 at peak SNR 12 / 24 / 36:

| MWC 758 13CO | flux (M0 ratio) | S5 GI rc_r | S5 GI line-centre error, m/s | S5 twin error, m/s | S5 twin invented, m/s | S6 rc_r vs raw | noise reduction | minutes |
|---|---|---|---|---|---|---|---|---|
| 512 px, native pixels (`base`, Oct 17) | 1.022 | 0.82 / 0.92 / 0.94 | 22 / 16 / 13 | 8 / 6 / 5 | (not stored) | 0.80 | 112x | 33 |
| 256 px field (6.4 arcsec) + zoom | 0.991 | 0.81 / 0.91 / 0.94 | 21 / 15 / 12 | 12 / 9 / 7 | 1.2 / 0.8 / 1.0 | 0.80 | 157x | 19 |
| **512 px + zoom** | 1.037 | **0.85 / 0.93 / 0.95** | **20 / 15 / 12** | 9 / 7 / 5 | **0.0 / 1.4 / 2.0** | 0.76 | 196x | 67 |

(dirty for reference: GI rc_r 0.04 / 0.16 / 0.47, error 65 / 37 / 26 m/s; Wiener GI rc_r 0.65 / 0.83 / 0.89.)
**Decision: 512 px field + zoom (beam to about 10 px) for every final ALMA run.** Reasons: the best injection-recovery of the three on the GI disc at every SNR; the twin's line-centre error stays at the 512 px level (9 / 7 / 5 m/s, against 12 / 9 / 7 on the 256 px field, which cuts the disc's outer emission); a fixed 6.4 arcsec field would not fit the larger discs (V4046 Sgr at 72 pc, DM Tau, LkCa 15), while 512 px (12.8 arcsec) is the field that kept flux on Oct 17. The cost is flux: 1.037, 3.7 percent over the raw total (inside the provisional 5 percent mark of ALMA_PLAN, but the furthest of the three from 1; cause not yet measured). It is reported, not hidden. The S1 bright-pixel M1 shift (median 73 m/s, p95 296; native 78 / 253) is above the provisional 20 / 50 m/s marks in every configuration, as it was in every MWC 758 run: it compares against the RAW M1, whose own noise on those pixels is about that size (S5's truth-based errors, 12 to 20 m/s, are the measure that counts). The Young duplicate run (`chiragbhayal/eval-young-test2`) came back identical to the misnamed one to the last digit (largest absolute difference 0): the inference is deterministic.

**Code** (`experiments/alma_unified.py`, `experiments/wiggle_scorecard.py`, kernels). (1) `--target-beam-px` (and `--max-scaled-px`, default 1024): the zoom factor is set per cube from its own beam (`target_scale`: beam to the target, never zoom out, grid capped for GPU memory), because a fixed `--scale 1.6667` fits only MWC 758's 6 px beam. DSHARP (10 mas pixels, beam about 7.7 px) gets 1.3x on the 512 / 768 px fields and none on the 1024 / 1280 px fields (cap). Test in `tests/test_alma_fov.py`. (2) Position angle (Jason: getting it right matters): the scorecard fits PA, centre and vsys from the raw M1 with the inclination fixed; the report's S6 block now stores the fitted geometry (`geom`: `pa_deg_fit`, `vsys_fit`, centre, M*) next to the published PA (`pa_lit_deg`: exoALMA I Table 2, discminer; DSHARP: Huang et al. 2018 Table 2, continuum, 0 to 180 deg so compared modulo 180), plus the PV numbers of the real cube (`pv`, `pv_mag_raw_ms`). The PA is not forced to the published value; the two are compared disc by disc in RESULTS. (3) `kernel_alma_all13co.py` now covers all 15 exoALMA discs (MWC 758 added), with Table 2 PAs and inclinations, S5 on MWC 758, HD 135344B, DM Tau and LkCa 15, `--crop 512 --target-beam-px 10`, and a `PART, NPART` split so the discs run over several kernels; `kernel_alma_dsharp.py` gets PAs, the zoom, and the same split. Both refuse a stale code overlay. A CPU smoke test on the local MWC 758 12CO cube (128 px) ran the whole path; its PV came back NaN because a 128 px crop of 12CO holds no usable minor-axis radii (the local file is also truncated, 128 of 301 planes), so PV on a full field is first checked on the Kaggle runs.

**Launched about 10:25 IST**, recipe B, one push per job with its folder printed: exoALMA `chiragbhayal17/alma-exoalma-final-p0` (MWC 758, HD 34282, HD 135344B, V4046 Sgr), `-p1` (DM Tau, CQ Tau, HD 143006, J1842), `chiragbhayal/alma-exoalma-final-p2` (AA Tau, SY Cha, J1604, J1852), `-p3` (LkCa 15, PDS 66, J1615); DSHARP `aditisah11/alma-dsharp-final-p0` (HD 143006, HD 142666, IM Lup), `-p1` (AS 209, HD 163296).
Numbers touched: none published. The Oct 13 / 14 exoALMA numbers (`aditisah11/alma-exoalma-all13co`, R4n models, continuum subtracted twice) are replaced by these runs.

## 2026-10-18 | bug + run | a watcher pushed the wrong code under a kernel name (bash 3.2); the 8 new Young discs: strong moments, weak wiggles not recovered; zoom on the 256 px field helps

**Bug (mine, found 04:53).** `watch_oct18b.sh` mapped job names to folders with `declare -A` (an associative array). macOS ships bash 3.2, which has no associative arrays: the keys were evaluated as arithmetic (`alma - mwc758 - zoom512` = 0), every assignment landed on index 0 and the last one won, so the job named `alma-mwc758-zoom512` was pushed with the folder `kaggle/eval_young_test2`. Caught because its Output held `eval_young_test2_B.json` instead of an ALMA report; confirmed by pulling the kernel source and by `kaggle/accounts/state.json` (`dir: .../kaggle/eval_young_test2`). Consequences: the MWC 758 zoom-512 test never ran (relaunched 04:55 as `chiragbhayal17/alma-mwc758-zoom512-v2`, folder checked first); the misnamed run is a complete, valid Young-test2 evaluation (archived as such, `results/unified/eval-young-test2_v1_2026-10-18/`); the intended Young run on chirag is a duplicate (reproducibility check). Nothing else used that script. Rule from now on: no bash associative arrays in these helpers (or run them with a modern bash); each push prints the folder it sends.

**8 new held-out Young discs** (TEST2; recipe B; 600 px; model / Wiener / dirty):

| disc (i deg) | rc_r | PV r, 5 px | PV magnitude model / clean, m/s | M0 / M1 / M2 % | PSNR model / Wiener |
|---|---|---|---|---|---|
| M0.1_R100_Mu1.4 (23) | minus 0.06 / 0.20 / 0.05 | 0.29 / minus 0.23 / 0.22 | 6 / 1 | 54 / 92 / 84 | 53.1 / 31.2 |
| M0.2_R100_Mu0.7 (49) | 0.51 / 0.36 / 0.39 | 0.54 / 0.68 / 0.36 | 15 / 10 | 50 / 93 / 96 | 50.7 / 33.2 |
| M0.3_R200_Mu1.0 (14) | minus 0.08 / 0.01 / minus 0.01 | 0.19 / 0.08 / 0.19 | 11 / 10 | 41 / 91 / 82 | 53.2 / 33.2 |
| M0.3_R50_Mu0.5 (49) | 0.50 / 0.37 / 0.37 | 0.20 / minus 0.39 / 0.41 | 13 / 14 | 50 / 92 / 96 | 51.4 / 32.5 |
| M0.5_R100_Mu0.3 (40) | 0.43 / 0.33 / 0.19 | 0.62 / 0.79 / 0.65 | 15 / 4 | 54 / 94 / 95 | 50.2 / 32.3 |
| M0.5_R200_Mu0.6 (53) | 0.61 / 0.30 / 0.38 | minus 0.03 / minus 0.10 / 0.18 | 11 / 6 | 42 / 90 / 94 | 49.9 / 33.8 |
| M0.7_R200_mu0.5 (42) | 0.55 / 0.25 / 0.21 | 0.39 / minus 0.74 / 0.49 | 11 / 7 | 47 / 92 / 96 | 49.5 / 32.7 |
| M1_R100_Mu0.3 (30) | 0.37 / 0.14 / 0.05 | 0.22 / 0.43 / 0.19 | 19 / 6 | 60 / 89 / 92 | 49.7 / 31.3 |

Means: rc_r 0.35 / 0.24 / 0.20 (600 px), 0.34 / 0.25 / 0.20 (300 px); PV 5 px 0.30 / 0.07 / 0.33 (600 px).
**Reading.** (1) Moments and PSNR: strong and consistent on every new disc (PSNR 49.5 to 53.2 vs Wiener 31.2 to 33.8; M1 89 to 94, M2 82 to 96). (2) The wiggle: these discs have WEAK minor-axis wiggles (clean PV magnitude 1 to 14 m/s, against 63 to 113 m/s for Speedie and the earlier Young discs M0.7 R50 and M1 R100). On them the model is modestly above Wiener and dirty in rc_r (0.35 vs 0.24 / 0.20), its PV correlation is no better than the noisy cube's (0.30 vs 0.33), and its PV magnitude exceeds the clean one by 1 to 13 m/s (ripple added at the level of the signal itself). The two near-face-on discs (14 and 23 deg) give nothing (rc_r minus 0.06, minus 0.08): little line-of-sight velocity to wiggle. (3) Together with the earlier set (strong wiggles recovered: 0.93, 0.80, 0.72), the honest statement is: the model recovers GI wiggles whose minor-axis amplitude is well above the noise (tens of m/s and more); wiggles of about 10 m/s or less, or seen nearly face-on, are not recovered by it or by anything else here. That is the claim the blog makes, with both tables.
**Zoom (Jason's "zoom in"), 256 px field, first half.** `aditisah11/alma-mwc758-zoom256`: the 6.4 arcsec field with `--scale 1.6667` (beam 6 to 10 px) against the same field without zoom: flux ratio 0.991 vs 0.980, S5 GI line-centre error 21 / 15 / 12 vs 23 / 16 / 13 m/s, S5 GI rc_r 0.81 / 0.91 / 0.94 vs 0.76 / 0.88 / 0.92, twin invented 1.2 / 0.8 / 1.0 vs 2.6 / 1.5 / 2.0 m/s, twin error 12 / 9 / 7 both. The zoom helps on every measure but one (equal). The 512 px field with zoom (`-v2`, running) decides the configuration for the rerun.
Numbers touched: none published.

## 2026-10-18 | run | hydro SNR ladder: 9074 is noise-limited at its native SNR AND short of what the model does on a training disc; the model over-smooths the hydro wiggle

`chiragbhayal17/eval-ladder-sg` v1 (40 cases; archive `results/unified/eval-ladder-sg_v1_2026-10-18/`). Recipe B. 9025 is a TRAINING disc: its numbers are a diagnostic, never a held-out result.

| rc_r, 600 px | native | SNR 11 | SNR 20 | SNR 40 | SNR 80 |
|---|---|---|---|---|---|
| 9074 noise-only: model / dirty / Wiener | 0.45 / 0.31 / 0.13 | 0.40 / 0.15 / 0.09 | 0.48 / 0.32 / 0.09 | 0.57 / 0.48 / 0.22 | 0.70 / 0.61 / 0.62 |
| 9025 noise-only (training disc): model / dirty | 0.35 / 0.08 | 0.57 / 0.13 | 0.66 / 0.29 | 0.78 / 0.42 | 0.82 / 0.57 |
| 9074 deconvolution: model / dirty | 0.35 / 0.13 | 0.36 / 0.21 | 0.38 / 0.34 | 0.56 / 0.47 | 0.64 / 0.63 |
| 9025 deconvolution: model / dirty | 0.39 / 0.13 | 0.45 / 0.32 | 0.36 / 0.38 | 0.48 / 0.44 | 0.56 / 0.56 |
| beam-only oracle (no noise): 9074 / 9025 | 0.93 / 0.80 | | | | |
| 9074 noise-only, model: spine / PV 5 px / agreement at wiggle scales / amplitude ratio | 0.76 / 0.27 / 0.29 / 0.60 | 0.77 / 0.36 / 0.28 / 0.65 | 0.79 / 0.43 / 0.35 / 0.67 | 0.82 / 0.53 / 0.44 / 0.64 | 0.86 / 0.80 / 0.61 / 0.68 |

300 px gives the same picture (9074_ai 0.34 / 0.47 / 0.56 / 0.66; 9025_ai 0.54 / 0.65 / 0.78 / 0.81 at SNR 11 / 20 / 40 / 80).
**Reading.** (1) **Noise is part of it:** on 9074 every score rises with SNR (noise-only rc_r 0.40 to 0.70, PV 0.36 to 0.80, spine 0.77 to 0.86, agreement at the wiggle's scale 0.28 to 0.61), so at its native SNR (9.5) Jason's "beam noise" reading is right. In the noise-only (ALMA) regime the model stays above dirty at every SNR (0.48 vs 0.32 at SNR 20). (2) **But not all of it: a generalisation gap.** At the same SNR the model recovers the training disc 9025 clearly better than the held-out 9074 (0.57 vs 0.40 at SNR 11, 0.66 vs 0.48 at SNR 20, 0.78 vs 0.57 at SNR 40), although 9025 is harder to resolve (beam-only oracle 0.80 vs 0.93). Even at SNR 80, 9074 reaches 0.70 against an oracle of 0.93. So part of the shortfall on a new hydro disc is the model not generalising from four hydro training discs; the fix would be more varied training discs, a new model idea, so it is reported as a limitation, not attempted (PLAN.md 0h rules). (3) **Systematic over-smoothing:** on 9074 the model's wiggle amplitude is 0.60 to 0.70 of the truth at every SNR, so it removes 30 to 40 percent of the hydro wiggle even when noise is no longer the problem; on the training disc the amplitude ratio is above 1 at low SNR (1.4 to 2.0), i.e. on a disc it has seen it adds structure. (4) **Deconvolution regime:** at SNR 40 and above the model is no better than dirty on either disc (9074 0.64 vs 0.63 at SNR 80), so its gain there is noise removal only; the size fix and the noise-only path (the ALMA case) are where it adds. (5) The 9074 PV score in the deconvolution regime stays unreliable (the large trend, PROGRESS 2026-10-17); the noise-only PV rises cleanly (0.36 to 0.80).
**What it changes in the write-up.** 9074 is presented as: noise-limited at its native SNR (true), with a measured generalisation gap of about 0.2 in wiggle score against a training disc at the same SNR, and a systematic 30 to 40 percent amplitude loss. Not hidden, not overstated: at ALMA-like SNR (20) on the noise-only path the model is still 0.16 above dirty and 0.39 above Wiener.
Numbers touched: none published.

## 2026-10-18 | run | THE ONE-TIME FINAL TEST (Speedie r1.5, never used before): the model beats Wiener and dirty at SNR 12 and 24, below its selection-disc numbers; residual noise on the twin

`chiragbhayal17/eval-final-test` v1, run once on the recipe frozen at 03:00, exactly as pre-registered on 2026-10-15. Archive `results/unified/eval-final-test_v1_2026-10-18/` (JSON, log, all 12 figures). Reported as it came out; not to be re-run.

| Speedie r1.5 | SNR 6 | SNR 12 | SNR 24 |
|---|---|---|---|
| wiggle rc_r: **model** / Wiener / dirty | **0.06** / 0.05 / 0.00 | **0.44** / 0.17 / 0.03 | **0.70** / 0.41 / 0.16 |
| (r0.5 selection disc, model; Wiener) | (0.14; 0.07) | (0.60; 0.18) | (0.83; 0.43) |
| spine_r: model / Wiener / dirty | 0.16 / minus 0.09 / minus 0.07 | 0.25 / minus 0.02 / minus 0.02 | 0.31 / 0.16 / 0.01 |
| PV r (1 px; 5 px): **model** | **0.16; 0.16** | **0.35; 0.35** | **0.71; 0.71** |
| PV r: Wiener / dirty | 0.04 / 0.03 | 0.01 / 0.12 | 0.27 / 0.64 |
| PV magnitude GI disc, m/s (clean 63): model / Wiener / dirty | 102 / 197 / 1130 | 94 / 72 / 847 | 60 / 32 / 88 |
| PV magnitude on the twin, m/s (clean 3): model / Wiener / dirty | 81 / 233 / 1004 | 84 / 24 / 336 | 49 / 27 / 75 |
| twin rotation-curve excess, m/s: model / Wiener / dirty | 55 / 137 / 1535 | 30 / 28 / 789 | 11 / 4 / 170 |
| moments M0 / M1 / M2 % (model), PSNR | 58 / 90 / 85, 51.2 dB | 64 / 87 / 87, 50.0 dB | 47 / 77 / 88, 49.8 dB |

**Reading.** (1) On a disc used for no decision, the model recovers the wiggle well above Wiener and dirty at SNR 12 (rc_r 0.44 vs 0.17, PV 0.35 vs 0.01) and SNR 24 (0.70 vs 0.41, PV 0.71 vs 0.27); the PV figure at SNR 24 shows the model following the clean minor-axis curve where Wiener flattens it. At SNR 6 nothing recovers it (0.06; PV 0.16), as on the selection disc. (2) **Selection optimism, measured:** the r1.5 numbers are 0.13 to 0.16 below the r0.5 numbers the recipe was chosen on (0.44 vs 0.60 at SNR 12, 0.70 vs 0.83 at SNR 24), while Wiener is about the same on both discs (0.17 vs 0.18, 0.41 vs 0.43): part of the r0.5 margin was selection. The r1.5 curve is the one to report; the r0.5 curve is labelled selection data. (3) **On the twin the model's PV curve is not flat:** about plus or minus 20 m/s beyond 300 au and larger swings at the inner edge below 300 au (`figs/speedie_ladder_Kep_r1.5_snr24_1024_pv.png`), so its PV magnitude is 49 m/s at SNR 24 against 3 for clean and 27 for Wiener; most of that number is the inner-edge swing. The ripple follows the dirty cube's bumps (near 320, 520, 650 au), so it is leftover smoothed noise, not an invented GI pattern, and it is smaller than the true GI wiggle on the same disc (63 m/s magnitude, plus or minus 100 to 150 m/s); but at SNR 12 the twin's 84 m/s is comparable to the GI disc's, so at SNR 12 a PV magnitude alone cannot tell a wiggle from noise on this disc: the correlation with the true curve (0.35) is what carries the claim, not the amplitude. (4) The SNR 6 magnitudes (model 102 vs clean 63) are noise.
Numbers touched: the reported wiggle-vs-SNR curve is now this one (0.06 / 0.44 / 0.70), replacing the r0.5 selection curve (0.14 / 0.60 / 0.83) as the headline; the deck's slide 10 and the scorecard row "Speedie SNR 12 0.60" become selection numbers (refresh Oct 22).

## 2026-10-18 | decision + run | recipe B frozen; R6 s33 not pursued; the one-time final test, the hydro SNR ladder and 8 more held-out Young discs launched; zoom runs queued

**R6 s33 (updated 03:15, Krishan: we have seven accounts, why drop it):** restarted from scratch as `jensipaneliya/unified-r6-s33b` (same recipe and seed 33, fresh account and slug, so neither the main-account mount failure nor the unattachable v1 output is in the way); about 9.5 h + one resume. It cannot change recipe B (the rule failed at seed 32), but it makes R6 a proper 3-seed set for the one-or-two-checkpoints decision. Original entry: v1 stopped at round 67 of 80 (deadline); the resumes v2 (16:47) and v3 (21:47) both crashed at startup with no inputs mounted (`IndexError` on `MANIFEST.json`), and after a failed version Kaggle's "own output" points at the empty failed run, so v1's output cannot be attached any more. Not pursued: the adoption rule needed the 9074 gain to repeat (0.43) and seed 32 already did not (0.38), so a third seed cannot make it consistent; seeds 31 and 32 answer the alternative-checkpoint question. Partial checkpoint kept: `models/unified/unified_r6_k15_s33_round67.pth`. Archive `results/unified/r6-k15-s33_v1-v3_2026-10-17_stopped/`. The cause of the empty-input crash on the main account is not understood (the same push worked for R6 s32 on anshika); noted as a risk for any further resume on main.
**Recipe B frozen** (`results/final/RECIPES.md`, "FROZEN 2026-10-18"): unchanged; no Oct 16 to 17 candidate met its rule.
**Launched on chirag2** (code overlay re-synced and checked ready each time; each kernel asserts the code it needs is present, after the stale-overlay crash of Oct 17):
- `eval-ladder-sg` (`kaggle/eval_ladder_sg/`): recipe B on 9074 and 9025 at native SNR and peak SNR 11 / 20 / 40 / 80 after the beam, deconvolution and noise-only, 300 and 600 px. New trainer options `--sg-ladder`, `--sg-ladder-base` (tested locally on the real 9025 cube: every case hits its target SNR exactly). 9025 is a TRAINING disc: diagnostic only (noise limit vs generalisation gap), never a held-out result.
- `eval-final-test` (`kaggle/eval_final_test/`): THE ONE-TIME FINAL TEST, Speedie r1.5 GI + twin at SNR 6 / 12 / 24, exactly as pre-registered on 2026-10-15. Run once; reported whatever it gives.
- `sph-cubes-test2` (CPU): 8 more whole Young et al. simulations held out (`TEST2` in `make_sph_cubes.py`: M0.1_R100_Mu1.4, M0.2_R100_Mu0.7, M0.3_R200_Mu1.0, M0.3_R50_Mu0.5, M0.5_R100_Mu0.3, M0.5_R200_Mu0.6, M0.7_R200_mu0.5, M1_R100_Mu0.3; the two Mu0.3 are low-mass controls); disjoint from TRAIN and TEST (checked); `make_sph_cubes.py` added to the code overlay (it was not in it: the kernel would have used the git copy and labelled the new runs "train").
- Queued by a watcher for the next free chirag2 GPU slot: `alma-mwc758-zoom512` (512 px field, `--scale 1.6667`: the 6 px beam to 10 px) and `alma-mwc758-zoom256` (6.4 arcsec field + the same zoom), Jason's "zoom in" (crop AND enlarge). `alma_unified.run_scaled` now runs the model on the scaled grid for the real cube AND for S5 (S5 used to ignore `--scale`), test added (`tests/test_alma_fov.py`: the model sees 200 px and a 10 px beam at scale 1.67, round trip within 2.6 % of the peak).
**Measured native SNR** (peak of beam (*) clean over the rms of dirty minus that): 9074 **9.5** (not "about 11" as in the deck), 9025 7.7, 9015 7.6, 9019 9.3, 9032 33.7, an07 19.7.
Numbers touched: the deck's "9074 peak SNR about 11" becomes 9.5 (refresh Oct 22).

## 2026-10-17 | meeting (re-read) | Oct 16 transcript read again in full: what Jason said, corrected where the first reading was loose

Krishan supplied the full transcript again. Points the first entry (2026-10-16 | meeting) missed or read too loosely:
- **"Zoom in" is not only a crop.** His reasoning: the exoALMA field is huge, "the network will see the features relative to the size of it, they're just so much smaller", and "I bet the beam is tiny on that, too"; "crop it and then you can expand it later". The MWC 758 13CO beam is 6 px at 25 mas, below the 8 to 20 px our training cubes span, and the size fix only ever SHRINKS a cube whose beam is wider than 12 px; it never enlarges one whose beam is small. So his suggestion is crop + resample up (beam to about 10 px, disc filling the network's view), then expand back. The Oct 17 field-of-view tests (128 / 256 / 512 px, PROGRESS.md same day) only cropped, at native pixels, so they do not test what he meant: the "keep 512 px" decision stands for crop-only, and crop + upsample is still untested. Test it before telling him the crop does not help.
- **"Beam noise" was about 9074.** Krishan showed the held-out hydro disc where the models do not recover the wiggle ("nowhere near clean, even for different pixel sizes") and Jason said "that looks like beam noise to me": the structure in those model rows is at the beam scale, not wiggle. It agrees with the Oct 17 PV result on 9074 (600 px, the beam spans about 27 px of a small disc): 9074 is a limit case, not the headline.
- **What the PV curve is, in his words:** "what is the moment 1 value, or fit the velocity peak using ... the collapse quadratic method ... just down the minor axis, so where that spine is, what is the moment 1 value along that line"; "a slice straight through the cube, sometimes people take finite widths, maybe 4 or 5 pixels wide, either one". `pv_curve.py` does exactly this (quadratic M1, slice 1 px and slit 5 px).
- **Why PV:** it is "in the Speedie paper and the Longarini Elias 2-27 paper; that's how they fit the wiggle ... we need to be able to replicate that if we want to say we're contributing to the standard", and "it also avoids things like ... the second row there ... that's just shooting out way to the side" (the spine tracker running off on the noisy dirty row). The spine is not dropped: "it's good that you're extracting those spines, and ... some of the spines clearly have wiggles on the more obviously self-gravitating discs, so ... something's going right"; "even if you're still using my spine-fitting technique ... you recover the curve very well".
- **Not to do:** no GIGGLE fitting routine (Longarini's analytic model; "takes a long time") and no discminer fitting ("takes even longer"); "just visually, if we can get a really good looking position-velocity curve ... then we could really start considering what's next".
- **Results he liked:** "it's good you're getting the position angle right ... that's really important"; the Young M1 R100 and Speedie panels "look good"; "the moment 1 residual ... clearly self-gravitating"; "looks like you're getting sub-beam level resolution on the second to bottom one"; "the moments ... look a lot better than the dirty one, so it's definitely success"; "these are really good models now". Not settled: Krishan said "two of the models, I was confused which one to take" and Jason answered "I think that second one ... sub-beam level resolution on the second to bottom one" about a figure we cannot identify from the transcript (which figure and which row was on screen is for Krishan to say), so this is not an answer to the one-or-two-checkpoints question.
- **Planets:** he sees a planet-like feature "around 7:00 or 8:00 in some of the moment 1 maps" of the run he called "2560" (probably run_0002_00560) and in Young M1 R100. Not checked yet.
- **SNR:** "you can assume an SNR probably above 15 to 20 in most cases"; "you tend to get less improvement because the observations are already good"; rescaling is optional; SNR 6 "is not really relevant to modern observations". **Continuum:** "most of the exoALMA data ... will" be subtracted, "I don't think the DSHARP data by default has continuum subtracted", "don't subtract the continuum yourself ... you'd be double subtracting". **Models:** keep training in the background, no qualitative jump expected, focus on post-inference analysis. **MWC 758:** "that's a whole different thing" (not taken as a GI disc). **Seven Kaggle accounts:** "hopefully they don't catch you": a real risk (Kaggle's terms), noted in PLAN.md 0f risks.
Numbers touched: none. Plan change: a crop + upsample test on MWC 758 (Oct 18) before the Oct 20 to 21 rerun.

## 2026-10-17 | run | FINAL EVALUATION of recipe B at five sizes, with the first PV curves on the test discs: strong on GI discs, not meaningful on 9074

`chiragbhayal17/eval-final-b` v1 (95 cases, none failed; archive `results/unified/eval-final-b_v1_2026-10-17/`, eight PV figures in `figs_pv/`). Recipe B, both levers, gate. **Five line holdouts (mean), model vs Wiener PSNR (30.5 to 30.6 at every size):**

| size px | PSNR dB | M0 / M1 / M2 % | rc_r | spine_r |
|---|---|---|---|---|
| 256 | 51.22 | 62 / 89 / 90 | 0.87 | n/a |
| 300 | 51.78 | 63 / 89 / 91 | 0.85 | n/a |
| 320 | 51.66 | 63 / 89 / 91 | 0.77 | 0.72 |
| 480 | 51.70 | 62 / 88 / 91 | 0.74 | 0.75 |
| 600 | 51.74 | 62 / 89 / 91 | 0.76 | 0.75 |

The previous frozen model fell with size (PSNR 48.2 to 49.3, M0 52 to 47, rc_r 0.88 to 0.68); recipe B is flat in PSNR and moments at every size and keeps most of the wiggle (0.87 to 0.76). The size fix does what it was built for. The 256 and 300 px spine score is n/a (no spine channel near the systemic velocity at those sizes in the scorecard, as for the previous model); this is the first measurement of recipe B at 300, 320 and 480 px.
**PV curves (M1 quadratic along the minor axis, slice 1 px / slit 5 px; `pv_r` = correlation with clean after a quadratic detrend; `pv_mag_ms` = Speedie's wiggle magnitude):**

| case | dirty r | Wiener r | model r | magnitude: clean / dirty / Wiener / model m/s |
|---|---|---|---|---|
| Speedie GI r0.5, native | 0.23 | 0.81 | **0.87** | 71 / 183 / 20 / 54 |
| Speedie ladder GI SNR 24 | 0.16 | 0.44 | **0.77** | 71 / 213 / 24 / 54 |
| Speedie ladder GI SNR 12 | 0.03 | 0.12 | **0.44** | 71 / 601 / 49 / 59 |
| Speedie ladder GI SNR 6 | minus 0.10 | minus 0.07 | 0.11 | 71 / 1283 / 203 / 71 |
| an07, 600 px | 0.07 | 0.03 | **0.90** | 21 / 40 / 5 / 11 |
| Young M1 R100, 600 px | 0.73 | 0.45 | **0.80** | 91 / 116 / 72 / 84 |
| Young M0.7 R50, 600 px | 0.75 | 0.69 | **0.93** | 113 / 140 / 57 / 82 |
| Young M0.5 R200, 600 px | minus 0.06 | minus 0.14 | minus 0.12 | 10 / 39 / 32 / 11 |
| 9074, 600 px (deconvolution) | 0.27 | 0.21 | 0.06 | 160 / 202 / 37 / 15 |
| 9074_ai, 600 px | minus 0.24 | 0.61 | 0.32 | 160 / 239 / 35 / 11 |
| Speedie Keplerian twin, SNR 12 | 0.13 | 0.30 | minus 0.78 | 11 / 791 / 65 / 32 |

(1) **On GI discs the model recovers the minor-axis PV wiggle where Wiener flattens it** (Speedie 0.87 vs 0.81 native, 0.77 vs 0.44 at SNR 24, 0.44 vs 0.12 at SNR 12; an07 0.90 vs 0.03): the figure `figs_pv/speedie_GI_r0.5_1024_pv.png` shows the model following the clean oscillations near 470 and 590 au that the Wiener curve smooths away. It underestimates the magnitude (54 against 71 m/s native, about 75 percent), as a smoothing denoiser would; Wiener gets 20 (28 percent). The dirty cube's magnitude (183, 213, 601, 1283 m/s) is noise, which is the useful contrast: the magnitude is only meaningful for a denoised curve.
(2) **9074 at 600 px: the PV score cannot be read as a model failure.** The clean 9074 minor-axis curve is dominated by a large step-like trend of about 600 m/s (`figs_pv/9074_600_pv.png`), not an oscillation, so the quadratic detrend leaves the step and the correlation is set by that; even the beam-only oracle (clean convolved with the beam, the best any denoiser could do) scores minus 0.35 / 0.19 there. The model's 0.06 (dirty 0.27, Wiener 0.21) and magnitude 15 against 160 m/s only say the model smooths the step. At 300 px the model is above both (0.39 vs 0.27, 0.31). For 9074 the spine (0.67) and per-pixel (0.30) scores stay the readable ones; the PV curve needs a different detrend there (decision for Oct 19 after seeing the other 9074 sizes), and the 9074 PV figure should be shown with that caveat or not at all.
(3) **Invented wiggle on the twin, measured the PV way, is larger than the earlier rotation-curve measure said.** At SNR 12 the model's twin PV magnitude is 32 m/s against 11 for clean (the twin rc rms excess was +6 m/s): about 20 m/s of false minor-axis wiggle, 30 percent of the true GI wiggle (71 m/s) at that SNR; Wiener has 65 (and dirty 791) so the model is still the quietest, but this is the number Jason's metric gives and the one to quote. The twin correlation (minus 0.78) is meaningless (the clean twin curve is a tiny sub-beam pattern, 11 m/s). Not yet separated here: how much of the 32 m/s is network invention and how much is residual noise.
(4) Young M0.5 R200 (a disc with a weak wiggle, magnitude 10 m/s) is no better than noise (minus 0.12); the other four Young discs and an07 are strongly positive.
Numbers touched: none published; this adds recipe B's five-size table and the first PV table. The deck's "all five sizes" slide can now show recipe B at every size (update Oct 19, RESULTS.md at the freeze).

## 2026-10-17 | run | R9 seed 62 complete: same verdict as seed 61, and a seed-spread warning; recipe B evaluation finished; R6 s33 resume failed once and was relaunched

**R9 s62** (`deepsceneloc/unified-r9-s62` v2, 604 min, train and eval complete, no resume needed; archive `results/unified/r9-k15-s62_v2_2026-10-17/`, checkpoint `models/unified/unified_r9_k15_s62.pth`). Levers on, like s61 and recipes A and B. 600 px: line PSNR 50.81 (s61 51.37, B 51.74), M0 / M1 / M2 59.1 / 90.0 / 91.3, rc_r 0.76; 9074 / 9074_ai rc_r 0.32 / 0.47, spine 0.44 / 0.68 (B 0.67 / 0.76); hydro M0 minus 13 (B 43); analytic M2 76.9 (B 83.5). Speedie ladder GI rc_r SNR 6 / 12 / 24: 0.10 / **0.32** / 0.82 against s61 0.14 / **0.61** / 0.75 and B 0.14 / 0.60 / 0.83; twin invented wiggle 238 / 19 / 7 m/s. **Not adopted** (rule: a clear gain on 9074 or the ladder; none). The two R9 seeds differ by 0.29 in SNR 12 wiggle with an identical recipe: the seed-to-seed spread of the SNR 12 wiggle is as large as 0.3 (R4n seeds 0.41 to 0.60, R6 0.57 and 0.61, R9 0.32 and 0.61), a caution for reading any single-seed difference in that metric, including the R8 weights in the entry below.
**R6 s33**: the resume (v2, 16:47 IST) crashed at startup ("IndexError" on `MANIFEST.json`: the code dataset was not mounted in the kernel's inputs; no training happened; a one-off mounting failure on Kaggle's side, the same push as the working R6 s32 resume). Relaunched with `orchestrator.py resume` at 21:47 IST as v3 (about 3 h).
Numbers touched: none.

## 2026-10-17 | run | R8 loss weights 0.003 and 0.03 complete: neither beats the 0.01 in recipe B; R6 seed 33 stopped at its deadline and is resuming

**R8 from R6, M1 loss weight** (, , v1, 20 rounds each, train and eval complete, archives , checkpoints in ). Lever-free evaluation like R6 seed 31 and the existing R8-from-R6 (weight 0.01). Speedie ladder GI r0.5 rc_r at SNR 6 / 12 / 24: w 0.003 0.08 / 0.30 / 0.81, w 0.03 0.10 / 0.42 / 0.83, **w 0.01 (recipe B's R8) 0.10 / 0.64 / 0.83**, R6 s31 0.09 / 0.61 / 0.81. Twin invented wiggle at SNR 6 / 12: 294 / 11 (w 0.003), 170 / 13 (w 0.03), 60 / 10 (w 0.01). 9074 600 px rc_r 0.39 / 0.39 vs 0.38 (w 0.01) and 0.43 (R6 s31); spine 0.57 / 0.57 vs 0.59. Line holdouts at 600 px: run_0002 M0 22 / 28 vs 19; run_0026 M0 47 / 57 vs 52. **Verdict:** the weight in recipe B (0.01) is the best of the three; the SNR 12 wiggle falls sharply at both other weights (0.30 and 0.42 against 0.64), which supports 0.01 as a real optimum and not a free choice. No change to recipe B. Single seed each, so the SNR 12 gap (0.2 to 0.3) is larger than the seed spread seen for R4n (0.41 to 0.60 over three seeds) only for w 0.003; w 0.03 at 0.42 is inside that spread's low end, so "worse" is clear for 0.003 and probable for 0.03.
**R6 s33** (): v1 stopped at round 67 of 80 at its deadline (RESUME_NEEDED); resume v2 pushed 16:48 IST on main with its own output attached (about 3 h). R9 s62 (backup, v2) is still running; it will need its own resume at about 18:30 to 19:00.
Numbers touched: none.

## 2026-10-17 | run | DSHARP spectral test result: the model must run on 0.1 km/s channels (`--spec-interp 0.1`); no new model needed

`chiragbhayal17/alma-dsharp-test` v2 (archive `results/alma/dsharp-hd142666_spectral-test_v2_2026-10-17/`): DSHARP HD 142666 12CO (0.35 km/s channels), recipe B, S5 injection of the Speedie disc binned to 0.35 km/s, noise synthesised from the cube's own spectrum (lag-1 channel correlation 0.68; only 12 of 60 planes are line-free).

| S5, Speedie GI r0.5 | SNR 12 | SNR 24 | SNR 36 |
|---|---|---|---|
| line-centre error, dirty | 21 | 10 | 7 m/s |
| line-centre error, model on native 0.35 km/s channels | 24 | 21 | 20 |
| line-centre error, model on channels interpolated to 0.1 km/s | **12** | **8** | **6** |
| wiggle rc_r: dirty / Wiener | 0.11 / 0.30 | 0.20 / 0.44 | 0.27 / 0.57 |
| wiggle rc_r: model native / interpolated | 0.52 / **0.82** | 0.67 / **0.92** | 0.71 / **0.96** |
| Keplerian twin line-centre error, native / interpolated (dirty 19 / 9 / 6) | 27 / 10 | 24 / 7 | 22 / 5 |
| Keplerian twin rc_r, native / interpolated | 0.14 / 0.74 | 0.33 / 0.89 | 0.40 / 0.94 |

Real disc (HD 142666, blind peak SNR 6.6, rms 1.3 mJy/beam): M0 flux ratio 1.078 native, 1.040 interpolated; median line-centre shift on the bright pixels 728 m/s native, 287 m/s interpolated (exoALMA MWC 758: about 55); fraction above 4 sigma in line-free planes 2.2e-4 native, 1.1e-4 interpolated, raw 2.1e-4 (nothing invented); continuum check: none found (far-plane peaks 8.3 / 7.4 sigma, each 1.5 / 0.7 sigma on the other side).
**Verdict by the pre-set rule** (native within a factor 1.5 of exoALMA's 20 / 14 / 12 m/s, else interpolated, else a fine-tune decision): native fails (worse than the dirty cube at SNR 24 and 36); interpolated passes (12 / 8 / 6 m/s, better than dirty at every SNR, and the twin stays clean). DSHARP therefore runs with `--spec-interp 0.1`; no retraining and no new model idea. The risk raised in the plan (a line 1 channel wide instead of 4, a window 3.5 times too wide) is real and the interpolation removes it.
**Caveats, not hidden.** (1) The injected disc's dirty errors are small here (21 / 10 / 7 m/s) because 0.35 km/s channels make the line centre coarse but stable; the model's gain over dirty is clear at SNR 12 (12 vs 21) and small at 24 to 36 (8 vs 10, 6 vs 7), so at high SNR on DSHARP the line-centre gain is modest and the wiggle gain (rc_r 0.92 vs 0.20 at SNR 24) is the large effect. (2) The real HD 142666 cube is faint (blind peak SNR 6.6, below the SNR 12 of the smallest S5 case) and its median line-centre shift of 287 m/s is large: in that regime the model changes M1 a lot where the raw M1 is noise, and the S5 numbers at SNR 12 and up do not cover it. Brighter discs (HD 163296, IM Lup, AS 209) are the main DSHARP results; HD 142666 stays the scientific case with this caveat. (3) S5 used synthetic noise (variance, spatial spectrum and lag-1 correlation of the real planes), not real noise planes: a second-order difference from the exoALMA S5.
Numbers touched: none published. Next: DSHARP kernel for the chosen discs with `--spec-interp 0.1`, S5 at SNR 6 / 8 / 12 / 24, blind SNR reported per disc (Oct 19).

## 2026-10-17 | added | DSHARP spectral test started early (`--spec-interp`, `kernel_alma_dsharp.py`, `chiragbhayal17/alma-dsharp-test` v1)

Because the 0.35 km/s vs 0.1 km/s risk decides whether DSHARP works without a new model, the test was pulled forward from Oct 19. `alma_unified.py`: `--spec-interp 0.1` (cube and Wiener linearly interpolated along velocity onto a grid r = round(dv / 0.1) = 4 times finer, the model run there, the output averaged back onto the original channels; helpers `spec_up` / `spec_down`, test passes: a velocity ramp round-trips exactly, original channels sit at c x r); `inject()` now trims the truth's empty edge channels after binning (so the injected cube fits within the DSHARP cube's line-free planes; its line-free count is unknown until the run: a skip is printed if too few). The kernel downloads `HD142666_CO.fits` (540 MB, NRAO) once and runs two modes, native channels and interpolated, each with S5 injection (Speedie GI r0.5 and twin, SNR 12 / 24 / 36, binned to 0.35 km/s into the cube's own noise). Inclination 58 deg and distance 148 pc are approximate and only enter S6 / PV of the real disc, not S5; the exact values come from Huang et al. 2018 (Oct 19). Code overlay re-synced. Result and decision rule: if the native S5 line-centre error is within a factor 1.5 of the exoALMA (0.1 km/s) S5 numbers at similar SNR (model 20 / 14 / 12 m/s on GI), DSHARP runs as is; if only the interpolated mode is, DSHARP uses that mode; if neither, DSHARP needs Krishan's decision on a fine-tune.
Numbers touched: none.

## 2026-10-17 | decision | Krishan: DSHARP is used as well; plan, data and the one open risk

DSHARP goes back into scope (dropped on Oct 11 by Krishan; Jason in September: "absolutely, as well", after exoALMA). `results/PLAN.md` 0f row 16 and the Oct 19 to 21 rows, `context.md` section 0 and `research.md` 2026-10-17 (data, header, what it needs) are updated. Data: 12CO J=2-1 fiducial images, 20 discs, NRAO bulk server, 0.35 km/s channels, 10 mas pixels, beam about 7.7 x 6.1 px. The one open risk: the models were trained on 0.1 km/s channels and the K=15 window becomes +-5.3 km/s at 0.35 km/s (3.5 times too wide); the S5 injection binned to 0.35 km/s decides it before any DSHARP result is believed. Numbers touched: none.

## 2026-10-17 | bug | a second continuum subtraction inside the S5 injection test (small; removed)

Krishan asked whether any other place double-subtracts after Jason's note. Audit of every mean / continuum subtraction on the inference path: (1) `alma_unified.main` on the real cube: replaced this morning by `--cont-sub auto` (see the entry above). (2) **`inject()` (S5): `dirty - dirty[lf_i].mean(0)` fed to the Wiener and to the model, while the scored "dirty" row stayed unsubtracted.** The noise planes injected into come from `cube`, which has already gone through the one continuum decision, so this was a second subtraction, and it treated the model's input differently from the dirty row it is compared with. Removed; the injected cube is used as is. Effect on past numbers: the controls with and without the main subtraction gave identical S5 rows (flux ratio 1.022 / 1.020, line-centre errors 22 / 16 / 13 and 8 / 6 / 5 m/s), because the leftover is about 0.002 against a noise of 0.005, so the S5 numbers in the deck (Oct 13 to 16 runs) move by well under their precision; they are recomputed in the Oct 20 to 21 rerun anyway. (3) Training and evaluation of the line-emission cubes (`continuum_sub` in `train_unified.py`): correct, those cubes contain a real continuum floor and the model was trained on subtracted data; exoALMA, already subtracted, is the same kind of input, so nothing is double subtracted there. (4) `native_wiener`, `moments`, the scorecard: no subtraction. Code overlay re-synced; `tests/test_alma_fov.py` passes.
Numbers touched: none published beyond the S5 rows above (negligible).

## 2026-10-17 | run | final evaluation of recipe B launched a day early (all five sizes, PV curves); Speedie r1.5 kept untouched

`chiragbhayal17/eval-final-b` v1 (`kaggle/eval_final_b/kernel_final_b.py`, derived from `kernel_combo.py` = combo-b): recipe B (R4n-long seeds 3, 11, 23 + R8-from-R6), both levers, gate on, five line holdouts, five held-out Young simulations, 9074 (+ `_ai`), an07, an08 and SG v2 at 256 / 300 / 320 / 480 / 600 px, Speedie r0.5 GI and twin native, ladder 3 / 6 / 12 / 24; PV curves and `pv_r`, `pv_mag_ms` for every case through the scorecard. It does not depend on the four jobs still running, because none of the new seeds met its adoption rule (R9 s61, R6 s32) so recipe B is unchanged; a seed that does later is a new row, not a change to this run. The ONE-TIME final test (Speedie r1.5 ladder at SNR 6 / 12 / 24) is deliberately not in it and stays for the freeze. `--eval-speedie` here is `GI_r0.5,Kep_r0.5` only (combo-b also scored r1.5 native; this run does not).
Pushed 2026-10-17 ~10:30 IST on chirag2. Share done today: `anshi2904/unified-r6-s32` to chiragbhayal17 (visible, checkpoint listed).

## 2026-10-17 | run | R6 seed 32 complete after its resume: the 9074 gain does not repeat at 600 px; Oct 17 launches (one network failure, relaunched)

**R6 s32**: v1 stopped at round 70 (deadline), v2 resumed from its own output at round 70 and finished with eval (231 min); archive `results/unified/r6-k15-s32_v1-v2_2026-10-17/`, checkpoint `models/unified/unified_r6_k15_s32.pth`. First real use of `orchestrator.py resume` on a deadline-stopped training job: it worked (log: "RESUMED ... round 70/80, step 21000"). Lever-free, like R6 s31 and R4n s3. 9074 per-pixel rc_r at 600 px 0.38 (s31 0.43, R4n 0.34), spine 0.53 (0.54, 0.60); at 300 px 0.42 (0.45, 0.30). Speedie SNR 12 0.57 (R4n 0.60), twin SNR 6 31 m/s (s31 293). Pre-set rule (join B only on repeating 0.43): not met. Two seeds put R6's 600 px per-pixel gain on 9074 at about +0.06 over R4n, not +0.09; the 300 px gain holds in both. This weakens the case for a separate kinematics checkpoint (question 4 to Jason); seed 33 is the third look.
**Oct 17 launches**: the scheduled launcher started at 05:55 IST (not 05:32: the Mac was asleep despite `caffeinate`, or the wait loop woke late) and ended 07:17. Pushed: `krishanyadav333/unified-r6-s33`, `chiragbhayal/unified-r8-w003`, `chiragbhayal/unified-r8-w030`. `deepsceneloc/unified-r9-s62` failed with "Connection reset by peer"; relaunched at 08:47 IST, and Kaggle answered "version 2", so the first push may have created a version 1. Both versions were running (Krishan, 10:16 IST). Krishan stopped v1 on the website at 10:17 and kept v2: the resume pushes attach the kernel's own latest output, so the latest version must be the running one (cancelling v2 would leave a cancelled empty latest version and risk the resume). Cost: about 3 h of one seed's compute.
Numbers touched: none published.

## 2026-10-17 | run | MWC 758: field-of-view and continuum controls; the tight crop costs flux and adds false wiggle, the 512 px field stays

Five MWC 758 13CO runs, recipe B, inference only, same S5 injection (Speedie GI r0.5 and its Keplerian twin, SNR 12 / 24 / 36); archives `results/alma/mwc758-13co_{base,nofov,fov-cs,fov,fov256}_v1_2026-10-17/`.

| field | continuum | M0 flux ratio | mask px | S6 rc_r | GI line-centre error, model (dirty) m/s | GI rc_r, model | twin error, model m/s | twin invented wiggle m/s |
|---|---|---|---|---|---|---|---|---|
| 512 px (12.8") | subtracted (old) | 1.022 | 3055 | 0.80 | 22 / 16 / 13 (67 / 37 / 26) | 0.82 / 0.92 / 0.94 | 8 / 6 / 5 | 0 / -1 / -1 |
| 512 px | not subtracted | 1.020 | 3044 | 0.77 | 22 / 16 / 13 (67 / 37 / 26) | 0.82 / 0.92 / 0.94 | 8 / 6 / 5 | 0 / -1 / -1 |
| 256 px (6.4") | auto: none found | 0.980 | 3043 | 0.81 | 23 / 16 / 13 (57 / 37 / 27) | 0.76 / 0.88 / 0.92 | 12 / 9 / 7 | 3 / 1 / 2 |
| 128 px (3.2", `--auto-fov`) | subtracted | 0.962 | 3054 | 0.82 | 20 / 14 / 12 (58 / 37 / 27) | 0.71 / 0.83 / 0.88 | 15 / 12 / 10 | 11 / 6 / 5 |
| 128 px | not subtracted | 0.960 | 3043 | 0.81 | 20 / 14 / 12 (58 / 37 / 27) | 0.71 / 0.83 / 0.88 | 15 / 12 / 10 | 11 / 6 / 5 |

(1) **Continuum: no effect.** Subtracting the line-free mean or not changes nothing beyond the third decimal of the flux ratio; `continuum_check` on the real cube finds no continuum (far-plane peaks 5.7 / 5.3 sigma, but each side's peak is -1.2 / 0.4 sigma on the other side: not the same structure), so the 6.7 sigma leftover is line wings or noise and the cube is kept as delivered. Jason's instruction costs nothing here.
(2) **Field of view: smaller is worse, monotonically.** The real disc loses flux as the field shrinks (1.02, 0.98, 0.96; the signal mask is the same, so the disc is inside every field), and the no-wiggle twin gains false wiggle (0, 2, 11 m/s at SNR 12). The S5 dirty errors differ between fields because a smaller field also cuts the injected disc (Speedie's disc at MWC 758's distance is about 20 arcsec across), so the S5 rows compare slightly different test regions; the flux ratio, on the real disc and the same mask, does not have that problem. Likely causes (not tested separately): the network was trained on discs with sky around them, and the blind Wiener measures the noise spectrum from the cube, which a tight field leaves with little empty sky.
(3) **Decision:** keep the 512 px field (12.8 arcsec, already a crop of the full exoALMA field, disc about a quarter of it) for the rerun; both ALMA kernels go back to `--crop 512` without `--auto-fov`. The crop options stay in `alma_unified.py`. To tell Jason: the crop he suggested was tested, the 512 px crop is kept, the tighter crops cost flux and add false wiggle on MWC 758 (this table). A disc larger than 12.8 arcsec (LkCa 15 may be) needs a wider field, not a tighter one: check the M0 at the field edge in the rerun.
Numbers touched: the deck's MWC 758 flux ratio 1.02 and S5 errors stay valid (they are the 512 px rows). The earlier "crop to the disc" plan line (PLAN.md 0f row 3) is answered: tested, not adopted.

## 2026-10-17 | added | continuum: subtract only when the cube really has continuum left (`continuum_check`, default `--cont-sub auto`)

Krishan asked what to do with a disc whose line-free mean shows a leftover. A leftover is either real continuum (a cube that was never subtracted) or faint line wings / noise in planes called line-free. Continuum is flat in frequency, so it sits at the same place in the line-free planes far from the line on BOTH sides; line wings sit next to the line and, by the disc's rotation, at different places on the blue and red side. `alma_unified.continuum_check`: mean of the farther half of the blue and red line-free planes, each in units of its error; continuum only if both peaks exceed 5 sigma AND each side's peak reappears on the other side, same sign, above 3 sigma. Then it is subtracted once (not double subtraction: nothing was subtracted before); otherwise the cube is kept as delivered. `--cont-sub auto` (default) uses it; `off` never subtracts; `on` is the old behaviour. The report records the decision, the reason and the numbers.
A first version used the whole-map correlation of the two sides as the "alike" test; on the test cube a clear 8 sigma compact continuum gave correlation 0.33 (the empty sky dominates), so it would have missed a real continuum. Replaced by the peak test above before anything ran on it. `tests/test_alma_fov.py` covers: already subtracted (kept), never subtracted (subtracted, map amplitude 2.0 recovered), line wings next to the line (kept), strong but different structure on the two sides (kept), pure noise (never triggers, 0 of 10). Code overlay re-synced.
Numbers touched: none.

## 2026-10-17 | added | ALMA: crop to the disc, no second continuum subtraction; PV magnitude (code and tests; one validation kernel running)

`experiments/alma_unified.py`: (1) `--auto-fov` (and `--fov-arcsec`) crops the cube to a square field around the disc before the model (`fov_box`: connected region of the beam-smoothed line M0 above 5 robust sigma, its extent x 1.25 plus 2 beams, multiple of 8, at least 128 px, inside the image); the line-free planes are found again on the crop; the output cubes carry the matching WCS (`EXXAOX / EXXAOY` offsets move CRPIX); the report records the box. (2) Continuum subtraction is now OFF by default (`--cont-sub` turns it on): exoALMA is already subtracted and subtracting the line-free mean again double-subtracts (Jason, Oct 16). The residual is still measured (`residual_continuum_peak_sigma`, `continuum_subtracted_by_us` in the report) with a warning above 5 sigma. Both kernels (`kernel_alma.py`, `kernel_alma_all13co.py`) now pass `--auto-fov`; the 14-disc kernel injects S5 into HD 135344B, DM Tau and LkCa 15 (inclination 16 / 39 / 50 deg). Test `tests/test_alma_fov.py` passes (off-centre disc, tiny disc, corner disc, noise only, WCS), `tests/test_line_free.py` still passes.
`experiments/pv_curve.py` after reading Speedie 2024 and Longarini 2024 (`research.md` 2026-10-17): quadratic detrend, `pv_mag_ms` = their wiggle magnitude (37.4 m/s for AB Aur 13CO), carried into the scorecard rows as `pv_mag_ms` / `pv5_mag_ms`; the PV block in `score_cube` is fail-soft (an error leaves every other score intact and sets the PV keys to NaN; verified by forcing an exception).
Code overlay `exxa-wiggle-code` re-uploaded (main). Validation: `chiragbhayal17/alma-mwc758-fov` v1 (MWC 758, recipe B, cropped, no continuum subtraction, S5 injection), pushed 2026-10-17; its result decides whether the Oct 20 to 21 rerun uses these settings unchanged. Not yet measured on any real cube.
Numbers touched: none (the 14-disc numbers stay void until the rerun).

## 2026-10-17 | run | R9 seed 61 complete (not adopted); R6 seed 32 hit its deadline and is resuming

**R9 s61** (`anshi2904/unified-r9-s61`, v1, 644 min, train and eval complete; archive `results/unified/r9-k15-s61_v1_2026-10-17/`, checkpoint `models/unified/unified_r9_k15_s61.pth`, zip-verified). R9 = R6 recipe with the beam resampled to 8 to 12 px in training (`--train-beam-px 8,12`), evaluated with both levers on. The first comparison I made put it next to the lever-free R4n and R6 evaluations, which was not like for like and made it look better than it is; caught by reading the RUNS row and the log ("infer: beam ... network on a 320 px grid"), corrected before anything was recorded elsewhere. Like for like against recipe A (3-seed average) and B (4-model average), all with the levers, 600 px: line PSNR 51.37 vs 51.40 / 51.74, M0 / M1 / M2 56.6 / 89.9 / 90.3 vs 60.9 / 88.2 / 90.8, rc_r 0.74 vs 0.77; hydro 9074 spine 0.41 vs 0.72 and M0 / M1 / M2 23 / 34 / 51 vs 42 / 46 / 65; analytic M2 69 vs 83; Speedie SNR 12 rc_r 0.61 vs 0.60 (no gain); twin invented wiggle SNR 6 / 12 / 24 42 / 13 / 5 m/s. Verdict by the pre-set rule (PLAN.md 0f row 6): not adopted. One R9 model with the levers equals a 3-model average on line PSNR but loses on the hydro and analytic cubes. Seed 62 runs today.
Numbers touched: none published (the recipe B numbers do not use R9).

**R6 s32** (`anshi2904/unified-r6-s32`): v1 stopped at its 12 h deadline after round 70 of 80 (560 min, `RESUME_NEEDED`, train_complete false, no eval), as designed (RULES.md 14). Partial checkpoint downloaded (349 MB, zip-verified, kept in the scratchpad until the resume supersedes it). `orchestrator.py resume` pushed v2 with its own output attached at 02:30 IST; it needs about 3 h (10 rounds plus eval). Expect the same deadline-stop for each Oct 17 job: plan a resume session for each, about 3 h, on its own account.

## 2026-10-17 | decision | master plan: old schedule and Jason's meeting list merged; all six Oct 17 jobs stand

`results/PLAN.md` section 0f is now the one plan to follow: a 15-row requirement checklist (project goal, Jason's meeting list, GSoC deliverables, records and housekeeping), the day-by-day table to Nov 3, accounts, gates, risks and the definition of done. 0d and 0e stay as the record of how it was reached. My earlier suggestion to trim the Oct 17 launches to three jobs is not adopted: Krishan's earlier choice of all six stands, and chirag2, jensi and aditi stay free for evaluation and the ALMA rerun. Numbers touched: none.

## 2026-10-16 | added | position-velocity (PV) curve along the minor axis (code and test only; no run yet)

`experiments/pv_curve.py`: M1 (quadratic) minus vsys sampled along both halves of the minor axis with the geometry `wiggle_scorecard` already fits, slit 1 px and 5 px; scores `pv_r` (correlation with clean after a cubic detrend, both halves pooled) and `pv_rms_ms`; two-half figure. `wiggle_scorecard.score_cube` now also writes `pv_r`, `pv_rms_ms`, `pv5_r`, `pv5_rms_ms` per method and `<cube>_pv.png`. Test `tests/test_pv_curve.py` passes: a 60 m/s synthetic wiggle is recovered to 0.9 m/s; noisy copy pv_r 0.79 at 1 px and 0.94 at 5 px; pure noise near 0; an empty radius range gives NaN, not an error. A synthetic cube through `score_cube` also runs (dirty row pv_r 0.69, pv5_r 0.87). `--selftest` of the scorecard passes. Added to the Kaggle code overlay (`sync_account.py` CODE_FILES; re-sync before the Oct 18 evaluation). `compare_models.py` METRICS not changed: older eval JSONs lack the pv keys. Numbers touched: none (new metric; no published number uses it). Not yet measured on any real cube.

## 2026-10-16 | meeting | Jason: models are good enough; next is position-velocity curves and a cropped exoALMA field of view

Mentor meeting, 22-slide deck (Canva). Source: Krishan's transcript of the call. What Jason said, in his words reduced to the decisions:
- **Position-velocity (PV) curves are the standard now, not the spine.** Speedie 2024 and Longarini et al. (Elias 2-27) fit the GI wiggle with PV curves, so "we need to be able to replicate that if we want to say we are contributing to the
  standard." The curve is the moment-1 value (quadratic, i.e. the velocity of the line peak) along the minor axis: a slice straight through the cube, or 4 to 5 pixels wide. His own spine method "has been used" but is not what people use
  now. Highest priority for next week; no wiggle-fitting routine and no disc-fitting routine, only a good-looking PV curve for clean / dirty / Wiener / model. Krishan committed to it for next week.
- **exoALMA: crop the field of view.** It is huge and the beam tiny, so the disc is a small part of the cube; the network sees features relative to cube size. Crop, run, expand later. Second priority.
- **exoALMA is mostly already continuum subtracted, so do not subtract it again** (DSHARP is not by default). `experiments/alma_unified.py` line 223 still subtracts the line-free mean (about 0 on a subtracted cube; the run reports the residual in sigma). To change: off by default for exoALMA, report the residual only. Not yet done.
- **Scaling to the beam (the size fix):** he asked that the beam keep the same angular size when the image is rescaled. `infer()` resamples the pixel grid of the cube and of its Wiener estimate together and resamples the output back, so the sky field and the angular beam are unchanged and only the pixel size changes; the beam in pixels shrinks. Still worth stating to him in writing.
- **Models are good; do not expect a qualitative jump.** Keep the seed runs going in the background, spend the week on post-inference analysis. "These are really good models now"; the moment-1 residual structure is "clearly self-gravitating". Even the existing slide (clean / dirty / model) "is a really good slide".
- **SNR:** real ALMA discs are above about 15 to 20 peak SNR; SNR 6 is "bad, 20 years ago bad", not relevant to modern observations; at high SNR the observations are already good so the gain is smaller. Rescaling exoALMA to other SNR is optional.
- Features he noticed: run_0002 (probably run_0002_00560) looks like it has a planet; so does the Young M1 R100 simulation. On the 9074 panel the model rows look like "beam noise" to him.
- Submission: "you have plenty to put a submission in for." The two items asked for next week are the PV curves and the cropped exoALMA look.

Not discussed, so still open with him: second held-out GI cube, one checkpoint or two, publishing the checkpoints trained on private data. Question 1 (which wiggle score to headline) is overtaken: the PV curve is the field standard, so it becomes a
third wiggle view next to rc_r and spine_r. Question 2 (a curve against SNR as the bar) is in effect answered: SNR below about 6 is not a case to design for.
Numbers touched: none changed. The deck's slide 13 line "do not use below SNR 6" and slide 10 stay true but are of less weight. Plan changes: `results/PLAN.md` section 0e.

## 2026-10-16 | decision | Krishan: store A, use B

Recipe B (R4n seeds 3 + 11 + 23 + R8-from-R6, both levers, gate) is the deliverable; recipe A (the three R4n seeds, same flags) is stored as the 3-model alternative. Both written down exactly in
`results/final/RECIPES.md`. The ALMA kernels (`kernel_alma.py`, `kernel_alma_all13co.py`) now average B's four checkpoints; their metadata lists the two extra kernel sources.

## 2026-10-16 | decision | Krishan: push; shares being done; Oct 17 batch = all six jobs; ship the best recipe and keep the alternative documented

Push approved (midterm-prep only, no PRs, no co-author trailer). Oct 17: all six prepared jobs (R6 seeds 32 / 33, R9 seeds 61 / 62, R8 weight 0.003 / 0.03), not the reduced set I suggested; a seventh
account is coming, which covers the GPU the final evaluation and the ALMA rerun need. Checkpoints: the main deliverable is the frozen combo-b recipe; the best-wiggle alternative (R6 family, and R8-from-R6) is
kept, archived and reported next to it with its trade-off (more real-GI wiggle, about 2.5 dB less PSNR), not dropped. Figures of eval-final (155 MB) and R7 s42 (24 MB) are not committed (size); they stay
in `results/` locally and in the kernel Outputs.

## 2026-10-16 | run | MWC 758 re-measured with the fixed line-free finder (CPU, partial): Gate A S5 holds; the 'noise reduction' number is not a useful check

`aditisah11/alma-mwc758-recheck` v1, R4n-long seeds 3 + 11, crop 512, `results/alma/mwc758-13co_recheck_v1_2026-10-16_partial/`. Fixed finder: 239 of 301 planes line-free (258-259 before; MWC 758's line is narrow, so the bug moved
less here than on the broad 14-disc lines). Config `fixed_only` complete (inference 100 min on CPU): S5 injection-recovery with the cube's own noise, line-centre error vs truth, dirty -> model:
GI r0.5 SNR 12 / 24 / 36: 67 -> 25, 37 -> 18, 26 -> 14 m/s (rc_r dirty 0.03 / 0.10 / 0.29, model 0.75 / 0.89 / 0.93, Wiener 0.57 / 0.78 / 0.86); Keplerian twin SNR 12 / 24 / 36: 54 -> 8, 29 -> 7, 20 -> 6 m/s.
S1: M0 flux ratio 1.02, M1 shift on the brightest 10 % of the mask median 79 m/s (p95 228), S2 > 4 sigma pixels in line-free planes 5.7e-5 raw -> 0 denoised. S5 stands, no invented wiggle on the twin.
Config `levers` (size fix inert at 6 px beams; two-pass Wiener): S5 GI r0.5 SNR 12 / 24 / 36 rc_r 0.78 / 0.90 / 0.94, line-centre error 23 / 17 / 14 m/s (a small gain over fixed_only); the CPU session was
cancelled by Kaggle at 164 min into it, so its S1 / S2 / S5 Kep / S6 are missing; the 14-disc rerun (GPU, ~5 min per cube) covers the levers on this cube too.
Not usable as evidence: S1 "noise reduction" now reads 108x. The denoised line-free planes are essentially zero (the network and Wiener flatten empty planes), so raw rms over denoised rms mostly measures
that, not noise removed from the signal; S5 and the M0 ratio are the checks that mean something. The earlier "25x" and "5-32x" figures should not be quoted either.

## 2026-10-16 | decision | Frozen inference recipe: combo-b (both levers + R8-from-R6 in the ensemble), size fix restricted to the noise-only regime (Krishan: keep both)

Decided on the regime argument, before the untouched final test (Speedie r1.5 ladder at SNR 6 / 12 / 24) is run. Size fix (`--infer-beam-px 10 --infer-mode full`) applies only where the beam is already in the
cube and only noise is removed (kind "line" / "*_ai": line holdouts, Young sims, Speedie, 9074_ai, every ALMA cube); beam-deconvolution cubes (9074, an07, an08, SG v2) keep the cube as given
(`train_unified.infer(..., ai=...)`, `tests/test_infer.py`). Reason: combo-a / b lost 0.05 wiggle on the analytic discs and 6 / 8 points of M1 / M2 on SG v2 @600 only there. Those rows now equal the gated
frozen model by construction, so combo-b's selection-set numbers for them are the frozen ones. Two-pass Wiener was already noise-only. Ensemble for the freeze: R4n seeds 3 + 11 + 23 + R8-from-R6, plus whatever the
Oct 17 R6-family seeds add; the r1.5 test is run once, after that ensemble is fixed.

## 2026-10-15 | run | Combos back: both inference levers + R8-from-R6 in the ensemble (combo-b) is the best recipe so far; it gains on line emission, Young, Speedie and costs a little on analytic and SG v2

Same cases as eval-final; "frozen" = eval-final's plain model rows (R4n s3 + s11 + s23; gate applied in the report). combo-a = levers on the same 3 seeds, combo-b = levers + R8-from-R6 (4 nets). Gate on in both.
- Line holdouts: @256 PSNR 48.2 -> 51.2 dB, M0/M1/M2 52/87/85 -> 62/89/90, rc_r 0.88 -> 0.87; @600 49.3 -> 51.7 dB, 47/86/88 -> 62/89/91, rc_r 0.68 -> 0.76, spine_r 0.78 -> 0.75.
- Young held-out: M0 @256 26 -> 42 %, @600 12 -> 47 % (Wiener ~21), M1/M2 unchanged, wiggle unchanged (rc 0.55-0.62, spine 0.68-0.69).
- Speedie ladder GI: SNR 12 rc_r 0.55 -> 0.60 (spine 0.32 -> 0.30), SNR 6 0.13 -> 0.14 (spine 0.13 -> 0.18), SNR 24 unchanged 0.83; twin invented wiggle at SNR 6 +52 -> +25 m/s, SNR 3 worse (+1170 -> +1473; the model is not usable at SNR 3 either way).
- Speedie r1.5 (SNR ~135, gate active): same as the gated frozen model (GI M0/M2 -0/+49).
- 9074: rc_r @600 0.33 -> 0.30, spine 0.60 -> 0.67; 9074_ai rc 0.37 -> 0.45, spine 0.72 -> 0.76.
- Costs: analytic rc_r 0.88 -> 0.83 and PSNR -0.6 dB; SG v2 M1/M2 @600 17/28 -> 11/20 and spine 0.59 -> 0.52. Both are deconvolution-regime cubes where the size fix triggers (analytic beam 13.7 px at 256 px).
- The size fix also triggered on Speedie (major-axis beam 19.5 px at 1024 px, not 12 as assumed): grid 528 px.
Decision candidate: combo-b's recipe. Open: whether to restrict the size fix to the noise-only regime (the ALMA case), decided on the regime argument before the final r1.5 test is run.

## 2026-10-15 | finding | Measured with the channel-map spine (Terry et al. 2022), the frozen model recovers most of 9074's wiggle; the per-pixel rc_r undersold it. Asked Jason which measure to report

The scorecard has always computed both: rc_r (per-pixel correlation of M1 minus the axisymmetric rotation curve) and spine_r (the Terry et al. 2022 channel-map spine near v_sys, plus the
wiggle amplitude A and wavelength lambda from its strongest Fourier component). Only rc_r was ever reported. Frozen model (eval-final, R4n seeds 3 + 11 + 23, gate on):
9074 spine_r 0.60-0.70 (256-600 px; 9074_ai 0.72-0.77) against rc_r 0.29-0.33; at 9074_ai@300 spine_r model / Wiener / dirty 0.76 / 0.38 / 0.15, wiggle amplitude 12.7 au (clean 12.3), wavelength
174 au (clean 183). 600 px group means (RESULTS.md, column added): line emission spine_r 0.78 (Wiener 0.35, dirty 0.19), hydro 0.66 (0.24 / 0.19), analytic 0.87 (0.42 / 0.17).
On the Speedie ladder the spine is weaker: 0.13 / 0.32 / 0.31 at SNR 6 / 12 / 24 (Wiener -0.13 / -0.05 / 0.11), so the two measures disagree there and neither is chosen yet.
Likely reason for the gap on 9074: per-pixel M1 carries noise below the beam scale (600 px: beam ~27 px) that rc_r counts and the spine, a beam-scale tracker, does not.
Consequence: 9074 is not the failure it looked like on rc_r. Email sent to Jason 2026-10-15 ~17:10 IST with two plots (`results/email_2026-10-15/`): which measure to report, whether "above the
best linear filter, with the curve against SNR" is an acceptable bar, a second hydro GI test disc, two checkpoints. `compare_models.METRICS` and RESULTS.md now carry spine_r next to rc_r.

## 2026-10-15 | audit | Check of today's work against RULES.md and the plan: four gaps fixed, one decision corrected

1. RULES.md #10: none of today's run folders had a README, R7 seed 43 was never archived, R7 seed 42 had no log: READMEs written for all nine folders, s43 archived, s42 log added.
2. RULES.md #12: R7 checkpoints existed only on Kaggle: s41 / s42 / s43 downloaded to `models/unified/`. The first s43 download was silently truncated (245 of 349 MB, not a valid zip):
   every local checkpoint is now checked with `zipfile.testzip` and s43 was re-fetched with a size check. All nine local checkpoints pass. R8 and R7 indexed in `models/README.md`.
3. Decision corrected: PLAN.md had "adopt the size fix" although it failed two of the adoption criteria fixed BEFORE the test (line wiggle @600 0.77 < 0.80; 9074 0.31 -> 0.25 where the bar was
   "not lower"). Now: not adopted on its own; `combo-a` decides it with the two-pass Wiener, and any narrowing of when it triggers is chosen before the final test is looked at.
4. Test contamination: the Speedie r0.5 ladder, 9074, the line holdouts and the Young sims have all been used to choose between models, so they are selection data. The final reported wiggle-vs-SNR
   curve will come from the Speedie r1.5 disc rescaled to SNR 6 / 12 / 24, used for nothing so far.
Open risks: all work since `02c8ff4` is uncommitted (local disk + the Kaggle code overlay only); `--train-beam-px` (R9) has only a unit test, not a run on real cubes; the Oct 17 batch needs the pending shares.

## 2026-10-15 | code | The two levers are in the ALMA adapter; MWC 758 re-measurement launched; Oct 17 batch built

`alma_unified.py`: `run_model` = `train_unified.infer` (network on a ~10 px-beam grid when the beam is wider than 12 px; exoALMA's ~6 px beams leave it inert) then a second pass from `prior_wiener`
(`--infer-beam-px 10`, `--no-wiener2` to switch off); the report records both. `alma_unified.py` was not in the `exxa-wiggle-code` overlay, so Kaggle ran the last PUSHED version (without the line-free fix):
added to `CODE_FILES` in `sync_account.py` and re-uploaded. `alma-mwc758-recheck` (CPU, kernel `kaggle/alma_mwc_recheck/`) re-measures MWC 758 with the fixed finder, with and without the levers.
`kaggle/accounts/make_unified_job.py --oct17` writes the batch for the quota reset: R6 seeds 32 / 33, R9 seeds 61 / 62 (`--train-beam-px 8,12`, evaluated with the levers), R8 weight bracket 0.003 / 0.03 warm from R6.
Not yet done: the mentor deck (its numbers move until the freeze), the 14-disc rerun (needs the frozen model).

## 2026-10-15 | run | Five jobs back: two inference-only levers work (size fix, two-pass Wiener), the M1 loss (R8) does not clearly, the oracle says what linear filtering can do

Archived: `results/unified/{size-test,wiener2-test,r8-m1-r4n,r8-m1-r6}_v1_2026-10-15/`, `results/oracle-speedie_v2_2026-10-15/`. Means over cubes, one training seed per row unless said.

1. **Oracle linear filter on Speedie** (knows clean's and the noise's spectra; `experiments/oracle_limit.py` logic, kernel `oracle-speedie`): GI r0.5 native 0.80 (our model 0.89, dirty 0.36);
   ladder SNR 3 / 6 / 12 / 24: 0.02 / 0.20 / 0.45 / 0.68 against the model's 0.00 / 0.13 / 0.55 / 0.83. The model is ABOVE the best linear filter at every SNR from 12 up, by 0.10-0.15, so
   its wiggle is not what a linear filter gives; 0.85 at SNR 12 is 0.40 above the oracle and 0.30 above the model: not reachable by any method we have, and the measured curve is the result to report.
2. **Size fix** (`aditisah11/size-test`, seeds 3 + 11, gate on; network on a grid where the beam spans ~10 px, `--infer-beam-px 10`): line holdouts @600 wiggle 0.67 plain -> 0.76 (corr) / **0.77 (full)** / 0.75 (b8 corr);
   run_0026 0.66 -> **0.89** (Wiener 0.85), run_0002_rt_01 0.63 -> 0.76; Young M0 @600 12 -> **29 %** (Wiener 21), @480 19 -> 29; line PSNR 49.3 -> 48.8 dB (full), 48.1 (corr), 46.8 (b8); M1/M2 +-1; 9074 0.31 -> 0.25 (the one case it hurts;
   it has a 13.8 px beam so it triggers). Not reached: line wiggle >= 0.80 (0.77; run_0002_rt_04 stays 0.57, a cube whose dirty score flips between 0.22 and 0.03 across sizes: metric-unstable).
3. **Two-pass Wiener** (`aditisah11/wiener2-test`, seeds 3 + 11): line holdouts PSNR @256 48.0 -> **50.9** dB, @600 49.3 -> **51.3**; M0/M1/M2 55/88/86 -> 63/89/91 (@256), 51/86/88 -> 61/87/90 (@600); wiggle @600 0.67 -> 0.71 (run_0026 0.66 -> 0.77);
   ladder SNR 12 0.50 -> 0.57, 6 0.12 -> 0.13, 24 unchanged; 9074_ai 0.34 -> 0.43 @600; twin spurious unchanged (+3.9 -> +4.9 at SNR 12, +46.9 -> +32.9 at SNR 6). Free, inference only; applies to the noise-only (A = I) cases.
4. **R8** (M1 loss, `--w-m1 0.01`, 20 rounds, lr 5e-5, one seed each): from R4n s3: ladder SNR 12 **0.35** (R4n seeds 0.60 / 0.41 / 0.56; worse), line @600 M0 58, wiggle 0.73, twin SNR 6 +152 m/s: not adopted. From R6: SNR 12 / 24 0.64 / 0.83,
   twin spurious SNR 6 / 12 / 24 +60 / +9.5 / +3.3 (R6: +293 / +11.9 / +4.7), 9074 0.40 / 0.38 (R6 0.47 / 0.43), line @600 45.8 dB M0 47 M1 86 M2 84 (R6 44 / 84 / 84). A small, single-seed improvement in invented wiggle and moments over R6 at a small cost in 9074 wiggle; inside seed noise for the SNR-12 number.
Next: `combo-a` / `combo-b` (levers 2 + 3 together, with and without R8-from-R6 in the ensemble) on the full decision set.

## 2026-10-15 | code | Two more wiggle levers: a second inference pass from a model-informed Wiener (`--wiener2`), and beam-matched training (`--train-beam-px`)

Krishan: best results, not one hopeful experiment. `prior_wiener`: the blind Wiener the network starts from estimates the signal spectrum as the dirty cube's minus a
noise model, which is poor at low SNR (9074: blind 0.16 vs oracle 0.38); the second pass builds the Wiener from the first pass's own output spectrum and runs the
network again (noise-only cases; the reported "wiener" row stays the blind one). Test `aditisah11/wiener2-test` (plain vs second pass in the same run).
`--train-beam-px lo,hi`: every redraw resamples the cube so its beam spans U(lo, hi) px, i.e. trains where `--infer-beam-px` runs the network; for R9 if the size
test passes. Tests: `tests/test_prior_wiener.py` (oracle prior mse 0.05 vs 1.00 noisy; self prior changes nothing; redraw lands on the beam-matched grid).

## 2026-10-15 | bug | ALMA adapter: the line-free plane finder called LINE planes line-free on the exoALMA 13CO cubes; every 14-disc number that uses those planes is void

What was wrong: `alma_unified.line_free` sigma-clipped the spectrum summed over the central half of the field. When the line's integrated profile is broad
and flat-topped, no plane is a 3-MAD outlier and the line planes pass as noise: 241-301 of 301 planes were called line-free on the 14 discs (AA Tau all 301,
PDS 66 296 of 300, DM Tau 282). How caught: the run's own diagnostic panel "a line-free plane" shows the disc's iso-velocity strip at full brightness (AA Tau),
and the "residual continuum" it reports was 7-48 sigma, i.e. the mean of the line itself. Consequences: the line's mean was subtracted from every plane as
continuum (AA Tau M0 ratio 0.83); the Wiener's noise spectrum was measured on line planes; S1 noise reduction and S2 (structure in line-free planes) were measured
on line planes; S6 used a mask built on top of that. Fix: per-plane detection (peak of the beam-smoothed plane over the plane's own MAD noise, > 6), the main run
of detected planes grown by 5 channels is the line, the rest line-free; `tests/test_line_free.py` reproduces the failure (old finder: 301 of 301 line-free,
all 121 line planes among them) and passes with the new one (line 85-215 for a true 90-210; noise-only cube all line-free).
Numbers touched: everything from `aditisah11/alma-exoalma-all13co` (noise reduction 5-32x, S2, S6, M0 ratios; reported in chat 2026-10-14, not in the blog).
Also the 2026-10-13 MWC 758 Gate A (258-259 of 301 planes line-free there, plausible for its narrower line but unverified): its S1 / S5 figures are to be
re-measured with the new finder before they are quoted. Correction in the same entry: the "3 of 14 discs failed" (chat, 2026-10-14) was wrong; all 14
exited 0; the CLI download stopped at its first page of the 140 output files.

## 2026-10-15 | code | Two fixes for the wiggle built: network on a ~10 px beam grid (inference), and a line-centre (M1) training loss; size test launched

Why the size fix: the same 5 line holdouts score wiggle 0.80-1.00 at 256 px and 0.49-0.69 at 600 px (run_0026 0.90 -> 0.66, below its Wiener's 0.85). The test sizes
are the native 600 px cube area-averaged down; native beam 15-30 px, so at 256 px it spans ~8 px, at 600 px ~20. Speedie at native 1024 px (beam 12 x 9 px) scores
0.89, so the network works at a beam scale, not an image size. `train_unified.py --infer-beam-px B --infer-mode corr|full` (`infer()`): if the beam is wider than
1.2 B px the network runs on a grid where it spans B px; `corr` upsamples only its correction onto the full-resolution Wiener, `full` the whole output.
Test: `tests/test_infer.py`. Kaggle `aditisah11/size-test` (RUNS.md).

Why the M1 loss: every round so far trained on the cube (L1 + velocity differences) and only measured the wiggle afterwards. `loss_m1`: on crops carrying all
2 x 10 + 1 channels around a line (+-1 km/s), the intensity-weighted line centre of the prediction against clean's, L1 in channels, where clean has a line peaking
inside the band. A wiggle the model invents on a Keplerian disc is an M1 error too, so this also penalises R6's spurious signal (11.9 m/s at SNR 12). Separate
forward/backward per step (memory). `--warm` starts from a checkpoint's weights with a fresh optimiser. Tests: `tests/test_loss_m1.py` (shift recovered exactly,
masked pixels ignored) and a synthetic one-step check (loss 0.58 channels on a noisy input, 0 on a perfect one, gradients flow). A full local smoke run was killed
for memory (8 GB machine): all further runs are on Kaggle only. R8 (fine-tune from R6 and from R4n seed 3) waits for account chirag2's shares; chirag and the
other accounts are out of weekly quota until Oct 17 05:30 IST.

Local checkpoint copies (RULES.md #12): R4n-long seeds 3 / 11 / 23 and R6 downloaded to `models/unified/` (git-ignored).

## 2026-10-15 | run | R6 (256 px crop training, 1 seed) done: best real-GI wiggle so far, above the oracle linear limit on 9074, but invents more wiggle on the Keplerian twin and loses ~2.5 dB at 600 px

`chiragbhayal/unified-r6-k15-crop256` v2 (resumed after its 11 h stop; train 80/80 rounds, eval 72/72). Archived `results/unified/r6-k15-crop256_v2_2026-10-15/`. One seed; compared with the
three R4n-long seeds (3 / 11 / 23) and their average (`final`), same cases, means over the 5 line holdouts:

| | 9074 rc_r (256-600 px) | Speedie SNR 12 / 24 | twin spurious SNR 6 / 12 / 24 (m/s) | line @600 PSNR, M0/M1/M2 |
|---|---|---|---|---|
| R6 | **0.43-0.47** | **0.61** / 0.81 | 293 / 11.9 / 4.7 | 45.8 dB, 44/84/84 |
| R4n s3 / s11 / s23 | 0.26-0.36 | 0.60 / 0.41 / 0.56; 0.80-0.82 | 67-82 / 4.8-8.0 / -1.4-3.7 | 47.9-49.3 dB |
| R4n average + gate | 0.29-0.33 | 0.55 / 0.83 | 52 / 3.8 / 0.2 | 49.3 dB, 47/86/88 |
| oracle linear (2026-10-14) | 0.38 | - | - | - |

Reading: crop training raises the real-GI wiggle beyond what any linear filter can reach (0.45 vs 0.38), so the 9074 limit is NOT the oracle figure; but the same model
invents more structure on a disc with no wiggle (SNR 12: 11.9 m/s vs the 5 m/s bar) and is worse for line emission at 600 px. Not adoptable alone. Candidate uses:
a wiggle checkpoint separate from the moment checkpoint (PLAN.md allows 2-3 checkpoints split by purpose), or an R6 + R4n average if averaging keeps the gain and cuts the
spurious signal (one eval job). R6 is one seed: its 0.61 at SNR 12 sits inside R4n's seed spread, the 9074 gain (above every R4n seed at every size) does not.

## 2026-10-14 | run | How much wiggle is recoverable at the test noise: the frozen model already sits at the best LINEAR filter's limit

`experiments/oracle_limit.py` (local, CPU): an oracle 3D Wiener deconvolution that KNOWS clean's and the noise's radially averaged power spectra (not clean
itself), i.e. the best shift-invariant linear filter possible for that cube, scored with the same rc_r as everything else. Log: `results/oracle-limit_2026-10-14/run_log.txt`.

| cube | dirty | blind Wiener | **oracle linear** | frozen model (256-600 px) | noise-free beam |
|---|---|---|---|---|---|
| 9074 (hydro GI, held out, SNR ~11.5) | 0.13 | 0.16 | **0.38** | 0.29-0.33 | 0.93 |
| an07 | 0.59 | 0.59 | **0.88** | 0.86-0.90 | 0.97 |
| an08 | 0.49 | 0.48 | **0.84** | 0.84-0.98 | 0.95 |
| 9015 (trained on) | 0.04 | 0.21 | **0.24** | - | 0.82 |
| 9019 (trained on) | 0.72 | 0.52 | **0.80** | - | 0.98 |

Consequence: the model matches or slightly beats the best possible linear filter on every cube where both exist. On 9074 the gap to the 0.85 target is not a training failure the
data can show: a linear method with perfect spectral knowledge reaches 0.38. Only a non-linear method with a much stronger prior on what GI wiggles look like could go
further, and how far is unmeasured. The noise-free beam figure (0.93) is NOT an achievable target at SNR ~11; the write-up should quote the oracle-linear figure next to it.
Speedie (SNR-12 ladder, 0.55) is not local; same check to run on Kaggle.

## 2026-10-14 | run | eval-final done (R4n-long seeds 3+11+23 averaged, gate in) and R7 seed 42 done: R7 is not a clear win on the SNR-12 wiggle

`eval-final` (chirag, 362 min, exit 0, 97 cases): `results/final/RESULTS.md` is the frozen-model report, gate (40,120,0.25) applied from stored mixes. Only the two
Speedie r1.5 cubes (blind SNR 133 / 121) are changed by the gate. Averaged model, 5 line holdouts, mean over cubes (not seeds): 256 px PSNR 48.2, M0/M1/M2 51.8/87.3/84.9 %;
600 px PSNR 49.4, M0/M1/M2 47.3/85.6/88.2 %, rc_r 0.68 (dirty 0.21, Wiener 0.39). SG v2 (1 cube, SNR ~63): M0 -6 %, rc_r 0.89 vs dirty 0.88, so equal to dirty (item 8, ceiling-limited).
Speedie ladder (GI r0.5, averaged model): wiggle rc_r 0.00 / 0.13 / 0.55 / 0.83 at peak SNR 3 / 6 / 12 / 24 (Wiener 0.00 / 0.07 / 0.18 / 0.43); twin spurious +1170 / +52 / +3.8 / +0.2 m/s.
**The 0.85 target at SNR 12 is NOT met (0.55); at SNR 24 it is 0.83.** Twin at SNR 3 is a failure (+1170 m/s); the model should not be trusted below SNR ~6.

R7 seed 42 (jensi, training SNR 3-40; seeds 41 and 43 not done): ladder wiggle SNR 12 = 0.53, SNR 24 = 0.81, SNR 6 = 0.24 (R4n seeds 0.11-0.13), twin SNR 12 / 24 = 0.36 / 0.57 (R4n 0.27-0.31 / 0.46-0.53);
line @600 PSNR 43.5 / 49.8 / 52.1 / 52.4 / 43.4 vs seeds 3 / 11 / 23 mean about 1 dB higher. One seed: SNR-12 wiggle inside the R4n range (0.41-0.60), so no gain there; SNR 6 gain real-looking but one seed, and
line emission slightly lower. Not adoptable on this evidence. Seed 41 is stopped at its budget and needs jensi's quota (reset Oct 17); seed 43 running.
Archived: `results/unified/eval-final_v1_2026-10-14/`, `results/unified/r7-k15-snr40-s42_v1_2026-10-14/`.

## 2026-10-13 | run | Do-no-harm gate chosen and built: at peak SNR ~135 the plain model's M0 / M2 damage (-76 / -257 %) becomes -4 / +50 % with the wiggle at dirty's level; below SNR ~40 the plain model stays

`eval-gate` (aditi, R4n-long seeds 3 + 11 averaged; archived `results/unified/eval-gate_v1_2026-10-13/`): each case scored for a * model + (1 - a) * Wiener at a = 1, 0.75, 0.5, 0.25, 0, with the blind estimate
`snr_est` (Wiener-normalisation scale over the rms of what the Wiener removed). The estimate reads 133 / 121 on the two peak-SNR-135 cubes and 27 / 18 / 9 on cubes of true peak SNR 36 / 24 / 12 (SG v2, a hydro-like
deconvolution cube, reads 27 for a true 63): it under-reads by ~25 % at low SNR and is monotonic within the A = I (Speedie-like) regime, which is what the ALMA cubes are. M0 / M1 / M2 % and rc_r by a:
| case (est.) | a = 1 (plain) | 0.75 | 0.5 | 0.25 | 0 (Wiener) | dirty rc_r |
|---|---|---|---|---|---|---|
| Speedie GI r1.5 (133) | -76 / +40 / -257, 0.95 | -53 / +51 / +68, 0.94 | -21 / +46 / +60, 0.93 | -4 / +37 / +50, 0.91 | -4 / +25 / +40, 0.88 | 0.93 |
| twin r1.5 (121) | -109 / +60 / -156, 0.85 | -74 / +62 / +62, 0.85 | -30 / +54 / +51, 0.82 | -7 / +43 / +39, 0.76 | -9 / +30 / +27, 0.64 | 0.75 |
| Speedie GI r0.5 (27) | +7 / +83 / +90, 0.89 | +25 / +80 / +77, 0.87 | +31 / +73 / +67, 0.83 | +31 / +64 / +58, 0.74 | +23 / +55 / +49, 0.59 | 0.36 |
| ladder SNR 24 (18) | +38 / +88 / +90, 0.83 | +48 / +86 / +77, 0.81 | +51 / +80 / +63, 0.76 | +49 / +73 / +52, 0.65 | +41 / +66 / +42, 0.43 | 0.14 |
| ladder SNR 12 (9) | +74 / +93 / +90, 0.50 | +78 / +93 / +84, 0.54 | +79 / +91 / +65, 0.57 | +77 / +87 / +49, 0.45 | +72 / +83 / +38, 0.18 | 0.01 |
| SG v2 (27) | -1 / +12 / +18, 0.87 | +1 / +9 / +13, 0.85 | +2 / +5 / +8, 0.82 | +4 / 0 / +1, 0.78 | +6 / -7 / -7, 0.72 | 0.85 |
Reads. (1) Only at est. >~100 does the plain model lose to a blend: its M0 and M2 are far below dirty there while dirty is nearly the truth. (2) At est. <= 27 the plain model has the best wiggle, and the blend only buys M0 at the
price of M1 / M2 / wiggle: kept as is. (3) Gate: a = 1 for est. <= 40, linear to 0.25 at est. >= 120, 0.25 above (`gate_alpha`, `--gate 40,120,0.25`); the 40-120 ramp is an interpolation, no cube of that estimate
was measured (SG v2's true 63 reads 27, so it does not test it). (4) Cost at est. ~121-133: the wiggle returns to dirty's level (0.91 / 0.76 against dirty 0.93 / 0.75), the denoising of M0 / M1 / M2 becomes
a small gain or a wash, i.e. the pipeline reverts toward the Wiener output exactly where the data need no help. (5) One observation, not acted on: at est. 9 the 0.5 mix has the better wiggle (0.57 against 0.50) at the
cost of M2 (65 against 90); a single case, two seeds. Built: `train_unified.py --gate lo,hi,floor` (applied inside `denoise`, logs `gate_a`), `alma_unified.py` (default on, reports `snr_est` / `gate_a`),
`final_report.py --apply-gate` (selects the stored mix per case from `eval-final`'s `--gate-sweep`; changes only the two r1.5 cases in the sweep file). Local pipeline check: gate a = 0.554 reproduces the manual 0.5 blend
(M0 / M1 / M2 29.9 / 48.8 / 64.7 against 31.0 / 48.3 / 63.4). Queued: R7 (training SNR 3-40, 3 seeds) for the SNR-12 wiggle, on account jensi once the shares are in.

## 2026-10-13 | run | R4n-long seed 23 done (first RULES.md #14 stop-and-resume on Kaggle worked); three seeds in: seed 3's line-emission M0 (55) was the high outlier

Seed 23 trained all 24000 steps with 0 skipped steps, reached its 11 h budget while evaluating, wrote `status_*.json` (eval_complete false), and `orchestrator.py resume` relaunched it with its own output attached; the resumed
session scored the remaining cases (72 of 72, none failed). Three R4n-long seeds (3 / 11 / 23): Speedie r0.5 wiggle 0.87 / 0.88 / 0.87; ladder SNR 24 0.81 / 0.82 / 0.80; SNR 12 0.60 / 0.41 / 0.56; SNR 6 0.11 / 0.12 / 0.13;
twin spurious at SNR 6 +82 / +67 / +78 m/s; 9074 0.34 / 0.28 / 0.36; SG v2 0.86 / 0.85 / 0.85; Speedie r1.5 M0 -20 / -185 / -141; line emission 600 px PSNR 49.30 / 47.91 / 48.07 and M0/M1/M2 55/87/89, 34/84/85, 30/83/85.
Corrections to earlier headlines: the "49.3 dB, M0 55" quoted for R4n-long was seed 3 alone and the high outlier; the three-seed mean at 600 px is 48.4 dB, M0 40, M1 85, M2 86. The stable properties of the recipe across
seeds: Speedie r0.5 wiggle 0.87-0.88, SNR-24 wiggle 0.80-0.82, SNR-6 invented wiggle +67-82 m/s (not +233 / +519 as single R4 seeds gave), SG v2 0.85-0.86. The unstable ones: SNR-12 wiggle (0.41-0.60), line M0 at 600 px
(30-55), Speedie r1.5 M0 (-20 to -185). The gate (`eval-gate`, running) targets the last.

## 2026-10-13 | code | Open pre-ALMA items sorted (results/PLAN.md 0b); gate sweep launched; `final_report.py` built; seed 23 stopped at its 11 h budget and was resumed

Krishan: sort the open items first, then ALMA. Order and reasons in `results/PLAN.md` section 0b; the key sequencing decision is that the do-no-harm gate (high-SNR M0 damage, -20 to -185 % at peak SNR 135) goes in BEFORE the final
evaluation, so the frozen numbers describe the shipped pipeline. `train_unified.py --gate-sweep a,b,c` scores a * model + (1 - a) * Wiener next to the model and the Wiener on every case and logs `snr_est` (Wiener-normalisation scale
over the rms of what the Wiener removed, the network's own noise-map quantity); local check: it reads 12.0 on a line cube of true peak SNR 15.5 and 13.9 on an analytic disc of 19.7, i.e. ~25-30 % low but monotonic.
`eval-gate` (aditi, R4n-long seeds 3 + 11) scores it on ladder SNR 12 / 24, Speedie r0.5, SG v2, Speedie r1.5; the threshold is chosen from that table, not guessed. Wiener, not dirty, is the blend target because it
was never worse than dirty on the moments at r1.5 (M0 -4 / +17, M1 +25, M2 +40), while dirty is zero by construction. `experiments/final_report.py` writes the frozen model's table at all five sizes by group with the range
across individual seeds, the ladder and the gate sweep (tested on the existing seeds); `compare_models.py` gained the `young_test` group. 2026-10-13 15:30: seed 11 became visible to chirag, so the final evaluation (`eval-final`, chirag, seeds 3 + 11 + 23 averaged, all cases, `--gate-sweep` on every case, 5 held-out Young simulations) was launched without waiting for the gate decision: the gate is a selection among the mixes stored in that one run. Seed 23 (chirag) trained all 24000 steps, then stopped at its 11 h deadline with
3 cases unscored (status file `eval_complete: false`): the RULES.md #14 mechanism worked as designed; `orchestrator.py resume` relaunched it with its own output attached.

## 2026-10-13 | run | ALMA Gate A passes: on exoALMA's REAL noise with a known injected disc, the averaged R4n-long model recovers line centres to 6-25 m/s (dirty 20-67) and invents no wiggle; the 77 m/s shift on MWC 758 is raw-M1 noise

`alma-exoalma-s5` (account aditi, R4n-long seeds 3 + 11 averaged, MWC 758 13CO fiducial cropped to 512 px = 12.8 arcsec; archived `results/alma/mwc758-13co_s5_v1_2026-10-13/`).
S5 injection-recovery (truth: Speedie GI disc and Keplerian twin resampled to 25 mas and 100 m/s, injected into this cube's own line-free planes = real exoALMA noise; one simulated disc):

| case | wiggle rc_r dirty / Wiener / MODEL | line-centre error vs truth, median m/s, dirty / MODEL | spurious rc rms m/s, dirty / Wiener / MODEL | M1 model - M1 dirty, bright px |
|---|---|---|---|---|
| GI, SNR 12 | 0.03 / 0.56 / **0.75** | 67 / **25** | +1329 / -23 / -24 | 56 m/s |
| GI, SNR 24 | 0.09 / 0.78 / **0.89** | 37 / **18** | +445 / -19 / -18 | 35 m/s |
| GI, SNR 36 | 0.29 / 0.86 / **0.93** | 26 / **14** | +127 / -14 / -14 | 26 m/s |
| twin, SNR 12 | -0.00 / 0.22 / 0.64 | 54 / **8** | +1275 / +3.2 / **-0.1** | 51 m/s |
| twin, SNR 24 | 0.01 / 0.43 / 0.79 | 29 / **7** | +328 / -1.5 / **-1.3** | 31 m/s |
| twin, SNR 36 | 0.13 / 0.58 / 0.86 | 20 / **6** | +75 / -2.2 / **-1.3** | 22 m/s |

Reads. (1) The model's line centres are 2-8x closer to the truth than the raw cube's at every SNR, on real exoALMA noise, and on the wiggle-free twin within 6-8 m/s of truth (exoALMA's quoted
velocity precision is ~10 m/s). (2) The |M1 model - M1 raw| difference on the injected disc (22-56 m/s) is of the size seen on the real MWC 758 cube (77 m/s median on its bright pixels, raw
SNR lower), while the model's own error against truth is 6-25 m/s: the shift on the real cube is the raw M1's noise, not model bias. (3) No invented wiggle on the twin (-0.1 to -1.3 m/s); on the GI
disc the recovered residual rms is 14-24 m/s BELOW the truth's, i.e. the wiggle amplitude is slightly under-recovered, not exaggerated. (4) The injected-wiggle recovery (0.75 / 0.89 / 0.93 at
SNR 12 / 24 / 36) is higher than the synthetic SNR ladder's (single seeds 0.41-0.60 / 0.81-0.82 at 12 / 24). Not a contradiction to rely on: the injected truth carries Speedie's own
0.24 x 0.18 arcsec beam (~9.5 x 7 px here), wider than the 0.15 arcsec noise correlation, so signal and noise separate more easily than for a real 0.15 arcsec image; and it is a 2-seed average.
Real MWC 758 (same run): off-line noise 25.3x lower, M0 total ratio 1.02, S2 above-4-sigma fraction 7.5e-5 raw vs 2.2e-5 denoised; S6 now valid (restricted to the disc): the denoised
rotation-curve residual correlates 0.82 with the raw one (coherence 0.79), residual rms 101 vs 117 m/s. The 13CO disc is compact (fit R 47-149 au). Gate A (ALMA_PLAN.md): passed.

## 2026-10-13 | code | ALMA S5 injection-recovery and the S6 mask fix built and launched; R6 (256 px training crops) launched

S5 (`alma_unified.py --inject-dir`): the truth is Speedie et al.'s GI disc and Keplerian twin (speedie_prep clean cubes, test-only), resampled to the real cube's pixel scale (area) and
channel width (velocity binning), centred, padded with line-free planes, and added to blocks of the real cube's OWN line-free planes (real noise with its spatial and channel correlation)
scaled to peak SNR 12 / 24 / 36; then the same path as the real cube (line-free finder, continuum, velocity-aware Wiener, model). Scored against truth with the wiggle scorecard
(rc_r, line-centre error, spurious wiggle on the twin) and, for comparison with the real-cube S1 number, |M1 model - M1 dirty| on the brightest 10 %. If the model's line-centre error
against truth is small while |M1 model - M1 dirty| matches the 76-83 m/s seen on MWC 758, the real-cube shift is raw-M1 noise, not model bias. S6: both cubes restricted to the S1 signal
mask grown by one beam before the geometry fit. Local plumbing test only (stand-in truth, old K=3 model; numbers not quoted). Launched on aditi with R4n-long seeds 3 + 11 averaged.
R6 (PLAN.md track C): the R4n-long recipe with 256 px training crops (batch 4) instead of 128 px (batch 12), on the reasoning that the GI spiral spans hundreds of pixels and the model
has only ever seen 128 px windows; seed 31, 24000 steps, first full run on the RULES.md #14 kernel. Decision on Oct 15: adopted only if its wiggle at SNR 12-24 beats the R4n-long
seeds without hurting line emission. The averaged final evaluation of R4n-long is waiting for seed 23 and for seed 11 to be shared with chirag.

## 2026-10-13 | run | R4n-long seed 11: the recipe's Speedie and SNR-24 wiggle hold across seeds (0.87-0.88, 0.81-0.82); SNR 12 and line M0 at 600 px do not; MWC 758 cropped run

Seed 11 (account aditi, 24000 steps, 0 non-finite steps, archived `results/unified/r4n-k15-long-s11_v1_2026-10-13/`) against seed 3: Speedie r0.5 wiggle 0.88 vs 0.87, M0/M1/M2 3/81/89 vs 8/83/87;
ladder SNR 24 0.82 vs 0.81, SNR 12 0.41 vs 0.60, SNR 6 0.12 vs 0.11; invented wiggle on the twin at SNR 6 +67 vs +82 m/s; line emission 600 px 47.91 dB 34/84/85 vs 49.30 dB 55/87/89, 256 px
47.38 53/87/87 vs 47.79 53/87/84; 9074 0.28 vs 0.34, SG v2 0.85 vs 0.86, an07 0.79 vs 0.88, an08 0.77 vs 0.82. Reads: the recipe's gains on the Speedie wiggle at SNR ~36 and 24 and on line
emission at 256 px are stable across seeds and above every R4 seed; the SNR-12 wiggle (0.41-0.60) and line M0 at 600 px (34-55) vary as much as R4's did, so the final model is an average
of seeds (the third, seed 23, runs on chirag) and its numbers are reported with the seed range.
MWC 758 13CO cropped to 256 px (v2, archived `results/alma/mwc758-13co_r4n_v2_2026-10-13_crop256/`): off-line noise 24.6x lower, M0 total ratio 0.98, |M1 shift| on the brightest 10 % median 76 m/s
(p95 257), above-4-sigma fraction in line-free planes 7.9e-5 raw vs 2.1e-5 denoised. S6 is still invalid: the scorecard builds its geometry mask from the reference M0 at 5 % of the peak, and
with the RAW cube as reference that mask includes noise blobs across the field (rc_rms 6 km/s); fix: restrict S6 to the S1 signal mask. The denoised M0 ring looks smoother than the raw one;
whether that is noise removal or lost asymmetry is what injection-recovery (S5) is for.

## 2026-10-13 | run | First exoALMA cube through the unified denoiser (MWC 758 13CO, R4n-long, inference only): Gate A passes on noise, flux and invented structure; velocity shift not yet interpretable

`alma-exoalma-r4n` v1 on account chirag (archived `results/alma/mwc758-13co_r4n_v1_2026-10-13/`). Cube: 301 x 1024 x 1024, 25 mas/px, circular 0.15 arcsec beam = 6.0 px, 100 m/s channels; 258
line-free planes by sigma clipping (the line spans ~43 planes, 4.3 km/s, consistent with a nearly face-on disc, i ~21 deg); raw off-line rms 4.70 mJy/beam; the line-free planes' mean holds a
compact central residual (peak 9.9 sigma of the mean; a point-like source also visible in single line-free planes, raw and denoised), subtracted as continuum. Download 13 s, denoise 4.9 min on a T4.
S1: off-line noise 30.7x lower (4.70 to 0.153 mJy/beam); M0 total ratio 0.96 inside the 5-sigma mask (3862 px); |M1 shift| on the brightest 10 % median 83 m/s, 95th percentile 267 m/s.
S2: fraction of line-free-plane pixels above 4 sigma_raw 2.3e-4 raw, 1.2e-5 denoised (the denoiser removes noise peaks and does not create new ones at this threshold).
S6: INVALID as run: the 1024 px field is 25.6 arcsec and the 13CO disc ~1.5 arcsec, so the geometry fit, the 95 %-flux radius (2052 au) and the spine ran on empty sky (rc_rms 8-13 km/s).
Figure: the denoised 5.99 km/s channel shows the disc's emission crisp on a clean background; M0 keeps the ring shape. Reading: Gate A passes on noise, flux and invented structure; the
83 m/s median M1 shift is NOT yet interpretable, because the raw M1 at this SNR is itself noisy (no truth on a real cube): the injection-recovery check S5 (a known disc injected into this
cube's own line-free noise) is what can say whether the shift is the model's bias or raw-M1 noise. v2 (cropped to 256 px around the disc) launched for S6.

## 2026-10-13 | code | ALMA inference adapter for the unified model (`experiments/alma_unified.py`, kernel `kaggle/alma/`); first exoALMA run launched

Same path as the synthetic tests: real cube (truncated downloads read by whole planes; 4D Stokes squeezed; frequency axis to radio velocity; velocity made ascending; NaN outside the
primary beam -> 0; central crop) -> line-free planes by iterative 3-sigma clipping of the central spectrum (an edge-plane rule failed when the edges hold line) -> their per-pixel mean
removed (residual continuum, reported in sigma) -> velocity-aware Wiener on the native grid with noise from those planes (`native_wiener(..., "line", noise_chans)`, now honoured for
line emission) -> the model with its own K / norm / clip from the checkpoint (several checkpoints averaged) -> optional resampling (`--scale`) -> denoised cube + raw / denoised M0, M1, M2 as
FITS (DS9) + figure + `report.json`: S1 (off-line rms raw vs denoised, M0 total ratio in the 5-sigma mask, |M1 shift| on the brightest 10 % in m/s), S2 (fraction of line-free-plane pixels above
4 sigma_raw, raw as denominator), S6 (rotation-curve residual of the denoised M1 against the RAW M1's, given inclination and distance). Tests: the line-free finder on a synthetic cube keeps
0 of 60 line planes and all 140 line-free ones, also with the line touching the edge; plumbing on the local truncated MWC 758 12CO file (128 of 301 planes, -9.2 to +2.8 km/s, i.e. almost no
line: those numbers mean nothing and are not quoted). exoALMA's 0.15 arcsec beam is 6.0 px at 25 mas, the low end of the training beams; `--scale` exists for the S4 check. First real run:
MWC 758 13CO on account aditi with R5 seed 1 (main's weekly quota is used up and aditi cannot see R4n-long); its purpose is Gate A, not the final numbers.

## 2026-10-12 | code | RULES.md #14: every run saves before the 12 h limit and resumes (Krishan's standing rule)

`train_unified.py`: `--budget-h` (default 11 h from `EXXA_T0`, set by the kernel at its first line), `--eval-reserve-h` (default 1.5 h); training stops before a round that would not fit
and writes `status_<tag>.json` + a FULL-state checkpoint every round (weights, optimizer, LR schedule, loss scaler, numpy / torch / CUDA random state; atomic write); `--resume` continues from
its round (older weights-only checkpoints resume with a fresh optimizer and a fast-forwarded schedule). Evaluation stops before the deadline when the next case would not fit and continues on
rerun (cases already in the JSON are skipped). Kernels (`kernel_unified.py`, `kernel_eval_ensemble.py`, `kernel_eval_ladder.py`) set the clock and copy their own previous session's files
for their own tags only, then resume. `orchestrator.py status` marks a finished job `resume_needed` from its status file; `orchestrator.py resume` re-pushes it with its own output attached
(`--self-input`). Tested locally: a smoke run stopped by a 0.004 h budget after round 1, resumed to round 2 and evaluated; an evaluation stopped before its first case under a short budget and
the rerun scored all 9 cases. Not yet exercised on Kaggle (the self-input attach of a kernel's own output is Krishan's nb14 procedure, automated). Note: a resumed run restores the optimizer,
schedule and random state but redraws every training cube's noise at the first resumed round, so it is not bit-identical to an uninterrupted run.

## 2026-10-12 | run | R5 (Young GI data) does NOT help the Speedie wiggle at low SNR; R4n-long (robust norm, 2x training) is the best model so far

R5 seeds 0 and 1 (R4 recipe + robust normalisation + 14 Young GI training cubes) and R4n-long (R4 + robust normalisation, 24000 steps, seed 3, no Young data): all 0 non-finite steps, 0 failed
cases. Archived `results/unified/r5-k15-s{0,1}_v1_2026-10-12/`, `r4n-k15-long_v1_2026-10-12/`.

| case | R4 s0 / s1 / s2 | R4n-long | R5 s0 / s1 |
|---|---|---|---|
| Speedie GI r0.5 wiggle (SNR ~36) | 0.80 / 0.78 / 0.85 | **0.87** | 0.84 / 0.85 |
| Speedie GI r0.5 M0/M1/M2 % | 9/72/45, 29/80/82, 21/78/76 | 8/83/87 | 26/82/85, 40/82/80 |
| twin r0.5 spurious m/s | -1.3 / -1.5 / -1.6 | -3.1 | +7 / +7 |
| ladder SNR 24 wiggle | 0.76 / 0.74 / 0.80 | **0.81** | 0.76 / 0.76 |
| ladder SNR 12 wiggle | 0.61 / 0.62 / 0.50 | 0.60 | **0.29 / 0.49** |
| twin SNR 6 spurious m/s | +47 / +233 / +519 | +82 | +232 / +192 |
| Speedie r1.5 M0/M1/M2 % | -253/-190/-919, -91/12/-41, -105/29/-60 | -20/31/-280 | -57/20/-295, -101/45/-33 |
| 9074 @600 wiggle | 0.31 / 0.31 / 0.32 | 0.34 | 0.41 / 0.31 |
| line emission @600 PSNR, M0/M1/M2 | 47.3-47.8, 40-45/86-87/82-84 | **49.30, 55/87/89** | 47.98, 43/83/87; 49.65, 51/87/89 |

Held-out Young simulations (R5, 600 px): wiggle rc_r 0.71-0.72, 0.85-0.87, 0.65 where dirty is 0.49-0.66, and 0.14-0.22 where dirty is 0.03-0.14; PSNR 48-53 dB vs dirty 21-23.
Reads. (1) The Young data trains a model that recovers the held-out Young discs, but it LOWERS the Speedie wiggle at SNR 12 (0.29 / 0.49 vs 0.50-0.62 for R4, both seeds) and adds a small
invented wiggle on the Keplerian twin (+7 m/s vs -1 to -3). Likely cause: the Young discs are young, massive and chaotic (clean residual ~190 m/s rms, against ~70 m/s for Speedie) and the
converter is validated only above ~50 au, so the model learns a prior of large, coarse velocity residuals that does not fit Speedie's subtle wiggle. Not proven: one cause among several, and
the Young data stays out of the final recipe unless a later run shows a gain. R5 seed 2 (added 18:55): Speedie r0.5 0.83, SNR 24 0.75, SNR 12 0.56, 9074 0.35, line 47.15 dB 48/86/83; across the three R5 seeds SNR 12 = 0.29 / 0.49 / 0.56 (mean 0.45) against R4's 0.50 / 0.61 / 0.62 (mean 0.58): the R5 deficit survives three seeds. Only one of this round's jobs hit the 12 h session limit (`eval-ensemble`, partial results kept); every training run completed. (2) R4n-long (robust normalisation + 2x steps) is the best single model on most rows: Speedie r0.5 0.87 (equal
to R3b), SNR 24 0.81, the best line emission yet (49.3 dB, M0 55, M2 89), the least M0 damage at r1.5 (-20). One seed; a second (seed 11) is running on account aditi. (3) PLAN.md target
(wiggle >= 0.85 at SNR >= 13): met at SNR ~36 (0.87), close at 24 (0.81), not at 12 (0.60). Final-model recipe candidate: R4n-long, several seeds averaged.

## 2026-10-12 | run | Ensembles do not lift the wiggle: the SNR 12-24 gap is bias, not seed variance; TTA dropped; R4 seed 2 in

`eval-ensemble` (main, stopped at the 12 h limit; (a) and (b) complete, (c) partial) and R4 seed 2. Archived `results/unified/eval-ensemble_v1_2026-10-12/`, `r4-k15-s2_v1_2026-10-12/`.
Same cases, one simulated disc for Speedie (no spread across discs; the three seeds give the spread across training runs):

| case | R3b | R4 s0 / s1 / s2 | mean of s0 + s1 | + flips | + R3b, flips |
|---|---|---|---|---|---|
| Speedie GI r0.5 wiggle rc_r | 0.87 | 0.80 / 0.78 / 0.85 | 0.80 | 0.80 | 0.84 |
| Speedie GI r0.5 M0 / M1 / M2 % | 18 / 83 / 88 | 9/72/45, 29/80/82, 21/78/76 | 24 / 77 / 60 | 24 / 75 / 47 | 35 / 80 / 63 |
| ladder SNR 24 wiggle | - | 0.76 / 0.74 / 0.80 | 0.76 | 0.76 | - |
| ladder SNR 12 wiggle | 0.38 (eval-ladder) | 0.61 / 0.62 / 0.50 | 0.63 | 0.63 | - |
| twin spurious at SNR 6 (m/s) | +784 | +47 / +233 / +519 | +57 | +58 | - |
| line emission 600 px PSNR, M0/M1/M2 | 48.4, 41/86/88 | 47.3-47.8, 40-45/86-87/82-84 | 48.1, 46/87/83 | - | - |

Reads. (1) Averaging two seeds gives the wiggle of the BETTER seed (SNR 12: 0.63 vs 0.61 / 0.62), not more: the residual wiggle error is mostly bias shared by all seeds (what the model cannot
infer from the data), not seed noise. (2) Averaging does remove the worst cases: invented wiggle on the twin at SNR 6 +57 m/s vs +233 / +519 for single seeds, and line PSNR / M0 rise ~0.5 dB / ~3 points.
(3) Flip averaging (TTA) adds nothing and costs 4x: dropped. (4) Seed spread on the SNR-12 wiggle is 0.50-0.62 (3 seeds): one seed is not enough to rank recipes on it. (5) Consequence for the
final models: an average of seeds is worth shipping for stability, but the wiggle gap at SNR 12-24 has to be closed by the data / training (R5, running), not by averaging.

## 2026-10-12 | code | First push of this month's work; Kaggle auto-commits checked

Pulled 4 Kaggle auto-commits on midterm-prep before committing (RULES.md #2): 05 Versions 48 and 49 and 13 Version 18 change outputs only (cell sources identical to
HEAD, compared cell by cell), so nothing was reverted; 14 Version 8 ADDS `14-native600-loss-sweep.ipynb` to midterm-prep as a stale copy (the current notebook 14 lives
on `native600-loss-sweep`, with the resume and RAM fixes); kept, flagged in context.md, not used. Not committed: `16-checkpoint-evaluation.ipynb` (52 MB), the exoALMA
paper PDF, stray downloaded notebook copies (V47 / V17 / "(1)"), speaker-notes PDFs and `nb14_loss_sweep.csv` (provenance not checked).

## 2026-10-12 | code | GI training data from the Young et al. 2026 simulations (`make_sph_cubes.py`, kernel `sph-cubes`) and the `sph` family in the trainer

Why: the wiggle is the gap at SNR 12-24 (0.61-0.75 against the 0.85 target) and the model's prior of what GI spirals look like comes from 4 hydro discs; the Young set is the only public
mass-ratio sweep (research.md 2026-10-10). `make_sph_cubes.py`: per simulation the LAST dump of its archive (the first is the initial condition), one random view (i 10-60 deg, PA, in-plane
rotation, spin), 13CO, tau0 log-uniform 3-50, `sph_cube.render(smooth=True)` on 301 px with the disc's R95 filling 80 % of the field, 0.1 km/s channels covering every particle's velocity + 1 km/s
(the 99.5th-percentile range left 47 % of the peak in the edge channels, which the trainer removes as continuum; now 0.7 %), a Gaussian beam of 5-12 px FWHM (axis ratio <= 1.3), clean = sky (*) beam
(the CLEAN-product regime), dirty = clean + beam-shaped noise (shape only; the trainer redraws noise). 16 train / 6 held-out WHOLE simulations, chosen across mu 0.2-1.4, M* 0.1-1, R 50-200 au.
Local check on M0.3_R50_Mu0.6 (mu 0.57): renders in 12 s, 242 x 301 x 301; the clean cube's rotation-curve residual is 191 m/s rms (strong, large-scale GI spirals); dirty at SNR 20 already reaches rc_r 0.78.
Trainer: `--sph-dir DIR` adds the train split as family `sph` (beam inside clean, like line emission, but sampled as its own family); `--eval-sph` scores the test split at all sizes. Local smoke
passes (4 families). Running the render on the backup account (CPU). Caveats carried: the converter matches MCFOST only at >~50 au (sph-validation v3); Young's discs are young and massive, so
they serve as a mass-ratio sweep for the wiggle prior, not as the target population. A direct line-centre (M1) loss was considered and deferred: the model outputs one channel and the
loss already sees three adjacent ones (first and second velocity differences); a real M1 term needs the whole spectrum per pixel, an architecture change.

## 2026-10-12 | bug | R4-long died in round 1: normalising by the Wiener peak collapses at low SNR (affects every run trained to SNR 3); `--norm robust` added

`unified-r4-k15-long` (R4 recipe, seed 3, 24000 steps) reached L1 1.06 in round 1 (normal ~0.04) with 85 non-finite steps, then 51 in a row at step 575. Measured locally (`TrainCube.redraw`, 3 draws per
cube, 300 px): with the old normalisation (divide by the Wiener cube's peak) the target |clean/s| reaches 15.9 (line), 2.4 (9015), 6.9 (an01) and the input |dirty/s| 29 / 4 / 12 at peak SNR 3; at
SNR 10 2.2 / 1.4 / 4.0; at SNR 50 ~1.1. The Wiener filter shrinks a SNR-3 cube to a small fraction of its clean peak, so the scale collapses; the input clamp (+-10) then clips the input but not
the target. R4 seeds 0 and 1 (same recipe) survived it; seed 3 did not. Fix `--norm robust`: scale = max(Wiener peak, 99.99th percentile of the dirty cube), blind and identical at test time;
measured |clean/s| 0.73-1.52 and |dirty/s| 1.02-1.37 for SNR 3-150 on the same three cubes. Stored in the checkpoint args, so older checkpoints keep evaluating with the old normalisation; ensemble
members must share it. Consequence for numbers already logged: R4 seeds 0 / 1 were trained (and are evaluated) with the collapsing scale, so their SNR 3-6 ladder points include inputs clipped at
+-10; their SNR >= 12 points are unaffected (scale ~1-2 there). Relaunched as `unified-r4n-k15-long`. R5 will use `--norm robust`.

## 2026-10-11 | code | Ensemble evaluation (`--eval-ckpt a,b,c`, `--tta`) and two more R4 runs; compute is not the constraint (Krishan)

Krishan: do not hold back compute. Launched on the backup account: R4 seed 2, and R4 with 80 rounds (24000 steps, seed 3; R4's training loss was still falling at 12000). Built:
`train_unified.py --eval-ckpt a.pth,b.pth[,...]` averages the outputs of several checkpoints (same K / arm / base, asserted) and `--tta` also averages over the 4 sky-plane
flips; class `Ensemble`. Local check only (plumbing: the same checkpoint twice with flips gives a different, finite result). Kernel `kaggle/eval_ensemble/` scores three groups: R4 s0 + s1
(all holdouts at 5 sizes, Speedie, ladder), R4 s0 + s1 with flips, and R4 s0 + s1 + R3b with flips (Speedie and ladder). It runs on the main account and needs the backup
account's R4 kernels shared with the main account (403 until then). All results from both accounts are collected; notebook 14's Oct 9 run was Version 15, already archived.

## 2026-10-11 | run | `eval-ladder` v1 (R3b, R3d) and the figures: training down to SNR 3 is the right recipe; what the maps show at each noise level

Ladder for R3b (K=15, training SNR 10-150) and R3d (K=3), same cases as R4 (archived `results/unified/eval-ladder_v1_2026-10-11/`, `snr_ladder_all.md`). Wiggle rc_r at peak SNR 24 / 12 / 6:
R4 seeds 0.74-0.76 / 0.61-0.62 / 0.09-0.14; R3b 0.79 / 0.38 / 0.03; R3d 0.75 / 0.21 / 0.06. Twin spurious at SNR 12: R4 +10, R3b +15, R3d +29 m/s; at SNR 6: R4 +47 / +233, R3b +784, R3d +513.
M2 at SNR 3: R4 +74 / +80, R3b -3, R3d -24. Decision: the R4 recipe (training SNR 3-150) is kept; equal at SNR 24, far better at 6-12 on wiggle, moments and invented wiggle.
What the figures show (R4 seed 1, `results/unified/figures_2026-10-11/`): (a) Speedie SNR 12: the model's rotation-curve residual shows the spiral as a faint, broken pattern (r 0.62,
coherence 0.33); dirty is pure noise (r 0.01); the velocity-aware Wiener smooths M0 into a featureless blob and leaves a four-lobe artefact at the centre of its residual (r 0.18). The Terry
spine follows clean's first two fingers and loses the outer ones. The model's M0 keeps the inner cavity and ring but loses the outer spiral arms in intensity. (b) Speedie SNR 6: the model's
residual is mostly noise with a faint ring; the spiral is gone. (c) Hydro 9074 (SNR 9.5, deconvolution task): clean's residual is a large-scale kinked pattern; the model's is blotchy at the beam
scale (r 0.31) while the noise-free beam reference reaches 0.93; Wiener shows a dipole (rotation-curve mismatch). (d) Line-emission holdout run_0026 (SNR 7.9, i 53 deg): the model is close to
clean (rc_r 0.89 vs dirty 0.15, Wiener 0.85; spine r 0.93) and its M0 keeps the four-lobe morphology that dirty hides. Reading: at the SAME peak SNR the model succeeds on a compact
line-emission disc and fails on a large GI disc; the GI wiggle sits in the faint outer disc, where the LOCAL SNR is several times below the cube's peak SNR. Peak SNR alone does not
predict wiggle recovery; the per-radius SNR of the wiggle region should be reported next to it (not yet measured).

## 2026-10-11 | run | R4 (2 seeds), R3d, R3e and the first SNR ladder: wiggle recovery is SNR-limited (0.75 at SNR 24, 0.61 at 12, ~0 at 6); seed spread is as large as the effects we were ranking

All four ran 12000 steps with 0 non-finite steps and 0 failed cases; the fp32 stream fix holds (R3d / R3e / R4 used it). R4 = R3b recipe (K=15, lr 1.5e-4, clip 10) + training SNR
3-150, run as seeds 0 and 1 on the backup account (first cross-account use of a shared kernel output, the Speedie cubes: works). Archived `results/unified/r4-k15-s{0,1}_v1_2026-10-11/`
(tables at all 5 sizes, `snr_ladder.md`), `real-k3-fp32_v1_2026-10-11/`, `real-k3-fp32-sgv2_v1_2026-10-11/`.

SNR ladder (Speedie GI r0.5 with its own measured noise rescaled; one simulated disc, so no spread across discs; two seeds of R4 shown as a range):

| peak SNR | wiggle rc_r R4 (dirty / Wiener) | M0 / M1 / M2 % R4 | median line-centre error m/s R4 (dirty / Wiener) | twin spurious m/s R4 (dirty / Wiener) |
|---|---|---|---|---|
| 24 | 0.74-0.76 (0.14 / 0.43) | 44-49 / 86-87 / 82-88 | 32-33 (48 / 42) | +1 to +2 (+204 / +5) |
| 12 | 0.61-0.62 (0.01 / 0.18) | 71-73 / 94 / 88-91 | 36-38 (79 / 49) | +10 (+882 / +30) |
| 6 | 0.09-0.14 (-0.02 / 0.07) | 47-71 / 96 / 86-89 | 47-52 (215 / 58) | +47 to +233 (+1681 / +59) |
| 3 | 0.01-0.04 (-0.02 / 0.00) | 20-47 / 89-92 / 74-80 | 83-142 (1555 / 71) | ~+2000 (+2179 / +588) |

Reads. (1) Moments survive to SNR 3 (M1 89-96 %, M2 74-91 %); the WIGGLE does not: 0.75 at 24, 0.61 at 12, gone by 6. exoALMA's fiducial set (13-50) maps to roughly 0.6-0.85; the
high-resolution set (3-10) to <0.3. The PLAN.md target rc_r >= 0.85 at SNR >= 13 is NOT met (0.61 at 12, 0.75 at 24). (2) At SNR 12-24 the model invents almost no wiggle on the
Keplerian twin (+1 to +10 m/s against Wiener +5 to +30 and dirty +200 to +880); at SNR 6 one seed already invents +233 m/s; at SNR 3 everything fails. (3) Seed spread is large: at Speedie
r0.5 R4 seeds give M2 +45 / +82 and M0 +9 / +29; at r1.5 (SNR 135) M2 -919 / -41. R3c and R3e (same recipe, K=3 + SG v2, different numerics) differ by 13 points in line M0. So every
single-seed difference in this log under ~20 points of M0 / M2 is within noise, including R3b's headline 18 / 83 / 88 (one seed). (4) SG v2 in training (R3e vs R3d, both K=3 fp32): no
gain on its own wiggle (0.87 vs 0.86) and worse M2 (Speedie r0.5 -145 vs -13, analytic -89 vs +63): SG v2 stays OUT of training. (5) K=15 against K=3 at 42 m/s channels: Speedie r0.5 M2
+45 / +82 (R4 seeds) and +88 (R3b) against -13 (R3d) and -96 / -145 (K=3 + SG v2): K=15 kept. (6) Training SNR floor 3 (R4) vs 10 (R3b): no wiggle gain on 9074 (0.31 vs 0.28), higher
PSNR / M0 on the hydro holdouts (PSNR 36.2-36.8 vs 34.7 at 600 px), Speedie native wiggle 0.78-0.80 against R3b's 0.87; undecided until R3b's own ladder (kernel `eval-ladder` on main,
running). (7) High SNR (r1.5) is seed-unstable: the do-no-harm gate (PLAN.md A3) is needed there. Not changed: line-emission PSNR / moments are strong for every K=15 run (600 px: 47.3-48.4 dB,
M0 40-45, M1 86-87, M2 82-88).

## 2026-10-11 | code | SNR ladder on the Speedie disc (`--eval-ladder`, `ladder_report.py`, kernel `kaggle/eval_ladder/`): PLAN.md phase 1 built, not yet run

`train_unified.py --eval-ladder 3,6,12,24` rescales the Speedie cubes' own measured noise (dirty - clean, so its spatial and Hanning channel correlation are kept) so that
peak(clean) / rms(noise) hits each target; truth, geometry and beam unchanged; GI disc and Keplerian twin; native pixels. `experiments/ladder_report.py name=eval.json ...` prints the table per SNR
(PSNR, M0/M1/M2, rc_r with dirty and Wiener, twin spurious rotation-curve rms). Kernel `eval-ladder` scores unified-wiener v1, R3b (K=15) and R3c (K=3 + SG v2) from their kernel Outputs
(~20-30 min each on a T4; 8 cases each). Plumbing tested locally on a synthetic stand-in only (numbers meaningless). Why: the test cubes sit at peak SNR 8-18 (line emission, hydro) and 20-36
(analytic, Speedie r0.5), exoALMA's high-resolution set at 3-10, and no cube with a trustworthy wiggle truth exists below 36; the ladder is the missing curve. Note: the "peak SNR" of the ladder
is peak(clean) / rms(dirty - clean) over the whole cube, the same definition as the `--snr` training draw and as the measured table of 2026-10-10. To run it on the `backup` account the three
model kernels must be shared with it (`orchestrator.py check` lists them as not visible); on `main` it needs a free GPU slot (R3d / R3e hold them).

## 2026-10-11 | code | Multi-account Kaggle runner (`kaggle/accounts/`), owner-approved accounts only

Krishan asked to spread runs over their backup and two friends' Kaggle accounts (used by hand before). `kaggle_as.sh <name> ...` runs the CLI as an account from
`~/.kaggle-<name>/kaggle.json` (chmod 600, created by the owner; keys are never pasted into chat or kept in the repo); `orchestrator.py` submits a kernel folder to the first
account with a free GPU slot and unused weekly quota, polls status and collects outputs and logs; state in `state.json` (git-ignored, no secrets). The Kaggle API has no sharing
or collaborator endpoint (searched CLI 1.7.4.5 and kagglesdk), so sharing the datasets (`line-emission-data`, `exxa-sg-synth-pairs`, `self-gravitating-v2`, `exxa-wiggle-code`, and a
Speedie dataset made from the `speedie-prep-fast` output) with each account stays a one-time manual step; `kernel_sources` do not cross accounts, so jobs for other accounts must not
use them. Quota left is not readable through the API; it is inferred from push replies. Tested: account discovery (main only so far), a `--dry-run` submit, reset-time logic.
Added `sync_account.py` (2026-10-11): with an account's own key, it uploads that account's OWN copies of the datasets (11.8 GB, cloned locally, `--dir-mode zip` keeps the layout) and rebuilds `exxa-wiggle-code` from the repo, so no website sharing is needed; `orchestrator.py submit` rewrites dataset / kernel source owners to the target account, and speedie-prep-fast is run per account on CPU. Account `backup` (deepsceneloc, Krishan's own backup) is configured (key moved out of the repo tree to `~/.kaggle-backup/kaggle.json`, chmod 600); `orchestrator.py check` shows all five shared items visible to it; a smoke job (`unified-smoke-backup`) was submitted through the orchestrator at ~02:00 IST to test mounting and the GPU slot. Two friends' accounts not yet configured. A key file had been left inside the repo working tree and was moved out before it could be committed. 2026-10-12: account `aditi` (aditisah11) configured the same way (key moved from the repo tree to `~/.kaggle-aditi/`; it was also pasted into the chat, so it should be rotated); it sees only `line-emission-data` so far, so the R5 launcher is restricted to `backup` until the other shares are in place. 2026-10-13: account `chirag` (chiragbhayal) added the same way (key moved from the repo root to `~/.kaggle-chirag/`, never tracked). Weekly GPU quota of main and backup is used up until the Saturday 17 Oct 05:30 IST reset; aditi (~8 h left) and chirag (30 h) carry the work until then. Caveat recorded: Kaggle's terms restrict multiple accounts and credential sharing; a suspension would fall on the account holders. Nothing has been run on another account yet.

## 2026-10-10 | finding | Every test cube placed on one SNR axis: the wiggle is solved at SNR >~ 20 and NOT at ~8-10; two of my claims corrected

Peak SNR = peak of the noise-free sky (beam (*) clean for the simulations, continuum-subtracted clean for line emission) over the rms of (dirty - sky), same definition as the
`--snr` training draw: hydro 9015 7.6, 9025 7.7, 9074 9.5; analytic an07 19.7, an08 21.5; SG v2 63.3; line-emission holdouts 7.9 / 11.0 / 12.4 / 15.5 / 18.1; Speedie r0.5 ~36,
r1.5 ~135 (speedie_prep). Corrections: (a) I said today that nothing below SNR 36 is tested; wrong, the line holdouts (8-18) and the hydro discs (8-10) are all inside exoALMA's
high-resolution range (3-10) or just above it; (b) "line emission peak SNR ~29" (an older figure) does not hold by this definition (8-18). Consequences: R3/R3b/R3c drew training
SNR from 10-150, so the hydro training discs were never shown at their own native SNR (7.6-9.5); this may be part of why R3b's 9074 wiggle (0.28) is below v1's (0.37), whose training
used the native noise. The wiggle results line up with SNR: 9074 (9.5) 0.28-0.43 against a noise-free reference of 0.88; an07/an08 (20-22) 0.78-0.86; Speedie r0.5 (36) 0.87; SG v2 (63)
and Speedie r1.5 (135) level with dirty, which is already good there. Line-emission moments are strong at SNR 8-18 (no GI wiggle in those cubes). The open regime is the GI wiggle at
SNR <~ 15.

## 2026-10-10 | run | R3b `unified-wiener-real-k15` and R3c `unified-wiener-real-sgv2`: the realistic-noise recipe fixes Speedie at the exoALMA-like noise level; K=15 is the better window; SG v2 teaches nothing visible

Both ran 12000 steps with ZERO skipped steps and no scoring failures (R3b 247 min, R3c 179 min); neither used `--fp32-stream`. R3 (same recipe as R3c without SG v2, K=3, lr 3e-4) overflowed at step 2013, so the
overflow is path-dependent, not a fixed step. Archived `results/unified/real-k15_v1_2026-10-10/` and `real-k3-sgv2_v1_2026-10-10/` (JSONs, logs, kernel scripts, tables at 600 and 256 px, figures).
IMPORTANT CAVEAT on references: the "wiener" row of `compare_models` tables is v1's channel-independent Wiener; the new runs' own JSONs hold the VELOCITY-AWARE Wiener, which scores differently
(Speedie GI r0.5: PSNR 34.7 vs 32.4, M0/M1/M2 23 / 55 / 49 vs 44 / 60 / 61, rc_r 0.59 vs 0.85). The velocity-aware Wiener is therefore worse on the wiggle than the old one on this disc (the Hanning
correlation puts noise power at low velocity frequency, where the line signal also lives, and alpha = 2 over-shrinks it): a regression in the non-learned baseline to carry into every later table.
Moments are signal-masked, noise-clipped improvements over dirty (RULES.md #6); Speedie = one simulation at two noise levels, so no spread; line = mean over the 5 holdouts.

| 600 px, line emission (5 holdouts) | PSNR | M0 | M1 | M2 | rc_r | dv m/s |
|---|---|---|---|---|---|---|
| unified-wiener v1 (K=3) | 46.25 | 33 | 85 | 83 | 0.70 | 49 |
| R3c K=3 + SG v2 | 48.53 | 45 | 87 | 68 | 0.64 | 43 |
| R3b K=15 | 48.38 | 41 | 86 | 88 | 0.56 | 41 |

| Speedie GI (held out, never trained) | model | PSNR (dirty 24.3 / 28.3) | M0 | M1 | M2 | rc_r (dirty) | invented |
|---|---|---|---|---|---|---|---|
| r0.5, exoALMA-like (SNR 36) | v1 | 41.0 | -36 | 71 | 58 | 0.84 (0.36) | 0.13 |
| | R3c K=3 | 50.6 | 21 | 66 | -96 | 0.81 | 0.00 |
| | **R3b K=15** | **51.7** | **18** | **83** | **88** | **0.87** | **0.03** |
| r1.5 (SNR 135, beam x1.65) | v1 | 32.7 | -167 | -165 | -952 | 0.91 (0.93) | 0.20 |
| | R3c K=3 | 42.5 | -53 | -69 | -628 | 0.89 | 0.00 |
| | R3b K=15 | 47.8 | -70 | 40 | 32 | 0.93 | 0.01 |

Keplerian twin, spurious rotation-curve rms (m/s, 0 = invents nothing; dirty +22.9): v1 +7.0, R3c +2.8, R3b -2.4. 9074 at 600 px (the only unseen hydro disc): rc_r dirty 0.13, v1 0.37, R3c 0.43, R3b 0.28 (K=3 models
ahead of K=15 here; PSNR 33.4 / 29.1 / 33.0). SG v2 at 600 px: rc_r dirty 0.85, v1 0.83, R3c 0.82 (TRAINED on it, so seen), R3b 0.85. Analytic an07 / an08: rc_r 0.83-0.86 / 0.78-0.83 against dirty 0.59 / 0.50.
Reads. (1) At the exoALMA-like noise (r0.5) R3b passes PLAN.md's Speedie bars: all three moments above dirty (18 / 83 / 88), wiggle 0.87 against dirty 0.36 and the old Wiener 0.85, twin spurious -2.4 m/s, invented
fraction 0.03 (v1 0.13). The figure (`..._speedie_GI_r0.5_1024.png`, bottom row) shows the spiral velocity residual recovered where dirty is noise and Wiener smooths it away (r 0.87 against 0.59, coherence 0.67
against 0.11). (2) It does NOT pass at r1.5: M0 -70 % there (M1 40, M2 32), wiggle only level with dirty (0.93): the high-SNR, wide-beam, low-noise regime still damages M0. The SNR draw (10-150) and the beam
broadening (to 1.6x) were meant to cover it; they did not. (3) K=15 against K=3 on Speedie: M2 +88 / +32 against -96 / -628. At 42 m/s channels the K=3 window is +-0.25 km/s; it cannot see the line width. K=15
is also best on line-emission M2 (88 vs 83 v1, 68 K=3 + SG v2). On the 9074 wiggle K=3 is ahead (0.37 / 0.43 vs 0.28). Confounded: R3b differs from R3c in K, lr, clamp AND SG v2. (4) SG v2 in training: its own rc_r
did not rise above dirty (0.82), so no evidence it helps; R3d (K=3, no SG v2, fp32 stream) against R3c isolates it (running). (5) Line-emission rc_r fell 0.70 to 0.56-0.64 while PSNR and M0-M2 rose: the
rotation-curve residual correlation on cubes with no GI is dominated by noise-level velocity structure; not yet understood, flagged. Gates: Gate 1 (Speedie all moments above dirty) is met at r0.5 and failed on M0 at r1.5.
Not yet met: SG v2 bar (below Wiener / at dirty), 9074 vs the models' best, r1.5. Checkpoints kept (RULES.md #12).

## 2026-10-10 | run | `sph-validation` v3 (SPH-kernel smoothing): gate as written still not met; the metric was too harsh across codes, and the converter matches the large-scale velocity structure

Smoothing changed almost nothing at full resolution (twin residual 52-63 m/s against MCFOST's 17.5; rc_r -0.05 to +0.09 over tau0 1-100): shot noise was not the main cause, and the v2 entry's
reading is withdrawn. Why the gate metric fails: `rc_r` subtracts MCFOST's own fitted rotation curve, so any small axisymmetric difference between two radiative-transfer codes (M1 slope
1.014-1.023 on the GI disc, 1.07-1.08 on the twin, i.e. 1.5-8 % of a +-1.1 km/s field) is a 50-60 m/s residual of its own, as large as the ~70 m/s wiggle. A direct comparison of the saved M1
maps (v2 render, no smoothing; high-pass at three scales, r 25-330 px): GI 0.25 at 25 au, 0.64 at 50 au, 0.89 at 100 au (rms 12.9 / 30.3 / 76.8 m/s ours against 10.4 / 30.1 / 78.0 MCFOST,
so amplitudes agree); Keplerian twin 0.79 / 0.96 / 0.99 (smooth rotation, not wiggle). Reads: the spiral velocity structure is reproduced at amplitude parity and correlation ~0.9 at 100 au,
~0.6 at 50 au and not below the beam (37 au). The orientation scan's high-pass correlation at the chosen PA is 0.97 on the coarse grid. Conclusion for use: as TRAINING TRUTH the render
needs to be a physically derived, self-consistent clean cube, not a pixel match to MCFOST; what it cannot yet be trusted for is anything finer than ~50 au. Caveat to carry: a model
trained on it may learn the structure of OUR emission surface; the MCFOST cubes (Speedie GI and twin) stay test-only for exactly this reason. Not decided: whether to relax gate B2 to
"amplitude parity and high-pass corr >= 0.85 at 100 au" and proceed to the Young download; waiting on Krishan. A rotation-curve-free metric (each cube against its OWN fitted curve) is the
better B2 gate and is not yet implemented in `validate_sph_speedie.py`.

## 2026-10-10 | bug | R3 `unified-wiener-real` v1 died at round 7: the U-Net's residual stream overflows fp16 (root cause of the k15 failures too, most likely)

Rounds 1-6 healthy (L1 0.0139 at round 6, better than unified-wiener v1's 0.018-0.025 on the older, easier noise), then non-finite losses from step 2013 on ORDINARY inputs (-1..2.8), 70 of
300 steps in round 7, 51 in a row, abort at step 2242. The saved weights are finite; a forward pass of the round-7 checkpoint on real batches (fp32) gives an output of at most 0.67 but
reaches 1.27e5 and 9.9e4 in `dec_blocks.2.0.shortcut` / `dec_blocks.3.0.shortcut` (fp16 max 65504). The residual stream is not normalised (`out = h + shortcut(x)`), so it grows
with training and overflows whichever fp16 conv reads it raw. This replaces the k15 v3 reading (input outliers at the peak learning rate): the +-35 inputs there were real but not the cause,
and R3 failed with inputs of +-3. Fix (`--fp32-stream`, new runs only): shortcut convs, strided / transposed convs and the residual add in fp32; verified identical to the original blocks (max
difference 0.0, same keys), so old checkpoints evaluate unchanged. Smoke on MPS reproduces the previous numbers exactly. Queued: R3d `unified-wiener-real-fp32` (R3 + fix, the control) and
R3e `unified-wiener-real-fp32-sgv2` (R3d + SG v2); R3c (`-sgv2`, launched on the unfixed code at 15:32) and R3b (`-real-k15`, lr 1.5e-4 + input clamp, still unfixed) were expected to fail the
same way; at 16:59 both were still running (R3c 87 min in, past where R3 aborted), so the overflow is not deterministic at a fixed step and the "stop R3c" advice is withdrawn;
their logs, with the skipped-step counts, arrive when they end. Speed cost of the fp32 pieces on a T4 not yet measured. No published number touched. Archived
`results/unified/unified-wiener-real_v1_2026-10-10_failed/` with the probe.

## 2026-10-10 | run | `sph-validation` v2: bulk velocity matches MCFOST (M1 corr 0.996), wiggle does not yet: particle shot noise, SPH-kernel smoothing added (v3)

With the PA fitted continuously (best rotation -146 / +31 deg depending on the spin / mirror combination; handedness chosen on the high-pass M1: 0.85 against 0.35 for the wrong sense)
the GI render's M1 correlates 0.996 with MCFOST's with slope 1.015 (v1: 0.92 / 0.95). Wiggle: rc_r 0.04-0.06 for tau0 1-100, residual rms 55-58 m/s against clean's 68.8, so the AMPLITUDE
matches but the pattern does not; the Keplerian twin shows 57-63 m/s against MCFOST's 17.5 and rc_r ~0. Reading: the same 55-60 m/s appears on both discs, so it is not the GI wiggle: it is
noise from binning ~10 particles per pixel, as large as the signal (~70 m/s). `render(smooth=True)` now spreads each particle over its smoothing length (Gaussian sigma 0.6 h in
the sky plane, grouped in log2 bins; h from the dump). On the coarse test grid (12.5 au/px) the small-scale M1 scatter of the render falls 71 to 47 m/s; the effect is larger on the real
3.1 au/px grid where 0.6 h is about 1 px. Gate unchanged (PLAN.md B2): the converter is not used for training until the twin residual approaches MCFOST's and rc_r is well above dirty's
0.65. v3 launched (CPU). tau0 still does not matter (rc_r varies by 0.01 across 1-100 at fixed noise). Archived `results/sph-validation/v2_2026-10-10_pa_fixed_no_smoothing/`.

## 2026-10-10 | code | SG v2 as a training cube (`--train-sgv2`), and what was measured about it first (R3c queued)

Question (Krishan): should SG v2 be trained on to reach ALMA-ready results? Measured before deciding. SG v2: dirty peak SNR 69 per pixel and channel (0.033 km/s; inside R3's 10-150 range;
a first estimate of 0.6 was a units error, clean is per pixel and dirty per beam); noise is half a static pattern (static / dynamic variance 1.15) with channel-independent dynamic
noise (lag-1 -0.013, unlike Speedie's +0.64); M0 peak SNR ~500. Clean and dirty are in different units: in the bright channels dirty = g * (beam (*) clean), g ~ 344 (355 from the top
20 channels), no extra blur (best extra blur 0 px in every channel tested); clean carries a continuum floor (~0.43 summed per plane) that the dirty cube lacks, removed with its own
edge mean as the eval already does. `load_sgv2_pair` applies exactly that (one scalar gain for the whole cube, not per channel). Why it was held out until now: one disc, q = 0.75
(Speedie 0.29, real discs lower), our only cube with this beam / pixel scale / channel width, and every learned model falls below dirty on it, which training on it would hide. Reasons to
train on it that held up: it is the only cube where that failure is specific and nothing else pushes against it; its static pattern is real-data-like. Decision: an experiment, not an
assumption. R3c = R3 exactly (same recipe, K=3, SNR 10-150, chan-corr 0.5, beam-max 1.6, no clamp) + SG v2 in training; compare Speedie GI wiggle / moments and 9074 against R3. If Speedie
improves it goes into the final model (selection frozen first, its own test numbers reported from the model without it); if not, it only memorises. R3c's own SG v2 numbers are
seen-data scores and must not be quoted as held-out. Note also: R3/R3b draw peak SNR 10-150 while exoALMA's high-resolution set is 3-10 (research.md 2026-10-08); R4 should reach ~3.
Local MPS smoke with `--train-sgv2` runs end to end (4 cubes; SG v2 loaded in 8 s).

## 2026-10-10 | bug | Round 2 (`unified-wiener-k15`) v3: not a single bad step but a divergence at step 1532; K=15 is unstable at lr 3e-4

v3 (OOM handler and skip logic in place) got past step 1532, which is what showed the real picture. Rounds 1-5 healthy (L1 0.046 to 0.030). At step 1532 the first non-finite loss; round 6 L1
jumped to 0.052 and the run NEVER recovered (0.045-0.054 for rounds 6-12; v1's K=3 run was at 0.018-0.025 on the same data); skips grew to 23 per round, then at step ~5000 every step was
non-finite or out of memory (51 in a row) and it aborted at round 13 of 40. Input ranges in the bad batches: most -3..+5 (normal) but a group at -30..+40, ~10x the usual,
which overflows fp16 in the 63-channel input of the K=15 model. Reading: the first event was an input outlier at the peak learning rate; the weights then moved to a worse region and the
network stayed marginal. The three earlier diagnoses (fp16 overflow, allocator fragmentation, graph held by the skip path) were each partial: the graph-hold was a real bug
(fixed) and the OOMs of the last rounds are its consequence, but the cause is the unstable run. The kernel's saved checkpoint is the degraded round-13 model, not round 5 (every
round overwrites it); noted for next time. Fix in `train_unified.py`: `--clip-in` clamps the network input (training batches and `denoise`; read back from the checkpoint at
evaluation), lr 1.5e-4 for K=15. K=3 (v1) did not show this at 3e-4. Relaunched as R3b `unified-wiener-real-k15`: the K=15 comparison now runs under the realistic-noise recipe of R3,
which is the comparison PLAN.md needs. Archived `results/unified/unified-wiener-k15_v3_2026-10-10_failed/`. No published number touched.

## 2026-10-10 | run | `sph-validation` v1: converter reproduces MCFOST's bulk rotation but not the wiggle, because of a sky-PA offset (found, v2 launched)

Speedie dump rendered on MCFOST's grid, beam and geometry (r0.5, GI and Keplerian twin, tau0 1-100). Archived `results/sph-validation/v1_2026-10-10_first_run/`.
As run: M1 correlation 0.92, M1 slope 0.95 (GI) / 1.00 (twin), velocity range +-1.12 vs +-1.13 km/s, so the physics and the length-scale / velocity scaling are right; the
spiral morphology of M0 is visibly similar. But rc_r against MCFOST's rotation-curve residual is -0.05 (GI) and ~0 (twin), and the residual rms is 230 m/s (twin 227) against clean's 69
(18) m/s: a large-scale dipole, not missing wiggle. tau0 from 1 to 100 changes nothing. Gate (PLAN.md B2, rc_r well above dirty's 0.65) failed as run.
Cause: the sky PA. Rotating the saved M1 map by -24 deg lifts the M1 correlation 0.91 to 0.995 and cuts the rms difference to MCFOST from 0.242 to 0.061 km/s (inside r < 330 px,
~1000 au). The discrete PA set I scanned (+-236.7, +180) did not contain it. A second flaw found while testing the fix on a planted offset: bulk M1 correlation cannot tell the four
(spin flip, mirror) combinations apart (all 0.99) but their spiral handedness differs and the wiggle correlation of the wrong one is ~0.03 against 0.93 for the right one, so the
combination must be chosen on the high-pass M1. Both fixed in `validate_sph_speedie.py` (continuous rotation scan per combination, then render-confirmed and refined to 1 deg) and
launched as v2. This is the sky-orientation CONVENTION, not a fit of physics: PA, spin and mirror are discrete or one-angle choices made once on the GI disc and reused for the twin.
No published number touched. The converter is not used for training until v2 passes.

## 2026-10-10 | code | SPH-to-cube converter (`src/data/sph_cube.py`, PLAN.md B1) and its validation against Speedie's MCFOST cube launched

Turns a Phantom dump into a line cube with radiative transfer along each line of sight: I(v) = sum_p T_p (1 - exp(-dtau_p)) exp(-tau_front), particles ordered
nearest-first per pixel, dtau from a Gaussian thermal profile (13CO mass) scaled to a target optical depth tau0 (median peak tau over the disc). An optically thick line therefore
shows the velocity of the layer facing the observer. Geometry knobs: inclination, PA, in-plane rotation, spin flip; `rescale` for scale-free (beta-cooled) runs. Leaves out:
SPH kernel smoothing inside a pixel, dust, freeze-out, non-LTE; MCFOST computes its own temperature by radiative equilibrium while we take the SPH temperature (Young dumps
carry `temperature`; Speedie and Longarini carry `u`). `tests/test_sph_cube.py` (PASS): on a synthetic Keplerian particle disc the M1 map is within 0.2 % (median) of the analytic
projected field, intensity never exceeds the hottest particle's T, the beam keeps flux, and a flipped / rotated render keeps flux to 5 %.
Validation (`experiments/validate_sph_speedie.py`, Kaggle CPU kernel `sph-validation`): render the Speedie dump with MCFOST's own settings (length scale 1.5, from the
parameter file's command line; 155.9 pc, i = 23.2 deg, PA 236.7 deg) on the clean cube's grid and beam, choose the sky-orientation convention (8 combinations of PA sense, mirror,
spin) by M1 correlation, then score our render's rotation-curve residual against MCFOST's (rc_r) for tau0 1, 3, 10, 30, 100, for the GI disc and its Keplerian twin (radial and
vertical velocity removed, as MCFOST -no_vr -no_vz). Local plumbing check on a self-rendered reference: the scan recovers the planted orientation (M1 corr 0.997) and the twin's rc
residual drops from 61 to 37 m/s. Gate (PLAN.md B2): rc_r well above dirty's 0.65 on Speedie; if not, the converter is not used for training.

## 2026-10-10 | code | Plan adopted (`results/PLAN.md`); workstream A (realistic noise) built and launched as R3 `unified-wiener-real`

The Speedie result below made the gate explicit: no new data until the learned stage survives a realistic noise regime. `results/PLAN.md` holds the pass bars, workstreams,
rounds and gates (Krishan approved 2026-10-10). Workstream A in `train_unified.py`, all behind flags that default off so earlier runs reproduce:
`--snr lo,hi` (peak SNR per redraw, log-uniform), `--chan-corr a_max` (training noise convolved along velocity with [a, 1, a], variance-preserving, a ~ U(0, a_max); a = 0.5 is
Hanning, lag-1 +0.67 / lag-2 +0.17, against Speedie's measured +0.64 / +0.15), `--beam-max` (circular Gaussian broadening of clean AND of the noise spectrum by r ~ U(1, beam_max) in the
noise-only tasks, the CLEAN-product regime). The Wiener step now always estimates the lag-1/lag-2 noise correlation blind (`velocity_corr`, off-source pixels picked on a map
smoothed to n/20 px) and uses the matching noise power along velocity frequency (`wiggle_scorecard.wiener3d`, optional `vspec`); on channel-independent noise it estimates ~0 and
reduces to the old filter. `tests/test_noise_realism.py` (PASS): estimator +0.660 / +0.151 for a true +0.667 / +0.167 and +0.000 / -0.006 for independent noise; the velocity-aware
Wiener has 31 % lower rms error than the channel-independent one on Hanning noise (0.0282 vs 0.0410, synthetic disk); a requested SNR of 40 comes out at 40.0. A first version ranked
pixels by their own noise and read lag-2 0.10 for a true 0.17 (selection bias), fixed before use. Not built: the do-no-harm gate (PLAN.md A3), deferred because the SNR draw
attacks the same failure; it is added if R3 still falls below Wiener at high SNR. Local MPS smoke with all three flags: runs end to end, zero-init identity holds.
R3 (`unified-wiener-real` v1): K=3 (direct comparison with `unified-wiener` v1), snr 10-150, chan-corr 0.5, beam-max 1.6, 12000 steps, then all holdouts at 5 sizes plus Speedie GI/Kep
r0.5/r1.5 at native pixels. Gate 1: Speedie M0/M1/M2 all above dirty. Speedie stays test-only. Note: the Wiener change also alters the Wiener input any OLD checkpoint sees at eval
if re-scored with this code; on the line-emission cubes (channel-independent noise) the estimate should be ~0 and the change negligible, not yet measured.

## 2026-10-10 | run | `eval-speedie` v1: on the realistic Speedie disk every learned model is WORSE than dirty on M0/M2 at low noise, and plain Wiener has the best wiggle

First real-physics, held-out, non-exoALMA test (Speedie et al. 2024 GI disk and its Keplerian twin, native 1024 px crop, 42 m/s channels, true geometry 155.9 pc / 23.2 deg; r0.5 = beam
0.237 x 0.175 arcsec, peak SNR 36, the exoALMA-like case; r1.5 = beam 0.39 x 0.27 arcsec, SNR ~135). Archived `results/speedie/eval_v1_2026-10-10/`. Moments are signal-masked,
noise-clipped improvements over dirty (RULES.md #6); one cube per case, so no spread; the 4 numbers per model are 4 cases.

| case | model | PSNR (dirty) | M0 % | M1 % | M2 % | invented_frac |
|---|---|---|---|---|---|---|
| GI r0.5 | Wiener (no learning) | 32.4 (24.3) | 44 | 60 | 61 | |
| GI r0.5 | unified-wiener | 41.0 | -36 | 71 | 58 | 0.13 |
| GI r0.5 | kin_mae_ft (mixed) | 41.8 | -66 | 4 | -193 | 0.15 |
| GI r0.5 | kin_mae_ft (nb16 original) | 40.9 | -692 | -61 | -219 | 0.20 |
| GI r1.5 | Wiener | 31.7 (28.3) | 20 | 29 | 45 | |
| GI r1.5 | unified-wiener | 32.7 | -167 | -165 | -952 | 0.20 |
| GI r1.5 | kin_mae_ft (mixed) | 34.4 | -2697 | -330 | -765 | 0.20 |
| GI r1.5 | kin_mae_ft (nb16 original) | 33.2 | -7969 | -1502 | -1317 | 0.22 |

Wiggle (mean of r0.5 and r1.5, `compare_models.py`): rc_r dirty 0.65, Wiener 0.91, unified-wiener 0.87, kin_mae_ft 0.60, nb16 kin_mae_ft 0.50. Keplerian twin, spurious rotation-curve
rms (m/s, 0 = invents nothing): Wiener -4.6, unified-wiener +7.0, kin_mae_ft +11.5, nb16 +15.6.

Reads. (1) PSNR rises for every learned model while M0/M2 collapse: PSNR is not the quantity that matters here. At r1.5 the learned models are far worse than dirty on all three moments
(M2 -615 to -1317 %), while the non-learned Wiener still improves them. (2) The failure is a regime shift, not a ranking: the r0.5 numbers for unified-wiener (M1 71, M2 58) are
usable, r1.5 is not. r1.5 differs from everything in training by a beam 1.65 x wider, a noise level 3.7 x lower relative to signal, and noise correlated between channels (+0.64 at lag 1,
which the training noise and the Wiener noise model both lack). Training only ever showed beams from the line-emission and sg_synth cubes and a noise level fixed at about half the
signal. (3) At r0.5 unified-wiener is ahead of plain Wiener on PSNR (41.0 vs 32.4) and M1 (71 vs 60), level on M2 (58 vs 61) and behind on M0 (-36 vs 44); at r1.5 plain Wiener is better on every moment. On wiggle (rc_r, mean of both noise levels) plain Wiener leads: 0.91 vs 0.87. (4) invented_frac 0.13-0.22
on a disk with true structure is high; the detector denominator was checked (RULES.md #8) with the Keplerian twin, which gets 0.13-0.23 too, so invented blobs are not specific
to the GI cube. Consequence: the learned stage must be trained on randomised beam width, noise level (SNR 10-150) and the measured channel correlation before it is ever
used on realistic cubes, and the do-no-harm blend (output = Wiener unless the correction is confidently better) moves from "nice to have" to required. Touches published numbers:
none of the earlier nb16 line-emission numbers change; it limits what they mean for real ALMA-like cubes (all had beam and noise level inside the training range).

## 2026-10-10 | bug | Round 2 (`unified-wiener-k15`) v2 died one step after a skipped batch: the skip path itself held the graph (CUDA OOM)

v2 (with the skip-non-finite fix) reproduced v1 exactly through round 5 (identical losses: the seed is fixed) and died at the same point, ~step 1532 in round 6, with
`torch.OutOfMemoryError` in the forward pass (`F.group_norm`; 14.51 GiB in use, 47 MiB free). Cause: the skip branch did `continue` without releasing `out` and `loss`, so
the skipped step's whole autograd graph stayed alive while the next forward pass allocated a second set of activations. The non-finite loss at step 1532 is the original
event (v1); the OOM was caused by my fix. A first write-up of this entry blamed allocator fragmentation from a "GPU Wiener" in the redraw: wrong, the Wiener runs in numpy on
the CPU (only the resampling uses the GPU), corrected the same hour. Fixes: the skip branch frees `out`/`loss` before continuing; a forward OOM is caught, frees the
graph and skips the step; a non-finite batch prints its min/max and shape (the cause of the step-1532 overflow is still unknown); `torch.cuda.empty_cache()` after each
redraw and `expandable_segments` (harmless, not the fix). v3 was launched before the graph-release line but with the OOM handler, which frees the same graph, so it
should get past step 1532 with two skipped steps. No published number touched.

## 2026-10-10 | run | `unified-ft-single` v1: fine-tuning the single-channel checkpoints on mixed data hurts line emission

Fine-tuned `winner_aug_res480_seed43`, `sweep_winner_p10_600` and `sg_k3_fresh` on the mixed data (same recipe as `unified-ft-kin`: lr 1e-4, 6000 steps).
Archived `results/unified/ft-single_v1_2026-10-10/` with per-size tables. Line = mean over the 5 line-emission holdouts (across cubes, not seeds); moments are
the signal-masked, noise-clipped improvement over dirty; rc_r the rotation-curve wiggle correlation.

| 600 px, line | PSNR | M0 % | M1 % | M2 % | rc_r | line-centre error m/s |
|---|---|---|---|---|---|---|
| `unified-wiener` (reference) | 46.25 | 33.0 | 85.4 | 83.3 | 0.70 | 49 |
| `kin_mae_ft` (mixed fine-tune) | 41.84 | 35.3 | 69.7 | 48.3 | 0.54 | 44 |
| res480 fine-tuned | 32.59 | -99.3 | 33.7 | -171.3 | 0.45 | 74 |
| p10_600 fine-tuned | 32.74 | -140.2 | 20.6 | -157.8 | 0.48 | 81 |
| sg_k3_fresh fine-tuned | 40.85 | -15.9 | 66.8 | -39.7 | 0.55 | 47 |

Reads. (1) The two single-channel models are refuted for the unified role: their d2 loss stays at ~0.034 for all 6000 steps (sg_k3_fresh: 0.005), so one
output channel cannot fit the velocity structure of the mixed data, and line-emission M0/M2 go strongly negative. Their only gain is SG v2 (rc_r 0.76 to 0.83).
(2) sg_k3_fresh improves on its own start (analytic rc_r 0.77 to 0.79, hydro PSNR 32.6 to 33.8, line M0 -49 to -16 % on the two shared cubes) but stays below
`unified-wiener` on every line metric. (3) SG v2 is not helped by anything: dirty already scores rc_r 0.85 there. The Wiener-fed model remains the base for
round 2 and later. Caveats: before/after JSONs cover different case subsets (2 vs 5 line cubes); 6 cases failed in every model with the old scorer
(`wiggle_fourier` polyfit on an empty track, guard already in the current code). Touches no published number. Checkpoints kept (RULES.md #12).

## 2026-10-10 | bug | Round 2 (`unified-wiener-k15` v1) died on one non-finite step

Step 1532 of 12000 (round 5/40, 33 min): `FloatingPointError: non-finite loss at step 1532: inputs finite=True`. The batch was finite, so
the overflow happened inside the fp16-autocast forward pass of the 63-channel model; the round losses before it (L1 0.030, d1 0.018,
d2 0.009) show no drift. The trainer's guard raised on the first bad step, which is stricter than GradScaler itself (that skips a step
with non-finite gradients and carries on). Fix in `train_unified.py`: skip the step (drop gradients, halve the loss scale), abort only
after 50 in a row or if the inputs are non-finite, average each round over finite steps, log the running skip count. Checked on MPS
by injecting two NaN losses into a smoke run: both skipped, run completed. Relaunched as v2 (accepted 07:14 IST when `unified-ft-single` freed its slot). Touches no published number: v1 produced only an unscored round-5 checkpoint. Archived
`results/unified/unified-wiener-k15_v1_2026-10-10_failed/`.

## 2026-10-10 | run | Speedie GI disk and its Keplerian twin prepared (Kaggle `speedie-prep-fast` v1, CPU): real ALMA-style noise is correlated between channels

58 min with 24 parallel range requests (the sequential first kernel managed 69 s per channel and would have needed ~20 h). Archived `results/speedie/` (noise report, log); the eight cubes
({GI,Kep}_r{0.5,1.5}_{clean,dirty}.fits, central 1024 px, 200 channels at 42 m/s) are in the kernel Output. Headers carry the WRONG geometry (162.9 pc / 23.5 deg; correct 155.9 pc / 23.2 deg / PA 236.7 deg, research.md
2026-10-10); the evaluation now ignores the header and uses the correct values.

| cube | beam (arcsec) | noise rms (mJy/beam) | peak SNR | noise correlation between channels, lag 1 / 2 / 3 | spatial corr. at 8 px | (dirty - clean) / noise, signal channels |
|---|---|---|---|---|---|---|
| GI, robust 0.5 | 0.237 x 0.175 | 2.12 | 36.4 | **+0.64** / +0.16 / 0.00 | 0.53 | 1.00 |
| Kep twin, robust 0.5 | 0.237 x 0.175 | 2.12 | 35.7 | +0.64 / +0.15 / 0.00 | 0.53 | 1.00 |
| GI, robust 1.5 | 0.390 x 0.274 | 1.26 | 137.7 | +0.64 / +0.15 / 0.00 | 0.79 | 1.05 |
| Kep twin, robust 1.5 | 0.390 x 0.274 | 1.26 | 131.5 | +0.64 / +0.15 / 0.00 | 0.79 | 1.07 |

Reads. (1) The truth is consistent: in the signal channels dirty minus the beam-convolved model is exactly at the noise level for robust 0.5 (5-7 % above it for robust 1.5, a small model / imaging mismatch).
(2) **Noise in a realistic ALMA-like cube is correlated between neighbouring channels: +0.64 at lag 1, +0.15 at lag 2, zero beyond** (whatever spectral response syndisk applies; real ALMA cubes are Hanning-smoothed
too). Everything we trained so far draws channel-INDEPENDENT noise, and the 3D Wiener filter's noise model is flat along velocity. Both are wrong for this regime: the fix is a velocity covariance in the training noise
and a noise spectrum along velocity in the Wiener step, both measured from line-free channels. (3) Peak SNR 36 (robust 0.5) sits inside exoALMA's fiducial 13-50; robust 1.5 (SNR ~135) is the easy end.
Next: score `unified-wiener`, the mixed-data `kin_gamma0_mae_ft` fine-tune and the original nb16 `kin_gamma0_mae_ft` on all four (kernel `eval-speedie`, queued for a GPU slot).

---

## 2026-10-10 | run | Fine-tuning kin_gamma0_mae_ft / starlet_ft on the mixed data (Kaggle `unified-ft-kin` v1): helps what it had never seen, does not beat the Wiener-fed model

Archived `results/unified/ft-kin_v1_2026-10-10/` (log, 4 eval JSONs). 6000 steps (48 min) + scoring (60 min) per checkpoint, lr 1e-4, dirty input in the checkpoint's own 31-channel format, data mix and sizes as in the
2026-10-09 plan entry. Evaluation here is at NATIVE size with no internal resize to 256 (nb16 resized), so absolute numbers differ from nb16's; compare within this harness. 6 cases per checkpoint failed to score (the
empty-spine-track bug, fixed since). One run, no seeds. Matched cases only (the "before" scoring covered 7 cases at 300 and 600 px), 600 px:

| case | metric | Wiener | kin_mae before to after | kin_starlet before to after | unified-wiener |
|---|---|---|---|---|---|
| run_0002_00560_rt_00 (line) | PSNR dB | 32.5 | 43.2 to 43.2 | 43.7 to 43.7 | 40.5 |
| | M1 % / M2 % | 33 / 24 | 57 to 71 / 43 to 85 | 69 to 72 / 81 to 87 | 63 / 79 |
| | line-centre m/s | 136 | 84 to 58 | 92 to 57 | 84 |
| run_0026_00005_rt_04 (line) | PSNR dB | 30.6 | 41.6 to 40.7 | 43.8 to 41.2 | 43.2 |
| | M1 % / M2 % | 69 / 61 | 63 to 84 / 15 to 73 | 87 to 86 / 77 to 75 | 86 / 81 |
| hydro 9074 | PSNR dB | 34.5 | 30.9 to 34.1 | 30.9 to 34.0 | 33.4 |
| | rc_r (dirty 0.13) | 0.16 | 0.50 to 0.43 | 0.43 to 0.39 | 0.37 |
| | line-centre m/s (dirty 308) | 293 | 258 to 189 | 255 to 187 | 263 |
| analytic an07 (deconv. / noise-only) | rc_r | 0.59 / 0.84 | 0.60 to 0.87 / 0.76 to 0.90 | 0.58 to 0.87 / 0.71 to 0.90 | 0.86 / n.a. |
| SG v2 | rc_r (dirty 0.85) | 0.86 | 0.63 to 0.78 | 0.59 to 0.75 | 0.83 |

Mean over all 5 line cubes after fine-tuning, 600 px: `kin_mae_ft` PSNR 41.8, M0/M1/M2 35 / 70 / 48 %, rc_r 0.54, line-centre 44 m/s; `kin_starlet_ft` 42.3, 38 / 72 / 57, 0.58, 42 m/s; the Wiener-fed `unified-wiener` 46.3, 33 / 85 / 83, 0.70, 49 m/s.

Reads. (1) Fine-tuning on the mixed data helps the starting checkpoints where they had never seen the data (analytic disks, hydro 9074 PSNR +3 dB, line-centre error down 25 to 40 %, M2 on both line cubes) and costs
line-emission PSNR on only one of two cubes (-0.9 and -2.6 dB). (2) It does not beat the model that is fed the Wiener cube: unified-wiener is ahead on line-emission PSNR (+4 dB), M1 (+13 to 15 points), M2 (+26 to 35) and
rc_r (+0.12 to 0.16), and on analytic M1 / M2 (81 / 87 vs 42 / 34); the fine-tunes keep the lower line-centre error on line cubes (42 to 44 vs 49 m/s), higher M0 (35 to 38 vs 33) and the lower line-centre error on 9074. (3) The original
`kin_gamma0_mae_ft` already had the best 9074 wiggle of anything scored (0.50), and fine-tuning lowered it (0.43): a single cube, so a lead. (4) SG v2 stays unsolved for every model: none reaches dirty (0.85) or Wiener (0.86).
Direction: the Wiener-fed arm is the base for round 2; the dirty-input fine-tunes stay as secondary candidates; `unified-ft-single` (winner_aug_res480, sweep_winner_p10_600, sg_k3_fresh) still running. Checkpoints (rule 12, models/README.md):
Kaggle kernel Outputs of `unified-ft-kin`, `unified-wiener`, `wiggle-correction`.

---

## 2026-10-10 | added | CPU-side work while the fine-tunes run: Speedie data prep, a spurious-wiggle metric, the model comparison script, the revised mentor deck; one correction

- **Speedie GI disk** (research.md 2026-10-09/10): `experiments/speedie_prep.py` (streams the deflate zips by HTTP range, crops 1024 px, builds beam-convolved truth, measures noise) with `tests/test_speedie_prep.py`
  (a 1 Jy point source peaks at exactly 1.000 Jy/beam, beam position angle, half-channel velocity resampling, noise-report recovery of a known lag-1 covariance, zip and range I/O against a local server: PASS).
  Runs as CPU-only Kaggle kernel `speedie-prep` (no GPU slot); output is attached to later kernels as a kernel source. `train_unified.py --speedie-dir --eval-speedie GI_r0.5,Kep_r0.5` scores them at native pixels
  (tested end to end on a toy stand-in; the real cubes are not yet in).
- **Spurious-wiggle metric.** `wiggle_scorecard` now also reports `rc_rms_ms` and `rc_rms_clean_ms`, the absolute rotation-curve residual rms in m/s for each method and for clean. On the Keplerian twin clean's is the floor, and a
  method's excess over it is structure the denoiser invented (`compare_models.py` prints it as "spurious m/s"). Correlation cannot say this: the twin has no wiggle to correlate with.
- **`experiments/compare_models.py`**: one table per size from the `eval_*.json` files (line / hydro / analytic / SG v2 / Speedie GI / Speedie Keplerian twin; PSNR, M0/M1/M2, rc_r, line-centre error), with dirty and Wiener as reference rows.
- **Mentor deck revised** (artifact "Recovering the GI Wiggle", now 9 slides): slides 2, 3, 4, 6, 8 rewritten and a new slide 7 (the unified model); slide 5 keeps its figure with a metric note. The Keplerian numbers (0.664, 0.925,
  0.733) are gone; every figure on the new slides is from PROGRESS.md 2026-10-08 to 2026-10-10.
- **Correction.** I wrote that notebook 14's from-scratch 600 px arms ranked "8th, 9th and 13th of 14" in notebook 16; those were mean-rank VALUES, not positions. By mean rank over 8 metrics they are 10th (`sweep_winner_p10_600`),
  12th (`sweep_winner_600`) and 14th (`sweep_winner_aug_600`). Fixed on the `native600-loss-sweep` branch (`80b9340`); the conclusion (native 600 from scratch has not helped) is unchanged and slightly stronger.
- **Speedie prep was 11x too slow, fixed (2026-10-10 ~05:30 IST).** The first kernel (`speedie-prep`, sequential range requests) read the model cube at 69 s per channel (0.23 MB/s) because
  Zenodo throttles PER CONNECTION: measured from the laptop, 1 request 0.26 MB/s, 8 parallel 1.72 MB/s, 16 parallel on real model-cube channel blocks 2.85 MB/s. Sequentially the job would have taken
  ~20 h and passed Kaggle's 12 h cap before producing the Keplerian twin (order was GI model, GI r0.5, GI r1.5, then Kep). `speedie_prep.py` now downloads with 24 parallel range requests, takes robust 0.5 for BOTH disks
  before robust 1.5, and downloads each zip to disk with a resumable chunk log; `tests/test_speedie_prep.py` covers the parallel crop, the parallel download (byte-identical), local-zip streaming and resume. Relaunched as
  `speedie-prep-fast` (CPU only); the original kernel is left running only until the fast one shows progress. Expected ~16 GB at ~3 MB/s, about 1.5 to 2 h, not yet confirmed on Kaggle's network.
- **Round 2 launched 2026-10-10 05:44 IST** (`unified-wiener-k15`): the same Wiener-fed model as `unified-wiener`, two changes at once (wider window 3 to 15 channels per side; each simulation also served as the noise-only task, p_ai 0.5), so a difference from v1 cannot be assigned to one of them; the ablations come later. Weekly GPU quota reset at about 05:30 IST Saturday.
- Housekeeping: the system cleared `/private/tmp` (scratch venv, the nb14 worktree); the branch was already pushed, the venv was rebuilt, the worktree re-added.

---

## 2026-10-10 | run | The learned correction works: leave-one-out and `unified-wiener` both finished (Kaggle `wiggle-correction` v2, `unified-wiener` v1)

Logs read 2026-10-10 00:10 IST; archive folders not yet written (RUNS.md rows pending, code not committed, see provenance caveat in the 2026-10-09 run entry).

**`wiggle-correction` v2** (Wiener + U-Net correction, 5 folds x 9000 steps, 605 min). Trained on the 4 other hydro disks + analytic an01-an06 (the 101-channel analytic cubes),
scored on the fold's disk, SG v2, an07, an08, none of them trained on. rc_r (rotation-curve wiggle), one run, no seeds:

| held out | dirty | Wiener | **corrected** | oracle (noise-free beam) | median line-centre error, Wiener to corrected |
|---|---|---|---|---|---|
| 9074 | 0.13 | 0.16 | **0.44** | 0.93 | 293 to 172 m/s |
| 9025 | 0.13 | 0.21 | **0.44** | 0.80 | 76 to 48 m/s |
| 9015 | 0.04 | 0.21 | **0.24** | 0.82 | 59 to 39 m/s |
| 9032 | 0.68 | 0.64 | 0.65 | 0.90 | 51 to 46 m/s |
| 9019 | 0.71 | 0.52 | **0.74** | 0.98 | 70 to 61 m/s |
| SG v2 (5 fold models) | 0.87 | 0.89 | 0.85-0.87 | 0.92 | 37 to 46-49 m/s (worse) |
| an07 / an08 (5 fold models) | 0.43 / 0.46 | 0.70 / 0.60 | 0.90-0.93 / 0.89-0.93 | 0.97 / 0.95 | 36 to 17-22 / 30 to 16-19 m/s |

First model to beat Wiener AND dirty on the two hard hydro disks (9025, 9074), the stated success bar (PROGRESS.md 2026-10-08). Still far from the oracle on the hydro disks
(0.44 vs 0.93 on 9074) and, on SG v2, no better than dirty. Gains on the analytic disks are large but those are a smooth single-mode family close to the training analytic disks.

**`unified-wiener` v1** (one model, all three data kinds, 12000 steps in 163 min, window +-0.3 km/s, deconvolution-task data only; built BEFORE the noise-only variants).
Scored at 256/300/320/480/600 px. Line-emission holdouts, mean of the cubes available at each size (4 at 256 and 300: run_0002_00560_rt_04 failed there, see below):

| size | PSNR dirty / Wiener / model | M0 / M1 / M2 % (Wiener) | M0 / M1 / M2 % (model) | rc_r dirty / Wiener / model |
|---|---|---|---|---|
| 256 | 20.3 / 30.5 / 41.3 | 20 / 54 / 46 | 20 / 81 / 84 | 0.26 / 0.51 / 0.79 |
| 320 | 20.1 / 30.5 / 43.2 | 19 / 59 / 50 | 27 / 85 / 86 | 0.21 / 0.40 / 0.73 |
| 480 | 20.1 / 30.5 / 45.2 | 19 / 59 / 51 | 32 / 86 / 85 | 0.22 / 0.38 / 0.69 |
| 600 | 20.1 / 30.5 / 46.3 | 19 / 59 / 51 | 33 / 85 / 83 | 0.21 / 0.39 / 0.70 |

Against notebook 16's best checkpoints on the same 5 holdouts (nb16 v6, its own Keplerian wiggle metric so the wiggle columns do not compare): `kin_gamma0_mae_ft` PSNR 45.7, M0 28, M1 78, M2 54;
`kin_gamma0_starlet_ft` 47.1, 30, 83, 69. At 600 px this single from-scratch model has 46.3, 33, 85, 83: level on PSNR and M0, ahead on M1 and M2, and it holds all five sizes.
The wiggle on line emission (rc_r) rises from Wiener's 0.39-0.51 to 0.70-0.79; only `run_0026_00005_rt_04` does not gain over Wiener (0.84 vs 0.82-0.83).

Weak spots, same run: SG v2 (the 0.033 km/s hydro cube) is not helped (rc_r 0.83-0.88, equal to dirty, 0.01-0.02 below Wiener; M1 improvement turns negative at 480/600, -13 and -18 %);
hydro 9074 PSNR falls below Wiener (33.4 vs 34.5 at 600) while rc_r rises 0.16 to 0.37-0.45, and its wiggle gain shrinks with size (0.45 at 256 to 0.37 at 600);
M0 is negative on run_0002_00560_rt_00 (-10 to -32 %), the cube where dirty is already good. Four cases FAILED in scoring (`run_0002_00560_rt_04` at 256 and 300, `an08` at 300 and 320):
`np.polyfit` on an empty spine track in `wiggle_scorecard.wiggle_fourier` (the fitted geometry left no channel within 1 km/s of v_sys or R_out < R_in at that size). Fixed 2026-10-10.

**Reads.** (1) A learned correction on top of Wiener is the first thing that recovers part of the hydro wiggle on unseen disks. (2) One model covers all five sizes with the wiggle gain stable for
line emission and the analytic disks, so no per-resolution models are needed on this evidence. (3) The remaining problem is SG v2 and the faint hydro disks; the window (+-0.3 km/s), more GI data
and the noise-only variants are the changes aimed at that.

---

## 2026-10-09 | plan | Fine-tune the best existing checkpoints on the unified data mix, plus fresh training of the same architectures; ALMA-matched noise-only task added

**Decision (user, 2026-10-09).** Use several existing checkpoints and train them again on the data the plan calls for, then go to exoALMA (evaluation only: Jason
does not want any model trained on exoALMA data). Fresh training of the same architectures is wanted too. Any checkpoint on the Kaggle account may be used.

**Shortlist** (from the nb16 v6 ranking over 14 checkpoints, 7 cubes, 8 metrics; mean ranks in PROGRESS.md / this session):
`kin_gamma0_mae_ft` (rank 5.1, best line-emission wiggle 0.757), `kin_gamma0_starlet_ft` (best M1/M2: 82.7 / 69.0 %), `winner_aug_res480_seed43` (best overall rank 4.75),
`sweep_winner_p10_600` (nb14's native-600 representative: nb16 SG v2 wiggle rank 2 of 14, low line-emission moments), `sg_k3_fresh` (not in nb16; best learned model on the
rotation-curve wiggle on unseen disks, confirmed by the user's own reading). `winner_res320_seed42` is not in any dataset on the account; skipped. nb14's native-600 arms
(PSNR 37.2 / 37.2 / 35.2 on 600 px images; `winner_patch_600` 30.18) are not a clear win over the 256 px models.

**New in `experiments/train_unified.py`:**
1. *Arm "legacy" (`--init CKPT`)*: the checkpoint keeps its own input/output format (dirty window with min-max shared from the centre channel; one centre channel out, or the
   whole window out for the 31-channel kin_gamma0 family) and is trained further with the same data mix, sizes and losses (L1 + first and second velocity differences).
   `--fresh` keeps the architecture and discards the weights. Window sizes are set in km/s so 0.033 km/s cubes see the same velocity span.
2. *ALMA-matched noise-only variant of every simulation* (`--p-ai`, default 0.5; held-out cases also scored as `<name>_ai`): real ALMA products (and exoALMA's fiducial set) are
   CLEAN images in which the beam is already inside the clean target, so the task is noise-only. sg_synth and the analytic pairs add a second beam (a deconvolution task); the
   variant uses clean + noise at the pair's own measured noise spectrum, no extra beam. One smoke-run lead (an07 only, 4 steps, 128 px): Wiener lifts the rotation-curve wiggle
   score from 0.61 (dirty) to 0.84 on the noise-only version, which it did not do on the deconvolution version. A lead, not a result.
3. Local smoke tests run on the Apple GPU (MPS) for the network (resampling stays on CPU: MPS cannot area-pool 600 to 256); MPS and CPU gave identical scores on the same smoke run.
4. `--before`: scores the starting checkpoint under the new protocol (subset, sizes 300 and 600) so fine-tune gains are measured against the same cubes.

**Smoke results (8 steps, 128/160 px, tiny model budget; plumbing only):** kin_gamma0_mae_ft (stack output) and sg_k3_fresh (single-channel, three-centre loss) both train and score.
The starting checkpoints look bad at 128/160 px (moments strongly negative), because those sizes are far from the 256 px they trained at; that is the mismatch the all-sizes
fine-tuning is meant to remove, and says nothing about their real quality.

**Kernels** (private, GPU, code via the `exxa-wiggle-code` dataset, checkpoints via `exxa-unified-ft-sources` (kin_gamma0_mae_ft / starlet_ft as `.ckpt`, RULES.md #3),
`exxa-13-checkpoint-sources` (sg_k3_fresh), `exxa-05-v38-recovered` (res480) and the nb14 kernel output (sweep_winner_p10_600)):
`unified-ft-kin` (2 checkpoints), `unified-ft-single` (3 checkpoints), 20 rounds x 300 steps each at lr 1e-4, before and after scoring on 12 cases x 5 sizes.
GPU slots: Kaggle allows 2 at once and both are in use (`wiggle-correction` LOO, ~7 h in; `unified-wiener` narrow-window control, ~1.5 h in), so an auto-launcher
pushes each kernel when a slot frees. Note `unified-wiener` v1 was built before the noise-only variants existed, so it only informs the deconvolution task and the window comparison.

---

## 2026-10-09 | plan + bug | One model for line emission + SG + analytic, scored on PSNR, moments AND the wiggle at 256/300/320/480/600 px

**Request (user, 2026-10-09).** The trained models score well on PSNR, M0/M1 and dispersion (M2) and fail the wiggle. Retrain on line
emission plus self-gravitating plus wiggle-bearing data, ready for ALMA, as a few properly trained checkpoints that hold at 256, 300, 320,
480 and 600 px and are best on everything at once.

**Design** (`experiments/train_unified.py`, Kaggle `kaggle/unified/`):
- Data: 9 line-emission cubes (cube_split train + val; the standard 5 holdouts never trained on), hydro 9015/9019/9025/9032 (9074 held
  out), analytic an01-an06 (an07/an08 held out), sampled one third each. Fresh noise every round from each cube's own measured 2D noise
  spectrum (the pair residual), so line emission keeps its beam-correlated noise (the CLEAN / real-ALMA regime) and the SG cubes their
  beam-shaped noise.
- Resolution: every draw is resampled to a random size from {256, 300, 320, 480, 600}; scoring is done at each of the five.
- Two arms (checkpoints): `wiener` = native-grid 3D Wiener + zero-initialised U-Net correction; `direct` = dirty + correction (ablation).
- Loss: L1 on three adjacent channels + L1 on the first velocity difference (line centre: M1, wiggle) + L1 on the second (line width: M2).
- Scoring per holdout x size: `checkpoint_eval.score` (PSNR, SSIM, M0/M1/M2 signal-masked, artifacts; the nb16 definitions) and
  `wiggle_scorecard.score_cube` (rc_r, finger coherence, line-centre error, Terry spine), each against dirty and Wiener on the same cube.

**Bugs found while building it (both would have produced wrong numbers, neither reached a published table):**
1. *Edge-channel continuum subtraction destroys a deconvolving Wiener filter.* Subtracting the mean of the edge channels adds the same
   noise pattern (sigma / sqrt(16)) to every channel; the 3D Wiener reads a channel-constant pattern as signal and the beam deconvolution then
   amplifies it: an07 Wiener PSNR 34.5 dB without it, 15.6 dB with it. Fix: continuum is removed only from line emission (real continuum, no
   deconvolution there, where it costs nothing: run_0002 Wiener 23.9 to 33.2 dB); the simulations have none and are left alone.
2. *The hydro disks have line flux in their edge channels* (9074: 92% of peak in the first/last 8 channels; the old analytic cubes 12%), so an
   edge-channel noise estimate is invalid for them. SG noise stays on `blind_noise2d` (quietest channels at high k). The analytic disks are
   now generated with 161 channels (+-8 km/s) so their edges are line-free; **the LOO run (wiggle-correction v2) used the earlier 101-channel
   analytic cubes**, so its analytic numbers are not on the same cubes as anything from the unified run.
3. *Wiener on a resampled cube amplifies the resampling's own artifacts* (an07 at 256 px: 1.3 dB; 9074 upsampled to 600: 19.1 dB). Fix: Wiener
   runs once on the native grid, where beam and noise are defined, and the dirty and Wiener cubes are then resampled to the test size. Verified:
   an07 29.2 to 34.5 dB, 9074 21.3 to 33.3 dB, run_0002 23.9 to 33.2 dB at every one of the five sizes. A real cube is never resampled, so this
   is also the ALMA path.

Early sign to watch: on line emission Wiener lowers the wiggle score (run_0002 at 128/160 px, rc_r dirty 0.76, Wiener 0.69), so on that data
the learned stage starts below dirty on the wiggle.

| Kaggle kernel | ver | mode | status |
|---|---:|---|---|
| `unified-smoke` | 1 | wiener, 2 rounds x 40 steps, eval run_0002 + an07 at 256/600 | COMPLETE in 14.8 min, plumbing ok (14 line cubes found, 19 training cubes loaded in 7 min, eval at both sizes) but **loss NaN from step 1**: hydro 9019 is in different units (peak 1.4e10; every other cube < 0.3), which overflowed the float16 cache. Its model numbers are meaningless. Fixed: caches normalised by each cube's Wiener peak before the float16 cast, and a non-finite loss now stops the run. Verified locally on 9019 (finite, targets <= 1.42). The LOO run used float32 caches and is not affected. |
| `unified-wiener` | 1 | full: 40 rounds x 300 steps, 5 line holdouts + 9074 + SG v2 + an07/an08 x 5 sizes | launched 2026-10-09 ~20:00 IST, ~4 h |
| `unified-direct` | - | same, direct arm | waiting: Kaggle allows 2 GPU sessions; launches when the LOO run ends |

---

## 2026-10-09 | run | Learned correction on 3D Wiener, leave-one-hydro-disk-out, on Kaggle (`krishanyadav333/wiggle-correction`)

New script kernel (`kaggle/wiggle_correction/`), private, GPU, no GitHub-linked notebook so RULES.md #2 (cells vs `src/`) does not apply.
Trainer `experiments/train_wiggle_correction.py`: out = Wiener(a=2) + U-Net([7-channel dirty window, 7-channel Wiener window,
noise-level map]), last conv zero-initialised (checked: step-1 L1 equals Wiener's L1 exactly, so the untrained model IS Wiener),
loss = L1 to clean on three adjacent centres + L1 on their central velocity difference, fresh noise every round (clean (*) beam +
beam (*) new noise at each cube's measured level), fixed budget (30 rounds x 300 steps, batch 12 x 3 centres, 128 px crops, base 48,
AdamW one-cycle 3e-4, fp16), no early stopping, no selection on the holdout. The spectral window is set in km/s so the 0.033 km/s
cubes (9019, SG v2) see the same velocity span as the 0.1 km/s ones. Per fold: train on the 4 other hydro disks + analytic an01-an06;
held out and scored with the scorecard: the fold's hydro disk, SG v2, and analytic an07/an08 (never trained on).

**Provenance caveat.** Tracked files come from a clone of `midterm-prep` at launch time; `analytic_gi.py`, `wiggle_scorecard.py`,
`make_analytic_gi_cubes.py` and `train_wiggle_correction.py` were not committed, so they ride in the private dataset
`krishanyadav333/exxa-wiggle-code` (MANIFEST.json holds sha256 sums; local HEAD 5b04466 plus those files). Commit them and cite the
SHA in RUNS.md once the run lands. The checkpoint of every round is written straight to `/kaggle/working/wiggle_correction/` (RULES.md #1).

| Kaggle ver | mode | outcome |
|---:|---|---|
| 1 | smoke (fold 9074, 2 rounds x 40 steps, eval 9074 + an07), 2.7 min | COMPLETE. Plumbing verified on GPU. After only 80 steps rc_r: 9074 Wiener 0.158 to **0.257** (dirty 0.132), an07 Wiener 0.697 to **0.814** (dirty 0.434); line-centre median 293 to 242 m/s and 36 to 29 m/s. One fold, a tiny budget, one run: it shows the pipeline works and the sign, not a result. |
| 2 | full, 5 folds x 9000 steps, eval 4 cubes per fold | launched 2026-10-09 ~14:00 IST; expected ~5 h; log is only readable after it ends |

A first attempt cloned the repo into `/kaggle/working` and put the whole 1.5 GB clone in the Output; version 2 clones to `/tmp`.

---

## 2026-10-09 | finding | Existing checkpoints re-scored on the rotation-curve metric: none comes near the noise-free beam (0.89); the high-SNR SG v2 cube is where every learned model loses to dirty

`experiments/wiggle_scorecard.py` now runs `winner_aug_seed43`, `sg_k3_fresh` and `kin_gamma0` (each through its own training
preprocessing, via `wiggle_domain_split.py`'s loaders) beside dirty, 3D Wiener a=2 and the noise-free beam oracle, on all 12 cubes
(5 hydro `sg_synth`, SG v2, 6 analytic). Output `results/wiggle-scorecard/2026-10-09/` (scorecard.json / .md / run_log.txt / 12 PNGs; one run, no
seeds, ~3 h wall clock including a laptop sleep). `rc_r` = correlation with clean of M1 minus clean's own rotation curve, R >= 2 beams.

| method | hydro 5 disks (mean +/- std across disks) | SG v2 | analytic 6 (mean +/- std) | paired wins vs dirty (hydro / SG v2 / analytic) |
|---|---|---|---|---|
| noise-free beam (oracle) | 0.885 +/- 0.076 | 0.916 | 0.936 +/- 0.060 | n/a |
| dirty | 0.341 +/- 0.329 | 0.873 | 0.421 +/- 0.181 | n/a |
| 3D Wiener a=2 | 0.349 +/- 0.218 | 0.892 | 0.557 +/- 0.292 | 3/5, 1/1, 5/6 |
| winner_aug_seed43 | 0.377 +/- 0.225 | 0.792 | 0.397 +/- 0.171 | 3/5, 0/1, 1/6 |
| sg_k3_fresh | 0.496 +/- 0.246 † | 0.813 | 0.537 +/- 0.231 | 4/5 †, 0/1, 6/6 |
| kin_gamma0 | 0.446 +/- 0.232 | 0.770 | 0.424 +/- 0.225 | 4/5, 0/1, 3/6 |

† `sg_k3_fresh` was trained on 9015, 9019, 9032 and validated on 9025 (RUNS.md 12-sg-spectral-context); only 9074 is a genuine holdout for it
(rc_r 0.362 vs dirty 0.132). Its hydro mean is inflated by the disks it saw; the analytic disks and SG v2 are unseen for it. `winner_aug` and
`kin_gamma0` were trained on line emission only, so every cube here is unseen for them. For SG v2 the static noise pattern was removed before any
method ran, which if anything favours the learned models.

What it says:
1. **The target is not met by anything we have.** The best honest hydro number is Wiener / kin_gamma0 / winner_aug at 0.35-0.45 against the oracle's 0.885 and
   dirty's 0.341. Per-disk wins are the fair read: the across-disk std (0.2-0.3) is as large as every between-method gap on 5 disks, so means alone do not
   rank the methods. 9015 and 9025 (dirty rc_r 0.04 and 0.13) stay at 0.08-0.35 for every method.
2. **On SG v2 (dirty already 0.873, oracle 0.916, headroom 0.043) every learned model is worse than dirty** (0.77-0.81) and Wiener is the only method above it
   (0.892). The learned models lose where the data are good, which is the standing finding (RUNS.md 12) now confirmed on the finger metric.
3. **sg_k3_fresh transfers best of the learned models**: 6/6 on the analytic disks (0.537, level with Wiener's 0.557, unseen for it) and the best
   `spine_r` among learned models on hydro, but it loses on SG v2 and its single clean hydro holdout is a win. A spectral window (k=3) plus SG-domain
   training is the only learned recipe that helps on cubes it has not seen; line-emission-only training (winner_aug, kin_gamma0) does not (winner_aug 1/6 on
   analytic, kin_gamma0 3/6).
4. **Line centres are still 5 to 10 times off the 10 m/s spec on hydro disks**: median |dM1| 14 m/s for the beam oracle, 76-135 m/s for dirty and every method
   (best sg_k3_fresh 76, contaminated; Wiener 110, kin_gamma0 90, winner_aug 111).
5. Metric reads: `spine_r` (Terry spine agreement) ranks Wiener first on hydro (0.634) and dirty near the bottom (0.305); A_wiggle is inflated 2-4x by dirty on the
   analytic disks (see the analytic entry below), so a wiggle amplitude measured on a raw dirty cube is not usable without denoising.

**Published numbers this touches.** `models/best_models/README.md` wiggle column (resid_r 0.760 / 0.681 / 0.8155 / 0.568-0.587), RUNS.md 08/09/10/11/12 wiggle
numbers, and the deck slides quoting them: all Keplerian `resid_r`, superseded for ranking by the `rc_r` table above (their own numbers stand as measured).
Rank change worth noting: under `resid_r` kin_gamma0 was "the strongest in-domain wiggle result" and DDRM the worst; under `rc_r` kin_gamma0 is mid-pack
(0.446 hydro, 0.770 SG v2). DDRM and `winner_beam` / `winner_p10` / `ddpm_seed42` were not re-scored here (DDRM needs the sampler; the rest are untested
single-channel U-Nets of the winner_aug family).

Next: the learned correction stage has to beat Wiener on `rc_r` on 9015/9025/9074 and not fall below dirty on SG v2; start from `sg_k3_fresh`'s recipe (SG-domain
data plus spectral window), add Wiener as input and the analytic disks as extra training volume, validate leave-one-out on the five hydro disks.

---

## 2026-10-09 | added + finding | Longarini's analytic GI model: exact-truth test disks; 3D Wiener beats dirty on 4 of 6 analytic disks but the hydro disks stay hard

**Arrival.** `giggle_functions.py` (C. Longarini; the "new GI wiggle function"), found in `~/Downloads`, copied unmodified to
`references/longarini_giggle_functions.py`. It computes the rotation curve of a self-gravitating disk (Keplerian plus a disc term through
elliptic integrals) and adds an m-armed log-spiral velocity perturbation that scales with q = md/ms and beta^(-1/2); the sky velocity is
(u_phi cos(phi) + u_r sin(phi)) sin(i). Run as shipped it needs `from scipy import ndimage` and fails on scipy >= 1.14 (`simps` removed). Bugs left
unported, none touch the velocity path: `get_masses` is an `if/if/else` that prints an error for statements -1 and 0 and uses `b[j]` with `j` undefined
when `statement == 0`; `amplitude_central_channel` calls `momentone` without `off`; the `*eta` variants have no azimuthal spiral term; the sign of the
log phase differs between `ur` and `urC`; the plot labels are pixel-hardcoded. Ask Cristiano whether a fixed version exists.

**Ported** as `src/data/analytic_gi.py` (vectorised, same physics; the emission model, beam and noise are ours and are documented as chosen, not derived),
`tests/test_analytic_gi.py` (PASS: Keplerian limit at md to 0 within 1e-6, disc speeds the rotation up 2% / 7% / 16% at 30 / 100 / 250 au for md = 0.35,
the quadratic estimator recovers the exact v_los to 0.1 m/s on an unbeamed cube, so the estimator is not what limits any score) and
`experiments/make_analytic_gi_cubes.py`. Six disks `an01`-`an06` (md 0.15-0.60, m 2-3, beta 3-5, pitch 13-20 deg, i 20-30 deg), laid out like
`sg_synth` (301 px, 1.99 au/px, 101 channels at 0.1 km/s, clean carries the hydro cubes' own 16.1 x 12.9 px beam, dirty = recovered beam (*) clean +
recovered beam (*) noise). Noise is scaled per disk so rmsdiff/rms = 0.44-0.52, inside the hydro band (0.41-0.57); at the sg_synth recipe's
NOISE_FRAC = 0.35 it came out 0.78-0.95 because the emission is compact, so that was changed and is the one knob that differs. Cubes live in
`self-gravitating cube and dirty cube/sg_analytic/` (gitignored, regenerate with the script; fixed seeds 1000+k), plus an exact `*_vlos_truth.npy`.

**Scorecard on the six analytic disks** (`results/wiggle-scorecard/2026-10-09/`, same metrics as 2026-10-08; mean +/- std across disks, no seeds):

| method | rc_r | rc_coh (40-160 au) | median abs line-centre error | spine_r | A_wiggle / clean |
|---|---|---|---|---|---|
| noise-free beam (oracle) | 0.936 +/- 0.060 | 0.961 | 11 m/s | 0.93 | 0.85 |
| dirty | 0.421 +/- 0.181 | 0.515 | 58 m/s | 0.09 | 2.9 |
| 3D Wiener a=2 | **0.557 +/- 0.292** | 0.578 | 28 m/s | 0.32 | 1.36 |
| 3D Wiener a=4 | 0.322 +/- 0.263 | 0.337 | 40 m/s | 0.21 | 1.07 |

Per disk, Wiener a=2 vs dirty on rc_r: an02 0.629 vs 0.428, an03 0.669 vs 0.462, an05 0.709 vs 0.485, an06 0.904 vs 0.684 (wins), an04 0.338 vs 0.328
(tie), an01 0.092 vs 0.139 (loses; md = 0.15, the weakest wiggle, noise dominates: dirty's rotation-curve residual has 6.0x clean's rms).

What it says:
1. Same ordering as the hydro disks on the oracle (beam alone keeps 0.94) and on dirty (noise destroys it), so the analytic disks are a fair
   stand-in for the noise problem, with an exact truth.
2. Here Wiener helps (0.557 vs 0.421, 4 wins, 1 tie, 1 loss); on the five hydro disks it did not (0.349 vs 0.341). The analytic spiral is one smooth mode,
   which a linear filter can keep; the hydro fingers are messier and faint. So Wiener's hydro failure is a statement about hydro structure, and a learned
   stage trained on analytic disks alone would be tested on an easier problem than the target. Use analytic disks for training volume and scoring on
   them as a unit test, but select on the hydro disks.
3. On an03 (`an03.png`) Wiener alpha=2 turns the noisy spine of dirty into a nearly straight one: the clean spine swings by about +/-20 au over 170 au,
   Wiener's stays within a few au (spine_r 0.44), so its A_wiggle ratio near 1 (25 vs 22 au) is not evidence the spine wiggle survived. Read A_ratio together
   with spine_r. alpha=4 oversmooths (rc_r 0.32).

**Published numbers this touches:** none new. Caveat for anyone quoting these: the analytic velocity amplitude is weak (spiral rms 0.07 km/s inside the
mask for md = 0.35, i = 30) next to the hydro cubes, the rotation-curve metric is what keeps it measurable.

---

## 2026-10-08 | bug + finding | GI wiggle scorecard: the Keplerian resid_r mostly scores the axisymmetric rotation curve, not the fingers; on the finger signal, 3D Wiener does not beat dirty

`experiments/wiggle_scorecard.py` (CPU, ~1 min for all six cubes) scores any method on every GI cube
we have, the same way, with figures. Output `results/wiggle-scorecard/2026-10-08/` (scorecard.json,
scorecard.md, one PNG per cube with M0, M1, residual, systemic channel + Terry spine, spine curve vs
clean; run_log.txt). Cubes: the 5 `sg_synth` disks and SG v2 (static pattern removed). The shipped
`_v3_extract` run_sg_* pairs are left out: run_sg_00019's dirty equals its clean (relative difference 3e-12
on channel 300); the other four differ from their clean by only 0.3 to 30 % rms, very low noise (corrected 2026-10-10, first
written as if all five were identical), so none separates methods.

**The bug (how caught).** On 9015 the clean cube's own Keplerian residual (M1 minus the fitted
Keplerian, the `compare_wiggles()` definition used since 2026-08-28) is a global dipole, not
fingers: a q = 0.25-0.75 disk is not Keplerian (self-gravity and pressure bend the rotation curve,
and the emission surface is elevated). Any smooth cube reproduces that dipole, so resid_r is high
for anything that keeps the large-scale velocity field. The scorecard therefore adds `rc_*`: subtract
the clean cube's own rotation curve v_phi(R), fitted freely per beam/2-wide radial bin, so only
non-axisymmetric motion is left. Clean's rc residual then shows the spiral fingers directly (9015,
9025 figures); 3D Wiener's shows an m = 2 butterfly that clean does not have, plus a monotonic drift
of the systemic-channel spine. R < 2 beam FWHM is excluded from rc metrics (beam-smeared even in clean).

| cube | beam_only (oracle) rc_r / coh | dirty rc_r / coh | Wiener a=2 rc_r / coh | Wiener a=4 rc_r | old Keplerian resid_r dirty / Wiener a=2 |
|---|---|---|---|---|---|
| 9015 | 0.815 / 0.871 | 0.040 / 0.087 | 0.214 / 0.381 | 0.206 | 0.838 / 0.992 |
| 9019 | 0.980 / 0.980 | **0.715** / 0.685 | 0.522 / 0.311 | 0.275 | 0.996 / 0.995 |
| 9025 | 0.801 / 0.792 | 0.134 / 0.226 | 0.208 / 0.275 | 0.141 | 0.963 / 0.995 |
| 9032 | 0.902 / 0.897 | **0.683** / 0.638 | 0.642 / 0.606 | 0.537 | 0.725 / 0.695 |
| 9074 | 0.925 / 0.953 | 0.132 / 0.127 | 0.158 / 0.116 | 0.091 | 0.245 / 0.298 |
| mean of 5 (std across disks) | **0.885 +/- 0.076** | 0.341 +/- 0.329 | 0.349 +/- 0.218 | 0.250 +/- 0.175 | 0.753 / 0.795 |
| SG v2 | 0.916 / 0.944 | 0.873 / 0.911 | **0.892** / 0.925 | 0.875 | 0.998 / 0.998 |

coh = Fourier coherence with clean's residual at 40-160 au (Terry+2022's lambda_wiggle range).
`beam_only` is clean (*) beam with no noise: an oracle, not a method. One run, deterministic (no seeds).

What it says:
1. **Noise, not resolution, destroys the wiggle on the synthetic disks.** The noise-free extra beam
   keeps rc_r 0.80-0.98; the same beam plus noise drops it to 0.04-0.72. A denoiser has headroom
   from 0.34 to ~0.89 mean, before any deconvolution.
2. **3D Wiener does not recover it.** Mean rc_r 0.349 vs dirty 0.341. It wins 3/5 but only where both
   are below 0.22 (9015, 9025, 9074), and loses where dirty is decent (9019 0.715 to 0.522, 9032 0.683
   to 0.642). Wiener's large old gains (9015 0.838 to 0.994,
   9025 0.334 to 0.657) were the axisymmetric part. Its real gains are in amplitude and line centres:
   median |dM1| 116 to 59 m/s on 9015, 136 to 76 on 9025; spine_r 0.31 to 0.63 mean.
3. **Line centres are far from exoALMA's ~10 m/s need.** Median |M1 - M1_clean|: beam_only 5-30 m/s,
   dirty 41-308, Wiener 37-293; 15-82% of dirty mask pixels are off by more than 100 m/s.
4. Terry+2022 spine (Eq. 3, both halves, 20-60 channels within 1 km/s of vsys): dirty's spine jumps
   between noise peaks (spine_r 0.21-0.52, A_wiggle 1.2-4.7x clean's: the noise reads as wiggle).
   Wiener smooths the jumps (spine_r 0.39-0.88) but over-estimates A by 1.1-1.7x on 4 of 6 cubes.

**Published numbers this touches.** Every Keplerian `resid_r`/`compare_wiggles()` number since
2026-08-28: the 2026-10-02 entries below (3D Wiener 0.925, the 5-disk table 0.733 vs 0.628, "4/5 wins"),
`research.md` 2026-10-02, `results/RUNS.md` sections 09-wiggle-scoring, 10-sg-training, 11-sg-loo
(sg_loo_wiggle.json), 12-sg-spectral-context, and `models/best_models/README.md`'s wiggle resid_r
column. They were computed as stated; what they measure is mostly the axisymmetric velocity field,
so rankings among methods that keep it (dirty, Wiener, beam-only) are unreliable. Not yet re-scored:
U-Net / DDRM / kin_gamma checkpoints under rc_r (next step; the scorecard takes any cube).

Deviations from Terry+2022 and limits, stated in the script docstring: y_n linearly detrended before
the FFT; R_in = 2 beam FWHM (they used h/H < 0.25 from the SPH); R_out = 95% of clean M0 (they used 95%
of disc mass), clipped to the image; their centroid window (5 px at 1.33 au) is converted to au. Track
lengths 220-400 au make the Fourier grid coarse (self-test: a 12 au, 80 au sinusoid comes back as 10.6 au,
73.6 au), so A_ratio and lam_ratio are comparisons between methods, not absolute Terry amplitudes.
The rotation curve is fitted on clean (shared model, as before); on real data it must come from the
cube being scored.

Consequence for the plan: the success bar for the learned correction is rc_r, not Keplerian resid_r.
Target: from dirty's 0.34 toward beam_only's 0.89 on the 5 disks, without Wiener's m = 2 artifact.

---

## 2026-10-02 | finding | SG v2 wiggle: beam-aware Wiener deconvolution beats dirty at every scale >= 6 px; the U-Net fails at the finger scales, not by plain blur

`experiments/wiggle_wiener_deconv.py` (CPU, numpy only, reuses `wiggle_all_methods_step1.npz`'s
shared Keplerian model and mask; channels 240-360). Its quadratic centroid reproduces the stored
bettermoments M1 maps exactly (corr 1.00000, median diff 1.7e-16 km/s) before anything is compared.
Wiener filter per channel with the recovered beam (`dirty_beam_recovered_v2.fits`), noise PSD from
line-free channels 0-59 and 541-600, no clean data used.

| method | resid_r | coherence with clean residual: 50-100 / 25-50 / 12-25 / 6-12 px |
|---|---|---|
| dirty | 0.892 | 0.94 / 0.85 / 0.79 / 0.69 |
| Wiener, alpha=1 | **0.900** | **0.97 / 0.89 / 0.82 / 0.71** |
| noise-free beam-only (ceiling for pure denoising) | 0.920 | 0.96 / 0.89 / 0.85 / 0.76 |
| U-Net (winner_aug) | 0.805 | 0.77 / 0.73 / 0.76 / 0.70 |
| DDRM | 0.587 | 0.70 / 0.53 / 0.58 / 0.45 |

alpha sweep 0.5/1/2/4/8: 0.877/0.900/0.886/0.859/0.781. Single cube, single run, no LOO yet: a
lead, not a result.

Three things measured along the way:
1. The U-Net is worse than dirty exactly at the GI-finger scales (25-100 px) and adds power at
   12-50 px (1.21-1.37x clean): it replaces the pattern rather than only smoothing it.
2. Binning error on the Keplerian velocity (i.e. on channel / isovelocity contour) explains 11.9%
   of the U-Net's error variance, against 4.0% for dirty and 2.1% for Wiener (radius: 6.1%). That
   is the radial-spoke artifact: channels denoised independently put their errors along isovelocity
   lobes.
3. Hypothesis tested and refuted: the model does not quantize line centres to channel centres.
   Pixel-locking exists (|frac|<0.1 of a channel: clean 0.203, dirty 0.299, U-Net 0.274), but it is
   the noise acting on the quadratic estimator, larger in dirty than in the U-Net output.

Also: realistic per-band fusion of dirty + U-Net (weights fitted on one half of the disk, applied
to the other) gives 0.890 vs 0.882 and 0.922 vs 0.902 on the held-out halves. Small but above both.

Consequence: the blur relative to clean on SG data is mostly the beam (coherence <6 px is 0.38 even
noise-free), and the current networks never learned deconvolution (trained on A=I line emission).
Next is a non-blind deconvolution network (Wiener step in feature space, Dong et al. 2020 DWDN;
unfolded data/prior steps, Zhang et al. 2020 USRNet) with a noise-level map input (FFDNet), spectral
context, and a residual on top of the Wiener estimate, trained on SG synth pairs across noise levels.
Figure: `results/self-gravitating/wiggle_wiener_vs_unet.png`; numbers: `wiener_results.json`.

**Same day, follow-up (`experiments/wiggle_wiener3d.py`): 3D Wiener reaches the linear ceiling.**
The dirty noise is 0.53 correlated between channels at every lag (1, 2, 3, 5): a static pattern
identical across the cube (mean of ch 0-59 vs mean of ch 541-600: r = 0.986), 53.7% of each
channel's noise variance; clean is ~1e-6 there. Subtracting the per-pixel mean of the line-free
channels leaves channel-independent noise (lag-1 corr -0.015). That does not move dirty's own
centroid (0.892 both ways: a constant offset cancels in the parabola) but helps any spatial filter.
Then Wiener in 3D (beam deconvolution + filtering along velocity, block ch 180-420, scored 240-360):

| method | resid_r | 50-100 / 25-50 / 12-25 / 6-12 px |
|---|---|---|
| 2D Wiener, static removed | 0.902 | 0.96 / 0.88 / 0.81 / 0.69 |
| **3D Wiener, alpha=1** | **0.925** | **0.98 / 0.93 / 0.86 / 0.73** |
| oracle 3D Wiener (clean's power spectrum; linear ceiling) | 0.923 | 0.98 / 0.92 / 0.84 / 0.74 |

Data-driven 3D Wiener sits at the linear ceiling and above the noise-free beam-only cube (0.920).
alpha 2/4: 0.912/0.902. Channel maps (v = +1.17 km/s) show arms and inner ring close to clean,
dirty a blob: `results/self-gravitating/wiggle_3d_wiener.png`. Below ~6 px coherence stays ~0.35
even for the oracle: information the beam removed. Caveats: one cube; the static pattern may be how
this synthetic dirty cube was made (on real ALMA the matching step is continuum subtraction); not
yet run on the 5 SG LOO disks or the line-emission holdouts.

**5-disk check of 3D Wiener (`experiments/wiggle_wiener3d_loo.py`, `results/self-gravitating/wiener3d_loo.json`).**
The five `sg_synth` disks (dirty = beam (*) clean + beam (*) noise, unit-gain beam), shared Keplerian fit
from clean with the true inclination fixed, `signal_mask` frac 0.05, no clean data in the filter. A first
pass had a beam-normalisation and noise-spectrum bug (beam summed to 355); the numbers below are the fixed run.

| disk | dirty r / PSNR | Wiener a=2 r / PSNR | Wiener a=1 r |
|---|---|---|---|
| 9015 | 0.838 / 18.4 | **0.994** / 29.3 | 0.337 |
| 9019 | 0.996 / 20.3 | 0.995 / 33.1 | 0.993 |
| 9025 | 0.334 / 19.3 | **0.657** / 29.7 | 0.053 |
| 9032 | 0.728 / 31.0 | 0.730 / 35.3 | 0.577 |
| 9074 | 0.245 / 21.3 | 0.288 / 33.0 | 0.040 |
| mean | 0.628 +/- 0.325 | **0.733 +/- 0.292** (4/5 wins, 1 tie) | 0.400 +/- 0.399 (0/5) |

PSNR rises on every disk (+4 to +14 dB). The wiggle gain is large only where dirty's own r is middling
(9015, 9025) and negligible where dirty is near-perfect (9019) or where both are poor (9074: 0.245 -> 0.288).
Estimated noise spectrum (from dirty only) matches the oracle one (0.733 vs 0.733), so the noise estimate is
not the problem. alpha=1 loses on all 5 (deconvolution amplifies noise); alpha=2 is stable; alpha above 2 and a
per-cube rule were not swept. Honest read: 3D Wiener is a strong no-training baseline for noise and amplitude,
but it does NOT solve the wiggle on the hard disks (9025, 9074); that is what the learned stage has to do.
**Superseded 2026-10-08 (entry above):** this resid_r is Keplerian-subtracted and mostly scores the axisymmetric
rotation curve. On the rotation-curve residual (rc_r) Wiener a=2 is 0.349 vs dirty 0.341 mean, not a win.

**Line-emission data is a different problem (checked on `run_0002_00560_rt_00`).** 201 ch at 0.1 km/s
(exoALMA's fiducial spacing). dirty - clean is independent between channels (lag 1/2/5/20 corr
-0.003/+0.001/+0.001/+0.002), no static part (0.005 of variance), spatially beam-correlated (power
224 at k<0.02, 0.64 at 0.05-0.1, ~1e-14 above 0.2), and the signal has no beam (<D C*>/<|C|^2> ~1
at all k, Phase 0's A=I). 3D Wiener with the noise spectrum taken from a different cube
(`run_0006_00170_rt_00`), no training: rel. error 0.574 -> 0.195, PSNR 29.36 -> 38.75 (alpha=2),
M1 corr 0.9812 -> 0.9895 (alpha=1). Above V12 (32.95), near the best 1-channel U-Nets; only the
spectral-context models (kin_gamma0 ~42.9) beat it.

**Correction (2026-10-08).** The sentence that stood here said models trained on line emission "are trained on the wrong forward model for
exoALMA" because their signal has no beam while a real CLEAN cube's does. That was wrong. Checked on `run_0002_00560_rt_00`: the clean AND the dirty
cube carry the same beam in the header (BMAJ 0.185", BMIN 0.126", BPA 149 deg, 18.6 x 12.7 px at 0.01"/px), the clean channel's spatial power
is band-limited like a beam-smoothed image (1.4e3 at k=0.01, 1.8e-2 at 0.065, 1.3e-7 at 0.115 cycles/px, then the float32 floor), and the
dirty-minus-clean noise is band-limited too (power 1.26 at 0.065, 6e-5 at 0.115): beam-correlated noise, as in a CLEAN image. "A = I" (Phase 0)
is right about clean to dirty, because the beam is already in the clean cube. So the line-emission pairs are the same kind of object as exoALMA's
fiducial products (0.15" circular beam via uv-taper, 100 m/s channels, ~1.5 K rms; exoALMA I, research.md 2026-10-08), at matching channel
spacing (CDELT3 = 0.1 km/s) and a similar per-channel peak SNR (clean peak 0.0374 over noise rms 0.00127, about 29; exoALMA 12CO/13CO about 13 to 50).
What differs: beam shape (elliptical 0.185 x 0.126 vs circular 0.15"), pixel scale, and the noise realisation (ours synthetic).

---

## 2026-10-02 | added | collect_outputs() now zips each run for one-click download

`zip_patterns` kwarg added to `collect_outputs()`: `run_dir` is always zipped
(`<notebook_id>_outputs.zip` at the top of `/kaggle/working`), and an optional
extra glob (checkpoints, which `patterns` deliberately excludes to avoid
duplicating them loose in Output) is folded into the same zip without being
copied into `run_dir`. Wired into notebooks 05 (`zip_patterns=['nb05_*.pth']`)
and 13 (`zip_patterns=['nb13_*.pth']`) on this branch, and 14
(`zip_patterns=['nb14_*.pth']`) on `native600-loss-sweep`. Notebook 16 needed
no notebook-side change — it already only writes CSVs/PNGs via `patterns`, so
the always-zip behavior covers it from the shared helper alone.

---

## 2026-09-25 | plan | notebook 16 runs 2 and 3

**Run 2 (next):** the 9 image-chosen checkpoints plus every resolution arm (`winner_aug_res320` s43, `winner_res320` s42, `winner_aug_res480` s43, `sweep_winner_600`, `sweep_winner_aug_600`,
`sweep_winner_p10_600`), 14 in all, on all 5 line-emission cubes and both SG cases (98 rows). Set in the repo's cell 6 (`PROFILE = 'full'`, `INCLUDE` regex), so a fresh import needs no override.
Estimate from measured row times: ~55 min scoring + ~15 min figures. **Run 3:** `INCLUDE = None`, every checkpoint, resuming from run 2's Output; pages stay clean + dirty + 6 checkpoints.

---

## 2026-09-25 | analysis | notebook 16 quick run read as IMAGES: the kinematic-input models are the closest to clean; pick for the 5-cube run

Read from the 12 `rt_00` sheets and the full CSV (`results/16-checkpoint-evaluation/v5_2026-09-25_quick_6e4e139/`). One cube for the images, two for the numbers, one seed per arm: **a shortlist, not a result.**

**What the pictures show that the tables did not.**
- **Sharpness (M1 gradient).** Clean is a smooth radial fan with a few fine rays. Every single-channel U-Net draws a web of closed contour lines across the disk (gradE/clean 1.8 to 2.6): M1 comes out
  terraced, so on line emission the U-Nets ADD velocity structure rather than smoothing it (the opposite of SG v2, where gradE/clean is 0.4 to 0.8). Only the `kin_gamma0` family (31-channel input) looks like clean
  (`kin_gamma0_starlet_ft` 0.83, `kin_gamma0_mae_ft` 0.81, `kin_gamma0` 0.95).
- **Calibration at the systemic channel.** `kin_gamma0_starlet_ft` (slope 0.92) and `kin_gamma0_mae_ft` (0.94) are a thin line on the diagonal. Most U-Nets are wide, and many show a break at clean ~0.006 Jy/beam
  (a lower branch of pixels whose bright emission was cut), visible in `starlet_p10_ft`, `mae_p10_ft`, `hybrid_p10_ft`, `wavelet_p10_ft`, `gradient_p10_ft`, the p10 seeds, `res320`. The aug seed 43 and `res480` are the
  best U-Nets here (0.99, 0.97), close to the line.
- **Channel error at the systemic channel (rms).** `kin_gamma0_mae_ft` 1.49e-4 and `kin_gamma0_starlet_ft` 1.61e-4, then `aug_res480` 2.05e-4, `kin_gamma0_starlet_fresh` 2.17e-4; the U-Net loss arms 2.5 to 3.7e-4.
- **M1 error map.** Every model carries the same dipole at the disk centre (unresolved inner Keplerian rise). The **p10-sourced fine-tunes all share a red wedge below the centre**, inherited from
  `sweep_winner_p10`: the loss did not remove it. Lowest mean |err|: aug seed 44 0.0545, aug seed 43 0.0564, `kin_gamma0_starlet_ft` 0.0579, `res480` 0.0608; `starlet_p10_ft` is 0.0848.
- **Wiggle error ratio** (below 1 beats dirty): the `kin_gamma0` family is best on both cubes (0.88 on `rt_00`, 0.10 on `rt_01`), `res480` next (0.92). `starlet_p10_ft`, 5-cube M0 leader in 05, is 1.14 on `rt_00`:
  **worse than dirty on the wiggle**.
- **Failures are visible, not just numbers:** `aug_res320` s43 tilts the whole velocity field (red left, blue right) and brightens all of M0: a scale failure of that training run. `kin_gamma0_mae_fresh` and `sg_k3_fresh`
  are broken on line emission. The 600 px arms leak noise at the disk edge (`aug_600` gradE 5.7).
- **Ensemble.** Averaging all 34 gives M1 mean |err| 0.069, worse than the best single models (0.055 to 0.058), so an ensemble of everything is not a shortcut.
**Suspect (RULES.md #8), CORRECTED 2026-09-25:** `invented_frac` is NOT exactly 0 (that was the log's 2-decimal rounding). From `nb16_eval_rows.csv`: at most 0.022 on `rt_00`, at most 0.0013 on `sg_v2`, but up to 0.28 on `rt_01`. A tenfold gap between cubes, so the detector's background definition still needs checking before any invented-structure number is quoted.
**Figure bug found while reading:** tile labels dropped the seed, so the four sweep-winner seeds were all drawn "winner" and the aug/p10 seeds likewise; the two `best_models` byte copies took extra tiles. Fixed (`short`
keeps `sNN`, `_drop_copies`).

**Decision: which checkpoints to score next, chosen from the images (no ALMA pick yet).** Kept because their pictures are closest to clean: `kin_gamma0_starlet_ft`, `kin_gamma0_mae_ft` (clean-like
sharpness, calibration on the diagonal, lowest channel error), `kin_gamma0_starlet_fresh` (calibration near the line, low channel error), `kin_gamma0` (the family's base, as its control), `kin_gamma0_wavelet_ft` and
`kin_gamma0_gradient_ft` (same family, never imaged), `sweep_winner_aug_seed43` and `sweep_winner_aug_seed44` (best U-Net M1 error maps, calibration near the line), `winner_aug_res480_seed43` (best U-Net channel error and
wiggle ratio). Dropped on sight: every p10-sourced arm (shared wedge, calibration break), the 600 px arms (edge noise), `aug_res320` s43 (tilted field), `winner_res320` s42 and `wavelet_fresh` (calibration break),
`beam` and the non-aug sweep-winner seeds (tilted M1 error), `sg_k3_fresh` and `kin_gamma0_mae_fresh` (broken). Based on ONE cube's images; the 5-cube run is what tests it. `tools/alma_infer.py` only accepts
1-channel models, noted for later.
---

## 2026-09-25 | fix | notebook 16 figures: every checkpoint, 8 disks per page at 150 dpi, tight non-square crop

Requested layout: clean + dirty + 6 checkpoints = **8 disks per page**, each large. `DISKS_PER_PAGE = 8`, 4 x 2, 150 dpi, tiles 4.2 in wide (each disk ~630 px); the error and invented-structure sheets
carry a third reference tile (dirty minus clean, or the dirty invented map), so they show 5 checkpoints a page to keep 8 disks. The classic wiggle page is clean + dirty + 2 checkpoints, each with its M1 and
its residual (4 columns x 2 rows = 8 disks). **The crop is now a tight, non-square box around the disk** (10% margin): an inclined disk is an ellipse and the old square crop left over a third of every tile black,
so the same page area shows the disk about 1.4x larger. EVERY checkpoint is on every sheet (an earlier top-16 cut dropped 18 checkpoints and was reverted). Sheet kinds in the default set: M0, M1, M2, M1 error, sharpness
(smoothing), invented structure (hallucination), classic wiggle, plus the singles (spectra, radial/power, calibration, integrated spectrum, error histogram, ensemble, top-k). Channel maps, M0 error and the
residual contact sheet need `FIG_EXTRA = True`. **Image count for 34 checkpoints:** per case 24 (M0, M1, M2, sharpness x 6 pages) + 14 (M1 error, invented x 7 pages) + 17 (classic) + 7 = 62; 3 cases + 6
dashboards = **192**. `FIG_TOP = n` shows only the best n checkpoints and cuts that roughly in proportion. Defaults live in `src/`, so a stale Kaggle cell gets them. Figures only; no published number touched.

---

## 2026-09-25 | fix | notebook 16's contact sheets tiled all 34 checkpoints in one figure; now 8 per page

**What was wrong.** Every M0/M1/M2, error, channel, wiggle, sharpness and invented-structure sheet put clean, dirty and ALL checkpoints on one figure at 7 columns, so each disk was about
1/8 of an 11 inch page and the fine structure (the thing the sheets exist to show) was not legible. **How it was caught.** The user looked at the sheets from the first Kaggle run against the
one-checkpoint-per-figure images from the older notebooks. **Fix.** `contact_sheet` pages: 8 checkpoints per page, 4 columns, 3 inch tiles, the reference tiles (clean, dirty, or the error to beat)
repeated at the top of every page (the error, channel-error and invented-structure sheets open each page with clean and dirty on their own scales, then the error to beat), one shared colour scale across pages (`..._p01.png` ...; a sheet that fits keeps its old name). The notebook display cell shows page 1 of each and the classic wiggle
pages 1 and 2; every page is in the Output. A test builds 34 panels and asserts 5 pages. **Published numbers it touches:** none (figures only). The first run's Output still has the old
unpaged sheets; the next run, or a local rebuild from the downloaded `nb16_maps/*.npz` with `cf.build_all`, produces the paged ones.

---

## 2026-09-25 | run | notebook 16 first Kaggle run (quick): 34 checkpoints, 3 cases, 102 rows in 49 min; 320 px is NOT inherently bad, 600 px does not help

Archived as `results/16-checkpoint-evaluation/v5_2026-09-25_quick_6e4e139/` (Kaggle Version 5). Line-emission means are over **2 cubes** (`rt_00`, `rt_01`), sd across cubes:
**not comparable to the 5-cube 05 tables**; the harness reproduces the published v20 row for `sweep_winner_aug_seed43` on `rt_00` (M0 +31.3 / M1 +77.2 / M2 +84.6), so it is scoring consistently.
A display cell raised after all scoring and stopped `collect_outputs`; scoring/figures are in the Output, the CSV must be downloaded. Fixed.

**Resolution (ask 2), 2-cube means M0 / M1 / M2:**
| model | M0 | M1 | M2 |
|---|---|---|---|
| aug at 256, 3 seeds (mean) | 30.4 | 80.0 | 51.6 |
| aug at 480 | 44.5 | 81.0 | 68.9 |
| earlier 320 px arm `winner_res320_seed42` | 44.8 | 80.9 | 67.3 |
| aug at 320 `winner_aug_res320_seed43` | -39.6 (sd 105) | 24.2 | -56.9 |
| 600 px: plain / aug / p10 | 5.3 / -1.7 / -13.4 | 60.7 / 60.6 / 67.6 | -23.4 / 45.3 / -6.1 |

**Corrects the 05 v47 reading ("320 is clearly worse").** A different 320 px model, `winner_res320_seed42`, scores like 480 (44.8 / 80.9 / 67.3), so 320 px is not inherently worse:
`winner_aug_res320_seed43` is one bad training run (it fails on one cube by -114). Two 320 px models, one good and one bad, is a seed spread, not a resolution effect. 480 is level with or slightly above
256 (M0 +14 is inside the 256 px seed spread of about 11 to 30 depending on the cube pair). **600 px is worse than 256 on M1 (61 to 68 against 71 to 84) and M0**, with lower PSNR (38 to 39 against 40 to 42) and it costs 74 s per cube per checkpoint against 21: the extra resolution is not recovered.
One seed per resolution arm.

**Loss controls (p10 source, 2 cubes) M0 / M1 / M2:** hybrid control 39.3 / 79.3 / 70.5; mae 35.5 / 75.4 / 84.2; wavelet 35.6 / 79.8 / 75.1; starlet 45.2 / 78.6 / 81.2; gradient 30.7 / 81.9 / 77.8.
Same shape as the 5-cube reading: only starlet (M0 +6, M2 +11) and mae (M2 +14) sit above the control, and only on some moments; one seed. Not established.

**kin_gamma0 family (stack_kin, 31 neighbour channels):** `kin_gamma0_starlet_ft` is the best line-emission checkpoint here on all three moments (50.4 / 84.1 / 72.8, PSNR 47.3; M0 sd across the 2 cubes 19.6) and `kin_gamma0_mae_ft` next (46.6 / 80.0 / 49.8). The base `kin_gamma0` is -23.1 on M0, and `kin_gamma0_mae_fresh` is a failed training (M0 -266.6, sharpness ratio 173). **A best-of-34 pick from two cubes; the 5-cube run must confirm it.**
**PSNR vs the moments:** Spearman across checkpoints +0.66 (M0), +0.73 (M1), +0.81 (M2). That is dominated by the failed arms (`sg_k3_fresh`, `kin_gamma0_mae_fresh`); the top PSNR arm is also the top M0 arm here, unlike the 05 result where `winner_k1` had the best PSNR and a poor M2. Do not read either as PSNR ranking or not ranking from this alone.

**SG v2 (cross-domain, 1 cube) is a suspect result (RULES.md #8), not a finding.** Every line-emission model has M1 -40 to -98 and M2 -23 to -93, and every wiggle gain is negative (worse than dirty, consistent with the earlier
"every model loses to dirty on the wiggle here"). But `sg_k3_fresh`, trained on SG data, scores M0 -66, M1 -57, M2 -123 on it, and is among the worst; SG v2 is a different cube from its training family and its amplitude is rescaled by `match_amplitude`, so a domain gap and a harness artefact are both possible.
Needs the per-cube figures and `amp_scale` from the CSV before it is quoted.
**Speed:** measured 21 s (256 px U-Net), 33 s (stack_kin), 74 s (600 px) per row: a full run (about 66 supported checkpoints x 7 cases, plus slower DDPM rows) is now about 4 to 6 h, not the 15 to 30 h guessed earlier.
**Published numbers it touches:** the 2026-09-25 v47 entry and BLOG_PROGRESS.md section 2 said 320 px was worse; both corrected.

---

## 2026-09-25 | run | 05 v47: 320 and 480 px scored at their own size; ask 2 answered for the aug recipe, one seed each

Clean scoring-only session (no training, 38 checkpoints restored, fixes live at `499030b`). Means over 5 holdout cubes, clipped + signal-masked, one seed per resolution arm
against a 256 px baseline whose spread is across 3 seeds:

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| aug at 256 (3 seeds) | 39.30 | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| aug at 320 | 37.36 | -17.8 | 48.0 | -3.9 |
| aug at 480 | 40.18 | 39.7 | 73.5 | 67.3 |

**Reading.** 320 is clearly worse: M1 26 points and M2 59 points under the baseline, and 2 of 5 cubes fail outright (M0 -114 and -114). 480 is level with 256: M1 the same,
M0 +10 and M2 +12, both inside one seed spread of the baseline (7.2 and 13.9), so **no established gain from less downsampling at 480**. The 320 failure being worse than 256
while 480 is not is not monotonic in resolution, which points at this one seed (or the 320 recipe) rather than at resolution itself; a second seed would settle it. Native 600 px
is not in 05 (gated, never trained); the only 600 px models are notebook 14's three, scored in notebook 16.
**Published numbers it touches:** the 2026-09-25 rows for `winner_aug_res320`/`res480` (scored at 256 in v42/v43) are superseded; BLOG_PROGRESS.md section 2 updated. Notebook 16 had a
second version of the same defect for 14's arms (label `sweep_winner_600` read as 256); fixed in `_train_size`.
**Archived:** `results/05-unet-line-emission/v47_2026-09-25_773b0ce/`. Versions 45 and 46 are not in git.

---

## 2026-09-25 | added | notebook 16 set to a QUICK first run: 3 cases, ~25 checkpoints

Full inventory (~65 scoreable checkpoints) x 7 cases is ~455 rows; the only timing is the local CPU preview (230 to 275 s per row), so a full run is a guess of 15 to 30 h
(2 to 4 sessions). Cell 6 now has `PROFILE = 'quick'` (2 line-emission cubes + SG v2) and an `INCLUDE` regex for the checkpoints the loss, resolution and
best-model decision needs: the loss arms `_p10_ft` and `_fresh`, both hybrid controls, aug/p10/non-aug seed baselines, `winner_beam`, `res320`, `res480`, the three
600 px arms, `kin_gamma0` (+ mae/starlet), `sg_k3_fresh`. ~25 checkpoints x 3 cases ~ 75 rows, guessed at 2.5 to 5 h. **Unmeasured until the first Kaggle session
prints `wall_s`.** Not included on purpose: DDPM/DDRM, the older 08/10/11/12 families, `winner_k1/k2`, patch. A later session sets `INCLUDE = None`, `PROFILE = 'full'`
and resumes from this Output. No published number touched.

---

## 2026-09-25 | run + bug | 05 v44 FAILED (CUDA OOM scoring res480), but the control arms ran: much of the fine-tune gain is extra training, not the loss

**Run.** Fresh Kaggle import, cells current (`1b1ee7f`). Trained `winner_hybrid_ft` (from aug, PSNR 39.574) and `winner_hybrid_p10_ft` (from p10,
PSNR 40.253), 30 epochs each, both persisted. Then the scoring step died: `OutOfMemoryError: Tried to allocate 3.96 GiB` in cell 18, scoring
`winner_aug_res480`. **Cause:** the 2026-09-25 per-arm-size fix resized the image but kept `BS = 32` channels per forward, which fits at 256 and 320
and not at 480 (or 600). **Fix:** `bs = max(1, int(BS*(TARGET_SIZE/size)**2))`, now in `tools/reapply_05_fixes.py`. **Published numbers it touches:**
none; `res480` and `native600` still have no valid moment row.

**The control arms answer the confound flagged on 2026-09-25 (RULES.md #4).** Same source, same 30 epochs at 0.1x lr, ORIGINAL hybrid loss. Means over
the 5 holdout cubes, one seed each, against a baseline whose spread is across 3 seeds:

| arm | M0 | M1 | M2 |
|---|---|---|---|
| aug baseline (3 seeds) | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| **control: hybrid, from aug** | **39.7** | **80.9** | **69.9** |
| mae / wavelet / starlet / gradient, from aug | 26.3 / 38.8 / 42.6 / 32.1 | 77.4 / 76.1 / 80.5 / 78.7 | 70.5 / 70.0 / 81.1 / 77.3 |
| p10 baseline (3 seeds) | 33.5 +/-9.6 | 70.7 +/-6.7 | 31.8 +/-11.1 |
| **control: hybrid, from p10** | **40.4** | **76.2** | **68.4** |
| mae / wavelet / starlet / gradient, from p10 | 43.5 / 41.4 / 44.5 / 41.7 | 79.6 / 78.1 / 77.1 / 80.3 | 83.8 / 73.1 / 79.6 / 70.5 |

**Reading.** Extra training alone (no new loss) moves aug by +10.5 / +6.9 / +14.9 pp and p10 by +6.9 / +5.5 / +36.6 pp. From aug, no new loss beats that
control except starlet on M2 (81.1 vs 69.9) and starlet on M0 (42.6 vs 39.7, inside noise); MAE is BELOW it on M0 (26.3 vs 39.7). From p10 the losses sit
at or above the control by 0 to +4 on M0, 0 to +4 on M1, and +2 to +15 on M2. So **most of the `_ft` gain in the 2026-09-25 table was continued training, not the
loss**, and the honest claim left is small: starlet and MAE (from p10) may add a little on M2, one seed, not established. The FRESH arms (which had a clean
control) are unaffected by this: they still beat the non-aug baseline on M1/M2.
**Published numbers it touches:** the 2026-09-25 "MAE helps M2 and M1 more than M0" paragraph and BLOG_PROGRESS.md section 1, both corrected. Also the res320 row:
scored at its own 320 px it is M0 -114.0 / +34.8 / +38.3 / -113.5 / +65.4 per cube (cubes 00 and 0025 fail), single seed; PSNR 37.36 is the lowest of any arm.
**RAM breakdown (first live reading).** `main` grows ~0.1 GB/epoch (2.4 to 8.6 GB across the two arms); workers' RSS stays flat (12.4 then 39.7 GB, shared pages).
The leak is in the main process. Cause still unknown. **Archived:** `results/05-unet-line-emission/v44_2026-09-25_failed_oom/`.
**Left:** a session to score `res480` and `native600` (fix in), and getting the two control checkpoints out of v44's Output (failed versions cannot be attached; download, re-upload as `.ckpt`).

---

## 2026-09-25 | run | 05 v43: last spectral-context arm done; Kaggle's push overwrote the 05 fixes again, restored by script

Ran the pre-fix cells (cell 0b pulled `ad600d2`), so nothing from the day's fixes was in it. One arm, `winner_k2` (two spectral neighbours),
seed 42, early stop at epoch 50: PSNR 42.81 (best of any arm), M0 +20.2 / M1 +65.9 / M2 +36.8 (5 cubes, one seed, clipped + signal-masked).
Same pattern as `winner_k1` (42.59, +33.3/+75.6/+40.5): spectral context is a pixel win and a moment loss, and the notebook's own line says
ranking by PSNR would have picked it. Two neighbours are no better than one. Archived: `results/05-unet-line-emission/v43_2026-09-25_6fb324d/`.
**Bug (RULES.md #2, third time).** Kaggle pushed v43 (`6fb324d`) over the notebook, reverting the per-arm scoring size, `STALE_MOMENT_ARMS`
and the two control arms. Caught by grepping the pulled file for `_arm_size`. Restored with `tools/reapply_05_fixes.py` (8 edits, idempotent).
**Published numbers it touches:** none new; the res320/480 rows in v42/v43 tables remain the invalid 256 px scores.
**Left:** `winner_hybrid_ft`, `winner_hybrid_p10_ft` and the res re-score, in one session, from a fresh Kaggle GitHub import.

---

## 2026-09-25 | bug | notebook 16's amplitude checks were meaningless on the SG v2 cube: clean and dirty are on different scales

Found by looking at the first SG v2 calibration plot (denoised against clean, pixel by pixel): slopes of ~300 and empty tiles. The
SG v2 cube's dirty is ~336x its clean at the systemic channel (~136x at the M0 peak), which this project already knew (PROGRESS.md
notes "dirty ~110x"; `kinematic_data_v2_amplitude_check.png`) but the harness did not account for. A model returns values in its INPUT's
units, so every amplitude comparison against an unscaled clean measured the data's units: PSNR (16.5 dB on that cube), M0 improvement,
channel and M0 error maps, calibration, invented structure, overshoot. Velocity quantities (M1, wiggle, M1 sharpness) are scale-free and
were never affected. The line-emission cubes are at 1.01 and are unaffected.

**Fixed.** `match_amplitude` puts clean in the input's units with the least-squares factor of dirty on clean, but only outside
0.8 to 1.25, so a cube whose halves already agree is left exactly as it was and the validated line-emission numbers (M0 +31.3 / M1 +77.2 /
M2 +84.6) cannot move. The factor is in every row (`amp_scale`) and on every amplitude figure's title. The correction is an assumption
(a single linear factor between the two cubes) and any SG v2 amplitude number should say it was rescaled.
**Published numbers it touches:** none. No published SG v2 result uses amplitude; they are all wiggle (velocity).

**What does and does not reproduce on SG v2.** The classic wiggle figure reproduces the earlier one to two decimals: residual RMS
clean 0.23 / dirty 0.21 / U-Net 0.22, and the smoothing is visible (the concentric ring and blotches in clean's residual are gone from
the models'). The wiggle CORRELATIONS are close but not identical to the published ones (dirty 0.873 against 0.862; `winner_aug_seed43`
0.790 against 0.760, and `sg_k3_fresh` and `winner_beam_seed42` swap order), so this is not a reproduction of that table. Notebook 16
continuum-subtracts every cube and starts the fit from the emission's shape; the older script did neither. The finding they support,
that every model loses to dirty on the wiggle here, holds: the new error ratio is 1.29 to 1.47 (all above 1).

---

## 2026-09-25 | bug | the line-emission wiggle geometry was degenerate on 2 of 5 cubes; the published nb08 wiggle numbers include them

**What was wrong.** `score_08_kinematic.py` (and most `experiments/` wiggle scripts) call `fit_keplerian(m1, mask, au_per_px)`
without `m0=`, so the fit starts from the MASK's shape, which `fit_keplerian`'s own docstring calls "the degenerate end of the
mass-inclination valley". **How it was caught.** Building the first real figures for notebook 16: the M0 radial profile was zero out
to 200 px and then rose, which meant the disk centre I had passed was wrong. The stored geometry had `cy = 600.0`, the image edge.
Refitting the same cube with `m0=` moved the centre to (299.9, 300.5), mass from 18.7 to 1.46 Msun (truth 1.0), residual rms from
9.4 to 4.4 km/s, and the fit cost to a third. Then the same refit on every case (about 45 s, no models needed):

| cube | as published (no m0) | with m0 |
|---|---|---|
| `run_0002_00560_rt_00` | centre (374, **600**), incl 73.0, mstar 18.7 | centre (300, 300), incl 48.9, mstar 1.46 |
| `run_0002_00560_rt_04` | centre (378, **600**), incl 74.2, mstar 0.80 | centre right, but incl 1.8, mstar **50.0 at the bound** (true incl 11.7: the documented mass-inclination degeneracy) |
| `rt_01`, `run_0025`, `run_0026`, and `sg_v2` | fine | identical |

So it is not a general failure: 3 of 5 line-emission cubes and the SG v2 cube were unaffected, and SG v2's dirty 0.87 (the anchor
of "doing nothing beats the model") stands. `nb08_kinematic_wiggle.json` stores exactly this: cubes `rt_00` and `rt_04` have `cy = 600.0`.

**Published numbers it touches.** The `nb08` kinematic wiggle table for those two cubes, and every mean over all five: kin_gamma0
**0.8155 against dirty 0.4284** (PROGRESS.md 2026-09-11 onward, the Phase J summary, and the line-emission points of
`headroom_scatter.png`, two of whose five points come from these fits, including the one at dirty 0.996). Not touched: SG v2, the SG
LOO folds and the SG holdout (inclination fixed at the true value, or fits identical either way).
**What still holds.** On the 3 cubes with valid free fits, kin_gamma0 (gamma 0) scores 0.7198 / 0.4993 / 0.8643 (mean **0.694**)
against dirty's 0.1182 / 0.2746 / 0.2062 (mean **0.200**): the win is intact and the gap is larger. What is no longer supported is the
pair of headline means, and the two high-dirty points used as evidence that models lose when dirty is already good.
**Read from stored geometry and refits, not from re-running any model**, so kin_gamma0's score on `rt_00` and `rt_04` under a valid
geometry is not yet known; notebook 16 will produce it.

**A second, separate limit found on the way.** Even with a valid geometry, `run_0002_00560_rt_00` (incl 63.7 deg) has dirty at 0.9966:
its residual against the flat Keplerian model is 4.4 km/s rms, dominated by the disk's flaring, and that shared error correlates
between clean and dirty whatever the denoiser does. The wiggle correlation is uninformative on that cube (no gain or loss can be read
from it). Notebook 16 now records `ref_resid_rms` so this is visible.
**Fixed with a metric that shared error cannot fool.** Notebook 16 also reports `resid_err_ratio` = rms(model residual - clean residual) /
rms(dirty residual - clean residual): subtracting clean's residual removes whatever the flat model leaves that clean and dirty share.
< 1 means the model brings the wiggle closer to clean's than dirty is. On that cube's saved maps, correlation gives 0.9907 / 0.9966 /
0.9967 for `sg_k3_fresh` / `winner_aug_seed43` / `winner_beam_seed42` (indistinguishable), while the error ratio gives 1.58 / 0.97 / 0.96:
`sg_k3_fresh` is worse than doing nothing there. (Computed from that preview's maps, which predate the geometry fix, so read it as a
demonstration that the metric discriminates, not as a result.)

**Fixed in notebook 16, not in the old scripts.** `checkpoint_eval.prepare` starts the fit from the emission's shape and, for the
line-emission cubes, holds the inclination at the true `.para` value (what `fit_keplerian` prescribes when it is known independently).
Under that protocol all 5 cubes fit sensibly (centre within 4.2 px, mass 1.2 to 1.75, none at a bound) and dirty's wiggle correlation
is 0.9966 / 0.1161 / **0.0435** / 0.2806 / 0.2165 (mean 0.331, against 0.428): `rt_04` is a heavily degraded cube, not the middling one
the degenerate fit made it. A `geom_ok` flag (fit converged, mass not pinned, centre within a tenth of the disk's extent of the M0
centroid) blanks the wiggle numbers when a fit fails instead of averaging them in (RULES.md #8). A related silent failure found and fixed
in the same pass: the `.para` lookup used `ho["folder"]`, which is a NAME, so the distance had fallen back to 140 pc for every cube and
the inclination was never fixed. The old scripts are left as the record; they are not edited.

---

## 2026-09-25 | added | notebook 16 now compares every checkpoint as images, not only numbers

The first version of 16 produced a results table, five aggregate figures and one stand-alone moment panel per checkpoint on
two cubes: to compare 40 checkpoints by eye you had to open 40 files, and the wiggle, the error and the channel views were
numbers or absent. Now scoring keeps each checkpoint's maps, blue/systemic/red channels, line profiles at three pixels and an
invented-structure map (small: a few MB per checkpoint per case), and `src/evaluation/checkpoint_figures.py` builds, for every
case, side-by-side sheets of ALL checkpoints next to clean and dirty on one colour scale taken from clean, never from a checkpoint:
M0, M1 and M2; error maps (denoised minus clean, scaled by dirty's own error, so "better than doing nothing" is visible); the
Keplerian residual of M1, the wiggle itself; the three channels and the systemic-channel error; an M1 gradient-magnitude
sharpness sheet; an invented-structure sheet; line profiles; the M0 radial profile and the M1 power spectrum against clean's;
and a large detail figure for the best checkpoints by M0. Across the run: the score tables as heat-coloured images, a scoreboard
of every metric (PSNR, SSIM, M0/M1/M2, wiggle, sharpness, invented blobs) with spread across cubes, the loss x source matrix
(Jason's asks 1 and 3), and a checkpoint x cube heatmap. About 20 figures per case, for all 7 cases by default.
Resume restores earlier sessions' maps; a scored row whose maps are missing is redone once so no checkpoint is a hole in a sheet.

Tested with `tests/test_checkpoint_figures.py` (all 19 per-case figure types plus the tables and matrices, from synthetic
artifacts, in seconds) and by building the real figures from three checkpoints on one full cube. Still not run on Kaggle.

---

## 2026-09-25 | bug | 05 scored the 320/480px arms at 256px: ask 2 (less downsampling) is still unanswered

**What was wrong.** Cell 18's `denoise_cube` did `F.interpolate(t, (TARGET_SIZE, TARGET_SIZE))` for every arm, so
`winner_aug_res320` and `winner_aug_res480`, trained at 320 and 480 px, were fed 256 px inputs when their moment maps
were computed. Their moment rows in v42 (M0 -12.3 / +9.7, M1 +50.0 / +66.1, M2 -4.4 / +54.5) are therefore not a test of
resolution. **How it was caught.** Reading `denoise_cube` while checking how the table treated resolution, before writing
up Jason's ask 2. The PSNR path was already right: `train_unet` and the resume path score on the arm's own view
(`VIEWS[view]()[1]`), so 37.36 and 40.18 dB stand, but PSNR does not compare across resolutions anyway.
**Published numbers it touches:** those two moment rows, in the v42 log and in anything quoting them. It does **not**
touch any 256px arm: their scoring size was correct. `winner_aug_native600` is gated off and has no row.

**Fixed.** `denoise_cube(..., size=)` with `_arm_size(name)` (320 / 480 / 600 for the resolution views, else 256), passed
from both callers. The res arms are added to `STALE_MOMENT_ARMS`, the mechanism built for the same class of bug when
`winner_beam` was scored without its beam vector, so their stored rows are dropped and re-scored at the right size on the next
run (checkpoints are in the recovered v38 dataset). Resolution changes and batch size changes together (16 at 256, 8 at 320, 6
at 480, forced by T4 memory, lr unchanged), so even a clean re-score will not isolate resolution; that confound is to be stated.
`checkpoint_eval.py` (notebook 16) takes each arm's size from its label, so it scores them correctly.

---

## 2026-09-25 | run | 05 v40, v41, v42: the loss sweep is done except one arm; MAE helps M2 and M1 more than M0

Three clean sessions, three new arms each (the session cap), archived under `results/05-unet-line-emission/v40_..`,
`v41_..`, `v42_..`. v42's moment table (5 holdout cubes, one seed per arm, clipped + signal-masked, mean across cubes)
gives Jason's asks 1 and 3 their first real answers. **Every number below is a mean over the 5 cubes, from one seed, against
a baseline whose spread is across 3 seeds (RULES.md #6); differences smaller than that spread are not established.**

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| `sweep_winner_aug` baseline (3 seeds) | 39.30 | 29.2 +/-7.2 | 74.0 +/-2.0 | 55.0 +/-13.9 |
| `sweep_winner_p10` baseline (3 seeds) | 39.27 | 33.5 +/-9.6 | 70.7 +/-6.7 | 31.8 +/-11.1 |
| `sweep_winner` non-aug baseline (4 seeds) | 37.52 | 11.4 +/-27.4 | 55.6 +/-9.0 | 6.0 +/-35.1 |
| mae / wavelet / starlet / gradient, fine-tuned from aug | 39.78 / 40.13 / 40.32 / 40.14 | 26.3 / 38.8 / 42.6 / 32.1 | 77.4 / 76.1 / 80.5 / 78.7 | 70.5 / 70.0 / 81.1 / 77.3 |
| same four, fine-tuned from p10 | 40.34 / 40.41 / 40.20 / 40.12 | 43.5 / 41.4 / 44.5 / 41.7 | 79.6 / 78.1 / 77.1 / 80.3 | 83.8 / 73.1 / 79.6 / 70.5 |
| same four, fresh (no aug) | 39.96 / 40.03 / 39.90 / 39.73 | 36.0 / 20.4 / 38.1 / 33.9 | 75.2 / 66.1 / 77.8 / 75.6 | 69.1 / 60.9 / 75.6 / 56.4 |
| beam-sourced (mae / wavelet / starlet / gradient) | 39.99 / 40.12 / 39.62 / 40.22 | 36.4 / 29.3 / 31.5 / 31.2 | 76.9 / 71.6 / 72.4 / 74.1 | 66.4 / 70.9 / 66.2 / 42.9 |
| `winner_k1` (one spectral neighbour) | **42.59** | 33.3 | 75.6 | 40.5 |

**Ask 1, MAE.** It did not lift M0: fine-tuned from aug it is 26.3 against 29.2 (inside the seed spread), from p10 43.5
against 33.5 (about one spread). It lifts M1 (+3.4 and +8.9 pp) and M2 (+15.5 and +52 pp) more clearly. So the simple loss helped,
but on the velocity moments, not on M0.
**Ask 3, the other losses.** All four raise M2 above their source, and the fresh (from-scratch) arms, which have a clean
control in `sweep_winner` (same recipe, same view, only the loss differs), beat it on M1 by +10 to +22 pp (its spread is 9)
and on M2 by +50 to +70 pp. **No loss clearly beats the others**: the M0 range across the four (26 to 45) is inside single-seed noise.
Best M0 is `winner_starlet_p10_ft` (+44.5) and `winner_mae_p10_ft` (+43.5), indistinguishable.
**Unresolved confound.** Every `_ft` arm changes the loss AND continues a converged model for 30+ epochs at 0.1x lr, so part of
its gain may be the extra training. The fresh arms have a clean control; the `_ft` arms did not. Two control arms are now in 05
(`winner_hybrid_ft`, `winner_hybrid_p10_ft`: same source, same budget, original hybrid loss). Until they run, "the loss helped" is
established for the fresh arms and only suggested for the `_ft` ones.
**PSNR again does not rank.** `winner_k1` is far ahead on PSNR (42.59, +2.3 dB) and behind on M2 (40.5 against 55 to 84 for
the loss arms); the notebook's own line says ranking by PSNR would have picked it. That matches 2026-08-21: spectral context is a
pixel win, not a moment win. Beam-conditioned arms show no gain.

**Left in 05:** `winner_k2`, plus the two control arms, exactly one session at the cap of 3. `winner_aug_native600` stays gated.
The RAM leak was mild at 256px (27.7 to 24.7 GB free over 30 epochs, ~0.1 GB/epoch) and cost nothing; the per-epoch
`main/workers/cache` diagnostic is not live on Kaggle because that commit is unpushed.

---

## 2026-09-25 | added | 16-checkpoint-evaluation.ipynb: one protocol for every checkpoint, validated against v20

Until now every checkpoint family had its own scoring script and preprocessing (`score_08_kinematic.py`,
`wiggle_domain_split.py`, `score_sg_wiggle.py`, `m1_rendering_audit.py`), so no two checkpoints had ever been
put through the same checks on the same cubes, and none of the loss-sweep arms had been scored beyond PSNR (13's)
or beyond moments (05's). `src/evaluation/checkpoint_eval.py` gives every checkpoint one interface (single channel,
31-channel stack, 7-channel stack, beam-conditioned U-Net, conditional DDPM) and runs the same battery, composing only
functions that already have published numbers: pixel PSNR/SSIM, `moment_improvement` (M0/M1/M2), the quadratic-M1
Keplerian wiggle against one shared geometry per cube, gradient-energy and Laplacian sharpness of raw M1 as a ratio to
clean, `channel_artifacts` (invented blobs, overshoot, floor leak), the moment-map panel, and the (dirty resid_r, gain)
pair for the headroom scatter. Cross-domain transfer needs no separate code: a line-emission checkpoint on an SG case, and
the reverse, is the same call. Cases: the 5 line-emission holdouts, the SG v2 cube (channels 240-360), and the SG holdout
`run_9074_00025_rt_00` (inclination fixed at 20 deg). Resumable, row persisted the moment it exists.

**Validated before trusting it.** On `run_0002_00560_rt_00`, full 201 channels, `winner_aug_seed43` scores **M0 +31.3,
M1 +77.2, M2 +84.6**, identical to v20's published per-cube numbers for `sweep_winner_aug` seed 43 on that cube. (A first run
on a 60-channel slice gave M0 -60.8 / M1 -76.0: it cut the line. A slice is not a smoke test for moments.)

**Not yet run on Kaggle.** Local smoke test only: four families (aug43, kin_gamma0, sg_k3_fresh, winner_beam) on 60 channels of
one cube, plus the full-cube validation above (~3.5 min CPU for one checkpoint on one cube). Timing on a T4 is unmeasured.
Estimate, to be replaced by the first session's printed timings: inference is seconds per cube, but moment maps and artifact
counting are CPU-side, so roughly a minute or two per (checkpoint, cube) and several hours for the full ~40 checkpoints x 7
cases, across sessions.

**Limits, all stated in the notebook.** DDRM is not scored (it restores through the beam operator, not wired into this
protocol) and is listed with that reason; DDPM runs on one cube only (sampling is ~100x a U-Net forward), 25 steps, one draw.
Every case is continuum-subtracted, so the SG-trained family (trained without it) sees the closest available input on
line-emission cubes. PSNR here is per channel on the dirty-scale normalisation over the holdout cubes, not the 256px validation
set behind the PSNR in 05's tables, so the two are not comparable. SSIM is on every 2nd channel. `val_loss` is never compared
(RULES.md #4). Tables give spread across cubes, not seeds (RULES.md #6).

One thing already visible on the validation cube: dirty's own wiggle correlation is 0.996 there, so no model can gain on it,
the headroom effect again. `tests/test_checkpoint_eval.py` covers naming, discovery and architecture detection.

---

## 2026-09-25 | plan | PLAN.md rewritten around what Jason asked for, plus the ML4SCI org deadlines

The 2026-09-10 plan was built on our own reading of what mattered, and by 2026-09-24 its Block 2 had become a
simulated degradation sweep Jason never requested. Re-reading the 2026-09-12 transcript, the plan now starts from
his nine asks in his words (MAE first, less downsampling to 320/480, wavelet/starlet/NLL, focus on the best two or
three models, inference on real exoALMA fiducial 13CO `.image.fits`, MWC 758, DSHARP second, VLT optional, DS9, and
"a really good ALMA pipeline ... scientific insights" as the success measure), then the org email's dates (blog to
mentors Sep 25 17:00 US Central, PR to `ML4SCI/EXXA` and a 3-minute talk Sep 29, submission Nov 3), then a dated
timeline. Notebook 14 (native 600px), the `simobserve` sweep, NLL and VLT are listed as optional. Block 1 is kept
as history.

Status of his asks found while writing it: MAE done and did not lift M0 (+26.3% against the +29.2% baseline);
wavelet and starlet done and did (+38.8%, +42.6%); 320/480 **trained but not scored on moments**, and PSNR does
not compare across resolutions, so ask 2 has no answer yet; NLL not tried; exoALMA inference built, not run.

---

## 2026-09-24 | run + bug | 05 v38: first winner_aug res arms finished, RAM leak NOT fixed, 09-20 root cause refuted

Pulled `9856f17`, cells current (guard, `SEED_OVERRIDE`, `winner_aug_res*` all present, so
v36's stale-cell failure did not repeat). Archived `results/05-unet-line-emission/
v38_2026-09-24_crashed/`. Killed by host RAM at epoch 19 of the third new arm.

**Finished, the arms the mentor asked for:** `winner_aug_res320` seed 43 PSNR 37.3597 SSIM
0.9932; `winner_aug_res480` seed 43 PSNR 40.1815 SSIM 0.9969. From scratch, augmented,
winner_aug's hyperparameters. PSNR across 256/320/480 is on different pixel grids, so it does
not rank resolutions; only the 600px moments in section 6 do, and this run never got there.
Both are stranded in a failed version's Output (README has the by-hand recovery).

**Correction to 2026-09-20 ("RAM leak root cause found and fixed").** That entry said the
DataLoader fork-storm was the cause and `persistent_workers=True` the fix, and marked it
unverified. v38 is the verification, and it failed: with the fix live, 480px still declines
0.29 GB/epoch, the same as v33's 0.3 before it. Slope also scales with image size (0.11 GB/epoch
at 256px, 0.22 at 320, 0.29 at 480), and no memory is returned at arm boundaries (free RAM 17.1
then 16.4, and 2.3 then 2.3). The fork-storm theory is refuted; `persistent_workers` stays in
because it is harmless, but it is not the fix. The cause is still unknown.

*Published numbers this touches:* none of the PSNR values. But `context.md` and any draft that
says the leak is fixed is wrong, and 05/13/06/07's per-arm cap `MAX_NEW_ARMS_PER_SESSION` is
what keeps runs alive, not a repair. 13 finishing all 28 arms on v17 was the cap working over
many sessions, not evidence the leak is gone.

**Diagnostic added:** `_host_ram_note()` in `src/training/sweep.py` now appends `main / workers /
cache` (this process's RSS, its DataLoader children's summed RSS, kernel page cache) to every
epoch line. Hot-reloads through cell 0b, no cell edit. Whichever column grows with the epoch is
the leak. Not a fix; it makes the next run decisive instead of a third guess.

**Plan:** all remaining 05 arms except the gated `winner_aug_native600` are 256px (about 0.11
GB/epoch, 4 to 5 GB per arm), so three fit one fresh session. If the two res arms are not
recovered they retrain (about 2.2 h and 4.3 h) and must run with `MAX_NEW_ARMS_PER_SESSION = 1`.

---

## 2026-09-24 | run | 13 Version 17 -- all 28 loss-sweep arms complete, pushed as a merge not a clobber

Pushed the res-arm/lr/epoch fixes below, then `git push` rejected: Kaggle had already
pushed `aca35c5` (13 Version 17) to `midterm-prep` in the meantime. Fetched and checked
before merging, rather than force-pushing over it (RULES.md #2's usual failure runs the
other way: Kaggle's push overwriting committed fixes; this time the risk was symmetric,
my local fix overwriting Kaggle's finished training run if merged carelessly).

**v17 finished the whole notebook**: kin 8/8, sg 8/8, ddpm 8/8 (`gradient_ft`/`gradient_fresh`
were the last two, deferred since 09-20), ddrm 4/4 (`ddrm_l2_fresh` the final arm,
`best_val_loss` 24.35863, 50 epochs, no early stop). First time 13 has reached 28/28.

**v17's cells were the same stale-cell disease as 05 v36**, fingerprinted the same way:
`FINETUNE_LR_SCALE` still multiplying the diffusion fine-tune lr (dead code, since
`load_checkpoint` overwrites it either way -- see 2026-09-24 entry above), kin's
`max_epochs=45` not yet raised to 60. Neither affects v17's actual results: the lr fix is
comment-only (the real lr was always the source's saved value), and all 8 kin arms were
`SKIPPED, already done` this version, so `max_epochs` never executed.

**Resolved by merge, not overwrite.** `git checkout --theirs` took Kaggle's file (all 28
arms' outputs intact), then the lr-comment and kin-epoch fixes were reapplied by editing
only `source` fields at the same cell indices -- verified `cell 18`'s outputs (5985 chars)
survived byte-for-byte. Cell-order test still passes. No training data lost, no stale
code kept either.

**Numbers this touches:** none newly wrong. Per-arm PSNR/`best_val_loss` for the 27 arms
besides `ddrm_l2_fresh` are not yet in git -- resumed arms print `SKIPPED, already done`,
not their value, so pull `nb13_*.csv` from v17's Kaggle Output before quoting any of them
in RUNS.md, PROGRESS.md, or a blog draft.

---

## 2026-09-24 | bug | review of 05/13: three recipes were not what the log said they were

A full check of both notebooks against their own claims, before the Sep 25 meeting. Three
real defects, each found by reading what ran rather than what the code comments say.

**1. 05's resolution arms lost winner_aug's recipe on 09-16, and nobody noticed for five
runs.** The 09-15 entry below ("resolution arms rebuilt as winner_aug_seed43's recipe")
was true for one commit. `1832c54` (the host-RAM fix, 09-16) was built on a Kaggle copy of
05 downloaded before `090b5e8` landed, and silently reverted it: RULES.md #2's failure
mode, caused by the assistant, not by Kaggle's push. Caught by `git log -S SEED_OVERRIDE`:
the symbol appears in exactly two commits, the one that added it and the one that removed
it. Every res arm trained in v30 to v34 used the OLD recipe: no D4 augmentation, seed 42,
and `winner_res320`/`winner_res480` fine-tuned from `sweep_winner_aug` at 0.1x lr rather
than trained from scratch.

*Numbers this touches:* `winner_res320` 39.6676, `winner_res480` 40.1817,
`winner_res320_fresh` 38.9608 (09-20 entry, RUNS.md). They are valid measurements of the
old recipe, and must not be quoted as "winner_aug's recipe at 320/480". Fix: `090b5e8`
re-applied by hand (the patch no longer applies, context moved): `winner_aug_res320`/
`winner_aug_res480`/`winner_aug_native600`, WINNER hyperparameters, `augment=True` train
views, seed 43 via `SEED_OVERRIDE`, from scratch, default `min_epochs=50`. New arm names,
so resume trains them and the old rows stay as their own record. `winner_res480_fresh`
(never finished, RAM-killed in v33 and v34) is dropped, the new arms replace it.

**2. Diffusion fine-tune arms never used `FINETUNE_LR_SCALE`.**
`DenoisingDiffusion.load_checkpoint` restores the source checkpoint's Adam state, lr
included, overwriting the constructor's lr. Linear warmup cannot undo it, since `self.step`
comes back at 7656 and warmup only runs below one epoch's steps. Read directly from the
checkpoints (no torch needed, `.pth` is a zip): `ddpm_seed42` saved lr 2e-4, `ddrm_prior`
2e-5. So 13's `ddpm_*_ft` arms trained at 2e-4, not the 2e-5 the notebook claimed.

*Numbers this touches:* `ddpm_l1_ft` PSNR 37.2620 and every other finished `ddpm_*_ft`
arm: the lr is 2e-4, not 2e-5. The numbers themselves stand. Its first fine-tune epoch
shows no spike (val 1673 then 1123) and it beats `ddpm_l1_fresh` (35.7669), so this is
"continued training at the source's own lr under a new loss", a valid fine-tune, just not
the one described. Not retrained: every DDPM ft arm, finished or still to run, gets 2e-4 by
the same mechanism, so the family stays internally consistent, and the DDRM ft arms (not
yet run) get 2e-5 the same way. The fix below is behaviour-neutral on purpose. Fix:
13's dead `* FINETUNE_LR_SCALE` removed from both diffusion cells and the comments now say
what happens; `load_checkpoint` prints the effective lr so the next one is in the log.

**3. 13's kin fresh arms ran 45 epochs, below the 50-epoch minimum.** The kin cell passed
`min_epochs=50` with `max_epochs=45`. `train_unet` stops at max, and early stopping needs
`ep >= min_epochs`, so every kin fresh arm ran exactly 45 epochs with early stopping unable
to fire. *Numbers this touches:* `kin_gamma0_mae_fresh` 30.5207 and
`kin_gamma0_wavelet_fresh` 38.8636, plus the starlet/gradient fresh arms. The 30.52 in
particular may be budget-limited, and only a rerun would say. Fix: `max_epochs=60` (as the
sg cell already had). The finished kin rows are not retrained by this, since resume skips
them; retraining the four fresh arms is a separate, optional ~GPU cost.

---

## 2026-09-20 | bug | RAM leak root cause found and fixed: DataLoader worker fork-storm

05 Versions 33 and 34 (pulled `fe52be5`/Version 32, and `c7083fb`) both died on the exact
same arm, `winner_res480_fresh`, at RAM 0.9/31.3 GB free, no traceback -- host OOM-kill.
Archived at `results/05-unet-line-emission/v33_2026-09-20_crashed/` and
`v34_2026-09-20_crashed/`. Between them: all 12 loss-sweep + res-sweep arms reconfirmed
(V33's `winner_res480` PSNR 40.1817, `winner_res320_fresh` 38.9608; V34 retrained the same
set from scratch since a failed version's Output cannot be attached, same numbers), plus
`winner_gradient_ft` finally has a value (40.1396) after two sessions deferred.

**Root cause, found by inspection rather than another blind cap-and-retry:**
`train_unet`'s `DataLoader`s (`src/training/sweep.py`) never set `persistent_workers=True`.
With Kaggle's `num_workers=4` and PyTorch's default `persistent_workers=False`, BOTH loaders
tear down and respawn all 4 worker processes at the end of EVERY epoch's iteration -- a
fork-storm every epoch, for the whole run. This explains what three prior entries could only
describe: a steady ~0.3 GB/epoch decline, IDENTICAL regardless of arm or image resolution
(matching fixed per-process fork overhead, not data volume), that no `gc.collect()` or
`empty_cache()` call could touch because it isn't Python-heap or GPU memory.

**Fixed**, same pattern, everywhere a `DataLoader` sets `num_workers>0`: `sweep.py`
(`train_unet`, covers 05 and 13's kin/sg sections), notebook 13's DDPM/DDRM cells
(`_nw=4`), and notebooks 06 and 07 (`nw=4`/`num_workers=2`). Notebooks 10 and 11 use
`num_workers=0` (no workers spawned) -- not exposed, left alone.

Not yet re-verified against an actual Kaggle run. Next session's `RAM free` trend across
epochs is the test: flat (or only GPU-driver-noise-level drift) confirms the fix; still
declining means the leak has a second source.

## 2026-09-19 | run | both notebooks finish clean for the first time -- KeyError fix holds, RAM leak margin tightening

05 Version 30 (`22b01ae`) and 13 Version 5 (`6b3e6ea`), both pulled `d93803d`, both verified
against it (cell counts and every fix marker intact, nothing reverted), both archived at
`results/05-unet-line-emission/v30_2026-09-19_22b01ae/` and
`results/13-checkpoint-loss-sweep/v5_2026-09-19_6b3e6ea/`.

**05**: the crashed session's Output (2026-09-18, likely Version 29, never confirmed) was
never attached -- Kaggle can't attach a failed run and it wasn't manually recovered -- so
`winner_mae_ft`/`wavelet_ft`/`starlet_ft` retrained from scratch instead of resuming,
consuming the full `MAX_NEW_ARMS_PER_SESSION=3` cap on arms already done once. Numbers moved
slightly (expected, not a discrepancy) and all three still beat `sweep_winner_aug` (PSNR
39.30, M0 +29.2%, M1 +74.0%, M2 +55.0%) on every metric -- `winner_starlet_ft` (PSNR 40.32,
M0 +42.6%, M1 +80.5%, M2 +81.1%) is now the best arm in this notebook's history on every
metric. Section 6 completed this time; the `KeyError` fix (`d93803d`) held.

**13**: `kin_gamma0_mae_ft`/`_fresh` resumed correctly this time (CSV + checkpoint both
matched, `_import_prior_nb13` worked as designed). Trained `kin_gamma0_wavelet_ft` (PSNR
42.88, now the best kin_gamma0 arm) and `_fresh` (38.86), deferred starlet/gradient and all
of sg/ddpm/ddrm.

**RAM leak still not traced, and the margin is tightening.** 05: 28.2 -> 17.9 GB over 98
epochs (~105 MB/epoch). 13: 27.8 -> 3.3 GB over 86 epochs (~285 MB/epoch) -- closer to
exhaustion than Version 4's run (6.5 GB left at the same cap). The cap is sized in ARMS, not
epochs or GB, so it is not a fixed safety margin -- a future session with epoch-heavier arms
under the same cap could still hit zero. Worth an actual trace before it does.

## 2026-09-18 | bug | 05 crashed on a KeyError the session cap exposed; three loss arms beat winner_aug on every metric

Kaggle version unconfirmed (pulled `1832c54`), notebook downloaded and archived at
`results/05-unet-line-emission/v_pending_2026-09-17_1832c54/` pending the number (RULES.md
#10 -- rename once known). This is a real bug, not host RAM: `MAX_NEW_ARMS_PER_SESSION=3`
correctly trained 3 arms and deferred the rest (`winner_gradient_ft` included), then section
6's moment-map table crashed with `KeyError: 'M0'` trying to print `winner_gradient_ft`'s
row. `band_mom[name]` is built for every `CONFIGS` name unconditionally, valued `{}` when
nothing scored it yet; `if name not in band_mom: continue` never catches an empty-but-present
value, only a missing key. Section 6 onward, `collect_outputs` included, never ran.

Found the identical pattern in section 6d's headline figure (cell 24, `_arms = [n for n in
CONFIGS if n in band_mom]`) before it could cause a second crash on the next run -- fixed
both, same guard (`band_mom.get(name) and all(m in band_mom[name] for m in moments)`).

**Results that survived, before the crash -- all three beat `sweep_winner_aug` (PSNR 39.30,
M0 +29.2%, M1 +74.0%, M2 +55.0%) on every metric:**

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| `winner_mae_ft` | 39.93 | +40.1% | +77.1% | +70.0% |
| `winner_wavelet_ft` | 40.18 | +42.4% | +76.1% | +58.4% |
| `winner_starlet_ft` | 40.14 | +39.3% | +75.8% | +59.9% |

`moment_improvement`, clipped + signal-masked (RULES.md #6), 1 seed each -- not wiggle-scored
yet, this is the ranking metric, not the kinematic diagnostic that actually answers the
smoothing question. First time this project's loss sweep has moment-map evidence, not just
PSNR, and the direction agrees with PSNR for once: real, not just a pixel-metric artifact.

## 2026-09-18 | run | 13's first clean finish -- LR fix confirmed, RAM leak still present but contained

Kaggle Version 4 (exact number unconfirmed, author's next `Add Input` push will settle it),
`afb2bbd` pulled and verified live (`_arm_tag`/`no checkpoint -- will retrain` markers present
in the log). `MAX_NEW_ARMS_PER_SESSION=2` did its job: trained `kin_gamma0_mae_ft` and
`kin_gamma0_mae_fresh`, deferred the other 26 arms, finished clean, `collect_outputs` ran.

`kin_gamma0_mae_ft`: PSNR **42.1972**, SSIM 0.9954, 30 epochs, `lr` correctly scaled to 8.2e-5.
Up from **35.6845** at the broken full-lr run two sessions ago -- the `FINETUNE_LR_SCALE=0.1`
fix (`1832c54`) is confirmed doing real work, not a cosmetic change. `kin_gamma0_mae_fresh`:
PSNR 30.5207 (45 epochs), consistent with the prior fresh run's 30.3696. Neither wiggle-scored
yet; PSNR only, 31-channel val set, not comparable to 05's 1-channel PSNRs (RULES.md #4).

**RAM leak not located, but no longer fatal.** Free RAM 27.8 -> 6.5 GB over the two arms
(~2.3h, 75 epochs combined) -- roughly 280 MB/epoch, matching the slope from the crashed
runs. The session cap stopped training before it reached zero, so this is contained rather
than fixed. Worth a real trace once GPU-hour pressure allows it.

`sg_k3_fresh`/`ddpm_seed42`/`ddrm_prior` source checkpoints all located correctly from
`exxa-13-checkpoint-sources` -- those three sections are ready, just deferred by the cap.

## 2026-09-16 | bug | both notebooks killed by host RAM; fine-tune arms ran at from-scratch lr

**Runs:** 05 Version 28 died at 25367.8s, 13 Version 3 at 11907.4s. Both: Kaggle's "tried to
allocate more memory than is available", `DeadKernelError`, no Python traceback -- host RAM
(the ~30 GB CPU side), not GPU. So `a74022b`'s `gc.collect()`/`empty_cache()` after each arm
was aimed at the wrong memory. 05 v27 died at 25240s and v28 at 25368s despite different arms
and epoch budgets: a steady per-step leak, not one bad arm. 13 v3's `kin_gamma0_wavelet_finetune`
epoch times climbed 110s -> 165 -> 270 -> 334 -> 406s before the kill, the page-cache squeeze
of RAM filling; the two arms before it held ~100s for 75 epochs. **Leak not located.** Reading
`FITSChannelDataset` found no cache and no memmap views surviving `_to_native_float32`.

**Results that did survive (13 v3, persisted, attributable):** `kin_gamma0_mae_finetune` PSNR
35.6845 / SSIM 0.9917 (30 epochs, full lr -- see below, superseded recipe);
`kin_gamma0_mae_fresh` PSNR 30.3696 / SSIM 0.9724 (45 epochs). Pixel metrics only, not
wiggle-scored, 31-channel val set -- not comparable to 05's 1-channel PSNRs (RULES.md #4).

**Second bug, caught in the same log:** fine-tune arms used the arm's from-scratch lr
(8.2e-4). Both v3 fine-tune arms spiked at epoch 3 (mae val 0.0096 -> 0.0369; wavelet 0.0005 ->
0.0055) -- the pretrained weights knocked out, so "fine-tune" became a worse from-scratch run.
Likely also why `winner_mae` fell 39.81 -> 35.89 in v27. Notebook 10 already had
`FINETUNE_LR_SCALE = 0.1` for exactly this reason; 05 and 13 never adopted it. Also in 13:
the DDPM section ran FRESH arms at the fine-tune lr (2e-5), 10x below 06's from-scratch 2e-4.

**Fixed:** `FINETUNE_LR_SCALE = 0.1` in 05 and 13 (U-Net and diffusion); 13's DDPM base lr
2e-4. Fine-tune arms renamed (`_ft`) so v27/v28/v3 full-lr rows stay their own record and
retrain, rather than resume counting them as the current recipe. `MAX_NEW_ARMS_PER_SESSION`
(05: 3, 13: 2) stops a session cleanly before the leak can kill it; next session resumes.
`train_unet` now prints free host RAM every epoch (`/proc/meminfo`) and runs `gc.collect()`
per epoch, so the next log shows the leak's slope instead of just its end.

**Cost reality:** at 2-3 arms per session, 13's 28 arms need ~14 sessions and 05's ~18 new
arms ~6 -- well past one account's 30 GPU-h/week. Trimming scope is the author's call.

## 2026-09-15 | added | 05's resolution arms rebuilt as winner_aug_seed43's recipe, from scratch

Author direction: 320/480/600 should reproduce the best model's recipe exactly, trained from
scratch -- fine-tuning from 256px weights risks carrying that resolution's smoothing into the
new one. Checking the existing res arms against `winner_aug_seed43` found they were NOT the
same recipe: no D4 augmentation (the defining feature of `winner_aug`) and seed 42, not 43.

Replaced `winner_res320`/`_fresh`, `winner_res480`/`_fresh`, `winner_native600`/`_fresh` with
three arms: `winner_aug_res320`, `winner_aug_res480`, `winner_aug_native600`. WINNER
hyperparameters, `augment=True` on the 320/480/600 training views (val un-augmented), seed 43
via a new `SEED_OVERRIDE`, `min_epochs=50`, no `init_from`. One forced deviation: batch 8/6/4
instead of 16 for T4 memory, with lr kept identical rather than rescaled (noisier gradients --
a confound to name when reading the result). Seed 43 was winner_aug's best of three, inside
08's ~1 dB seed spread, so matching it keeps the recipe but won't carry that seed's luck.
`winner_aug_native600` still gated by `RUN_NATIVE600 = False`.

## 2026-09-15 | added | epoch budgets raised: fine-tune 6->30, fresh 20/15/10/6->50, across 05 and 13

`winner_mae`'s only completed fine-tune arm from the crashed run (see the bug entry just
below) stopped at epoch 10 on a 6-epoch floor -- author's call after seeing it: too thin to
trust the plateau, especially on a loss that changed scale entirely (MAE vs the original
hybrid MSE+SSIM). Raised uniformly: every fine-tune arm's `min_epochs` 6->30 (05's 14
loss/resolution arms; 13's kin/sg/ddpm/ddrm sections, replacing the earlier
per-checkpoint-matched floors of 15/10). Every fresh-init arm's `min_epochs` -> 50 (was 05's
shared default of 20; 13's ddpm/ddrm sections' 30).

**Cost, revised up substantially from the 20-35h estimate two entries below:** roughly 5x
the epoch floor on fine-tune arms and 1.7-8x on fresh arms (depending which arm's old
budget). Rough recompute for 05's new arms alone: ~9h floor on the 12 aug/p10/beam-source
fine-tune arms, ~3-4h on res320/res480 fine-tune, ~5h on the four fresh losses, ~4-7h on
res320/res480 fresh -- **21-25h just for 05's new arms' minimums**, before patience can push
any of them longer, and before 13's four sections (also raised) are counted at all. This is
a real, large increase, not a rounding change -- flagging it plainly rather than
understating it because the earlier estimate turned out low once already.

## 2026-09-15 | bug | 05 had no per-arm persistence, 7h run died and lost `winner_wavelet` entirely

`05-unet-line-emission.ipynb` NEVER had per-arm checkpoint persistence to `/kaggle/working`
-- section 9 was the only copy step, at the very end of the whole notebook, a pre-existing
gap that predates this session. Adding 20 loss/resolution-sweep arms onto it pushed one
session to ~7h (`DeadKernelError` at 25240.5s) before section 9 ever ran.

**Caught by:** reading the crash log the user pasted after the session died -- kernel death
traceback plus `papermill` writing `__notebook__.ipynb` (169905 bytes) as the final artifact,
confirming section 9 never executed.

**Consequence:** `winner_mae` seed 42 finished training (PSNR 35.8889) but its checkpoint
was never copied out of the ephemeral git clone -- the number survives only as printed text
in whatever Kaggle version the crash produced, not in a reloadable CSV or `.pth`. Nothing
downstream (moment maps, wiggle scoring) can be built on it without retraining.
`winner_wavelet` seed 42 was 5+ epochs in (val 0.0007, improving) when the kernel died;
`train_unet` only writes `ckpt_path` when the call **returns**, so nothing was ever written
for it at all -- total loss, retrains from scratch. Everything queued after it in `CONFIGS`
order never started. This is exactly notebook 06 v11's incident (RULES.md #1), in a
different notebook, because the fix from that incident was never backported here.

**Fixed:** `persist_ckpt()` defined in cell 6, called after every arm in both the
fresh-training and checkpoint-found-scoring branches of section 4's loop, copying the
checkpoint and `nb05_seed_repeats.csv` to `/kaggle/working` immediately. Cell-order test
passes. Not yet re-verified against an actual Kaggle run -- next session's crash-resistance
depends on this actually being pulled before running (RULES.md #2).

**Not yet done:** the crashed run itself needs archiving per RULES.md #10, failures included
-- Kaggle version number needed from the author, not visible from here.

No published numbers affected -- nothing from this run reached RUNS.md or the blog.

## 2026-09-15 | added | loss sweep expands to all 7 best_models checkpoints, plus a stale 06 restored

Continuation of the same day's earlier entry. Mentor direction (mentee call, 2026-09-12) was
to try MAE/wavelet/starlet/gradient losses and a 320/480px compromise; follow-up direction
this session was to run every arm BOTH fine-tuned from an existing best checkpoint AND fresh
(random init) -- not assumed to answer the same question -- and to cover all 7
`models/best_models/` checkpoints, not just `winner_aug_seed43`.

**`05-unet-line-emission.ipynb`**: the four losses now run from three fine-tune sources
(`winner_aug_seed43`, `winner_p10_seed44` -- highest untested PSNR/SSIM, `winner_beam_seed42`
-- needs the `beam` view) plus a `_fresh` random-init counterpart of each, so a source-name
comparison and a fine-tune-vs-fresh comparison are both possible without conflating them.
`winner_res320`/`winner_res480` get the same fine-tune/fresh split. 20 new arms total, on top
of the original 8. None need an external checkpoint upload -- all three fine-tune sources
train inside this same notebook, earlier in `CONFIGS`' dict order, so the checkpoint exists
by the time the loss arms need it.

**`13-checkpoint-loss-sweep.ipynb`** (new): covers the four checkpoints that don't fit 05's
1-channel line-emission U-Net shape -- `kin_gamma0` (31-ch spectral, kinematic_gamma pinned
at 0 to match its own training), `sg_k3_fresh` (7-ch spectral, self-gravitating domain, not
line-emission -- a scientific stretch, flagged as such rather than assumed to transfer),
`ddpm_seed42` and `ddrm_prior` (both diffusion). The diffusion pair needed a real design step,
not a mechanical loss swap: `noise_estimation_loss` (`src/training/diffusion.py`) gained
`loss_type` ("l1"/"l2" on the noise/v residual, the diffusion analogue of MAE) and
`aux_loss_name`/`aux_weight` (an optional wavelet/starlet/gradient term on the predicted-clean
estimate x0_hat, inverted from the noise/v prediction by the standard DDPM identity --
`WaveletLoss`/`StarletLoss`/`GradientLoss` have no meaning on a noise residual directly).
`AUX_WEIGHT` in the notebook is flagged explicitly as an order-of-magnitude guess (primary
loss sums squared error over ~65k pixels; the aux functions are per-pixel means), not tuned --
inspect the loss curve before trusting it.

Every arm in both notebooks: `finetune` (min_epochs=6, cheap) and `fresh` (min_epochs matching
that checkpoint's own original training budget). Fine-tune checkpoints for 13 staged at
`models/_kaggle-upload/13/` (gitignored, hardlinked, `.ckpt` extension per RULES.md #3) --
still needs manually creating the actual Kaggle Dataset and attaching it, which nothing here
can do without Kaggle credentials.

**Cost, not yet paid:** rough estimate from per-epoch timings elsewhere in this project puts
05's new arms at 7-15 GPU hours and 13's at 10-20+ GPU hours (the diffusion sections are the
uncertain part -- DDIM eval at K_AVG=4/25 steps is not cheap and nothing in this repo gives a
solid per-epoch number for it). 20-35 GPU hours combined exceeds one 12h Kaggle session for
either notebook alone; both resume correctly via their CSV skip-already-done logic, so this
is safe to spread across sessions but is not a one-run cost.

**Bug, separate from the above:** `06-ddpm-line-emission.ipynb`'s working-tree copy (present
before this session started, not produced by it) had reverted three committed fixes against
HEAD -- the `RESCALE_TO_DIRTY` pedestal fix, `RUN_SWEEP` flipped back to `True` (re-running a
sweep already marked "ANSWERED TWICE"), and the entire `## 13b. Diagnostics` section, dropping
cell count 47->45. Exactly the RULES.md #2 failure mode: Kaggle's own push sent a stale copy
back over committed work. Restored from HEAD (`git stash` first, the stale copy recoverable
if this diagnosis is wrong); cell-order test passes at 47 cells. Not yet re-verified against
Kaggle -- next session there needs the same check (rule 2: confirm what got pulled before
building on it).

## 2026-09-15 | added | 05 gains a loss-fn sweep and a resolution sweep, not yet run

Mentor direction (mentee call, 2026-09-12): the smoothing complaint on 256px output may be
loss-driven (try MAE), resolution-driven (600px trains huge, try a 320/480 compromise), or
both. `src/utils/losses.py` gained `MAELoss`, `WaveletLoss` (multi-level Haar DWT), `StarletLoss`
(a trous B3-spline, the transform the mentee already validated on real ALMA data), and
`GradientLoss` (Sobel edge L1, the pix2pix-style term without a discriminator). None need a
new dependency -- all fixed-kernel `conv2d`. `train_unet` (`src/training/sweep.py`) gained
`loss_name` (`LOSS_REGISTRY`), written into the checkpoint so `best_val_loss` stays
attributable to the objective that produced it (RULES.md #4/#6).

`05-unet-line-emission.ipynb` gained six exploratory (single-seed) arms: `winner_mae`,
`winner_wavelet`, `winner_starlet`, `winner_gradient` (all 256px, WINNER config, loss_name
swapped), `winner_res320`, `winner_res480` (WINNER config, new 320px/480px dataset views,
smaller batch). `winner_native600` stays gated off (`RUN_NATIVE600 = False`, RULES.md #1 --
has never once completed). None of the six are gated; they run by default alongside the
existing arms next session.

**Bug caught before running anything:** the "checkpoint found, scoring without retraining"
resume branch only special-cased `n_neighbors`/`use_beam` when picking the val set, so a
resumed `res320`/`res480`/`native600` run would score PSNR on the 256px `val_ds` while a
freshly-completed run of the same arm scores on its own view's val set -- same config name,
two metric bases, exactly what RULES.md #4 exists to prevent. Fixed: the resume branch now
uses `VIEWS[view]()[1]` for those three arms.

**Nothing measured yet.** Cell-order test passes; no Kaggle run has happened. Next: push
through Kaggle, verify a marker (`winner_gradient` in the loaded cell) per RULES.md #2, run.

## 2026-09-12 | added | PLAN.md and RUNS.md caught up to Phase J's close

Two doc gaps found while updating context.md yesterday, closed today. `PLAN.md`'s Block 2
still described the original single-cube ALMA verdict, which the headroom scatter now
predicts would likely just be an eighth instance of "doing nothing wins" -- real DSHARP dirty
data tends to sit in the low-degradation regime this project's own figure says models lose in.
Rewrote Block 2 as the degradation-axis redesign discussed 2026-09-11: inject the synthetic
signal into real DSHARP dirty at several noise/config levels, score recovery the same way
`headroom_scatter.png` does. Week 3 changes from a free overrun buffer to part of the sweep
itself, since this needs more `simobserve` runs than one. Exit criterion changed from "one
cube scored" to "at least 3 points on a real-noise degradation-vs-gain curve."

`results/RUNS.md` never had a section for notebook 12 (spectral context, the `k=3` result the
entire Phase J closing chain was triggered by) -- only `models/README.md` documented it.
Added, matching notebook 11's format, including the SG v2 non-transfer result and a link back
to the six-test chain. Flags an open item: notebook 12 has never been archived per RULES.md
#10 (no `results/12-sg-spectral-context/v<N>_.../` folder exists), unlike every other
notebook in the project. Deferred, not fixed today -- close before Block 3's cleanup pass.

---

## 2026-09-11 | run | headroom scatter: the SG thread's unifying figure, built from existing data only

Assembled from three already-logged JSON/PROGRESS sources, no new denoising: `sg_v2` (6
checkpoints, one cube, today's leaderboard + domain-split runs), `sg_loo_wiggle.json` (5 folds
x 3 trained arms, each fold's own dirty), `nb08_kinematic_wiggle.json` (5 cubes x
gamma=0/0.1, each cube's own dirty). Real per-point data, not per-regime means -- 31 points
total, each plotted at its OWN dirty resid_r rather than a pooled average, since the x-axis
IS the quantity the thesis is about and pooling would hide the within-regime spread that
makes it legible.

`experiments/headroom_scatter.py` -> `results/self-gravitating/headroom_scatter.png`.

Mean kinematic gain (model resid_r - dirty resid_r) by regime:

| regime | n | x range (dirty resid_r) | mean gain |
|---|---|---|---|
| line-emission holdouts (gamma=0/0.1) | 10 | 0.118 - 0.996 | **+0.391** |
| SG leave-one-out folds (frozen/finetune/fresh) | 15 | 0.468 - 0.996 | -0.089 |
| SG v2 cube (6 checkpoints) | 6 | 0.862 (fixed) | -0.163 |

**The scatter shows a monotonic decline in gain as dirty's own resid_r rises, not just three
separated clusters.** Every line-emission point sits above zero except one (`run_0002_00560_
rt_00`, dirty resid_r=0.996 -- the near-undetectable-wiggle cube flagged back in the original
kinematic-gamma sweep entry, sitting at gain~0 because there is no headroom there either,
consistent with the trend rather than an exception to it). LOO folds straddle zero, tilting
negative as their own dirty rises within the regime (compare the frozen arm at dirty=0.468,
gain=-0.01, against the same arm at dirty=0.723, gain=-0.08). SG v2's six points cluster at
the high-dirty end, all negative.

**This is the figure that reconciles the project's two headline results without either one
being wrong.** "Doing nothing beats every model" (SG v2, Phase H onward) and "kin_gamma0
crushes dirty" (notebook 08, today) are the same mechanism measured at opposite ends of one
axis, not a contradiction requiring a tie-breaker. Model value here is a property of how
degraded the input already is, not of architecture, training domain, or loss function --
every one of which was tested and shown NOT to explain the SG v2 losses in the preceding four
entries. This closes the SG investigation thread that opened with "why is the denoised image
smooth" three system-turns ago.

---

## 2026-09-11 | run | DDPM K_AVG=1 on SG v2: deliberately not run

The row was queued after v13 diagnosed the DDPM's moment collapse as posterior-mean averaging
over K draws, and K_AVG=1 is the natural counterfactual. Not run here because no outcome would
change a decision: DDRM, the diffusion family's measurement-consistent member, is the worst
row on this cube (0.568), so the family verdict on kinematics is already in from v13 and the
DDRM comparisons; and on this cube the headroom term (dirty at 0.862, headroom 0.109 to the
target ceiling) dominates any sampler-level effect, so a wiggle number here would confound the
pedestal mechanism with the input-quality regime rather than isolate it. Best realistic case
is DDPM moving from worst to mid-pack, still below dirty -- a seventh losing row bought at the
cost of porting the DDIM sampler (schedule reconstruction, EMA weights, normalisation
inversion), the exact class of work that has hidden a bug every time in this project (the
DotDict pickle break, the missing F import, the silent `beam=None`). The pedestal test belongs
where it was diagnosed, on the line-emission moment collapse, and remains open there. Not-run
with rationale is recorded instead of a number that cannot be interpreted.

---

## 2026-09-11 | run | leaderboard extends: winner_p10 and winner_beam also below dirty, beam OOD confirmed with real numbers

`experiments/wiggle_leaderboard_sg.py`, two more line-emission checkpoints on the SG v2 cube,
same protocol as every run in this thread. `winner_beam`'s beam vector taken directly from the
SG dirty header via `beam_features_of()`, confirmed real and non-zero before running
(`[0.983, 0.185, 0.160, 0.100]` -- BPA/BMAJ/BMIN are present in the header), not a degenerate
zero-vector test. `beam_dim=4` verified against the checkpoint's own stored value before
forcing it, and the state dict load asserted exact (no `strict=False` skip-on-mismatch).

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8620 |
| winner_aug_seed43 | 0.218 | 0.9821 | 0.7603 |
| winner_p10_seed44 | 0.219 | 0.9757 | 0.6912 |
| winner_beam_seed42 | 0.210 | 0.9794 | 0.7113 |

Both land below `winner_aug`, both below dirty. `winner_beam`'s residRMS (0.210) is actually
the lowest of the three trained models, closer to dirty's 0.213 than either of the others --
and it still loses on resid_r. Another instance of the RULES.md #6 lesson: RMS and the
wiggle's resid_r are not interchangeable, a method can look best on one and mid-pack on the
other because RMS doesn't see whether the residual PATTERN survives.

**Now six checkpoints tested on this cube** (winner_aug, winner_p10, winner_beam, sg_k3_fresh,
kin_gamma0, DDRM), spanning 3 architectures, 2 domains, 2 loss variants, a patience/early-stop
change, and a beam-conditioning arm run out of its training distribution. All six land in
0.568-0.760, all six lose to dirty's 0.862. This is now a strong pattern, not a handful of
coincidences: nothing tried has closed even half the gap to doing nothing on this specific
cube. The standing read from the last two entries holds and strengthens -- the SG v2 cube
cannot rank models, and the objective (or the input-quality regime, see the headroom table in
the domain-split entry) is where the remaining gap lives, not any one architecture or
checkpoint choice tried so far.

DDPM K_AVG=1 remains the one queued, unrun row (needs porting the DDIM sampler from
06-ddpm-line-emission.ipynb, not attempted inline).

---

## 2026-09-11 | run | rendering audit: the smoothness is in the M1 pixels, the renderer is exonerated

Before running any more checkpoints, settled whether the smooth look in the published M1
panels is in the arrays or in the mask/contour rendering. Logically the audit could only ever
change a presentation question -- mask, moment estimator and rendering are applied identically
to every panel, so they cancel in a between-method comparison and cannot manufacture the
dirty-0.862-vs-U-Net-0.760 gap, which is computed on raw arrays before any renderer runs.
`clean` is the control: if clean stays sharp through the same estimator, mask and rendering
while the U-Net goes smooth, the renderer is not the cause.

**First, a grep that shortcuts most of the question:** `wiggle_all_methods.py`,
`wiggle_patch_unet.py` and `wiggle_domain_split.py` contain **no `gaussian_filter` and no
`contour` call at all**. Every published panel is a raw `imshow` of the raw M1 with NaN
outside the mask. The contour-smoothing code in `moment_maps.py:435` is in a different
function that none of these scripts call. Contouring was never in the path.

`experiments/m1_rendering_audit.py`, plot-independent sharpness on the raw M1 arrays:

| cube | gradE (masked) | gradE (all px) | lapvar (frame) | hf>1/10 | hf>1/5 |
|---|---|---|---|---|---|
| clean | **4.994e-03** | 1.073e-03 | 7.569e-03 | 0.0308 | 0.0182 |
| dirty | **3.029e-03** | 1.447e-03 | 4.807e-03 | 0.0292 | 0.0172 |
| winner_aug (resize) | **1.928e-03** | 1.488e-03 | 5.324e-03 | 0.0289 | 0.0172 |

**Verdict: the masked gradient energy is decisive and the renderer is cleared.** Inside the
mask the U-Net's M1 carries 0.386 of clean's gradient energy against dirty's 0.607 -- a real,
large, monotonic difference in the RAW arrays with no mask or contour involved in the metric.
The ordering clean > dirty > U-Net matches both the visual impression and the resid_r ordering
(1.0 > 0.862 > 0.760) exactly.

**Honest note on the other three metrics: they are uninformative here, and should not be
quoted as if they corroborate.**
- `gradE (all px)` and `lapvar` are computed outside the mask as well, where M1 is fitted to
  noise and has no physical content. Both dirty and the U-Net exceed clean there simply
  because both carry noise where clean has none. The pre-registered mask-artifact trigger was
  "masked and all-pixel disagree for the U-Net ONLY"; they disagree for dirty (1.348) and the
  U-Net (1.386) almost equally, so the trigger does not fire and there is no mask artifact --
  but the column establishes that, not the main claim.
- The `hf` spectral fractions land within 0.0289-0.0308 for all three cubes. The mask-zeroing
  injects a hard edge whose broadband power dominates the FFT, and the mask is shared, so the
  ratio is set by the mask edge rather than by the disk. A null from a metric that turned out
  to be measuring the wrong thing, not evidence of similarity.

The 4x3 rendering matrix (`m1_rendering_audit.png`) shows the same ordering under imshow
no-mask, imshow masked (the published path), contourf-24 no-mask and contourf-24 masked. The
unmasked imshow column is the clearest: clean is visibly mottled with spiral texture, dirty
intermediate, the U-Net a smooth dipole with almost no texture. `_60lev.png` confirms level
quantization at 60 levels does not recover structure in the U-Net panel.

**Consequence: every published M1 figure stands as-is.** No reissue needed. The smoothness
readers see is a property of the model output, which is what those figures were drawn to show.

---

## 2026-09-11 | run | spectral context does NOT transfer to the SG v2 cube -- every trained checkpoint lands ~0.10 below dirty

Ran `kin_gamma0` (notebook 08's 31-channel stack, the only checkpoint with a large measured
win where there was headroom) on the SG v2 cube, through its own training preprocessing
(`subtract_continuum=True`, 31 clamped neighbours, min-max shared from the centre dirty
channel, `stack_target=True` so only the centre output channel is read, continuum added back
so the result lands in the same raw space as clean/dirty). 8.8 min on mps.
`experiments/wiggle_domain_split.py`, extended to carry all three checkpoints.

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| **dirty (do nothing)** | 0.213 | 0.9892 | **0.8620** |
| `winner_aug_seed43` (line-em, 1 channel) | 0.218 | 0.9821 | 0.7603 |
| `kin_gamma0` (line-em, 31-channel stack) | 0.217 | 0.9814 | 0.7581 |
| `sg_k3_fresh` (SG, 7-channel stack) | 0.242 | 0.9783 | 0.7053 |

**Spectral context does not transfer.** `kin_gamma0` scores 0.7581 here against
`winner_aug`'s 0.7603 -- indistinguishable, and slightly worse. The same checkpoint scores
0.8155 against its line-emission holdouts' dirty at 0.4284. The mechanism that wins there
does nothing here.

**The decisive pattern: three architecturally different models (1 / 7 / 31 input channels),
two training domains (line emission, self-gravitating), two loss variants (MSE, MSE +
kinematic term) all land in 0.705-0.760, and every one of them loses to dirty's 0.862.** The
model-to-model spread (0.055) is about half the gap from the best model to doing nothing
(0.104). When every model in a family that different clusters that tightly and all fall on the
same side of the baseline, the result is a property of the SETUP, not of any model's design.
The M1 panels show it directly: all three are visibly smoother than both clean and dirty and
nearly identical to each other.

**What is now tested and ruled out as the explanation**, each with a number:

| hypothesis | test | verdict |
|---|---|---|
| inference-time resize | patch inference | real but 0.025 |
| training resolution ceiling | clean through the 600->256->600 target path | 0.029, scores 0.9714 |
| training domain mismatch | SG-trained model on SG cube | refuted, it did WORSE |
| architecture / spectral context | `kin_gamma0` (k=15) on SG cube | refuted, matches the 1-channel model |

What remains is the objective and the benchmark's headroom, and those two are not separable
with what is on hand. Note the one genuinely non-MSE method tried on this cube, DDRM, is the
worst result in the entire comparison (0.568), so "swap the objective" is a hypothesis with a
discouraging first data point, not a demonstrated fix. State it that way in the writeup.

**Practical consequence: the SG v2 cube cannot rank models.** Every checkpoint scores
0.70-0.76 on it. It is a benchmark that says "do not denoise this cube", not one that says
which denoiser is better. Model selection has to happen on the line-emission holdouts and the
leave-one-out folds, where dirty sits at 0.428 and 0.664 and there is room to distinguish.

---

## 2026-09-11 | run | domain split: SG-trained loses to line-emission-trained on the SG cube -- domain is NOT the 0.186

Splits the 0.186 "loss/learning" term from today's resolution-vs-loss test into DOMAIN
(`winner_aug_seed43` trained on line emission, run on an SG cube) versus OBJECTIVE (MSE
posterior-mean averaging). `experiments/wiggle_domain_split.py`, same cube, same frac=0.05
mask, same shared geometry (mstar=0.643, matches every prior run of this comparison), each
checkpoint through its OWN training preprocessing read off the training code -- notebook 12's
`subtract_continuum=False`, 7 clamped neighbours, min-max shared from the centre dirty
channel, resize 256.

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8620 |
| `winner_aug_seed43` (line-emission trained) | 0.218 | 0.9821 | 0.7603 |
| `sg_k3_fresh` (SG trained, k=3) | 0.242 | 0.9783 | **0.7053** |

**The SG-trained model is WORSE here than the line-emission one, and both lose to dirty.**
Domain is not the explanation for the 0.186. `sg_k3_fresh`'s residual RMS (0.242) also
exceeds clean's own (0.233), so it is inflating residual amplitude rather than recovering
signal -- the same signature DDRM shows, and the M1 panel is visibly broader/smoother than
`winner_aug`'s.

**The cross-check is the real finding.** `sg_k3_fresh` BEAT dirty on its own notebook 12
holdout (0.681 vs that cube's dirty at 0.594). Here it LOSES to dirty (0.705 vs 0.862). Same
checkpoint, opposite verdict, different benchmark cube. What changed is how degraded the
input is:

| benchmark | dirty's own resid_r | headroom to the ~0.971 target ceiling |
|---|---|---|
| line-emission holdouts (notebook 08, n=5) | 0.4284 | ~0.54 |
| SG leave-one-out folds (notebook 11, n=5) | 0.6644 | ~0.31 |
| **SG v2 cube (this comparison)** | **0.8620** | **~0.109** |

**The SG v2 cube is by a wide margin the least-degraded benchmark in the project**, and it is
the one every "the model loses to dirty" headline comes from (Phase H onward). When dirty
already sits at 0.862, any model that smooths at all loses, because the smoothing is a
systematic error aligned with the signal while the noise it removes is not. When dirty sits
at 0.43, the same class of model wins by a wide margin (`kin_gamma0`: 0.8155).

**Consequence for the standing conclusions.** "Doing nothing beats both models" remains true
AS MEASURED, but its scope is now much narrower than it has been stated: it is a statement
about this one unusually clean cube, not about the models in general. The in-domain results
(notebook 08's `kin_gamma0`, notebook 12's k=3 on its own holdout) are not in conflict with
it and never were -- they are measured where there is headroom to win. Both belong in the
writeup, together, with the headroom column attached. Quoting either alone misrepresents the
result.

Objective/architecture remains the lever for the part that is real (spectral context, twice
confirmed). Domain: refuted. Resolution: 0.029. Inference resize: 0.025.

---

## 2026-09-11 | run | resolution-vs-loss split test: the objective is 6x the resolution term, native-res retrain is NOT the fix

Cheap decisive test, minutes of CPU, no training. Ran the CLEAN SG cube through the exact
training-target path (bilinear 600->256->600, per channel, what `FITSChannelDataset` does to
every clean target) and scored it against untouched clean with the usual shared geometry
(fit on clean, mstar=0.643, `at_bound=False`), frac=0.05, channels 240-360 step 1.

| signal | residRMS | resid r |
|---|---|---|
| clean (reference) | 0.233 | -- |
| clean@256 (the training-target path) | 0.228 | **0.9714** |
| dirty | 0.213 | 0.8620 |
| dirty@256 | 0.213 | 0.8617 |
| U-Net patch (native-res inference) | 0.211 | 0.7853 |
| U-Net resize (existing path) | 0.218 | 0.7603 |

**Decomposition of the model deficit:**

| term | cost in resid_r |
|---|---|
| resolution ceiling (clean -> clean@256) | 0.029 |
| loss / learning (clean@256 -> U-Net patch) | **0.186** |
| inference resize (patch -> resize) | 0.025 |

**The objective term is about 6x the resolution term.** A native-resolution retrain, the
GPU-weeks option that has been deferred twice, can recover at most 0.029 of resid_r on this
diagnostic. That is not where the deficit lives. The network is realizing only 0.785 of the
0.9714 its own 256px training targets already permitted, so the ceiling was never the data.

**Correction to this morning's entry ("resize round trip quantified: 89.6% of real structure
lost").** That measurement stands and is not withdrawn: a pure 600->256->600 round trip does
keep only 10.4% of the clean channel's Laplacian-variance sharpness. But that entry let a
pixel-scale SHARPNESS metric imply the resize was also the dominant cause of the WIGGLE
deficit, and it is not. `dirty@256` scores 0.8617 against `dirty`'s 0.8620 -- the resize is
worth 0.0003 on the wiggle. The GI wiggle evidently lives at a spatial scale coarser than the
~2.3px Nyquist cutoff a 2.34x downsample imposes, so it survives almost intact. Two different
questions with two different answers: **the resize costs visual sharpness, the objective costs
the wiggle.** Both matter, they need different fixes, and they should never again be quoted as
one problem.

**What this redirects.** Objective/architecture changes, not resolution, are where the GPU
time belongs -- which is exactly the direction notebook 12's spectral context (k=3, 0.681 vs
dirty's 0.594) and notebook 08's `kin_gamma0` (0.8155 mean vs dirty's 0.4284, in-domain) have
independently been pointing all along. Those two results and this test now agree.

Patch inference stays worth keeping (0.025, free, no retraining), it is just a small term, not
the lever.


## 2026-09-11 | run | `wiggle_patch_unet.py`: native-resolution patch inference, tested against the resize baseline

Direct test of the resize-vs-quality hypothesis from earlier today, on `winner_aug_seed43`
(no retraining, same checkpoint both ways). Same SG cube, same shared Keplerian geometry
(fit on clean, mstar=0.643, matches every prior run of this comparison), same frac=0.05 mask.
9 tiles/channel, Hann-window overlap-blended, ran on `mps`, resize-based 0.2 min, patch-based
1.9 min (the tile-count multiplier as expected).

| method | residRMS | raw r | resid r |
|---|---|---|---|
| clean | 0.233 | -- | -- |
| dirty | 0.213 | 0.9892 | 0.8620 |
| U-Net (resize, existing) | 0.218 | 0.9821 | 0.7603 |
| U-Net (patch, native res) | 0.211 | 0.9834 | **0.7853** |

**Real, measurable gain (resid_r +0.025, ~3% relative), but does not close the gap to dirty's
own 0.862, let alone approach clean.** This is the expected outcome given today's other
finding: `FITSChannelDataset` resizes to 256 on the TRAINING side too, for every checkpoint in
this project, so `winner_aug_seed43` has never been asked to represent detail finer than
256px. Patch inference removes the inference-side resize and recovers what the network
already learned, it cannot recover detail the network was never trained to produce. Confirms
the ceiling is primarily set during training, not at inference -- a native-resolution (or
patch-based) retrain is the only way past this, still deferred given the Nov 2 deadline and
Block 2 (ALMA) not yet started.

Figure: `results/self-gravitating/wiggle_patch_vs_resize.png`. Visually, the patch residual
panel carries more fine texture than the resize panel, consistent with the small quantitative
gain, still visibly coarser than dirty's own residual.

**Worth keeping regardless of the retrain decision**: patch inference is strictly better than
resize at zero extra cost (no retraining, ~10x runtime for this cube, U-Net only, not
DDRM-prohibitive). Candidate to become the default inference path for `winner_aug_seed43`,
not yet swapped into `wiggle_all_methods.py`/`score_08_kinematic.py`.

---

## 2026-09-11 | run | notebook 08 kinematic_gamma figures land, first run on mps

`figures_08_kinematic.py`, ran clean end to end on `device mps` (the Apple GPU fix landed
mid-way through the main sweep, this is the first script to actually pick it up). Denoised
one representative holdout cube (`run_0002_00560_rt_00`, the mstar=18.69 outlier fit flagged
earlier -- worth rerunning `--cube 1` or later for a less unusual example at some point) at
all 4 gammas, ~36 min total (9-11 min/gamma), faster than the CPU-only estimate but not the
dramatic speedup hoped for; mps has its own overhead on convolution-heavy nets.

`results/self-gravitating/nb08_kinematic_moments.png`: visually confirms the numbers.
gamma=0/0.1's M0 keeps the dirty cube's lobe structure while cleaning up the halo; M1 keeps
the smooth blue/red dipole. gamma=1's M0 nearly loses the lobes, M2 saturates solid yellow
across the whole field (same total-saturation failure mode already seen in DDPM's
sampler-collapse bug, worth noting as a recurring pattern in how these models fail, not
necessarily the same cause). gamma=10 is visibly noisy/grainy throughout, matches its high
per-cube variance in the numbers.

`results/self-gravitating/nb08_kinematic_vs_gamma.png`: one nuance the summary table alone
didn't show -- gamma=10 partially recovers above gamma=1 on 2 of 5 cubes (not a clean
monotonic collapse), though the mean stays well below gamma=0/0.1 either way. Confirms this
is a real per-cube effect, not a mean-hiding-a-clean-trend situation.

---

## 2026-09-11 | bug | resize round trip quantified: 89.6% of real structure lost, not the network's fault primarily

Follow-up to the 600->256->600 resize finding logged earlier today. That entry named the
resize as a second smoothing source alongside the known MSE regression-to-mean effect, but
did not separate how much each actually contributes. Isolated it directly: took the `clean`
channel (300) from `kinematic_data_v2`, noise-free, so any loss measured is genuine structure,
not noise being smoothed away, and ran it through a PURE bilinear 600->256->600 round trip
with NO network involved at all. Sharpness measured as Laplacian variance (`scipy.ndimage.
laplace`, standard no-reference high-frequency-content metric), cropped to the disk region
(the full 600x600 frame is mostly empty background and swamps the metric otherwise).

    clean, native 600x600:                 baseline
    clean, after resize round-trip only:   10.4% of native sharpness

**89.6% of the clean image's own real structure is destroyed by resize alone, zero network,
zero noise.** This is the dominant smoothing source, larger than the MSE regression-to-mean
effect flagged earlier, which only ever acts on an already-blurred 256px image -- the network
cannot recover detail that resize already discarded before inference began. Downsampling
600->256 is a 2.34x reduction, which by Nyquist removes everything finer than ~2.3px, and
that is exactly the scale spiral-wiggle substructure lives at in these disks.

**SCOPE CORRECTION (same day, see the split-test entry above):** this 89.6% figure is a
pixel-scale SHARPNESS measurement and stands as such, but it does NOT explain the wiggle
deficit. `dirty@256` scores resid_r 0.8617 against `dirty`'s 0.8620, so the resize is worth
0.0003 on the wiggle diagnostic. The wiggle lives at a coarser scale than the ~2.3px cutoff.
Read this entry as "resize costs visual sharpness", never as "resize costs the wiggle".

**Bigger implication: this is not only an inference-time artifact.** Every training pipeline
in this project (`FITSChannelDataset`, `target_size=256` default) resizes both dirty AND
clean to 256 for every training sample, on both line-emission and self-gravitating cubes,
both native 600x600. So no checkpoint currently in `models/best_models/` has ever been asked
to represent detail finer than 256px resolution, training-side. Patch-based inference
(`wiggle_patch_unet.py`, built earlier today, not yet run -- CPU still occupied by the
kinematic_gamma sweep and its figure generation) can only recover what a 256px-trained
network actually learned to reconstruct; it removes the inference-side resize but not the
training-side resolution ceiling underneath it. A full fix needs native-resolution (or
patch-based) TRAINING, not just inference, which is the "full retrain" option already
discussed and deferred given the Nov 2 deadline and Block 2 (ALMA) not yet started.

---

## 2026-09-11 | run | notebook 08's kinematic_gamma sweep, all 4 checkpoints scored on moments and the wiggle, 5 holdout cubes complete

`experiments/score_08_kinematic.py`, full run, 282 minutes on local CPU (mps fix landed
mid-sweep so this run itself stayed on the old CPU-only code path; the next run of this kind
gets the speedup). All 5 line-emission holdout cubes' Keplerian geometry fit free (no known
inclination for these, unlike the SG disks) and all 5 came back with `mstar_at_bound=False`
(RULES.md #8 checked, not assumed) -- one fit (`run_0002_00560_rt_00`, mstar=18.69) is a wide
outlier worth flagging but still clear of the 45 Msun (0.9x bound) cutoff.

Summary across the 5 holdouts (mean +/- std):

| gamma | M0 | M1 | M2 | wiggle resid_r |
|---|---|---|---|---|
| 0.0 (control) | -17.2+/-47.2 | +66.8+/-30.1 | +52.8+/-28.1 | **0.8155+/-0.2106** |
| 0.1 | +1.6+/-89.1 | +44.7+/-46.1 | +19.8+/-39.4 | **0.8242+/-0.1713** |
| 1.0 | -200.1+/-137.2 | -189.6+/-170.7 | -222.4+/-134.5 | 0.2323+/-0.2605 |
| 10.0 | -539.7+/-444.8 | -212.5+/-242.3 | -128.3+/-145.3 | 0.3728+/-0.4249 |
| dirty (no model) | -- | -- | -- | 0.4284+/-0.3556 |

**Headline: gamma=0 and gamma=0.1 both crush dirty on the wiggle (0.82 vs 0.43), but they are
statistically indistinguishable from each other.** gamma=0 is the architecture control --
`n_neighbors=15, out_channels=31`, the 31-channel spectral-context stack, with NO kinematic
loss term at all -- and it already gets 0.8155. Adding the kinematic loss at gamma=0.1 moves
that to 0.8242, well inside gamma=0's own std. **The real lever here is the wide spectral
context (31 neighbouring channels), not the kinematic loss term**, which mirrors notebook
12's finding on the self-gravitating side (spectral context wins, at a much smaller k=3).
This is a second, independent confirmation of the same mechanism on completely different data
(line emission vs self-gravitating) and a completely different architecture entry point
(fixed k=15 neighbour stack vs a k-sweep). The kinematic loss term itself is NOT shown to
help by this sweep -- it is not shown to hurt at gamma=0.1 either, that is the honest
reading, not "kinematic loss works."

gamma=1.0 and gamma=10.0 collapse hard, consistent with their exploding val_loss during
training (0.000721 -> 0.001721 -> 0.016901 -> 0.089597): the kinematic loss term dominates
the objective at high gamma and wrecks pixel-level reconstruction, which then wrecks moments
and the wiggle along with it. Not a subtle effect -- gamma=10 goes as low as -1260% M0 on one
cube.

One run-time anomaly: gamma=0.1 on `run_0002_00560_rt_01` took 69.5 minutes against every
other cube/gamma pair's 6-14 minutes. Not investigated further (the sweep as a whole still
finished and every number it produced is internally consistent with its neighbours), most
likely CPU contention from other work running on the same machine during that window rather
than anything wrong with that specific checkpoint or cube.

Saved to `results/self-gravitating/nb08_kinematic_wiggle.json`. Figures
(`figures_08_kinematic.py`) run immediately after, per the standing instruction not to run
them in parallel with the main sweep.

---

## 2026-09-11 | bug | U-Net/DDRM inference carries a 600->256->600 resize round trip, a second smoothing source beyond the loss function

Looking closely at `wiggle_all_methods.png` (regenerated by the notebook 09 v3 Kaggle rerun
below): the U-Net and DDRM M1 panels look visibly smoother than dirty and beam-only, more
than the known MSE regression-to-mean effect alone accounts for. Traced it to a second,
independent cause.

`self-gravitating cube and dirty cube/kinematic_data_v2/clean_sg.fits` is native
**600x600px**. `experiments/wiggle_all_methods.py`'s `unet_denoise()` and `ddrm_restore()`
both bilinear-resize the input down to `SIZE=256` before running the network, then resize
the output back up to 600x600 (`wiggle_all_methods.py:70,72,91,99`). That is a 2.34x
downsample followed by an upsample that cannot recover what was thrown away -- real spatial
detail below about 2.3px is gone before the network ever sees it, and the upsample just
interpolates smoothly between the coarse 256-grid points. `dirty` and `beam-only` never go
through this path; they stay at native 600x600 the whole time, only convolved with the beam
kernel. So this resize artifact hits exactly the two "denoised" columns and nothing else in
that comparison, which is consistent with what the figure shows.

Checked for other smoothing sources before settling on this explanation: the only
`gaussian_filter` call anywhere in `src/evaluation/` is `moment_maps.py:435`, and its own
comment says it draws a smoothed COPY for contour lines only, the displayed pixels are
untouched -- and this script does not even call that function, it builds its own `imshow`
panels directly from `rows[tag]["m1"]`. Ruled out, not the cause here.

**Two ways to fix it, neither of them a quick toggle:**
1. Retrain at native 600x600 (or patch-sized) resolution. Architecturally possible -- 600
   divides cleanly by 8, matching the current 3-downsample `MULTS=(1,2,4,8)` U-Net -- but
   every `TARGET_SIZE`/`SIZE` constant across notebooks 05/06/07/08/10/11/12 and
   `src/data/fits_cube_dataset.py`'s resize step would need to change, and every existing
   checkpoint would need retraining from scratch. Not something to take on lightly this
   close to the Nov 2 deadline with Block 2 (ALMA validation) not yet started.
2. Patch-based inference at native resolution using the checkpoints already trained:
   crop overlapping 256x256 patches directly out of the 600x600 cube (no resize needed, a
   patch already matches the net's trained input size), denoise each patch, stitch with
   overlap-blending. No retraining, only touches the inference/scoring scripts. Agreed as
   the next step to try, since it isolates whether resize is the dominant smoothing source
   without committing to a full retrain first.

Not yet implemented. Logging this now, before ALMA validation starts, since the same
inference pattern (global resize to a fixed square) is used everywhere in the scoring
scripts, not just this one comparison.

---

## 2026-09-10 | run | figure for notebook 11's leave-one-out wiggle (metrics-only, no re-denoise)

Notebook 11 had a JSON (`sg_loo_wiggle.json`, 5 folds) but no figure -- flagged as a gap
during a general audit of missing visual results across the SG-training notebooks. Built
`experiments/figures_sg_loo.py`, reading the JSON directly, no re-denoising: per-fold
`resid_r` bars (dirty/frozen/finetune/fresh) plus a mean +/- std summary panel, saved to
`results/self-gravitating/sg_loo_wiggle_vs_fold.png`. All 5 folds hold `mstar_at_bound=False`
(RULES.md #8 checked, not assumed).

Summary across the 5 leave-one-out folds:

| method | resid_r (mean +/- std) |
|---|---|
| dirty | 0.664 +/- 0.207 |
| frozen | 0.641 +/- 0.213 |
| finetune | 0.564 +/- 0.265 |
| fresh | 0.522 +/- 0.282 |

**None of the three trained arms beat the untouched dirty cube on this metric, and `fresh`
is the worst of the four, not the best.** This is consistent with the standing WITHDRAWAL
(`4e88cad`, notebook 10 V1's "fresh beats finetune" does not reproduce) rather than a new
finding: across genuine per-disk holdouts, SG training does not yet recover the wiggle
signature better than doing nothing. Per-fold spread is also large (std nearly half the
mean for finetune/fresh), and one fold (`run_9019_00019`) sits near 0.99 for every method
including dirty, meaning that disk's own wiggle is close to undetectable at this mask
regardless of denoising -- worth keeping in mind before averaging fold numbers together in
any future writeup.

---

## 2026-09-11 | run | frac sweep on `wiggle_all_methods.py`'s own comparison, confirms 0.05

Before trusting the frac=0.05 fix on this specific comparison (it had only been checked on
the SG holdout cube's mask sensitivity, not this line-emission-vs-SG cube one), swept it here
too, reusing the M1 maps cached in `experiments/wiggle_all_methods_step1.npz` (2026-08-28's
original-correction run) rather than re-denoising -- same trick as the earlier sweep, minutes
not hours.

| frac | mask% | dirty | beam-only | U-Net | DDRM | mstar |
|---|---|---|---|---|---|---|
| 0.02 (old) | 39.3% | 0.892 | 0.920 | 0.805 | 0.587 | 0.535 |
| 0.03 | 28.6% | 0.979 | 0.984 | 0.966 | 0.892 | **1.587** |
| **0.05 (new)** | 16.0% | 0.862 | 0.888 | 0.760 | 0.571 | 0.643 |
| 0.08 | 6.9% | 0.997 | 0.997 | 0.995 | 0.983 | 0.524 |
| 0.10 | 4.4% | 0.997 | 0.997 | 0.995 | 0.982 | 0.552 |
| 0.15 | 0.9% | 0.993 | 0.994 | 0.989 | 0.962 | **50.0 DEGEN** |
| 0.20 | 0.4% | 0.990 | 0.990 | 0.983 | 0.940 | **50.0 DEGEN** |

**Same three-zone shape as the SG mask-sensitivity sweep**, on independent data: loose (0.02)
pulls in a halo, 0.05 sits in the zone where the four methods actually separate, past ~0.08
the comparison flattens toward 1.0 (only the bright core survives, uninformative), past ~0.15
the fit degenerates.

**Caught one thing this sweep specifically exposed: `frac=0.03` is a bad fit, not a data
point.** `mstar` jumps to 1.587 there against 0.535 and 0.643 on either side, and every
correlation spikes with it -- the least-squares geometry fit landing on a different local
optimum at that mask size, not a real trend. Flagging it so it never gets read as a stronger
result than 0.05.

**Confirms 0.05 as the right choice for this comparison specifically**, not just inherited
from the SG holdout check. This is what the queued Kaggle rerun of `09-wiggle-scoring.ipynb`
will produce; the "third confirmation" table stays provisional until that actually lands.

---

## 2026-09-11 | bug | the "third confirmation" wiggle table was still on the flagged frac=0.02

The masking fix (Jason's flag, 0.02 -> 0.05, PROGRESS.md 2026-09-04) reached the SG training
scoring scripts but never reached `wiggle_all_methods.py`, the script actually behind the
"third confirmation" numbers (dirty 0.891 / beam-only 0.920 / U-Net 0.804 / DDRM 0.583,
reproduced 3x). All three of those reproductions, and `09-wiggle-scoring.ipynb`'s (its GPU
port), ran at the old threshold. Caught by checking every scoring script's actual `frac`
directly rather than assuming the fix had propagated.

Both fixed to `frac=0.05` now. `09-wiggle-scoring.ipynb`'s stale V1/V2 outputs cleared (same
reason as notebook 10's clearing on 2026-09-04: code and embedded output must not disagree in
a file about to be re-imported). Queued to rerun on Kaggle GPU. **The "third confirmation"
table is provisional until this reruns** -- the ordering held up to frac=0.10 in the earlier
mask-sensitivity sweep, but that was checked on the SG holdout cube, not this line-emission
comparison specifically, so it is not yet confirmed for this exact script.

---

## 2026-09-11 | run | scoring notebook 08's 4 checkpoints on moments AND the wiggle, started

`experiments/score_08_kinematic.py`, launched on local CPU. Sliding 31-channel window, centre-
channel readout (the standard read for a channel-stack model: one prediction per true target
position, not the whole predicted stack trusted at once). Normalisation matches training
exactly -- continuum-subtract first (mean of first/last 5 channels), then min-max both dirty
and clean using the CENTRE dirty channel's (lo, hi), shared across the neighbour stack, never
per-channel (`src/data/fits_cube_dataset.py`'s own documented reasoning for why: per-channel
normalisation would erase the relative amplitude along velocity that M1/M2 are computed from,
the same argument that already governs notebook 05's spectral-context arms).

Wiggle geometry is a FREE fit per cube (no stated ground truth for these disks, unlike the
self-gravitating ones), one fit on clean per cube, shared across dirty and all four gammas.
`mstar_at_bound` checked and printed per cube rather than assumed converged (RULES.md #8).

**Timed before committing to the full run:** 3.56 s/channel, so one cube x one gamma is ~12
min and the full 5-cube x 4-gamma sweep is ~4 hours on CPU -- the 31-channel model is far
heavier per forward pass than the SG scripts' 1-channel one. Not yet finished.

---

## 2026-09-11 | run | notebook 08's `gamma=10` rerun completes clean, the memory-clear fix held

Kaggle auto-push (`c1b2245`, "Version 5"), code `d9919bc`. `GAMMAS=[10.0]` only, per-arm
`gc.collect()` + `torch.cuda.empty_cache()` fix from the same day applied. No kernel death this
time: early stop at epoch 16 (best epoch 10, val_loss 0.0896), PSNR 24.333 / SSIM 0.9293 over
the full 31-channel stack (not comparable to single-channel PSNR elsewhere, the notebook's own
printed output says so).

All four arms of the sweep now complete:

| gamma | best epoch | val_loss |
|---|---|---|
| 0.0 | 27 | 0.000721 |
| 0.1 | 17 | 0.001721 |
| 1.0 | 28 | 0.016901 |
| 10.0 | 10 | 0.0896 |

`val_loss` is not comparable across gamma (RULES.md #4): the kinematic term's weight changes
the loss function's own scale, so a higher number at higher gamma does not mean worse pixel
performance on its own. Whether `kinematic_gamma>0` actually helps needs scoring on moments
and the wiggle, not read off this table.

**Not yet scored, and now lower priority than it was.** This run's original purpose (PLAN.md
Block 1) was a fast diagnostic on whether `KinematicLoss` was worth rebuilding for SG data
before spending days on it. Notebook 12's spectral-context result (two entries below) already
resolved that question positively without needing `KinematicLoss` at all, so this sweep is now
an informational data point on line emission rather than a blocking decision. Worth scoring
eventually, not urgent.

**`kin_gamma10.pth` downloaded and stored** in `models/08-kinematic/`, verified single-root,
strict-loadable, `epoch=10, val_loss=0.089597` matching this entry's table exactly. All four
gammas now in the local store, indexed in `models/README.md`.

**`kin_gamma10.pth` not yet pulled** (111.02 MB, Kaggle Output), the other three arms already
stored (`models/08-kinematic/`, entry earlier).

---

## 2026-09-11 | run | notebook 12, k=3: spectral context now exceeds doing nothing on the wiggle

Kaggle auto-push (`83c7ac0`, "Version 4"), single arm, `KS=[3]` only, reusing `k=0/1/2` from
the entry below rather than retraining them. Same holdout, same fixed-inclination geometry.

| k | in_channels | PSNR | M0 | M1 | M2 | wiggle resid_r |
|---|---|---|---|---|---|---|
| 0 | 1 | 30.053 | -26.5% | +4.2% | -44.4% | 0.366 |
| 1 | 3 | 32.828 | +29.2% | +31.0% | +62.5% | 0.590 |
| 2 | 5 | 33.576 | **+61.4%** | +31.4% | +67.6% | 0.506 |
| 3 | 7 | 35.114 | +53.0% | **+38.8%** | **+70.1%** | **0.681** |

**`k=3` exceeds dirty's own wiggle score (0.594) on this holdout**, not merely matches it, the
first SG-trained arm anywhere in this project to do so. Also the best PSNR, M1 and M2 in the
SG thread. `k=2`'s dip now reads as real non-monotonic structure rather than a fluke that
would wash out with one more point, since `k=3` recovered past it, though the mechanism for
why `k=2` specifically dips is not established.

**Block 1's decision gate (PLAN.md) is resolved: fast path, confirmed, not just directional.**
Spectral context is a working recipe that keeps the moment gains and improves the wiggle past
baseline. `k=3` is the strongest candidate on hand; `k=1` remains the cheapest (22 min vs
`k=3`'s 34) at a real but smaller cost (0.590 vs 0.681).

**Still n=1, one holdout disk, one seed, all `fresh`.** The magnitude and the above-baseline
result make this a genuine finding, but "`k=3` is the final recipe" isn't established past one
measurement, same caveat as `k=1`/`k=2` carried. A `k=4` point would say whether the trend
continues, plateaus, or was already past its peak at `k=3`; not run, lower priority now that
Block 1 has a working, positive result rather than an open question.

**Checkpoint not yet pulled** (`sg_k3_fresh.pth`, Kaggle Output).

---

## 2026-09-11 | run | notebook 12: spectral context fixes most of the wiggle loss, and improves the moments more than anything tried so far

Kaggle run (code `07f047f`, confirmed from cell 0b's log; no push commit, downloaded
manually). Same split as notebook 10 (train `{9015, 9019, 9032}`, val `9025`, holdout `9074`)
for direct comparability. All three arms `fresh` (random init), since no stored checkpoint
has `in_channels` matching `k>0` to fine-tune from. Holdout scored on moments AND the wiggle
directly (`fix_incl_deg=20`, this disk's `.para` truth), not moments alone.

| k | in_channels | PSNR | M0 | M1 | M2 | wiggle resid_r |
|---|---|---|---|---|---|---|
| 0 (control) | 1 | 30.053 | -26.5% | +4.2% | -44.4% | 0.366 |
| 1 | 3 | 32.828 | +29.2% | +31.0% | +62.5% | **0.590** |
| 2 | 5 | 33.576 | **+61.4%** | **+31.4%** | **+67.6%** | 0.506 |

**`k=1` recovers the wiggle to essentially the level of doing nothing.** dirty scored 0.594 on
this exact holdout (2026-09-10 entry); `k=1`'s 0.590 is a 0.004 difference, while every trained
arm from notebooks 10 and 11 sat well below that (`frozen` 0.487, `finetune` 0.465, `fresh`
0.278 on this same cube in notebook 11's fold 4). **And `k=1`/`k=2` post the largest moment
gains seen anywhere in the SG thread**, well past `k=0`'s own already-positive numbers.

**Mechanism, as predicted before this run (PLAN.md's Block 1 decision gate):** the `k=0` model
gets one dirty channel in, one clean channel out, with no information about neighbouring
velocity channels, so it cannot preserve a sub-channel velocity centroid it never sees.
Feeding `2k+1` channels while still predicting the centre channel gives it that information
directly, the same lever already proven on line emission (`winner_k1`/`k2`, 2026-08-20/21).

**Not monotonic, and this is reported rather than smoothed over: `k=2` has the best moments
and a LOWER wiggle than `k=1`** (0.506 vs 0.590), though still far above `k=0` and every prior
arm. Consistent with the mechanism, more context keeps helping pixel accuracy, but something
about `k=2`'s particular fit trades a little kinematic fidelity for it. On n=1 this cannot yet
be told apart from run-to-run noise the way `fresh`'s instability was shown to be real
(notebook 10 V1 vs V4). A third point (`k=3`) would say whether this is a real peak at `k=1`
or noise; not yet run.

**What this does NOT yet establish, same caveats as every SG result so far:** one holdout
disk, one seed, all `fresh` (no fine-tuned arm at `k>0` to compare against, since fine-tuning
from `winner_aug` needs a matching `in_channels=1` checkpoint that spectral context breaks by
construction). The magnitude here is large enough to be a real lever regardless, but "`k=1` is
the recipe" is not yet a claim past one measurement.

**Checkpoints not yet pulled** (`sg_k0_fresh.pth`, `sg_k1_fresh.pth`, `sg_k2_fresh.pth`,
Kaggle Output, `results/checkpoints/`), needed before storing per RULES.md #12.

---

## 2026-09-10 | code + added | notebook 12: spectral context on SG training, scored on the wiggle directly

Response to the finding above: the model gets one dirty channel in, one clean channel out,
zero information about neighbouring velocity channels, which is the documented mechanism
(2026-08-19 entry) for why per-channel denoising cannot preserve a sub-channel velocity
centroid. Spectral context (`n_neighbors=k`) already exists and was proven on line emission
(`winner_k1`/`k2`); this applies it to SG training for the first time, and scores the result
on the wiggle directly rather than only the moments.

Three arms, `k=0/1/2`, ALL `fresh` (random init): no stored checkpoint has `in_channels`
matching `k>0`, so fine-tuning from `winner_aug` is not available without first training a
matching-shape prior. `k0` is a same-day retrain of the fresh control, not a reused
checkpoint from notebook 10 or 11, so the comparison isn't confounded by which session
produced it. Same split as notebook 10 (train `{9015,9019,9032}`, val `9025`, holdout `9074`)
for direct comparability, inclination fixed at that disk's stated 20 deg throughout.

**Verified by executing the real cell sources** at reduced scale (64px, 1 epoch, 4
samples/cube, 10 scoring channels): shapes correct at each k (in_channels 1/3/5, out_channels
1 throughout), the neighbour-gathering denoise function (adapted from notebook 05's
`denoise_cube`, each neighbour normalised by the CENTRE channel's scale, ends clamped not
wrapped) runs without error at every k, wiggle scoring and the results table run. Numbers at
this scale are noise (10 channels sliced outside the line, `mstar` pins at its lower bound as
expected on flat signal). **Not yet run for real. No GPU numbers exist.**

---

## 2026-09-10 | run | sanity check: the wiggle finding is not a masking artifact

Before trusting the n=5 result below, checked whether fold 1's near-ceiling numbers (every
method within 0.006) were a `frac=0.05` masking effect that would dissolve at a tighter
threshold. `experiments/sanity_mask_frac_fold1.py`: denoised `run_9019` once, scored at
frac 0.05 / 0.10 / 0.15 without repeating the denoise per frac.

| frac | mask% | dirty | frozen | finetune | fresh |
|---|---|---|---|---|---|
| 0.05 | 24.4% | 0.9956 | 0.9956 | 0.9912 | 0.9901 |
| 0.10 | 5.1% | 0.9952 | 0.9950 | 0.9865 | 0.9827 |
| 0.15 | 1.4% | 0.9921 | 0.9917 | 0.9769 | 0.9687 |

**The gap widens as the mask tightens, it does not close.** No degenerate fit at any frac.
The ordering (dirty ~ frozen > finetune > fresh) is not a loose-mask artifact.

## 2026-09-10 | finding | across all 5 disks as genuine holdouts, SG training does not recover the wiggle, and `fresh` is worst

Extends the entry below from n=1 to n=5, using notebook 11's leave-one-out checkpoints
(`models/11-loo/`) so every disk is scored by a model that never trained on it.
`experiments/score_sg_wiggle_loo.py`, inclination held fixed at each disk's stated `.para`
truth throughout (20 deg for four disks, 30 for `run_9019`), one shared geometry per fold fit
on that fold's clean cube. 190 min total on CPU.

**resid r per fold (higher = wiggle preserved):**

| fold | holdout | dirty | frozen | finetune | fresh |
|---|---|---|---|---|---|
| 0 | run_9015 | 0.543 | 0.621 | 0.580 | 0.566 |
| 1 | run_9019 | 0.996 | 0.996 | 0.991 | 0.990 |
| 2 | run_9025 | 0.468 | 0.464 | 0.270 | 0.396 |
| 3 | run_9032 | 0.721 | 0.640 | 0.516 | 0.379 |
| 4 | run_9074 | 0.594 | 0.487 | 0.465 | 0.278 |
| **mean +/- std** | | **0.664 +/- 0.207** | **0.641 +/- 0.213** | **0.564 +/- 0.265** | **0.522 +/- 0.282** |

**Doing nothing wins on average, and training degrades the wiggle monotonically in 3 of 5
folds** (2, 3, 4: dirty > frozen > finetune > fresh, exactly the shape of the n=1 result
below). Folds 0 and 1 don't follow it: fold 0 has `frozen` slightly ahead of `dirty`, fold 1
has every method within 0.006 of ceiling (that disk's wiggle is essentially recoverable by
construction, mean resid RMS 1.45 km/s against clean's 1.448, barely denoised at all). Neither
exception reverses the aggregate: `dirty` and `frozen` sit within noise of each other, both
comfortably ahead of `finetune`, and `fresh` is worst on every single fold except fold 0.

**This is the opposite conclusion from notebook 10's headline, and it is not a contradiction,
it is a different metric.** M0/M1/M2 measure amplitude; the wiggle measures whether the
specific kinematic substructure survives. SG training makes the pixel-level moments better and
the kinematic diagnostic worse, on average, across every disk checked. The RETRACTION entry's
shape (2026-08-28: denoising improves moments, damages the diagnostic underneath them) now
holds for SG-trained models too, not just the original line-emission ones.

**`fresh` is confirmed as the least reliable arm on a second, independent measurement.** Its
std (0.282) is the largest of the four, consistent with the training-noise instability found
in notebook 10 V1 vs V4 (66pp swing on M0 between identical runs). Fold 4's fresh raw r is
0.0100, functionally uncorrelated with truth, worth flagging as a suspect value on its own
(RULES.md #8) even though `mstar_at_bound` was False in every fold's fit, so it is not the
degeneracy bug recurring.

**Caveats that still apply.** Three training disks per fold, one seed per fold, no seed
repeats, so cube variance and training variance remain confounded for `finetune` and `fresh`
exactly as flagged when this run was designed. `frozen`'s spread (0.213) is the one clean
cube-variance number in the table; it is comparable in size to `finetune`'s (0.265), meaning
most of what looks like "spread" here may be disk-to-disk kinematic difficulty rather than
training instability, except for `fresh`, whose spread exceeds even that baseline.

## 2026-09-10 | finding | SG training improves the moments and degrades the wiggle, on the one disk checked

Notebook 10's headline was scored on M0/M1/M2 amplitude. That is not the same question as
whether the kinematic signature (the Keplerian-subtracted residual) comes back, which is the
actual reason the SG thread exists. Scored it directly: `experiments/score_sg_wiggle.py`,
holdout `run_9074`, one shared geometry fit on clean with inclination FIXED at 20 deg (this
disk's stated `.para` truth), avoiding the mass-inclination degeneracy from the 2026-09-04
entry rather than re-triggering it.

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean (ref) | 2.341 | -- | -- |
| dirty | 2.746 | 0.267 | 0.594 |
| frozen | 3.461 | 0.209 | 0.487 |
| finetune | 3.569 | 0.183 | 0.459 |
| fresh | 3.627 | 0.143 | 0.424 |

**Doing nothing beats every trained model, and it degrades monotonically with how much the
model diverges from untouched: dirty > frozen > finetune > fresh.** The same shape as the
original line-emission-cube finding (RETRACTION entry, 2026-08-28): denoising improves pixel
moments and damages the kinematic diagnostic underneath them. Notebook 10's positive M1/M2
numbers do not mean the wiggle came back; on this disk it got worse the more the model was
trained.

**Not yet a claim, a data point.** n=1 holdout, and the baseline wiggle signal here is weak to
start: raw dirty-vs-clean r is only 0.267, against >0.9 on the disks used for the earlier line-
emission wiggle work. `run_9074` also has the smallest signal mask of the five disks (2.2% of
field, 2034 px), consistent with it being generally low-SNR. Could be a genuinely weak-signal
draw rather than a universal result. Needs scoring on at least one more disk with a stronger
baseline wiggle before this is reportable as more than "measured once, this is what it showed."

---

## 2026-09-10 | run | notebook 11 v2: leave-one-out, nothing is separable, but `frozen`'s spread isolates cube variance cleanly

5 folds, every cube holding out exactly once, seed fixed at 42. 128 min total. Version number
is author-reported (no push commit exists to verify it, unlike 10's V4); code `b7b140d`
confirmed from cell 0b's own log. Checkpoints NOT downloaded -- `loo*_*.pth` (10 files) still
only on Kaggle. Archived at `results/11-sg-loo/v2_2026-09-10_b7b140d/`.

| fold | holdout | frozen M0/M1/M2 | finetune M0/M1/M2 | fresh M0/M1/M2 |
|---|---|---|---|---|
| 0 | run_9015 | +23.6/-0.6/-22.3 | +35.0/+26.3/+9.3 | +37.4/+17.9/+7.4 |
| 1 | run_9019 | +23.8/+9.7/-76.3 | -24.1/-7.5/-44.0 | -6.3/-20.7/-139.7 |
| 2 | run_9025 | +4.6/+12.3/-9.5 | +13.2/+22.3/+7.1 | +14.2/+19.2/+7.7 |
| 3 | run_9032 | -111.3/-284.2/-112.8 | -68.2/-92.8/-36.9 | -288.7/-351.5/-246.5 |
| 4 | run_9074 | -10.3/-0.6/-43.6 | -15.1/+20.3/+6.2 | +10.7/+2.2/-48.4 |

**The notebook's own printed verdict: `fresh - finetune` is smaller than the fold-to-fold
spread on all three moments** (M0 -34.7+/-104.4pp, M1 -60.3+/-111.1pp, M2 -72.3+/-86.6pp).
Nothing separable at n=5, exactly what was expected going in.

**Fold 3 (`run_9032`) is catastrophic for every arm, `frozen` included.** Predicted in the
notebook before this run, from that cube's own diagnostics (rmsdiff 0.107 vs the others'
0.46-0.54, mask covering 98.9% of the field). Recorded in advance, not explained after.

**The one clean read: `frozen`'s fold-to-fold spread is pure cube variance**, since it never
trains (PSNR std 2.98, M0 std 56.3, M2 std 42.0). Against that baseline, `finetune` has LOWER
std on M0 and M2 and the best mean on all three moments -- "more stable and not worse," which
is a real, if modest, finding. "Finetune is better" is still not supported; the spreads remain
too large.

**Confirms the design caveat flagged before this ran.** Seed was fixed across folds to
isolate cube variance (RULES.md #6), but notebook 10 V1-vs-V4 showed `fresh` alone can move
66pp between IDENTICAL runs on ONE cube. So `fresh`'s per-fold numbers here mix cube variance
with that arm's own training noise, and cannot be read as a pure cube effect the way
`frozen`'s can. `finetune`'s spread is more trustworthy as a cube measurement, since notebook
10's logs showed its early-stopping point barely moved between V1 and V4 (epoch 32/24 both
times) where `fresh`'s did not (epoch 33/25 vs 23/15).

**Not yet done:** seed repeats per fold, which would actually separate cube variance from
training variance instead of confounding them for two of three arms. Checkpoints need pulling.

---

## 2026-09-04 | run | notebook 10 V4: V1's headline does not reproduce, `fresh` is high-variance

Re-ran 10 at identical settings (same seed, split, data; code `340e26c` vs V1's `be616fd`,
differing only by the `val_metrics` patch, which does not touch training) specifically to test
whether V1's `fresh` > `finetune` survived a second draw. Push `ee040ae`, Kaggle Version 4.
Archived at `results/10-sg-training/v4_2026-09-04_ee040ae/`.

| arm | PSNR V1 -> V4 | M0 | M1 | M2 |
|---|---|---|---|---|
| `frozen` | 29.032 -> 29.032 | -10.3 -> -10.3 | -0.6 -> -0.6 | -43.6 -> -43.6 |
| `finetune` | 30.859 -> 30.874 | -6.5 -> -7.0 | +36.5 -> +36.7 | +15.6 -> +15.3 |
| `fresh` | 30.024 -> 29.258 | **+5.0 -> -61.1** | +21.8 -> -27.3 | +26.0 -> -42.2 |

**WITHDRAWN: "`fresh` beats `finetune`" (2026-09-04, V1).** It was one draw of an arm that
moves 66 pp on M0 between identical runs. `fresh` in V4 is worse than the untrained baseline.

**Why this is trustworthy rather than just another noisy number:** `frozen` reproduced
*exactly* (it is inference only, so it must, and it did) and `finetune` reproduced to within
0.5 pp on every moment. The setup is not noisy in general; that arm is.

**Cause, and it is a known one.** v25 measured that early stopping on a noisy validation set
converts run-to-run nondeterminism into large performance swings. The logs show it directly:
`fresh` early-stopped at epoch 23 with best epoch 15 here, against epoch 33 / best 25 in V1 --
it stopped ten epochs sooner from a worse point. `finetune` stopped at 32 with best 24 in both
runs, because starting from trained weights it sits near a good solution and the stopping
decision is not delicate. Random init on three training disks is.

**What survives.** SG training closing the domain gap: `frozen` negative on all three moments
in both runs, `finetune` positive on M1 and M2 in both. That is the answer to Jason's
suggestion and it is now reproduced.

**Consequence for notebook 11.** Its LOO design fixes the seed across folds to isolate CUBE
variance. For `fresh` that isolates the wrong thing -- a fold-to-fold difference cannot be
read as a cube effect when the same cube moves 66 pp on its own. LOO stays valid for
`finetune`; for `fresh` it needs seed repeats per fold, or its per-fold numbers are a lower
bound on noise rather than a measurement. **This should be settled before 11 is run**, or its
`fresh` column will not mean what the notebook says it means.

**V4's checkpoints were not downloaded**, so `models/10-sg/` still holds V1's weights, which
is correct: the V1 archive's numbers were measured from exactly those (RULES.md #12).

## 2026-09-04 | code + added | notebook 11: leave-one-out, to test whether `fresh > finetune` survives

Notebook 10 V1's headline (`fresh` beating `finetune` on M0 and M2) rests on one holdout cube
and one seed, which is inside variance this project has already measured twice (V7/V9: M2
+18.4% -> +2.5% on the same cube with no config change; v25: 1.73 dB from an identical-seed
rerun). Five disks make leave-one-out available.

**5 folds, each 3 train / 1 val / 1 holdout, every cube holding out exactly once.** Training
seed fixed at 42 across folds, so the only thing varying is the fold: this measures spread
across CUBES, not seeds, and the notebook says so where it prints the number (RULES.md #6).
Folds are constructed explicitly rather than by reseeding `split_cubes`, so the assignment is
auditable instead of being a function of a seed, and each fold asserts no cube appears twice.

Reports `fresh - finetune` per moment against the fold-to-fold spread, compared against the
full standard deviation rather than the standard error, deliberately: n=5 supports no
significance claim and the wording must not imply one.

**Recorded before the run:** `run_9032` will produce a strange fold. Its synthesized pair came
out at rmsdiff 0.107 against the others' 0.46-0.54 and its signal mask covers 98.9% of the
field. Written down now so it is not explained away afterwards.

Results are written after every fold, so an interrupted session leaves usable partial data
rather than nothing. ~3 h expected (5 folds x 2 arms x ~18 min), 10 checkpoints at ~112 MB.

**Verified by executing the real cell sources** at reduced scale (64px, 1 epoch, 2 folds, 8
scoring channels): folds build and pass the leakage assertion, both arms train per fold, all
three arms score, per-fold persistence and the aggregate both run. **Not yet run for real.**
The reduced-scale artifacts were deleted rather than left in `results/` -- the same toy files
were briefly mistaken for results after notebook 10.

## 2026-09-04 | run | notebook 10 v1: SG training closes the domain gap, and `fresh` beats `finetune`

Kaggle Version 1 (push `cc194ef`), code `be616fd` confirmed from cell 0b. Dataset
`exxa-sg-synth-pairs`. Both arms converged and early-stopped (32 and 33 epochs, ~18 min each),
360 train items across 3 disks. Archived at `results/10-sg-training/v1_2026-09-04_cc194ef/`.

| arm | PSNR | SSIM | M0 | M1 | M2 |
|---|---|---|---|---|---|
| `frozen` | -- | -- | -10.3% | -0.6% | **-43.6%** |
| `finetune` | 30.859 | 0.98343 | -6.5% | **+36.5%** | +15.6% |
| `fresh` | 30.024 | 0.98055 | **+5.0%** | +21.8% | **+26.0%** |

**Training on SG data works.** `frozen` is negative on all three moments; both trained arms
are positive on M1 and M2. First measurement of that gap closing with a forward operator known
exactly rather than estimated.

**`fresh` beats `finetune` 2 of 3** (M0 by 11.5 pp, M2 by 10.4 pp; `finetune` takes M1 by
14.7 pp). **This contradicts the prediction written down before the run**, which was that
pretraining would help given only three disks. Recording that it was wrong rather than quietly
moving on.

**PSNR does not track the science, and with the baseline filled in this is the sharpest case
yet.** The run printed `nan` for `frozen` (it never goes through `train_unet`, so it has no
fixed-metric evaluation, and the results cell hardcoded nan -- a gap that put the hole exactly
where the baseline belongs). Scored afterwards with `val_metrics` itself on the same val split
(`experiments/frozen_val_metrics.py`; `finetune`/`fresh` reproduce to the digit, which is what
makes the baseline comparable):

| arm | PSNR | SSIM | MSE |
|---|---|---|---|
| `frozen` | 29.032 | 0.97815 | 0.001456 |
| `finetune` | **30.859** | **0.98343** | **0.000984** |
| `fresh` | 30.024 | 0.98055 | 0.001303 |

So `finetune` wins **every** pixel metric -- PSNR, SSIM, MSE and validation loss -- and still
loses M0 and M2 to `fresh`. Previously this pattern was PSNR being insensitive (05 v26's
spectral arms, the beam arm); here the training objective itself prefers the scientifically
worse model. And `frozen` is only 1.8 dB off the best arm while being ~60 pp worse on M2.
Notebook patched so future runs score all three arms instead of printing nan.

**What this does NOT establish.** One holdout cube, three training disks, one seed. V7/V9
measured M2 swinging +18.4% -> +2.5% on the same cube with no config change, and v25 measured
1.73 dB from an identical-seed rerun. A 10-15 pp gap on n=1 is inside that. `fresh > finetune`
is directional, not a finding. Also `frozen` scores M0 -10.3% here against -86.5% on Jason's
real pair, so the synthesized corruption is gentler than the real one and this understates the
true gap. And `run_9032`, one of the three training disks, came out at rmsdiff 0.107 against
the others' 0.46-0.54, so a third of the training data is easier than intended.

**Cleanup worth recording:** the local `results/checkpoints/sg_*.pth` and
`results/self-gravitating/sg_training_arms.json` left by the reduced-scale verification run
(1 epoch, 64px, 12 channels, M0 -111.6%) were deleted. They looked like results and were not,
which is the RULES.md #7 failure mode in miniature.

**Not yet done:** the real checkpoints are still only in the Kaggle Output and must be pulled
before the next run wipes them (RULES.md #1, #12), and these arms have not been scored on the
GI wiggle metric -- the kinematic question is what the SG data exists for, and `finetune`
winning M1 by 14.7 pp is the hint worth chasing.

## 2026-09-04 | code + added | notebook 10: frozen vs fine-tuned vs fresh on the synthesized SG pairs

Three arms against one baseline, on the pairs synthesized in the entry below:

| arm | init | trains on SG | the question it answers |
|---|---|---|---|
| `frozen` | `winner_aug` seed 43 | no | how large is the domain gap when the operator is known exactly |
| `finetune` | `winner_aug` seed 43 | yes | does SG training close it |
| `fresh` | random | yes | does line-emission pretraining help or hurt |

All three share `winner_aug`'s architecture and optimiser settings, so they differ only in
initialisation and whether they train. `finetune` runs at 0.1x the learning rate; at the full
rate the first steps overwrite the pretrained weights and the arm silently becomes an
expensive `fresh`.

**`train_unet` gained `init_state_dict=`** for this. It loads STRICTLY and raises on any
missing or unexpected key, because a silent partial load is the `winner_beam` failure again:
it trains, it reports numbers, and the numbers describe something other than the arm's name.

Scored on the holdout cube only, by moment improvement (RULES.md #4) rather than PSNR alone.

**Verified by executing the notebook's real cell sources** at reduced scale (64px, 1 epoch, 4
samples/cube, 12 scoring channels): the split is leakage-safe, `init_state_dict` loads its 256
tensors, both arms train and checkpoint, and the holdout scoring and results table run.
**Not yet run for real** -- no GPU numbers exist. Needs the `sg_synth` cubes (1.8 GB, of which
1.4 GB is the single 600x600x512 cube) plus the `winner_aug` checkpoint uploaded as a Kaggle
Dataset.

## 2026-09-04 | code | synthesized trainable SG pairs, since the shipped ones cannot train

Jason's suggestion (use SG data in training) is right in substance and blocked in practice:
the pairs he sent differ clean-to-dirty by 0.4-7% RMS against the training set's ~50%, entry
below. So the corruption gets applied here instead:
`dirty = beam (*) clean + beam (*) noise`, using the beam recovered from the v2 pair's
cross-spectrum. `experiments/synthesize_sg_pairs.py`, output under
`self-gravitating cube and dirty cube/sg_synth/` (1.8 GB, gitignored), manifest at
`results/self-gravitating/sg_synth_manifest.json`.

**Two bugs caught while building it, both in the same place: operator normalisation.**

1. The recovered beam **sums to 355**, not 1 -- the Jy/pixel-to-Jy/beam area factor, plus the
   129 px crop truncating the negative bowl. Convolving with it as-is multiplies the flux
   scale by 355, and the first run produced `rmsdiff/rms_clean` of ~347 on every cube: a scale
   factor wearing the costume of a corruption. That is the same signature the v2 pair shows
   (341), and it would have been actively harmful in training, because
   `fits_cube_dataset` normalises BOTH cubes by the DIRTY channel's range, so the target would
   have landed ~355x below the input. That is the Week-5 conditioning bug all over again.
   Fixed by normalising the beam to unit DC gain.
2. Convolving white noise with a unit-sum kernel suppresses its variance by a kernel-dependent
   factor (measured ~0.0384 here), so a sigma chosen before convolution silently tracks the
   beam normalisation rather than the requested level. Now calibrated after convolution:
   draw once, measure, rescale.

**Result, against the line-emission training set's 0.41-0.57 band:**

| run | source | channels | rmsdiff/rms_clean |
|---|---|---|---|
| `run_9019_00019_rt_00` | `run_sg_00019` | 512 | 0.4725 |
| `run_9015_00370_rt_00` | `run_sg_15` | 172 | 0.4908 |
| `run_9025_00370_rt_00` | `run_sg_25` | 172 | 0.5354 |
| `run_9032_00020_rt_00` | `run_sg_32` | 157 | **0.1065** |
| `run_9074_00025_rt_00` | `run_sg_74` | 201 | 0.4589 |

Four of five sit in the band. `run_9032` comes out much easier, consistent with it being an
odd cube generally: lowest in-signal RMS of the five by an order of magnitude, and the one
whose signal mask covered 98.9% of the field in the validation entry below, i.e. almost no
dynamic range in M0.

**Verified end to end through the real pipeline** (`experiments/check_sg_synth_pipeline.py`),
because being able to write FITS is not the same as being trainable: `split_cubes` discovers
all five with distinct RunIDs and splits them leakage-safely (3 train / 1 val / 1 holdout, no
cube in both), `FITSChannelDataset` returns matched finite (1, 256, 256) items, and the
**identity baseline scores 12.19 dB** -- against the U-Net's ~39 dB on line emission. There is
real signal to recover, which is precisely what the shipped pairs would have failed.

**What this is not.** Convolution plus correlated Gaussian noise is not an interferometer: no
uv sampling, no phase errors, no deconvolution residuals. It is a controlled approximation,
and it is exactly the forward model DDRM already assumes -- which is the upside, since `A` is
now known exactly rather than estimated at 0.80 held-out correlation. CASA `simobserve` on
these same clean cubes is the higher-fidelity version and remains the natural follow-up, and
would double as the first real work toward ALMA validation.

## 2026-09-04 | arrival + bug | September SG batch: unusable as training pairs, and it exposed
a degeneracy in the Keplerian fit

Jason sent more self-gravitating data (Drive, two zips, 3.7 GB) and said using SG data in
training would be a good idea. Extracted to
`self-gravitating cube and dirty cube/_v3_extract/`. Five new run folders plus the v2 pair
re-shipped, and MCFOST `.para` files alongside each, which turned out to matter more than the
cubes did.

**The pairs cannot train a denoiser as shipped.** Measured clean-vs-dirty difference at the
brightest channel, against the line-emission training set for scale:

| dataset | rmsdiff / rms_clean |
|---|---|
| line-emission training cubes | **0.41 - 0.57** |
| `run_sg_00019_rt_00019` | **0.0000** |
| `run_sg_74_00025_rt_00` | 0.0038 |
| `run_sg_15_00370_rt_rt00` | 0.0113 |
| `run_sg_32_00020_rt_00.` | 0.0265 |
| `run_sg_25_00370_rt_00` | 0.0708 |

The training set's dirty cubes differ from clean by ~50%; these differ by 0.4-7%. A network
trained on the latter learns the identity. `run_sg_00019`'s dirty is its clean: max absolute
difference 0.03125 against a 1e10 peak, a relative 3e-12, and that 1e10 Jy/beam scale is
itself wrong (v2's dirty peaks at 0.075). Nor are they a deconvolution task: clean and dirty
carry identical `BMAJ/BMIN` in every pair, so both sides already share a beam and there is no
operator between them. Phase 0 agrees (`NO_CONVOLUTION`), though the direct measurement above
is the evidence -- Phase 0 returned `A=nan`, which is a failed fit, not a verdict (RULES.md #8).

**What the `.para` files gave us is worth more: stated ground truth.** Stellar mass,
inclination, distance and grid size per run. Two configurations: 0.6 Msun / 30 deg / 140 pc,
and 1.0 Msun / 20 deg / 175.178 pc. The pixel scale they imply (600 AU over 301 px = 1.993
AU/px) matches what the code computes from the header, so the physical scale is confirmed
correct.

**First real validation of `fit_keplerian`, and it failed 3/5.** Until now the diagnostic had
exactly one check: 0.639 Msun recovered from the v2 cube against Hall+2020's 0.6, and that
0.6 came from a paper, not the data. Clean cubes, no noise, no beam, no model:

| run | true M | true incl | fitted M | fitted incl |
|---|---|---|---|---|
| `run_sg_00019` | 0.6 | 30 | 0.519 (-13%) | 31.8 |
| `run_sg_15` | 1.0 | 20 | **50.0 (bound)** | 3.3 |
| `run_sg_25` | 1.0 | 20 | 10.5 (+948%) | 6.8 |
| `run_sg_32` | 1.0 | 20 | 0.572 (-43%) | 22.6 |
| `run_sg_74` | 1.0 | 20 | **50.0 (bound)** | 2.2 |

**Cause: mass and inclination are nearly degenerate.** The line-of-sight velocity is
`sqrt(GM/r) sin(i) cos(theta)`, so the two enter as the single product `sqrt(M) sin(i)`. Only
the geometric deprojection, which stretches sky coordinates by `1/cos i`, separates them, and
that is a 6% effect at i = 20 deg against 15% at 30 deg. Hence the one disk at 30 deg
converging correctly (31.8 fitted against 30 true) while three at 20 deg collapsed toward
face-on and drove the mass to its bound to compensate.

Holding inclination at its stated value: -5.1% / +47% / +26% / -29% / still-pinned. Four of
five go from catastrophic to the right order of magnitude. The residual error is plausibly
physical rather than a bug, since these are *self-gravitating* disks and a point-mass
Keplerian model is the wrong model for one, but that is not demonstrated.

**Fixed in code.** `fit_keplerian` gains `m0=` (initial geometry from the emission's shape)
and `fix_incl_deg=` (hold inclination when it is known independently), and now returns
`mstar_at_bound` so a pinned fit cannot be quoted as a measurement. Case 8 in
`tests/test_gi_wiggle.py` covers it.

Also fixed a real but, as it turns out, *not causal* bug on the same line: the initial
geometry came from `disk_geometry_from_m0(np.ones_like(m1), mask)`, i.e. the shape of the
MASK rather than of the emission, which for a circular mask implies inclination ~0 and starts
the optimiser at the degenerate end of the valley. Passing the true M0 changed the results
barely at all (0.519 -> 0.519, 50 -> 50), so the degeneracy, not the initialisation, is what
breaks these fits. Worth fixing regardless.

**Which published numbers this touches.** The method comparison is safe: `compare_wiggles`
fits ONE geometry on the clean cube and subtracts that same model from every method, so a
mis-estimated absolute mass shifts all rows together and the ranking (beam preserves the
wiggle, U-Net degrades it, DDRM degrades it most) is unaffected. What weakens is the
standalone claim that recovering 0.639 Msun corroborates the cube's provenance: we now know
the fit is only trustworthy at favourable inclination, and even at 30 deg it is 5-13% off.
The v2 cube fitted 27-33 deg, in the favourable regime, so that number is not withdrawn --
but it should be quoted with the inclination caveat attached.

**`run_sg_74` fails even with inclination fixed** and has by far the smallest mask (2.2% of
the field, 2034 px) and the largest residual RMS. Separate problem, not yet diagnosed.

## 2026-08-29 | run + added | notebook 09 on Kaggle GPU: third independent confirmation

`09-wiggle-scoring.ipynb` run on GPU, first downloaded manually with outputs (exec counts
6-10, pulled commit `9f59221`), before its interactive session was ever committed on Kaggle,
so that run has no version number to attribute it to. Kaggle's GitHub integration then
auto-pushed a second, fresh-kernel run as `ce1b6ae` ("Kaggle Notebook | 09-wiggle-scoring |
Version 2", exec counts 1-5, pulled commit `7d73e2e`), which is the number this project uses
per RULES.md's own stated method: reconstruct the version from the push commit, not from
recollection. Both runs agree to the fourth decimal. Dataset resolved correctly at
`/kaggle/input/datasets/krishanyadav333/kaggle-wiggle-scoring-dataset/`.

**240-360 step 1 (121 channels), 7.2 min on GPU** (vs 140.9 min on CPU for the same config,
~19x): mstar=0.537, resid r: dirty 0.8907, beam-only 0.9198, U-Net 0.8040, DDRM 0.5833. Step 4
(31 channels), 1.8 min: same bunching-near-1.0 sampling artifact as every prior run of this
config, not a real finding.

**Third independent reproduction of the retraction's corrected table**, now across two
machines and both CPU and GPU: 0.891/0.920/0.804/0.583 (original), 0.892/0.920/0.805/0.587
(local CPU rerun), 0.891/0.920/0.804/0.583 (this run, rounded). No longer provisional in any
sense.

**Archived** at `results/09-wiggle-scoring/v2_2026-08-29_ce1b6ae/` and added to `RUNS.md`,
explicitly disambiguated from the unrelated pre-existing "09 -- Architecture comparison"
entry (same number, different notebook). Root `09-wiggle-scoring.ipynb` committed with its
outputs intact, matching how 07/08 already carry their own Kaggle-pushed outputs at root.

## 2026-08-28 | run | wiggle_all_methods.py confirms the retraction's corrected table

`compare_wiggles()` port of `experiments/wiggle_all_methods.py` (bug fix two entries below)
finished its first real run, both channel-sampling configs, 173 min total on CPU (the run it
replaces crashed at 111 min on a wrong key name, `vsys_kms` instead of `vsys`, fixed same
session).

**240-360 step 1 (121 channels), independent re-fit of the shared geometry (mstar 0.535 vs the
retraction entry's 0.538, same ballpark, not the same optimiser run):**

| method | resid RMS | raw r | resid r |
|---|---|---|---|
| clean | 0.182 | -- | -- |
| dirty | 0.171 | 0.9928 | 0.8916 |
| beam-only | 0.169 | 0.9947 | 0.9204 |
| U-Net | 0.169 | 0.9874 | 0.8051 |
| DDRM | 0.303 | 0.9527 | 0.5869 |

Matches the RETRACTION entry's corrected table (0.920 / 0.891 / 0.804 / 0.583) to within
fit noise. **This is now a reproduced result, not a one-off.** Ordering holds: beam-only and
dirty close together, U-Net worse, DDRM worst by a wide margin.

**240-360 step 4 (31 channels)**: all five methods bunch at 0.987-0.999, which is the
sampling artifact the script's own docstring warns about, not a real finding -- coarse
sampling degrades `quadratic_moment1`'s parabola fit for every cube alike, and that shared
artifact correlates spuriously regardless of what happened upstream. Do not read "the
methods are indistinguishable" out of this config; step 1 is the one that resolves them.

Figure regenerated at `results/self-gravitating/wiggle_all_methods.png`, overwriting the
stale pre-fix version. `wiggle_all_methods_step1.npz` regenerated alongside it.

Kaggle GPU version (`09-wiggle-scoring.ipynb`) still worth finishing and running once the
kinematic-loss checkpoints exist, since a 173-minute CPU turnaround does not scale to
re-scoring four more gamma arms.

## 2026-08-28 | code + added | KinematicLoss and notebook 08, aimed at the corrected failure

Direct response to the corrected result two entries below: U-Net degrades the wiggle residual
(0.804 vs dirty's 0.891), DDRM degrades it worse (0.583). Built rather than deferred, at the
user's push -- the "needs 2 days" estimate did not hold up; the objective and its test took
about 90 minutes.

`src/utils/losses.py`: `spectral_moment1()`, a differentiable moment-1 (biased intensity-
weighted mean, same estimator as `collapse_first`; the bias is shared between prediction and
target so it cancels in the loss, unlike in the standalone diagnostic where `collapse_first`
was replaced for exactly this reason). `KinematicLoss(alpha, beta, gamma)` combines it with
MSE and SSIM. 15 checks in `tests/test_kinematic_loss.py`.

`08-kinematic-loss.ipynb` (new, 15 cells): trains `winner_aug`-configuration U-Net with
`out_channels=31` (k=15 stack, matching the line's measured ~37-channel FWHM) and
`kinematic_gamma` swept over [0.0, 0.1, 1.0, 10.0]. gamma=0 is a fresh control at the same
31-channel architecture, not notebook 05's single-channel `winner_aug` -- comparing against
05 directly would confound the loss change with the output-channel change.

**Not extending notebook 05**: its `denoise_cube`/`val_metrics`/moment-table path assumes
single-channel output throughout, and the wiggle score needs the self-gravitating cube, which
05 never loads. Scoring stays local (`experiments/wiggle_all_methods.py`).

Verified by extracting and executing the notebook's real cell sources against local data at
reduced scale (k=3, 64px, 1 epoch): data shapes match, velocity axis builds correctly from
`CDELT3`, both gamma=0 and gamma=1 train and checkpoint. Not real numbers, plumbing only.

**Not yet run on Kaggle GPU.** Success criterion: wiggle residual correlation above the
U-Net's 0.804 without losing the M0/PSNR gains `winner_aug` already has.

---

## 2026-08-17 | run + added | 05 Kaggle Version 24 — beam arm re-scored, the bug was real

First run with `denoise_cube()` passing the beam vector. No training.

```
winner_beam            M0        M1        M2
before (no vector)   -95.7%    +14.1%    -27.3%
after  (with it)      +9.6%    +63.2%    +20.9%
```

**M0 moves 105 points and M2 changes sign.** Four of five cubes are strongly positive on M0;
`run_0025_01000_rt_04` at −133.8% carries the whole deficit alone.

**Consequence:** beam conditioning is **not refuted**. It is positive on all three moments
and simply below the winner at one seed. The earlier reading, second-best PSNR paired with
the worst M0 in the table, was an artifact. That reading is in the submitted midterm blog,
which stays as written; corrected in RUNS.md and in the v24 README.

`winner_patch` PSNR 33.96 -> 34.98 on the common full-image set. Moments unchanged.

Kaggle auto-pushed again mid-commit. Checked before merging, per rule 2: no fix reverted,
all five probes intact. Notebook archived with outputs, per rule 10.

## 2026-08-17 | bug | 05 — beam arm scored without its beam vector

`denoise_cube()` never passed a beam vector, and `UNet.forward` ignores `beam=None`
silently (it asserts only in the opposite direction), so `winner_beam` was scored on moments
with its conditioning branch dead. Section 4 scored the same checkpoint *with* the beam and
got 38.71 dB, second best in the table; section 6 scored it *without* and got M0 −95.7%,
worst in the table.

**Caught by** reading `denoise_cube`'s call signature against the arm's `beam_dim=4`, during
a review, not by any test or assertion.

**Touches** RUNS.md's beam row (M0 −95.7 / M1 +14.1 / M2 −27.3, now retracted) and the
published blog, whose closing cites "second-best PSNR and worst M0 by a factor of two" as
evidence for the pixel-vs-science thesis. The blog is submitted and stays as is.

**Fixed** in the notebook: the vector now comes from the dirty header via `beam_features_of`,
a missing header raises instead of zeroing the conditioning, and stale beam rows are dropped
on resume so the fix cannot be masked by resumed values. **The arm still needs re-scoring.**
On a toy beam model the vector shifts the output by 0.025 mean absolute, so the two paths
were scoring different functions.

Same review found `winner_patch`'s PSNR measured on 64px crops while every other arm used
256px full images, putting 33.96 dB in a column of 37 to 39. Also fixed; PSNR now uses a
common full-image set.

## 2026-08-17 | run + added | 05 Kaggle Version 22 — artifact diagnostics complete

Sections 7 and 8 finished for the first time. No training, no new moment scores; the moment
table is identical to v20's. 300 validation channels analysed, and the per-channel CSV
finally saved.

Three readings, in the run's README: invented structure is a **threshold not a gradient**
(~25 channels at ~100% invented area, everything else near 0, all below SNR ~0.5); the
negative floor leak is **systematic** (`denoised.min` clusters near −0.10 and −0.155, no
channel near zero); and the worst channels are where the **metric stops meaning anything**
(SNR 0.0 scored against 20% of a clean peak that is itself a noise spike).

Kaggle auto-pushed during the commit and the push was rejected. Checked before merging: it
had **not** reverted any code fix, only re-added stored outputs. Rebased, kept the stripped
notebook. That push is also what revealed the run was Kaggle Version 22, not the v21 it had
been filed as; folder renamed.

## 2026-08-16 | bug | 05 — section 7 CSV write discarded the whole diagnostic

`channel_artifacts()` returns `n_background_px`; the hand-kept `fieldnames` list did not
include it, and `DictWriter` raises only at write time. All 300 channels were analysed and
printed, then thrown away at the last line.

**Fixed** by deriving columns from the data. `n_background_px` is the field that should never
have been dropped: it is the denominator RULES.md #8 exists to make you check.

## 2026-08-14 | run + added | 05 v20 — first U-Net scores on the DDPM's metric

Zero arms trained. 12 checkpoints restored from 08, 3 from v19's Output, `winner_patch`
scored from stored weights, 15 rows over 6 arms. The whole session went to section 6.

**Why it mattered:** every U-Net number before it was on the raw or clip-only metric while
the DDPM was on mask + clip, so no cross-family comparison had been like for like. Best
U-Net arm beats the DDPM by 85 percentage points on M0 with a spread twenty times tighter.

Two negative results, both kept: beam conditioning worst on M0 (later retracted, see above),
64px patches at M0 −40.8%.

**Notebook not archived.** Lost. Kaggle did not auto-push this version and the local copy was
stripped before the gap was noticed. Part of why RULES.md #10 exists.

## 2026-08-27 | finding | robust estimator: wiggle recoverable raw, model destroys it more decisively than first thought

Two follow-ons to the same day's GI wiggle work, both from asking whether `collapse_first`
(intensity-weighted mean) was the wrong tool for a cube that is 51.55% negative pixels.

**Part 1: does the wiggle survive in the raw dirty cube, or did `collapse_first` just fail to
find it?** Swapped to `collapse_quadratic` (Teague & Foreman-Mackey 2018) --
`quadratic_moment1` in `src/evaluation/gi_wiggle.py` -- which fits a parabola to the peak
channel rather than averaging the whole spectrum, so a negative sidelobe elsewhere cannot pull
it off the true line centre. Validated on synthetic data with an injected negative sidelobe
before trusting it on anything real: naive weighted mean pulled 1341 m/s off the true centre,
quadratic estimator off by 21.5 m/s. 17 checks in `tests/test_gi_wiggle.py`.

On the real dirty cube the fit now converges (mstar 0.580 Msun against clean's 0.522,
inclination 27.5 against 33.4 degrees) where it previously ran to its 50 Msun bound. **The
wiggle is genuinely recoverable from the raw, undenoised dirty cube.** Residual correlation
with clean: 0.92.

**Part 2: does "denoising makes it worse" survive the same correct method?** Reran inference
(858s) to get the denoised cube's raw channels, since only its first-moment maps had been
saved. Full three-way comparison, quadratic throughout:

| | mstar (Msun) | incl (deg) | PA (deg) | vsys (km/s) | raw M1 r | residual r |
|---|---|---|---|---|---|---|
| clean | 0.522 | 33.4 | 178.8 | 0.100 | -- | -- |
| dirty | 0.580 | 27.5 | 175.2 | 0.083 | 0.773 | 0.921 |
| denoised | 0.578 | 69.8 | 67.3 | 1.918 | 0.250 | 0.281 |

**This supersedes the earlier first-moment comparison (0.72/0.53) with a sharper version of
the same conclusion, not a different one.** The gap between doing nothing and running the
model is larger under the correct method: dirty holds at r=0.77-0.92, denoised falls to
r=0.25-0.28. Denoised's mstar (0.578) looks close to clean's by coincidence; every other fit
parameter is essentially unrelated to the true geometry (inclination off by ~40 degrees,
position angle a genuinely different orientation not an aliasing artifact, systemic velocity
off by nearly 2 km/s). The optimiser found a local optimum with no physical resemblance to the
real disk.

**The 2026-08-27 first-moment numbers (0.72/0.53) should not be quoted going forward; this
comparison is now the authoritative one.**

## 2026-08-27 | finding | denoising makes the disk's kinematics WORSE than doing nothing

Extended the GI wiggle check to `dirty_cube.fits` and the `winner_aug` seed-43 denoised
output already computed in the OOD eval, reusing its saved moment maps.

**Clean, unconfounded result, no Keplerian fit needed: correlation of raw moment-1 with the
true clean M1 is 0.72 for the untouched dirty cube and 0.53 for the denoised one.** The
model degrades recoverable kinematics below doing nothing. Matches the OOD moment-improvement
result from earlier the same day (M0 -86.5%, M2 -168.3%) with an independent metric.

**Two `fit_keplerian` bugs found and fixed along the way**, both from using the same crude,
data-independent initial guess for every cube:

- Dirty's M1 has NaN pixels (moment-1's own 0/0, since `mask` is defined from clean's M0, not
  the dirty cube's own signal quality). `fit_keplerian` used to propagate that straight into
  the fit bounds and crash with "x0 is infeasible". Now drops non-finite samples before
  fitting and reports how many. 3 new regression cases (15 total in
  `tests/test_gi_wiggle.py`).
- Even with NaNs handled, dirty's fit still ran mass to its 50 Msun bound with a
  near-degenerate inclination, from the same fixed init used for every cube regardless of
  data quality. Re-fit using CLEAN's own converged geometry as the init for dirty/denoised
  (physically correct: same disk, so the true geometry is shared) -- and dirty STILL does not
  converge. That is now a real finding rather than an optimiser artifact: a naive per-pixel
  Keplerian fit to the raw dirty cube's moment-1 does not work. Consistent with why the
  literature (Hall+2021 in particular) works from individual channel maps rather than the
  collapsed M1, which is more noise-sensitive.

**A caveat worth keeping:** denoised's fit converges, to a disk at about a fifth of the true
mass with almost no real rotation (0.034 km/s median deviation vs 0.635 km/s truth). Its
residual correlates with clean's residual at 0.82, which looks like "the wiggle survived" --
it should not be read that way, since a near-flat fitted model leaves the residual close to
the raw M1 map itself. The raw-M1 correlation (0.53) is the number not confounded this way,
and it already answers the question in the opposite direction.

**Settled:** `winner_aug` should not be used to prepare a cube for GI wiggle analysis; it
removes kinematic information. **Not settled:** whether a better (channel-map-based) fit
could recover something from the raw dirty cube where the moment-1 fit failed.

Write-up: `results/self-gravitating/gi_wiggle_dirty_vs_denoised.md`, figure
`gi_wiggle_clean_vs_dirty_vs_denoised.png`.

## 2026-08-27 | finding | GI wiggle diagnostic built and run: a real, large, coherent residual, provenance corroborated

Jason redirected the priority: work properly on the self-gravitating data, and pointed at
five papers (Speedie+2024 Nature, Terry+2024 A&A, Hall+2021 MNRAS, Hall+2022 ApJL,
Hall+2020 ApJ). All five are built around one diagnostic, the "GI wiggle": fit the disk's own
Keplerian rotation, subtract it from the moment-1 map, and look at the residual.
`DISTPC=140 pc` in `dirty_cube.fits` matches Hall+2020's founding simulation exactly.

New module `src/evaluation/gi_wiggle.py`: geometry init from M0's image moments, a
least-squares Keplerian fit (centre, PA, inclination, vsys, stellar mass) against the moment-1
field, and the residual. Validated on synthetic data before running on real cubes: a pure
Keplerian case recovers every parameter exactly with zero residual; a known injected m=2
perturbation is recovered at r=0.85 correlation, with the caveat that amplitude is inflated
because the geometric fit itself absorbs some of the injected signal.

**Fitted on the clean cube (`lines.fits`, non-padded range [30,571)):** stellar mass
**0.639 Msun**, with no mass information anywhere in either header -- recovered purely from
the velocity field. That is close enough to Hall+2020's 0.6 Msun founding simulation that it
is independent corroboration of the DISTPC match: this cube is very likely built on that exact
pipeline.

**The residual: RMS 1.34 km/s against 0.64 km/s of typical local rotation** -- larger than the
rotation itself -- and it does not decay with radius (checked to r=100 px against a median
disk radius of 130 px, ruling out a 1/sqrt(r) fit artifact near the centre). One annulus
(r 120-140 px) shows a clean single-period (m=1) sinusoid in azimuth with essentially no
scatter. Large, coherent, global: consistent with what the papers describe.

**Not yet claiming a confirmed GI wiggle detection.** A pure radius-independent m=1 residual
is also what a slightly imperfect flat thin-disk geometric model produces on its own, and the
field is actively debating exactly this degeneracy (arXiv:2510.05601 argues infall can mimic
a wiggle in AB Aurigae, the same object Speedie+2024 reads as gravitational instability). What
is established: a validated pipeline, a large real residual, and a stellar-mass recovery that
independently corroborates the cube's provenance.

Not yet done: run the fit on `dirty_cube.fits` and on a denoised cube (does the wiggle survive
noise and the currently-failing U-Net); check the residual's radial structure against the
literature's "interlocking fingers" description, not just its azimuthal one; a proper
Fourier-decomposed amplitude if a q estimate is wanted, rather than the RMS/max proxy built
here.

Write-up: `results/self-gravitating/gi_wiggle_check.md`, figure `gi_wiggle_clean.png`.

## 2026-08-27 | run | first OOD test: the trained U-Net fails on a real dirty beam

`experiments/eval_self_gravitating.py`, `winner_aug` seed 43, no retraining. The out-of-
distribution test proposed 2026-08-21 and only now run: the model had never actually been
scored against the self-gravitating pair before this.

Result over the non-padded range (541 channels, see below): **M0 -86.5%, M1 -10.8%,
M2 -168.3%**, signal-masked. Every moment is negative -- the denoised cube is FURTHER from
truth than the raw dirty cube, not just less improved. Figure at
`results/self-gravitating/ood_moment_comparison.png`, write-up at
`results/self-gravitating/ood_eval.md`.

**Why this is expected and still worth having measured.** Every training cube has `A = I`
(Phase 0), so the model was trained to remove additive noise with no mechanism for inverting
a beam. Applying it to a cube with a REAL dirty beam (recovered 2026-08-25, peak 0.911,
-2.8% sidelobes) does not degrade gracefully, it actively damages the signal. That is now
measured rather than assumed, and it is the first concrete evidence of the domain gap between
this project's training regime and what real ALMA data will look like.

**Also found in building the eval:** channels 0-29 and 571-600 of `lines.fits` are
byte-identical repeats within each block, confirmed with `np.array_equal`, not a real
line-free baseline. That breaks both the training pipeline's continuum-subtraction assumption
and `bettermoments.estimate_RMS`, which reads `data[:N]`/`data[-N:]` literally. The eval trims
to the non-padded [30, 571) range and skips continuum subtraction rather than apply it to
padding.

## 2026-08-28 | code | velocity-aware objective, built to target the measured failure

The corrected comparison shows every model degrades the GI wiggle (beam-only 0.920, dirty
0.891, U-Net 0.804, DDRM 0.583). The U-Net's damage has an identifiable cause: it optimises
MSE + SSIM, pixel accuracy, while the wiggle is a sub-channel velocity perturbation that
per-channel smoothing shifts. Nothing in the objective ever asked it to preserve velocity
structure. This adds that.

`KinematicLoss` in `src/utils/losses.py`:

    L = alpha*MSE + beta*(1-SSIM) + gamma*|M1(pred) - M1(target)|

with `spectral_moment1` a differentiable intensity-weighted mean velocity over the channel
axis. It is not an unbiased M1 and does not need to be: it is computed identically on
prediction and target, so a finite-window bias cancels in the difference. What it must be is
sensitive to a line-centre shift, which it is.

Measured properties (`tests/test_kinematic_loss.py`, 15 checks): M1 tracks a shifted line,
is invariant to amplitude scaling (it measures velocity, not flux), is finite on empty
spectra, and the loss separates a velocity shift (0.0855) from a pure amplitude change
(1.1e-08) by seven orders of magnitude. `gamma=0` contributes nothing to the total while
still reporting the velocity term for monitoring.

Three pieces of plumbing this needed, all in code the existing arms share:
- `FITSChannelDataset(stack_target=True)` returns all 2k+1 CLEAN channels rather than the
  centre alone; a moment-1 penalty needs a line profile, not one slice. Default unchanged.
- `_build_unet(out_channels=N)` and `train_unet(out_channels=N)`: the model has to predict a
  stack for its M1 to be computable.
- `train_unet(kinematic_gamma=..., velax_kms=...)`, recorded in the checkpoint.

**Window size matters and is measured, not guessed:** the line FWHM is ~37 channels at a
typical bright spaxel, so a useful stack is ~31 channels (`n_neighbors=15`), not the 3 the
spectral-context arms used.

**Not yet trained.** Needs a Kaggle run: `n_neighbors=15, stack_target=True, out_channels=31,
kinematic_gamma` swept. The honest test is whether it improves the wiggle correlation above
the U-Net's 0.804 WITHOUT giving up the M0 and PSNR gains, since a model that only preserves
velocity by refusing to denoise would be useless.

## 2026-08-28 | RETRACTION | the wiggle comparisons were methodologically broken; the beam never erased the wiggle

Running the U-Net alongside DDRM at matched configuration exposed the flaw behind a chain of
wrong conclusions: **each cube was fitted its OWN Keplerian model before its residual was
compared to the truth's.** The residual then means something different per cube, so the
comparison measures fit disagreement, not the wiggle.

Beam-convolved clean data vs truth, across five channel ranges at step 1:

| range | own-fit r | shared-model r |
|---|---|---|
| 240-360 | 0.117 | 0.920 |
| 230-370 | 0.998 | 0.999 |
| 250-350 | 0.153 | 0.998 |
| 200-400 | 0.949 | 0.998 |
| 60-540 | 0.116 | 0.997 |

The own-fit column is noise. That instability was visible all along and went unchecked.

**WITHDRAWN: "the beam alone erases the GI wiggle" (2026-08-27).** Under a shared model the
beam-convolved cube correlates at **0.92-0.999** with truth and its residual RMS (0.168)
matches the truth's (0.182). The beam preserves the wiggle. **This finding motivated the whole
DDRM effort.**

**WITHDRAWN: all three DDRM verdicts issued on 2026-08-28** ("no recovery", "destroys signal",
and an unpublished "recovers"). All used per-method fits; two also compared across different
channel samplings.

**Corrected result** (240-360 step 1, one shared model from clean):

| method | resid r | resid RMS |
|---|---|---|
| clean (ref) | -- | 0.182 |
| beam-only | 0.920 | 0.168 |
| dirty | 0.891 | 0.170 |
| U-Net | 0.804 | 0.169 |
| DDRM | 0.583 | 0.303 |

The beam preserves the wiggle; noise costs a little; the U-Net degrades it; DDRM degrades it
most and inflates the residual amplitude 1.7x. The ordering "do nothing > U-Net > DDRM" holds,
so the qualitative claim that models hurt the kinematics survives. What does not survive is
the beam being the culprit, or DDRM ever having recovered anything.

**Fixed in code:** `compare_wiggles()` fits the reference once and subtracts that model from
everyone; regression case 7 covers it. Also fixed `fit_keplerian` raising "x0 is infeasible"
when its default initial guess fell outside its own bounds -- unreachable when an explicit
`init` is passed, which is why every prior test missed it.

**Lesson:** three conclusions were published in one day from a statistic whose instability was
measurable from the start. Validate a comparison metric against its own free parameters before
reading any result from it.

Write-up: `results/self-gravitating/RETRACTION_wiggle_methodology.md`.

## 2026-08-28 | correction | the DDRM comparison was invalid; DDRM DESTROYS signal, not merely fails to recover it

Published a DDRM result an hour ago comparing 0.1277 against a 0.116 floor and calling it "no
recovery". **The comparison was invalid and the conclusion understated.**

DDRM was scored on 31 channels at step 4; the 0.116 floor came from 481 channels at step 1.
Different configurations. Measured at DDRM's OWN configuration the beam-only floor is
**0.938**, so DDRM at 0.128 is far BELOW it: it destroys structure that survives the beam
untouched.

**Root cause: the wiggle residual is acutely sensitive to velocity sampling.**

| step | dv (km/s) | clean resid RMS | dirty resid RMS | resid r |
|---|---|---|---|---|
| 1 | 0.033 | 0.182 | 1.341 | 0.115 |
| 2 | 0.067 | 0.185 | 1.355 | 0.117 |
| 4 | 0.133 | **1.436** | 1.418 | **0.997** |
| 8 | 0.267 | 1.741 | 1.689 | 0.997 |

At step 4 the CLEAN cube's residual jumps 8x. That is parabola-fit error in
`quadratic_moment1` (it fits the peak channel and two neighbours), not signal, and being a
deterministic sampling artifact it is SHARED between clean and dirty -- hence the spurious
0.997.

**This also retracts a correction I made earlier today.** I attributed the 2026-08-27 value of
0.111 to "a 10-channel window collapsing the fit". Wrong: that value came from 481 channels at
step 1 and is sound for its configuration. The discrepancy was the sampling step, not the
window length.

**What stands:** the prior trained properly; DDRM's fitted geometry is physically incoherent
(8.2 Msun against 0.56, inclination 8 against 32 degrees); the beam passes 1.3% of modes above
1% gain; DDRM is not usable for this problem on this data. The qualitative conclusion is
unchanged and in fact stronger.

**Methodological rule going forward:** a GI wiggle correlation is only comparable to another at
the SAME channel range AND step. Three separate errors today came from comparing across
configurations. Record the configuration with every value.

## 2026-08-28 | finding | DDRM does not recover the wiggle: 0.128 against a 0.116 floor

Prior trained on Kaggle (60 epochs, unconditional, v-prediction, loss 8106.7 -> 16.5).
Restoration and scoring rerun locally on 31 channels (240-360), the range that actually covers
the line-centre variation.

| | mstar (Msun) | incl (deg) | raw M1 r | residual r |
|---|---|---|---|---|
| clean | 0.564 | 31.6 | -- | -- |
| dirty | 0.618 | 29.4 | 0.9939 | **0.9970** |
| DDRM | 8.205 | 8.0 | 0.9572 | **0.1277** |

**Beam-only floor 0.116, DDRM 0.1277. No recovery.** This is outcome 2 of the two written into
`ddrm_feasibility.md` BEFORE the prior was trained: the prior hallucinates plausible structure
that is not the truth.

Three things make this a real negative rather than a failed run: the prior trained properly
and was still improving at epoch 60; the signal WAS recoverable, since with a proper channel
window the dirty cube's own residual correlates at 0.997 with truth; and the failure is
physically diagnostic, with DDRM's fitted geometry badly wrong (8.2 Msun against 0.56,
inclination 8 deg against 32) while its raw M1 correlation stays high at 0.957. It produced a
disk-shaped velocity field whose implied physics is incoherent.

**Also corrects an earlier number.** The 2026-08-27 entries recorded the dirty cube's residual
correlation as 0.111 and read it as "the wiggle is not clearly recoverable from the dirty cube
in the first place". That was the 10-channel window collapsing the Keplerian fit. With the
correct window it is **0.997**: the wiggle IS strongly recoverable from the dirty cube. The
beam-only ablation (0.116) still stands, since it used the full 481-channel range.

Likely explanation, known in advance: the beam passes only 1.3% of Fourier modes above 1% of
peak gain, so DDRM had to invent ~99% of the spectrum. Measurement consistency is a weak
constraint when the instrument measured almost nothing.

Write-up `results/self-gravitating/ddrm_result.md`, figure `ddrm_restoration.png`.

## 2026-08-28 | run + bug | DDRM prior trained; the scoring cell reported a degenerate fit as "RECOVERY"

**Prior trained successfully.** 60 epochs, 115 min on T4 x2, 16.9M params unconditional, loss
8106 -> 17.7 train / 20.5 val, still improving at the end. Checkpoint persisted (271 MB).
That part of notebook 07 works.

**The scoring is wrong and its output must not be quoted.** It printed:

```
          mstar    incl    raw r   resid r
dirty    49.999    1.3     0.9766   0.7871
DDRM     50.000    1.5     0.8632   0.4578
  DDRM: 0.4578 -> RECOVERY
```

`mstar = 50.000` is `fit_keplerian`'s upper BOUND and `incl ~ 1.3 deg` is face-on degenerate:
both fits failed, so the correlations between them mean nothing. The giveaway is `dirty`
scoring 0.787 here against 0.111 in the validated 2026-08-27 run on the same cube.

**Cause: too few channels.** The notebook restored 10 channels (280-316). Measured, the line
CENTRE varies from channel 260 to 348 across the disk -- that variation IS the rotation.
`quadratic_moment1` finds each spectrum's peak channel, so with a narrower window most
spaxels' peaks land on the window edge, the velocity field flattens, and the fit runs to its
bound. Confirmed directly:

| channels | mstar | incl |
|---|---|---|
| 10 (280-316), as run | **50.000 (bound)** | 1.1 |
| 61 (240-360, step 2) | 0.529 | 33.1 |
| 481 (60-541), validated | 0.572 | 32.0 |

Fixed: `CHANNELS` now spans 240-360, and the scoring cell **raises** if any fit hits 90% of
the mstar bound rather than printing a verdict from it. A degenerate fit is a failure, not a
measurement, and it should never again be reportable as a result.

Also fixed `collect_outputs()` missing its required `patterns` argument, which errored in the
final cell.

**The DDRM question remains unanswered.** The restoration itself ran (10 channels, finite
output); only the scoring was invalid. Rerunning sections 5-7 with the corrected channel range
will answer it -- the prior does not need retraining.

## 2026-08-28 | bug | notebook 07's bootstrap pointed DATA_DIR at the wrong Dataset

First Kaggle run of notebook 07 failed immediately:
`FileNotFoundError: No valid cube pairs found under .../self-gravitating-v2`.

My bug. The bootstrap globbed `/kaggle/input/**/*dirty*.fits`, which matches
**`dirty_sg.fits`** from the self-gravitating Dataset, so `DATA_DIR` resolved to that Dataset
instead of the line-emission cubes. Notebook 06 used the stricter `*_dirty.fits` pattern,
which would not have matched; I dropped it when writing 07.

Fixed by locating the data through the `run_<id>_<step>_rt_<pp>` folder structure
`split_cubes` actually requires, and picking the directory containing the most such folders.
Verified locally: selects `Line Emission Data` with its 14 run folders, excludes the
self-gravitating Dataset.

**Then executed the notebook's real cell sources end to end** at reduced scale (64px, 1
epoch, 3 channels) against local cubes, rather than the separate dry-run script used before.
All stages run: data (56 train / 16 val), unconditional training (checkpoint written), DDRM
restoration (finite, correct shape), and scoring. The notebook's own relative paths
(`../results/...`, `../results/checkpoints/...`) resolve correctly from `notebooks/`, which is
where the Kaggle bootstrap `chdir`s.

Also hoisted the transfer-function build out of the per-channel loop (identical every
iteration) and dropped an unused `dirty_rs` list.

**Do not read anything into the dry run's "RECOVERY" line.** With a 1-epoch prior on 56
images at 64px, both fits are degenerate (mstar 0.001, inclination ~88 degrees) and the
numbers are noise. The plumbing is what was being checked.

## 2026-08-27 | code | DDRM dry run: caught an oversized-beam bug before it reached Kaggle

`experiments/ddrm_dryrun.py` runs notebook 07's whole pipeline at 64px on real cubes (2 cubes,
3 epochs) purely to catch plumbing failures. It immediately earned its keep: stage 3 crashed
with a broadcast error because `beam_transfer_function` assumed the beam is smaller than the
target grid, and the recovered beam is 129px against a 64px grid.

Worth noting the notebook would NOT have crashed at its 256px setting -- the 129px beam fits
there -- but the same code path would have mis-placed the beam silently, which is the worse
outcome. Fixed by cropping an oversized beam symmetrically about its centre. Verified the peak
gain is identical (~355) at 64/128/256/600px, and regression case 4b checks a centred delta
gives flat unit transfer at every grid size.

After the fix all four stages run on real data: data loading, unconditional training (loss
972.6 -> 920.9 in 10s), checkpointing (70 MB), DDRM sampling (finite output), and GI wiggle
scoring. The restoration numbers at this scale are meaningless by construction and are not
recorded as results.

One number IS worth carrying forward: the transfer function on the real beam passes **1.3% of
modes above 1% gain**, matching the independent measurement in `ddrm_feasibility.md`. That is
the same hard constraint from two different code paths.

**Notebook 07 is now safe to run on Kaggle.**

## 2026-08-27 | code | DDRM notebook (07), plus a checkpointing bug that broke every config

Notebook `07-ddrm-restoration.ipynb`: trains an unconditional diffusion prior over ~2800 clean
Jy/beam channel maps, then restores `dirty_sg.fits` with DDRM using the recovered beam, scored
on the GI wiggle residual against the beam-only floor of 0.116.

**Found a real bug in `DotDict.__getattr__` that broke checkpointing for the WHOLE project,
not just DDRM.** It returned `None` for any missing key, dunders included. `pickle` probes for
`__reduce_ex__`/`__getstate__`, got `None`, and tried to call it -- surfacing as
`TypeError: 'NoneType' object is not callable` from inside `torch.save`, with nothing in the
traceback pointing at `DotDict`. Any `save_checkpoint` call would hit this. Fixed by raising
`AttributeError` for dunder lookups while keeping the `None`-for-missing-key behaviour the
codebase relies on.

Three smaller fixes to make the unconditional path work at all, all in code the conditional
DDPM shares:
- `noise_estimation_loss` hardcoded the conditional concat; now branches on channel count.
- `_epoch_loss` unpacked `for x, _ in loader` and took `x[:, 1:]` for the clean channel, which
  is EMPTY for single-channel input. Both handled.
- Notebook API corrected against the real signatures: `n_epochs` not `epochs`,
  `train_losses`/`val_losses` not `train_loss`/`val_loss`.

Verified end to end on CPU: unconditional training runs, loss decreases (214.7 -> 191.6),
checkpoint writes, and `ddrm_steps` produces finite output. All five existing DDPM/architecture
tests still pass. `tests/test_ddrm.py` now has 5 cases including the pickling regression.

**Not yet run on Kaggle.** The prior needs GPU training.

## 2026-08-27 | finding | ablation: the BEAM alone erases the wiggle, not noise, not the denoiser

Tested the hypothesis left open by the v2-cube run. Convolved the CLEAN cube with the
recovered beam, added no noise, ran no model, then ran the identical Keplerian-fit pipeline.

| | residual RMS (km/s) | residual r vs clean |
|---|---|---|
| clean (reference) | 1.394 | -- |
| **beam-only, no noise, no model** | **0.174** | **0.116** |
| real dirty | 0.176 | 0.111 |
| real denoised | 0.170 | 0.108 |

**The beam alone reproduces the entire effect** -- 0.116 against the real dirty cube's 0.111
and denoised's 0.108, statistically indistinguishable. Raw M1 correlation stays at 0.988,
matching the real cubes' 0.98-0.99, so bulk rotation survives smoothing just as it does in
the real data.

**This settles the v2 cube's open question.** The wiggle is destroyed by the instrument
response before noise or denoising exist. The earlier result was never a model failure on
this cube -- there was nothing left to preserve. A denoiser cannot fix this in principle:
removing noise cannot restore what a convolution erased.

**It also sharpens the DDRM/VIREO case from "worth trying" to "the indicated approach".** A
measurement-consistency prior reconstructs structure the instrument did not measure, which is
categorically different from denoising, and the forward operator it needs is already
recovered and saved (`dirty_beam_recovered_v2.fits`).

Note this does NOT retract the original (wrong-script) cube's finding, where denoised raw-M1
correlation fell to 0.25 against dirty's 0.77 -- that was a genuine degradation on different
data. Here both sit at 0.98.

Caveat: uses the recovered beam, whose held-out validation on this cube was 0.80 rather than
the original pair's 0.994. The agreement with the real dirty cube is close enough that it is
clearly capturing the dominant effect, but a more accurate beam could smooth slightly
differently.

Figure: `results/self-gravitating/gi_wiggle_beam_only_ablation.png`. Write-up:
`results/self-gravitating/beam_only_ablation.md`.

## 2026-08-27 | run | corrected cube tested: a different failure mode, not a repeat

`winner_aug` seed 43 on `clean_sg.fits`/`dirty_sg.fits`, trimmed [60,541) (this cube's
padding is wider: 0-59 and 541-600, checked directly). One denoising pass fed both the
standard moment-improvement metric and the GI wiggle Keplerian fit.

**Moment improvement: M0 +7.5%, M1 -62.3%, M2 -79.5%** (signal-masked). Different from the
original cube's uniformly catastrophic M0 -86.5% / M1 -10.8% / M2 -168.3% -- here M0 is
genuinely positive, M1/M2 are not. Mixed, not uniformly bad.

**GI wiggle, quadratic estimator: all three fitted geometries agree closely** (mstar 0.572 /
0.579 / 0.500 Msun, inclination 32.0 / 30.7 / 32.3 deg, PA and vsys likewise close). No
unphysical local optimum on this cube's denoised fit, unlike the original's.

**Raw M1 correlates very well with truth for both dirty (0.986) and denoised (0.979). Residual
correlation is low for both (0.111 / 0.108), and nearly identical between them.** This is not
"denoising destroys the wiggle" -- it is "the wiggle is not clearly recoverable from the dirty
cube here, and denoising does not meaningfully change that either way." Clean's residual is
large and structured (RMS 1.39); dirty and denoised's are both nearly flat (RMS 0.18/0.17,
~8x smaller) -- their own Keplerian fits absorb almost all their variance.

**A hypothesis, not yet tested:** the beam convolution itself, not noise or denoising, may be
smoothing out the fine structure the wiggle consists of on this cube -- an information-loss
problem rather than a noise problem, which a plain denoiser cannot fix and which is exactly
the case DDRM/VIREO's measurement-consistency prior is designed for.

Figure + data: `results/self-gravitating/v2_cube_test.png` / `experiments/v2_cube_test_result.npz`.
Full write-up: `results/self-gravitating/v2_cube_test.md`.

## 2026-08-27 | run + bug | corrected self-gravitating pair downloaded, and a real Phase 0 numerical bug found

Jason's replacement cube (`clean_sg.fits`/`dirty_sg.fits`, same Drive folder) downloaded and
checked. Different SHA-256 from the originals, same 601x600x600 shape -- confirmed genuinely
different data, not a resend.

**Different setup from the first pair.** Both cubes are now `BUNIT=JY/BEAM` with identical
BMAJ/BMIN/BPA in both headers (the original had no beam info on its dirty cube at all).
Negative pixels in dirty: 19.25%, against the original's 51.55%. Simulation-recipe header
keys (DISTPC/HACNTR/TRKLEN/NTIME/DECDEG) are gone, replaced by RMS/PBCOR/SEED.

**Found and fixed a real numerical bug in Phase 0's fitter.** First run: `A=0.000`, Gaussian
match RMS in the billions, fitted sigma 0.81 px against the header's 5.63 px -- looked like a
data anomaly. Checked the raw spectra before believing it (RULES.md #8): `P_dirty` runs
~120,000x `P_clean` at the lowest k, confirmed visually
(`kinematic_data_v2_amplitude_check.png` -- same physical structure in both, dirty ~110x
brighter at peak). The least-squares solve in `fit_forward_model` was never tested at this
dynamic range and returned garbage. Fixed with column scaling before the solve (Jacobi/Ruiz
preconditioning); regression case 11 verifies a real ~1e10 power-scale factor no longer
breaks it, recovering both A and sigma within a few percent of the truth.

**Phase 0, after the fix: `NON_GAUSSIAN_CONVOLUTION`.** Fitted beam sigma 6.27 px against the
header's 5.63 px -- 11% agreement, "consistent". A real convolution, genuinely a
deconvolution problem, same category as the original self-gravitating pair (before it turned
out to be the wrong script). Figure: `phase0_v2_cube.png`.

**Beam recovery works, less cleanly than the original pair.** Peak 0.991, shallow -1.6%
sidelobe, but held-out validation gives correlation 0.80 (against the original's 0.994) with
60% residual. Reported honestly rather than smoothed over: likely the ~100x amplitude scale
factor between clean and dirty is not perfectly constant channel to channel, which a single
beam averaged across 12 fit channels cannot correct for. Not yet confirmed. Figure + data:
`dirty_beam_recovered_v2.png` / `.fits`.

**Not yet done:** the GI wiggle Keplerian fit and OOD-style denoising test, run on the
original pair, have not been repeated on this corrected one. Natural next step.

Write-up: `results/self-gravitating/kinematic_data_v2.md`.

## 2026-08-27 | correction | Jason: the self-gravitating cube was made with the wrong script

Reply to the 2026-08-27 email (thread "Doubt", jason.terry47@gmail.com, 13:34): "it turns out
there's a reason it's different: I made it with a completely different script. That's my
bad." He has sent an updated version via the same Drive folder link
(folders/1V33FGbjb8JsaQYSbnYkSEcrSSx-b2kfv), which per his note has "a lot more channels than
the other ones" but should otherwise match in structure.

**Not yet downloaded.** The Drive API cannot list this folder's contents for this account
(same limitation hit on 2026-08-07); it needs a human to open the link first. Whoever picks
this up next: download the new pair, confirm channel count and BUNIT on both files the same
way the first version was checked, before assuming anything else carries over.

**What this means for everything dated 2026-08-21 through 2026-08-27 under
`results/self-gravitating/`:** all of it -- the recovered dirty beam (peak 0.911, FWHM 5px),
the OOD moment result (M0 -86.5%, M2 -168.3%), the GI wiggle Keplerian fits (mstar 0.522 to
0.639 Msun, matching Hall+2020's 0.6 Msun), the quadratic-estimator three-way comparison
(0.92/0.28 residual correlation) -- was measured on the cube Jason now says used the wrong
script. None of it is retracted; it may still describe that specific (if unintended) cube
correctly. But none of it should be presented as the definitive characterisation of "the
self-gravitating disk" going forward, since a corrected version exists and has not yet been
examined.

**What is NOT affected:** every piece of general-purpose code built along the way --
`src/evaluation/gi_wiggle.py` (Keplerian fit, quadratic estimator, both validated on
synthetic data independent of this cube), `estimate_beam_from_pair` in
`forward_operator.py`, `plot_phase0_report` -- is reusable on the corrected cube unchanged.
The methods are not in question, only which cube they were pointed at today.

## 2026-08-25 | finding | the dirty beam is measured, so DDRM no longer waits on the mentor

Both datasets are now on the local machine: the line-emission set at
`DENOISING_DIFFUSION/Line Emission Data/` (14 cube pairs, 7.5 GB, splitting 7/2/5 exactly as
the notebook does) and the self-gravitating pair extracted alongside it.

I had recorded that a DDRM arm needs the PSF image from Jason. **That was wrong.** With both a
clean and a dirty cube in hand the operator can be measured from the data:
`B = <D conj(C)> / <|C|^2>`, now `estimate_beam_from_pair` in
`src/evaluation/forward_operator.py`.

| recovered beam | |
|---|---|
| peak | 0.9109, at the exact centre pixel |
| FWHM | 5 px |
| deepest sidelobe | -2.77% of peak, ring from r ~ 26 px |
| total flux, full field | ~0 |

Peak near 1 with zero net flux is the definition of an interferometric dirty beam, and the
negative ring is exactly what Phase 0's `non_gaussian_convolution` verdict was detecting. Two
independent cross-checks agree: Phase 0's Gaussian fit gave sigma 2.07 px (FWHM 4.9 px)
against the 5 px measured directly, and convolving `lines.fits` with the recovered beam
reproduces `dirty_cube.fits` at **correlation 0.9939 on channels 150-450, none of which were
used to fit it**, leaving a 16.6% residual that is the noise term.

Saved as `results/self-gravitating/dirty_beam_recovered.fits` (72 KB, committed with a
gitignore exception since it is a result and re-deriving it needs the 1.7 GB cubes). Note the
129 px crop sums to 26.5 rather than 0 because it truncates the outer negative bowl; anything
needing flux conservation should re-derive at full size.

**What this changes.** "Ask Jason for the PSF" is off the DDRM critical path. What still
blocks a DDRM arm is data volume: one pair cannot train a model, so Jason's "I'll give you
more later" is the ask that matters, not the PSF.

**What it does not change.** DDRM and VIREO's consistency term remain refuted for the
line-emission training data, where `A = I`. Nothing here contradicts that; the two datasets
are simply different problems.

## 2026-08-21 | finding | Jason's self-gravitating pair IS a deconvolution problem, unlike the training data

The cubes Jason shared on 2026-08-07 were downloaded on 08-08 and never opened. They are in
`DENOISING_DIFFUSION/self-gravitating cube and dirty cube/` (gitignored, 1.5 GB zip, two
601x600x600 cubes). Opening them changes the DDRM/VIREO conclusion for this data.

| | training cubes | Jason's new pair |
|---|---|---|
| clean | `BUNIT=JY/BEAM` | **`BUNIT=JY/PIXEL`** |
| dirty | `BUNIT=JY/BEAM` | `BUNIT=Jy/beam` |
| operator between them | none, `A = I` | **a real convolution** |

`lines.fits` is an unconvolved sky model: 0.00% negative pixels, minimum +41.55.
`dirty_cube.fits` is a genuine dirty image: **51.55% of its pixels are negative**, its minimum
is -0.34 of its maximum, and its total flux is ~0. That is the signature of a non-deconvolved
interferometric map, sidelobes pushing flux below zero with no zero-spacing baseline to carry
the total. A restored map convolved with a Gaussian would show neither.

Phase 0 on the pair returns `non_gaussian_convolution` with a fitted amplitude
**A = 293.5**, which is the beam area in pixels, exactly the Jy/beam-against-Jy/pixel factor
the report warns about. The Gaussian fit itself is poor (residual 0.66, RMS 10.7), which is
what a real sidelobed beam should do to a Gaussian model, so the label is right but the fitted
sigma of 2.07 px is not to be trusted.

**What this changes.** DDRM and VIREO's data-consistency term are refuted **for the
line-emission training data**, where both sides are already beam-convolved. They are not
refuted in general, and this pair is precisely the setup they were designed for. The v26
entry's "the physics-informed line is closed" applies to the data trained on so far, not to
this.

**What it would take.** This is a pivot, not a bolt-on. Training on this regime needs (1) more
than one such pair, and Jason said "I'll give you more later", and (2) the **dirty beam / PSF
image**, because with 51% negative pixels a Gaussian `A` would enforce badly wrong constraints.
Both are concrete asks rather than open questions.

Also worth noting: no header beam. `dirty_cube.fits` carries no BMAJ/BMIN/BPA at all, so
`beam_features_of` returns a zero vector and `beam_kernel_of` returns None on this cube. Any
beam-conditioned arm would silently run unconditioned on it.

## 2026-08-21 | run | 05 v26 — the band-limit idea is refuted, and spectral context is a pixel win only

First run with all three post-v25 fixes, so the first to show the spectral arms in the summary
table and the first to measure out-of-band error. Archived at
`results/05-unet-line-emission/v26_2026-08-21_df38336/`.

**The physics-informed line is now closed, on measurements rather than for lack of time.**

Median out-of-band excess **0.5x**, 1.1% of residual power above the beam cutoff. Excess below
1 means the model's error is if anything smoother than the truth, so a band-limit penalty
would be aimed at nothing. The invented structure is **beam-scale, not sharp** — the worse of
the two possibilities, because a beam-scale blob is exactly what a real detection looks like
and no band constraint can separate them.

With Phase 0's `A = I`, all three are struck: DDRM, VIREO-lite's data-consistency term, and
the band-limit variant. Each was killed by a number, not by the calendar.

**Spectral context, three runs per arm at seed 42:**

| arm | PSNR | M0 | M1 | M2 |
|---|---|---|---|---|
| winner_k1 | 41.26 ± 0.92 | 21.0 ± 9.3 | 63.4 ± 4.9 | 33.0 ± 16.3 |
| winner_k2 | 41.81 ± 0.55 | 16.6 ± 2.1 | 67.6 ± 6.1 | 23.8 ± 12.3 |
| winner_aug (3 seeds) | 39.30 ± 0.46 | 29.2 ± 7.2 | 74.0 ± 2.0 | 55.0 ± 13.9 |

About +2.4 dB of PSNR, and it **loses M0 and M2 to augmentation**. Three runs each, so this is
not the seed noise that made v25's reading unreliable. `winner_aug` is still the best arm on
the metric that matters.

That divergence is itself a result worth reporting: the two spectral arms are the clearest
case yet in this project of PSNR and moment reliability moving in opposite directions.

Unlike v25's, this version's push **preserved** the committed cells rather than reverting them.

## 2026-08-21 | run + bug | 05 v25, and the error bars in this project are too small

Second run of `winner_k1` and `winner_k2` at seed 42, same code, and it does not reproduce
the first. Archived at `results/05-unet-line-emission/v25_2026-08-21_889cd44/`.

| arm, seed 42 | first run | v25 | difference |
|---|---|---|---|
| k1 PSNR | 41.948 | 40.219 | 1.73 dB |
| k1 best epoch | 25 | 18 | |
| k1 M2 | +46.2% | +14.8% | 31.4 pp |
| k2 PSNR | 41.199 | 42.269 | 1.07 dB |

A third partial run of k1 hit 41.058 at epoch 22, so three fixed-seed runs span 40.219 to
41.948, standard deviation **0.865 dB**. `winner_aug`'s spread across three different SEEDS
is **0.455 dB**. Re-running one seed varies more than changing the seed.

Cause is GPU nondeterminism, not a notebook bug: `torch.manual_seed` does not fix cuDNN's
algorithm selection, and the T4 x2 setup splits batches across devices.

**Which published numbers this touches.** Every per-arm figure in this project is a single
run, so every band quoted as a seed spread is really a run spread and understates the total.
The 3-seed bands are the least affected because three runs at three seeds do sample it. The
1-seed rows, `winner_beam` (+9.6 / +63.2 / +20.9), `winner_patch`, and now both spectral arms,
have no error bar at all and should not be compared against anything at this resolution.

**Withdrawn:** the reading from the first spectral run that k1 lifts M1 and M2 beyond the
seed spread. v25's k1 gives M1 58.5 against the baseline's 55.6 and M2 14.8 against 6.0, both
well inside the spread. **What survives** is the PSNR gain over the un-augmented baseline,
40.2 to 41.9 against 37.5, consistent across all three runs.

**Also: Kaggle reverted committed work again (RULES.md #2, third time).** Commit `df38336`
pushed the kernel's notebook over `84b6cb6`, deleting 2248 lines: the CONFIGS-derived moment
table, the out-of-band check, and the section 7 centre-channel fix. Restored in `201b4ef`.
v25 itself ran at `889cd44`, before all three, so it has no out-of-band measurement and its
summary table again omits the spectral arms. The VIREO question is still open.

## 2026-08-20 | code | the spectral-context arms are wired into 05 and ready to run

Two new arms in notebook 05, `winner_k1` and `winner_k2`, identical to `sweep_winner` except
that the input is the channel plus k neighbours along velocity. This is the Phase 3a item and,
after Phase 0, the only physics-informed lead still standing.

Wired end to end rather than only in `src/`:

- `train_unet(n_neighbors=k)` forwards to `build_model`, and the checkpoint now records
  `in_channels`, so a resumed run rebuilds the right shape instead of assuming 1.
- Cell 12 builds the k=1 and k=2 views and asserts their centre channel is byte-identical to
  the k=0 item, so the arms stay a clean one-change ablation.
- Cell 14's score-without-retraining path reads `in_channels` from the checkpoint and scores a
  k>0 arm on its own val view, because a 2k+1-channel model cannot be fed the 1-channel
  loader at all.
- `denoise_cube` builds the neighbour stack for section 6, with neighbours normalised by the
  CENTRE channel's (min,max) and the cube's ends clamped, matching the dataset exactly.

That last one is where a silent bug would have lived. `denoise_cube` keeps a per-channel
`norm` array, and slicing it for neighbours would have given each its own scale, erasing the
relative amplitude along velocity, which is the whole signal these arms add. It would have
trained on one representation and scored on another with nothing raising, exactly the shape of
the `winner_beam` failure. `tests/test_denoise_cube_spectral.py` executes the notebook's real
`denoise_cube` source against a spy model and compares its input tensors to the dataset's:
8/8 channels identical, and the deliberately-wrong version is flagged 8/8, so the check can
fail (RULES.md #8).

Verified by training a k=1 arm for one epoch on synthetic cubes: trains, writes
`in_channels=3`, and the resume path rebuilds it and reproduces the same PSNR.

**Not yet run on real data.** Suggest k=1 and k=2 at one seed against `sweep_winner` at the
same seed. Expected effect is on M1 and M2 specifically; if they do not move, the
per-channel-independence explanation for their weakness is wrong and worth revisiting.

## 2026-08-20 | finding | the clean cubes are already beam-convolved

Run 4 came back `indeterminate` on all four cubes, and the diagnostic dump answered the
question outright.

**Both cubes are `BUNIT=JY/BEAM`.** A model or sky image would be Jy/PIXEL. Jy/beam means the
beam has already been applied, to the CLEAN cube as well as the dirty one. The spectra agree:
`P_clean` for run_0006 falls 1650 -> 4.6 -> 0.0069 -> 1e-6 -> 4e-11, thirteen orders by
k = 0.104, exactly where the header's 7.83 px beam cuts off, and then flattens onto the
float32 floor near 1e-12. An unconvolved model image would keep following a power law.

**So the answer to Phase 0 is that there is no operator BETWEEN the two cubes.** The pair is
`dirty = clean + noise`, with the same beam already inside both sides. The ratio behaves
accordingly: it climbs 1.009 -> 2.42 -> 8.79 -> 41.7 -> 58.6 and never dips, which is
`1 + N/P_sky` with `P_sky` falling.

Two code faults this exposed, both now fixed and both regression-tested:

- **The band included the float32 floor.** Past k ~ 0.10 neither spectrum is physical, yet
  their ratio settles near a plausible 2.2. Mixing that with the real region, where the ratio
  climbs to 58, left no (A, N) able to fit either, which is what produced sigma = 0 with
  residuals of 0.42 to 0.89. `_measurement_band` now cuts on the CLEAN floor as well as the
  dirty one.
- **The noise was modelled as white.** In a Jy/beam map the noise has been through the beam
  too, so its spectrum is beam-shaped; a flat term implied N running from 16 down to 2e-9
  across the band. Both a white and a beam-shaped term are now fitted together, so neither
  the caller nor the code has to guess which kind a cube has.

Also fixed a `KeyError: 'indeterminate'` in the notebook cell's verdict-explanation dict.

**What this means for the plan.** DDRM has nothing to invert: the instrument response is not
between the network's input and its target. VIREO-lite's image-plane data-consistency term is
dead for the same reason, since `A = I` makes it `||pred - dirty||`, which would train the
model toward the noise. Both are struck, and not for lack of time.

This is a result worth stating in the final blog rather than a dead end: **the problem here is
denoising, not deconvolution.** The network is learning to remove beam-correlated noise from
an already-convolved map. It also reframes the Friday questions: the useful ask is no longer
the PSF image but whether an unconvolved (Jy/pixel) sky cube exists, because that is what
would turn this into a deconvolution problem and put DDRM and VIREO back on the table.

## 2026-08-20 | bug | Phase 0 run 3: the input was half empty channels

`non_gaussian_convolution` 4/4 again, and this time visibly broken: best-fit beam **0.00 px**
with gain 1.00x, meaning the fit found no beam at all, yet it still landed in the convolution
branch and printed Gaussian RMS values up to 7e32.

**Two faults.**

1. **Input.** `phase0_from_fits` took evenly spaced channels via `np.linspace(0, n-1, 8)`,
   which includes channel 0 and channel n-1. Those are the extreme high-velocity ends, which
   the mentor's own sampling note (2026-06-18) calls "mostly continuum with little signal".
   Clean power there is near zero, so P_d/P_c explodes for reasons unrelated to any beam, and
   averaging them in with real channels left a model that fits neither way: no-beam residual
   0.5556 / 0.8029 / 0.8657 / 0.8992. Channels are now ranked by clean standard deviation and
   the line-bright ones used.
2. **Logic hole.** A fit landing on sigma = 0 found no beam, so it can never be a convolution.
   It now returns `indeterminate` with the reason, instead of falling through to a
   Gaussian-vs-sidelobe comparison against a flat transfer.

`phase0_diagnostics()` added and wired into the notebook cell: on an indeterminate verdict it
now dumps shapes, BUNIT, header keys compared between clean and dirty, the radial spectra as
a table, and the per-channel clean std. Three wrong verdicts came from a summary statistic
hiding what the spectra were doing, so a surprising answer should no longer be guessed at.

**Phase 0 still unanswered**, fourth run pending. Published numbers touched: none.

## 2026-08-20 | bug | Phase 0 got two more wrong verdicts before the method was right

Three versions of the discriminator, three confident wrong answers on the real cubes. Nothing
was recorded from any of them, and no published number is affected, but the pattern is worth
keeping: each failure came from choosing a normalisation and then testing a threshold against
it, rather than fitting the thing being measured.

1. **Raw ratio dips below 1?** Breaks when the two maps are on different intensity scales.
   Real cubes returned `no_convolution` 4/4 with minima 1.010 / 1.024 / 1.138 / 3.962.
2. **Normalise the ratio by its own low-k level, take the minimum over the band?** For a
   monotonically rising ratio the lowest bin is by construction below the median of the
   lowest bins, so a dip appears where nothing was suppressed. Real cubes flipped to
   `non_gaussian_convolution` 4/4, with `k_at_min` pinned at the lowest bin, 0.004, on every
   one, and Gaussian fit RMS of 1.59 to 3.47. It was reporting non-Gaussian because the fit
   had failed, not because it found sidelobes.
3. **Does a Gaussian beam beat the no-beam model?** A sidelobed beam is not Gaussian, so
   neither model fits, the comparison ties, and a real convolution reads `no_convolution`.

**What replaced them.** Fit the forward model directly,
`P_d(k) = A exp(-4 pi^2 sigma^2 k^2) P_c(k) + N`, and ask whether its no-beam version is
adequate ON ITS OWN. `A` is a free parameter, so intensity scale cannot mislead it; sigma is
recovered rather than inferred from a threshold. Given sigma the model is linear in (A, N),
so a scan over sigma with a linear solve at each step is exact and needs no optimiser.

Separation on synthetic cases is two orders of magnitude, so the threshold is not delicate:

| case | no-beam residual |
|---|---|
| additive noise, x1 to x25 | 0.0008 to 0.0107 |
| sidelobed beam | 0.8938 |
| Gaussian beam, sigma 1 to 3 | 2.86 to 3.21 |

Recovered sigma is exact to 0.3% and unchanged across a 200x range of intensity scale. All
three failures are regression cases in `tests/test_forward_operator.py`.

**Phase 0 is still unanswered.** Needs a third run.

## 2026-08-20 | bug | Phase 0's first real run was wrong: the check was not scale-invariant

First run on the project's cubes returned `no_convolution` on 4/4, which would have killed
DDRM and VIREO-lite outright. The numbers gave it away: minima of **1.010, 1.024, 1.138 and
3.962**. If `A = I` truly held, `dirty = clean + noise` puts every cube at ~1.00. A 3.96 means
that cube's dirty map carries 15.7x the clean power at the largest scales, which additive
noise cannot do, and a 1.01-to-3.96 spread is not one operator measured four times.

**Cause.** The verdict tested whether the raw ratio `P_d/P_c` dips below 1, which assumes
clean and dirty share an intensity scale. A dirty or restored map is conventionally Jy/BEAM
and a model image Jy/PIXEL, differing by the beam area in pixels, order 200 for these
headers. Any such factor multiplies the whole ratio and lifts a real convolution above 1.

**Reproduced before fixing.** Same synthetic Gaussian blur, four intensity scales, old code:

| scale | verdict | min |
|---|---|---|
| x1 | gaussian_convolution | 0.060 |
| x5 | non_gaussian_convolution | 0.291 |
| x25 | **no_convolution** | 1.456 |
| x200 | **no_convolution** | 11.824 |

The observed 1.01-3.96 sits inside that failure range.

**Fix.** Normalise the ratio by its own low-k level before the dip test. A convolution kernel
has unit sum so |B(k)| -> 1 as k -> 0, meaning the low-k level estimates the scale factor by
itself; dividing it out leaves the shape, and the shape is what separates the two cases. The
factor is now reported as `dirty/clean low-k amplitude`, flagged when it is far from 1, so a
units mismatch is visible rather than silently steering the verdict.

Six regression cases added, x1 through x200 convolved plus rescaled additive. Recovered beam
sigma is now 2.97 px across a 200x range of scales.

**Second look, same day: the scale factor was probably not the story.** Jy/beam against
Jy/pixel predicts a factor near 200; the cubes showed 1.01 to 3.96. And on all four the
minimum sat in the LOWEST k bin, so the ratio never fell anywhere, and dividing out a factor
that small cannot change that. The re-run will most likely return `no_convolution` again.

What was actually missing is a check that the measurement could have SEEN a beam. The band
ends where the dirty spectrum sinks into its own noise floor; the header's beam (sigma ~6.5
px) does not suppress anything below k ~ 0.033. A null from a band that stops short of that
is uninformative, not negative. `phase0_report` now reports the band's reach against the
header's own rolloff and returns `indeterminate` rather than `no_convolution` when the band
was too narrow to decide. RULES.md #8, applied to this diagnostic itself.

**Published numbers touched: none.** The bad verdict was never recorded anywhere; it was
caught in the same session it was produced. **Phase 0 is unanswered** and needs a re-run,
which will now say either `no_convolution` with the band coverage to back it, or
`indeterminate`.

## 2026-08-19 | code | 2.5D spectral context, the one Phase 3 item nothing gates

`FITSChannelDataset(n_neighbors=k)` now emits a `(2k+1, H, W)` dirty tensor, the sampled
channel plus k neighbours each side along velocity, with the clean target still the centre
channel alone. `_build_unet(n_neighbors=k)` sets `in_channels` to match; `UNet` already took
`in_channels`, so there was no model change to make.

**Why this one and not VIREO-lite.** VIREO-lite turns out to be gated by Phase 0 as well, not
just DDRM: if `A = I` the data-consistency term collapses to `||pred - dirty||`, which would
push the model toward the noisy input. Spectral continuity is gated by nothing, and the plan
already called it the highest-value item in the document. M0 is a spectral sum and scores
~+70%; M1 and M2 are spectral shape statistics and lag, because every channel is denoised
independently and nothing uses the axis those two are computed over.

22 checks in `tests/test_spectral_context.py`, and the existing 24-test data pipeline still
passes. `n_neighbors=0` reproduces the old items bit-for-bit, asserted with `torch.equal`
rather than a tolerance, so no existing result changes meaning.

Two decisions in it would have silently destroyed the point if made the other way: neighbours
share the CENTRE channel's scale (per-neighbour normalisation would erase relative amplitude
along velocity, which IS the added signal), and the cube's ends clamp rather than wrap (the
first and last channels are the line-free high-velocity ends and are unrelated).

**Not yet run.** The notebook builds its own datasets and cells do not sync from git
(RULES.md #2), so an arm needs `n_neighbors` threaded through 05 by hand.

## 2026-08-19 | code | Phase 0's forward-operator check exists, verdict still unknown

`src/evaluation/forward_operator.py` + `tests/test_forward_operator.py`. Settles the gate in
PHYSICS_INFORMED_PLAN.md that decides whether DDRM gets built: is `dirty = clean (*) beam +
noise`, or just `dirty = clean + noise`? In the second case `A = I` and DDRM collapses into
the conditional DDPM that already exists.

Discriminator is the radially averaged power-spectrum ratio, which dips below 1 only if
something suppressed spatial frequencies. Three verdicts, because a convolution by a real
dirty beam is not the same finding as a convolution by a Gaussian: the first needs the PSF
image from Jason before `A` can be written down.

Also lands `pixel_scale_arcsec` and `beam_kernel_of`, reading `CDELT`, which no code in
`src/` read before. `beam_features_of` takes BPA/BMAJ/BMIN only, which describes a beam in
angular units but cannot build a kernel in pixels.

Verified against three synthetic cases with known operators, 14 checks. Two wrong versions
were caught by them and both are now regression cases: a signal band set as a fraction of
peak power, which cuts off at k = 0.04 on a red spectrum and makes every beam look Gaussian;
and reading the verdict off the noise-subtracted ratio, whose floor estimate manufactures a
dip where nothing was suppressed.

**Nothing measured yet.** The check needs FITS data, which lives on Kaggle. Until it runs,
DDRM stays unbuilt, per the plan's own gate.

## 2026-08-14 | run | 05 v19 — crashed at 4.0 h, no moment scores

`DeadKernelError` at 14413s, inside Kaggle's 12 h limit, so a crash rather than a timeout.
`ConnectionResetError: [Errno 104] Connection reset by peer` in
`multiprocessing/resource_sharer.py`, after `winner_patch` early-stopped at epoch 26 with its
epoch times drifting 111 to 128s where every other arm held flat near 90s. Memory pressure,
most likely `FlatPatchDataset.__getitem__` decoding each image once per patch, 8x redundant.

Section 6 never ran, so the run produced **no moment scores at all**. Three checkpoints
trained and survived in the Output; `winner_patch`'s metric row did not, because the kernel
died between its last epoch and `val_metrics`.

**Consequence:** 05 gained a resume path (`_import_prior_nb05`), a score-without-retrain
branch for a checkpoint with no metric row, `RUN_NATIVE600 = False`, and a guard so section 5
cannot `NameError` when every arm resumes. **Notebook not archived. Lost.**

## Earlier

See `RUNS.md` for the full per-notebook index back to 05 v12 and 06 v11. This log starts at
the point the project began losing information that the run folders alone did not capture.
