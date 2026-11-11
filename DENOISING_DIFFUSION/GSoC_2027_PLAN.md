# GSoC 2027 plan: a telescope-independent kinematic denoiser, and what it can find

Author: Krishan Yadav. Written 2026-11-12. Status: draft for discussion with the EXXA mentors; nothing here is agreed yet.

GSoC 2026 is complete and passed. Its work is merged in this repository under
[`Final Submission/`](Final%20Submission/) and [`Mid-term Submission/`](Mid-term%20Submission/). This plan builds on that finished
project. It has two parts: small monthly contributions from November 2026 to April 2027, and the science of a GSoC 2027 proposal.

---

## 1. The scientific goal

The 2026 denoiser recovers the velocity wiggle of gravitationally unstable disks in ALMA line cubes, and its evaluation was built to catch
smoothing and hallucination. Three questions are still open, and they are the science of 2027:

1. **Does it depend on the data it was trained on?** The 2026 model used private simulations, so its weights cannot be released. Can a
   model trained only on public data do as well?
2. **Does it depend on the instrument?** Every 2026 result is ALMA, at the beam, channel width and noise of ALMA cubes. Which properties of
   an observation decide performance, and can one model handle a range of them?
3. **What can it find?** A denoiser is only useful if it reveals real structure that was hidden. Published planet kinks exist in three of
   the 20 real disks the 2026 pipeline already processed. Does denoising preserve them, and how far down in SNR can it recover them?

Each question below is stated with a hypothesis, the test, and a pass criterion fixed now, before any 2027 experiment is run, following
the 2026 practice of writing adoption rules before results are seen.

## 2. What 2026 established (the baseline every 2027 result is compared with)

Recipe B: blind 3D Wiener filter plus a U-Net correction, four networks averaged, do-no-harm gate, two passes, beam resampled to about
10 px. Numbers from [`Final Submission/README.md`](Final%20Submission/README.md) and its result folders.

| Measurement | Recipe B | Wiener alone |
|---|---|---|
| Final test disk, wiggle score, SNR 12 / 24 (mean of 4 noise draws) | 0.59 ± 0.10 / 0.79 ± 0.06 | 0.18 ± 0.03 / 0.42 ± 0.02 |
| 8 fresh analytic disks, beam to undo | 0.79 | 0.47 |
| 8 fresh Young et al. simulations | 0.56 | 0.33 |
| GI disk injected in real MWC 758 noise, SNR 12 / 24 | 0.844 / 0.926 | 0.638 / 0.831 |
| 20 real disks: fitted position angle within 3.9 deg of published | 19 of 20 | |
| Spurious wiggle on no-wiggle twins, SNR 12 / 24 | 27 ± 5 / 18 ± 3 m/s | |
| Flux within 1 beam on a bright-centre injected disk | 1.28 to 1.41 x truth | |
| AS 209 at SNR 12, native 0.35 km/s channels / interpolated to 0.1 | 0.19 / 0.41 | 0.65 (interpolated) |

The 2026 work also measured an information limit: an ideal linear filter that knows the true spectrum reaches 0.45 at SNR 12 on the
selection disk. Gains in 2027 are expected from generality and from what the model can be trusted to find, not from a large jump in score.

---

## 3. Research questions

### RQ1. Can a model trained only on public data match the 2026 model?

- **Why it matters.** Public data means public weights: anyone can use the denoiser, and every result can be reproduced.
- **Public training data.** The analytic GI disk generator (Longarini et al. 2021 model, no external data) and the Young et al. (2026) GI
  simulations (63 runs, public on Zenodo, record 22045440). In 2026, 38 of the 63 were used: 16 for a training candidate and 22 for
  testing (`experiments/make_sph_cubes.py`, splits TRAIN, TEST, TEST2, TEST3). **25 runs have never been rendered or seen.**
- **Evidence it is feasible.** In 2026, adding 16 Young runs to training gave 0.53 to 0.54 on the 8 fresh Young disks against 0.56 for the
  control, and 0.79 to 0.85 on the fresh analytic disks against 0.80. That model still also used private data; a public-only model has not
  been tried.
- **Test.** Split the 25 unused Young runs now into a 2027 training addition and a sealed 2027 test set (TEST4), and fix new analytic
  seeds for testing. Train three seeds on public data only. Evaluate once on TEST4, the new analytic disks and their twins.
- **Pass criterion.** Mean wiggle score on TEST4 and on the new analytic disks each at least recipe B's minus 0.03, and spurious wiggle on
  twins no more than 3 m/s above recipe B's. If it fails, the release uses recipe B's design retrained on public data anyway, with the gap
  reported.

### RQ2. Which properties of an observation decide performance, and can one model cover them?

- **Hypothesis.** Performance is set by a few measurable properties: beam size in pixels and its axis ratio, channel width relative to the
  line width, SNR per channel, channel-to-channel noise correlation, and how noise varies across the field (primary-beam correction).
  Training over a range of these (domain randomisation) gives a model that works at values it was not trained on.
- **Evidence.** In 2026, coarse channels cost most: AS 209 at native 0.35 km/s scored 0.19 against 0.41 after interpolation. Beams were
  trained with axis ratio up to 1.3; real beams up to about 1.35 were processed but the effect was never measured. Noise was assumed
  uniform across the field.
- **Test: an instrument ladder.** Render held-out disks under a grid: axis ratio 1.0 to 2.0, channel width 0.1 to 0.5 km/s, three noise
  correlation strengths, and a primary-beam noise ramp. Score recipe B and the randomised model at every grid point. Check the image-plane
  noise model against visibility-based simulations (CASA simobserve) for two array configurations, since the 2026 pipeline only
  approximated the instrument in the image plane.
- **Pass criterion.** The randomised model keeps at least 90 percent of its own in-distribution wiggle score across the grid, and on
  AS 209 at native channels it scores at least 0.41 (the 2026 interpolated result), removing the need to interpolate.
- **Other telescopes.** Real non-ALMA cubes (for example NOEMA, which observes northern disks) are the natural real-data test of RQ2.
  Using them is conditional on the mentors agreeing that other telescopes are in scope.

### RQ3. Does denoising make independent observations of the same disk agree, beyond what the model's own prior explains?

- **Why it matters.** Real data has no truth. Two independent observations of one disk are the best substitute: real structure should
  agree between them, noise should not.
- **Evidence.** In 2026, HD 143006 was observed by both exoALMA and DSHARP. The agreement of their rotation-curve substructure rose from
  0.20 (raw) to 0.80 (denoised). One disk, no significance test.
- **The trap.** A denoiser imposes its own smooth prior on both observations, which can create agreement that is not in the sky. The test
  must separate the two.
- **Test.** Pairs of independent observations of the same disk: HD 143006 (exoALMA and DSHARP), and AS 209, HD 163296 and IM Lup
  (DSHARP and MAPS, matching lines to be checked). Measure agreement raw and denoised. Null distributions: (a) one map rotated or shifted
  against the other, (b) denoised twin pairs with no signal and independent noise, which gives the agreement the prior alone produces.
- **Pass criterion.** Denoised agreement exceeds both nulls at p < 0.01 on at least 3 of the 4 pairs. All pairs are reported either way.

### RQ4. Can the uncertainty map be calibrated so its error bars mean what they say?

- **Evidence.** The 2026 `--boot` map gives a median M1 uncertainty of 12.6 m/s inside MWC 758, from the noise term only. The injection
  test in the same noise measured line-centre errors of 12 to 27 m/s. The difference is the model's own error, which the map omits.
- **Test.** Model the total error as the bootstrap term plus a model term that depends on radius in beams and local SNR, fitted on the
  2026 injection results. Check on new injections into noise sources not used for the fit.
- **Pass criterion.** In every SNR bin, the fraction of pixels whose true error is within 1 sigma is 68 percent plus or minus 5.

### RQ5. Can it preserve and recover planet kinks?

- **Targets with published answers.** Pinte et al. (2020) report nine localised deviations from Keplerian rotation (kinks) in eight
  DSHARP disks, including **HD 143006, HD 163296 and IM Lup, three of the 20 disks the 2026 pipeline already processed**. They note the
  kinks are usually seen in a single channel and need confirmation. HD 163296 has a blind SNR of 28 in the 2026 run.
- **Evidence and difficulty.** In 2026, a compact 171 m/s feature one beam wide was recovered by no method, and the HD 142666 kink could
  not be tested at its SNR of about 7. Compact features are the model's weakest case, so this is the most uncertain question.
- **Test.** (a) Preservation: denoise the three disks and check the kinks at the published positions in the channel maps. (b) Recovery
  curve: add noise to bring each disk to lower SNR and find where the kink is lost, with and without denoising. (c) False positives: the
  same procedure on twins and on mirror-symmetric control regions (the 2026 `kink_check.py` test). (d) Training: add analytic
  planet-wake velocity perturbations to training (candidate source: the public `wakeflow` code, to be checked) and repeat (a) to (c).
- **Pass criterion.** At least 2 of the 3 published kinks preserved at native SNR, a completeness curve with a false-positive rate on twins
  below 5 percent, and the SNR at which kinks are lost reported with and without denoising.
- **Output.** A completeness map (kink amplitude and size against SNR) and, applied to archival disks, a ranked candidate list with
  false-positive estimates. Candidates, not discovery claims.

### RQ6. Two engineering limits with measurable targets

- **Flux near the star.** Recipe B returns 1.28 to 1.41 times the true flux within one beam of a bright-centre disk. The 2026 candidate
  correction reached 0.91 to 1.03 but cost 6.5 points of M0 on smooth disks. 2027 target: a flux-conserving output (the correction
  constrained to integrate to zero flux in each annulus) within 0.95 to 1.05 inside one beam, with no more than 2 points of M0 lost on the
  analytic disks.
- **Coarse channels.** Covered by RQ2's native-channel criterion.

### Optional: galaxies

Streaming motions along spiral arms in galaxies are the galaxy counterpart of the GI wiggle. Public galaxy CO cubes (PHANGS-ALMA, about 90
nearby galaxies) could test whether the method transfers to a different kind of disk. Only if it fits with the separate galaxy
classification work Katia Matcheva proposed.

---

## 4. Data

| Data | Use | Access |
|---|---|---|
| Analytic GI disks and twins | training, testing, benchmark | generated, public |
| Young et al. 2026 GI simulations (25 unused runs) | training addition and sealed TEST4 | public, Zenodo 22045440 |
| exoALMA (15 disks) | real-data evaluation, inference only, never trained on | public |
| DSHARP (5 disks; kinks in HD 143006, HD 163296, IM Lup) | RQ3, RQ5 | public |
| MAPS (AS 209, HD 163296, IM Lup among its 5 disks) | RQ3 pairs | public, ALMA archive and alma-maps.info |
| Planet-wake models | RQ5 training | to be checked (`wakeflow`) |
| Non-ALMA cubes (for example NOEMA) | RQ2 real-data test | only with mentor agreement; NOEMA raw data are not distributed by the IRAM archive, so published cubes only |
| Private 2026 training data | not used in 2027 | not public |

## 5. How results will be judged

The 2026 protocol carries over: wiggle score, PV correlation and PV magnitude reported together, twin tests for invented structure,
injection into real noise, and one sealed test set run once.

New in 2027:
- **Pre-registration.** The TEST4 run names, the new analytic test seeds and every pass criterion in section 3 are committed before
  training. The commit hash is the record that they were fixed in advance.
- **Significance.** Every agreement or detection is reported against a null distribution (RQ3, RQ5), not as a bare number.
- **Every outcome reported.** A failed criterion is a result and goes in the report.

## 6. Before GSoC: one small pull request a month (November 2026 to April 2027)

Each PR prepares one research question, needs no new private data, and needs no permission beyond what 2026 had.

| Month | Pull request | Prepares |
|---|---|---|
| Nov | CI: GitHub Actions runs the 45 tests on every PR (CPU only, synthetic data) | all |
| Dec | Cube reader for any instrument: frequency or velocity axis, per-channel beams, units, primary-beam flag; tests on synthetic headers | RQ2 |
| Jan | Public analytic disk benchmark v1: versioned disks, twins, truth and a scoring script | RQ1, RQ2 |
| Feb | Uncertainty calibration pilot on the existing injection results | RQ4 |
| Mar | Kink pilot: check the published Pinte et al. kinks in the already-denoised HD 163296, HD 143006 and IM Lup cubes | RQ5 |
| Apr | One small fix while proposals are reviewed | |

Outside the PRs: in November, ask the mentors about other telescopes, releasing the 2026 weights and the 2027 idea; in December, read
the closest prior art (section 9); in March, write and submit the proposal (the 2026 application window was 16 to 31 March).

## 7. GSoC 2027 timeline (assuming 350 hours over 22 weeks, as in 2026)

| Weeks | Work | Gate |
|---|---|---|
| Bonding | agree the pre-registration with mentors; render the 25 unused Young runs; download MAPS pairs | pre-registration committed |
| 1 to 4 | RQ1: public-only training, three seeds | |
| 5 to 7 | RQ2: domain-randomised training and the instrument ladder | **Midterm: RQ1 criterion met, or the gap measured and explained** |
| 8 to 12 | RQ3 cross-observation tests; RQ4 calibration | |
| 13 to 18 | RQ5 kinks; RQ6 flux | |
| 19 to 22 | release, benchmark v2, report, paper draft | |

If the programme length is 12 weeks instead, the minimum version is RQ1, RQ2 on synthetic data, RQ3 on ALMA pairs, and the release.

**Compute.** Each 80-round training run took about 11 hours on a Kaggle P100 in 2026. Estimate: about 33 h for RQ1, 40 h for RQ2 plus 10 h
of ladder evaluation, 33 h for RQ5 training, 22 h for RQ6 and about 40 h of evaluation: roughly 180 GPU hours, about 8 a week over 22
weeks. Every job saves after each round and resumes across Kaggle's 12-hour sessions.

## 8. Deliverables

1. A denoiser trained only on public data, with **released weights**, installable with pip, reading cubes from any instrument.
2. The public analytic disk benchmark (v1 before GSoC, v2 with the instrument ladder and kinks).
3. A calibrated uncertainty map with a documented coverage test.
4. A report answering RQ1 to RQ6, each against its pre-registered criterion, failures included.
5. A kink completeness map and a candidate list for archival disks, with false-positive rates.
6. A paper draft, with the mentors.

## 9. Risks, and what happens if they occur

| Risk | Response |
|---|---|
| Public-only model falls short of recipe B (RQ1) | release it anyway with the measured gap; recipe B stays the reference |
| Kinks too compact to recover (RQ5, the 2026 weak case) | report the completeness curve and the SNR limit; a negative result still sets a limit others can use |
| Agreement in RQ3 explained by the prior | the twin null exists to detect exactly this; report it |
| Other telescopes not in scope, or no public non-ALMA cubes | RQ2 runs on the synthetic ladder and on ALMA surveys with different channel widths and beams |
| Overlap with recent learned denoisers for interferometric data | position against them before the proposal (section 10) |

## 10. Prior art to read before the proposal

- A 2026 learned denoiser for interferometric data trained on visibilities of 1,489 SPH gas-only disk simulations
  ([arXiv:2605.14971](https://arxiv.org/abs/2605.14971)): the closest work; the proposal must state what this project adds (kinematic
  evaluation with twins, real-noise injection, telescope-independence, calibrated uncertainty).
- Pinte et al. 2020, kinks in DSHARP disks ([arXiv:2001.07720](https://arxiv.org/abs/2001.07720)); Pinte et al. 2018 for HD 163296.
- Terry et al. 2022 and 2023 (machine-learning kinematic planet detection; HD 142666).
- MAPS I and II (Öberg et al. 2021, [arXiv:2109.06268](https://arxiv.org/abs/2109.06268); Czekala et al. 2021,
  [arXiv:2109.06188](https://arxiv.org/abs/2109.06188)) for the cross-observation pairs and the JvM correction.
- Kinematic fitting tools the pipeline should interoperate with rather than replace: eddy, discminer.

## 11. Decisions needed from the mentors

1. Are telescopes other than ALMA in scope for 2027, and if so which one first?
2. May the 2026 weights be released, or should 2027 rely only on the public-data model?
3. Is the kink search (RQ5) in scope, and how should candidates be followed up?
4. Large (350 h) or medium (175 h) project?
5. Does the galaxy extension belong here or in the separate galaxy classification project?
