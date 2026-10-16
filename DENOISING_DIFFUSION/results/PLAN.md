# Plan: a few checkpoints that hold at 256 / 300 / 320 / 480 / 600 px and work on realistic cubes

Adopted 2026-10-10 (Krishan approved). Progress against it is logged in `PROGRESS.md`; runs in `RUNS.md`.

## 0. Done means

**Krishan, 2026-10-11 (supersedes the table below where they differ):** at all 5 sizes, on held-out cubes: best PSNR and M0/M1/M2 (incl. dispersion);
wiggle above dirty and Wiener, and as close to clean as possible; trained on line emission, self-gravitating and wiggle data; works at exoALMA-like
noise; totally ready for ALMA. Made measurable as: the ceiling for the wiggle is the NOISE-FREE beam-convolved clean (rc_r 0.88-0.94 on our discs), not
clean itself; targets on realistic held-out GI discs (Speedie SNR ladder, held-out simulations): rc_r >= 0.85 at peak SNR >= 13 (exoALMA fiducial floor),
reported (not promised) at SNR 3-10, Keplerian twin spurious <= 5 m/s, median line-centre error <= 20 m/s at SNR >= 13. "Ready for ALMA": the
inference adapter runs on exoALMA fiducial cubes with checks S1, S2, S5, S6 of ALMA_PLAN.md passed.


| test set | metric | bar |
|---|---|---|
| 5 line-emission holdouts, all 5 sizes | PSNR, M0 / M1 / M2 | at least unified-wiener v1 (600 px: 46.3 dB, 33 / 85 / 83 %), no size more than a few dB or points worse |
| hydro 9074 (never trained), SG v2, an07, an08 | rc_r (wiggle) | at or above plain Wiener, never below dirty |
| Speedie GI r0.5 and r1.5 (never trained, realistic noise) | M0 / M1 / M2 | all above dirty (2026-10-10: -36 to -952 %) |
| Speedie | rc_r | at or above Wiener's 0.91 |
| Speedie Keplerian twin | spurious wiggle | near 0 m/s (2026-10-10: +7 to +16; Wiener -4.6) |

Every result reports the spread across cubes and seeds, with the Wiener and dirty rows. Selection per RULES.md #4: wiggle and
moments on held-out cubes, never PSNR alone. Deliverable: 2-3 checkpoints; two split by data type is an acceptable outcome.

## 0b. Open items before ALMA, in priority order (2026-10-13, Krishan: sort these first, then ALMA)

| # | item | why here | status |
|---|---|---|---|
| 1 | Seed 11 shared with chirag | blocks the averaged final evaluation | **DONE** |
| 2 | High-SNR M0 damage: do-no-harm gate | known flaw, known fix, no training; must be in BEFORE the final evaluation so the frozen numbers describe the shipped pipeline | **DONE 2026-10-13**: gate (40, 120, 0.25) chosen from `eval-gate`; in `train_unified.py --gate`, `alma_unified.py` (default on) and `final_report.py --apply-gate`; SNR 135: M0 / M2 -76 / -257 -> -4 / +50 |
| 3 | Seed 23 completes | running | **DONE** (stopped at its 11 h budget, resumed, 72 of 72 cases) |
| 4 | Final averaged evaluation (seeds 3 + 11 + 23, gate in), 5 sizes, Speedie, ladder, held-out Young sims | this IS the frozen model | `eval-final` RUNNING on chirag since 15:35 (~7-8 h; gate applied afterwards from its stored mixes) |
| 5 | Final figures and results table | needs 4 | `final_report.py` ready |
| 6 | R6 (256 px crops) decision | job running; adopt only if SNR 12-24 wiggle beats R4n-long seeds without hurting line emission | due Oct 15 |
| 7 | Wiggle at SNR 12 (0.41-0.60 vs 0.85) | two attempts: R6 (256 px crops) and R7 (training SNR 3-40, 3 seeds, new account); adopt the better one only if it beats the R4n-long seeds on SNR 12-24 without hurting line emission; else a measured limit with the SNR curve | R6 running; R7 queued for the new account |
| 8 | SG v2 equal to dirty | ceiling-limited: noise-free beam reference 0.92, dirty 0.85 (peak SNR 63); documented, no work | write-up |

Then: re-run ALMA (14 discs + MWC 758) with the frozen model (~5 min per cube).

## 0b2. Wiggle plan (2026-10-15, Krishan: fix the large-size wiggle and the real-GI wiggle before ALMA; freeze moves Oct 15 -> ~Oct 18)

**Problem 1, wiggle at 480 / 600 px.** Measured: same line cubes 0.80-1.00 at 256 px, 0.49-0.69 at 600 px; Speedie at 1024 px (beam 12 x 9 px) fine (0.89).
The network works at a beam scale (~8-12 px), not an image size; at 600 px the line beams span ~20 px. Fix: `--infer-beam-px B --infer-mode corr|full`.
Test `aditisah11/size-test` (seeds 3 + 11, plain / b10 corr / b10 full / b8 corr). Adopt if at 480 / 600 px: line wiggle mean >= 0.80, PSNR within 0.5 dB of
plain, M0/M1/M2 within 2 points, Young M0 >= Wiener, 9074 not lower. Then it goes into `alma_unified.py` and the final evaluation.

**Problem 2, real-GI wiggle at low SNR** (frozen model: 9074 0.29-0.33, Speedie SNR 12 0.55; R6 0.43-0.47 / 0.61; oracle linear 9074 0.38, Speedie: `oracle-speedie`).
Fix: R8 = fine-tune with the line-centre loss (`--w-m1`), the first training that optimises what the wiggle is made of. Two starts: R4n seed 3
(`chiragbhayal17/unified-r8-m1-r4n`, running) and R6 (`.../unified-r8-m1-r6`, needs R6 shared with chirag2). Adopt if, on the same cases: Speedie SNR 12 >= 0.65 and
SNR 24 >= 0.85, 9074 >= 0.45, twin spurious <= 5 m/s at SNR 12 and <= 2 at SNR 24, line @600 >= 48 dB with M1/M2 >= 84 %. Oct 17 (quota reset): 3 seeds of the
winner (one seed proves nothing: seed spread 0.41-0.60 at SNR 12), averaged, with the size fix; final evaluation Oct 18. If R8 lifts the wiggle but costs moments:
two checkpoints (moments = R4n average, kinematics = R8 average), allowed by 0. If R8 does not lift it: self-consistent Wiener at inference (signal spectrum
re-estimated from the model output, Wiener rebuilt, network re-run; blind vs oracle Wiener on 9074 is 0.16 vs 0.38), else report the limit against the oracle per SNR.

**Status 2026-10-15 (results back):** size fix FAILS two of its own adoption criteria set this morning (line wiggle @600 0.77 < 0.80; 9074 0.31 -> 0.25, criterion was "not lower"), while it gains elsewhere (Young M0 @600 12 -> 29 %, run_0026 0.66 -> 0.89, PSNR -0.5 dB).
So it is NOT adopted on that evidence; `combo-a` decides it together with the two-pass Wiener (which alone passes: +2-3 dB, +10 points M0, `wiener2-test`). If the combination still lowers 9074, the size fix is
restricted to the cubes it helps (beam > ~15 px, a threshold to be chosen BEFORE looking at the final test) or dropped, and the trade-off is reported. R8 (M1 loss) from R4n: not adopted (SNR 12 0.35); from R6: SNR 12 0.64, invented wiggle at SNR 6 293 -> 60 m/s, 9074 0.38-0.40
(R6 0.43-0.47): small, one seed. Oracle linear on Speedie 0.02 / 0.20 / 0.45 / 0.68 at SNR 3 / 6 / 12 / 24 against the model's 0.00 / 0.13 / 0.55 / 0.83: the 0.85 bar at SNR 12 is a reported limit.
Done: `combo-a` / `combo-b` (both levers, full decision set): combo-b adopted 2026-10-16, size fix restricted to noise-only cubes (PROGRESS.md 2026-10-16). Oct 17 batch (`make_unified_job.py --oct17`): R6 seeds 32 / 33, R9 (beam-matched training, `--train-beam-px 8,12`) seeds 61 / 62, R8 weight bracket 0.003 / 0.03.
**Final test, never used for a decision:** the Speedie r1.5 disc rescaled to SNR 6 / 12 / 24 (`--ladder-base GI_r1.5,Kep_r1.5`); every selection so far used the r0.5 ladder, 9074, the line
holdouts and the Young sims, so the frozen model's reported wiggle-vs-SNR curve comes from r1.5, with the r0.5 numbers labelled as selection data.

## 0c. After GSoC (Krishan, 2026-10-13): consolidate notebooks, datasets and checkpoints in one place

Decided to do this after Nov 3, not now. Plan when it is time: (1) `kaggle/accounts/archive_all.py` (written, untracked until then) pulls code as pushed, outputs and checkpoints of every project kernel on all
four accounts into `archive/` (git-ignored) and writes `results/KERNELS.md`; (2) publish the final checkpoints as one Kaggle Model on the main account so any account can attach them; (3) a public host (Hugging Face or a
GitHub release) for the final checkpoints; (4) datasets stay where they are (sharing works; no ownership transfer known). The mentor data (line emission, sg_synth, SG v2) stays private throughout. Caveat while waiting:
every checkpoint currently lives only in a Kaggle kernel output on one of four accounts, three of them not Krishan's own main account, and RULES.md #12 (keep every checkpoint until GSoC finishes) depends on those accounts surviving to Nov 3.

## 1. Workstreams

**A. Realistic noise and beam (code, no new data). Gate for everything else.**
1. Channel-correlated training noise ([a, 1, a], a ~ U(0, 0.5); a = 0.5 is Hanning) and a velocity noise spectrum in the Wiener, estimated blind.
2. Peak SNR drawn log-uniform 10-150; beam broadened up to 1.6x in the noise-only tasks.
3. Do-no-harm gate (output = Wiener + gated correction). Deferred: the SNR draw attacks the same failure; added only if R3 still falls below Wiener at high SNR.
4. Speedie (r0.5, r1.5, GI and twin) scored automatically after every training run.

**B. New GI data.**
1. SPH-to-cube converter for Phantom dumps (Young et al. 2026, 63 runs; Rowther et al. 2024, 5; Longarini et al. 2026, 7). research.md 2026-10-10.
2. Validate on the Speedie dump against Speedie's own MCFOST cube; stop if the velocity residuals disagree.
3. Render Young across mass ratios, plus inclinations, position angles, distances. Hold out whole simulations.
4. About 30 randomised analytic disks.
5. Speedie is test-only. No exoALMA data is ever trained on.

**C. Training rounds (Kaggle, 2 sessions, weekly quota).**

| round | what | purpose |
|---|---|---|
| R2 `unified-wiener-k15` | window +-1.5 km/s, noise-only task | baseline, K=15 vs K=3 |
| R3 `unified-wiener-real` | A on existing data, K=3 | does the pipeline survive Speedie? |
| R4 | R3 + B | does more GI data fix the wiggle? |
| R5 | 3 seeds of the winner | error bar, ensemble option |

**D. Final.** Select by the bars; figures (wiggle, moment maps) per model per size; archive (RULES.md #10); models/README, RUNS, PROGRESS.

**E. exoALMA.** Inference only, after the checkpoints are frozen. No tuning on it.

## 2. Gates

1. R3: Speedie moments all above dirty. If not, fix the noise model before adding data.
2. R4: on held-out whole simulations (leave-one-mass-ratio-out), wiggle at or above Wiener. If not, ship Wiener + gated correction, and say so.
3. R5 and selection, then E.

## 3. Risks

- The converter's emission model is a proxy for MCFOST; the Speedie validation catches a mismatch.
- Young's discs are young and massive; used as a mass-ratio sweep, not as the target population.
- The channel correlation measured on Speedie comes from its syndisk pipeline; training draws a range, not one value.
- GPU quota: R3/R4 ~3-4 h each; R5's seeds are the real cost.
