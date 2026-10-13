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
| 1 | Seed 11 shared with chirag | blocks the averaged final evaluation | needs the share (user) |
| 2 | High-SNR M0 damage: do-no-harm gate | known flaw, known fix, no training; must be in BEFORE the final evaluation so the frozen numbers describe the shipped pipeline | `eval-gate` running (a * model + (1 - a) * Wiener, a = 0.25 / 0.5 / 0.75, SNR 12-135) |
| 3 | Seed 23 completes | running | resumed after its 11 h budget stop |
| 4 | Final averaged evaluation (seeds 3 + 11 + 23, gate in), 5 sizes, Speedie, ladder, held-out Young sims | this IS the frozen model | after 1-3; `final_report.py` ready |
| 5 | Final figures and results table | needs 4 | `final_report.py` ready |
| 6 | R6 (256 px crops) decision | job running; adopt only if SNR 12-24 wiggle beats R4n-long seeds without hurting line emission | due Oct 15 |
| 7 | Wiggle at SNR 12 (0.41-0.60 vs 0.85) | moves only if 6 wins; else reported as a measured limit with the SNR curve | follows 6 |
| 8 | SG v2 equal to dirty | ceiling-limited: noise-free beam reference 0.92, dirty 0.85 (peak SNR 63); documented, no work | write-up |

Then: re-run ALMA (14 discs + MWC 758) with the frozen model (~5 min per cube).

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
