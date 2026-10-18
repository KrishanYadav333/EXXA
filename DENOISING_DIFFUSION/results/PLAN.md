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

## 0d. Schedule to the end of GSoC (2026-10-16; final submission window Oct 26 - Nov 3, blog draft to Jason Oct 27)

> Merged into the master plan 0f (with the Oct 16 mentor meeting). The six Oct 17 launches stand; all other dates are in 0f.

| dates | phase | work | done when | slack |
|---|---|---|---|---|
| Oct 16 | models | anshika shares; R6 s32 + R9 s61 launched on anshika | both running | - |
| Oct 17 05:30 | models | quota reset: R6 s33, R9 s62, R8 w0.003, R8 w0.03 on main / backup / jensi / chirag; smoke-check R9 in its first round | 6 jobs running | - |
| Oct 18 | models -> freeze | read the six; R6 seeds join B only if they repeat R6's 9074 gain; recipe fixed; ONE run of the untouched final test (Speedie r1.5 ladder 6 / 12 / 24) + full set at 5 sizes; RESULTS.md, RECIPES.md | recipe B (+ seeds) frozen, final numbers written | 1 day (Oct 19) |
| Oct 19 - 21 | ALMA | 14 discs + MWC 758 with the frozen recipe (GPU, ~5 min each); S5 injection on 3 discs; scale and model-agreement checks; compare M1 / residuals with exoALMA I-III published kinematics; Krishan's DS9 look | every cube has report + figures; S5 passes on all 3 | Oct 22 |
| Oct 22 - 24 | write-up | final figures (5-size table, SNR curve with oracle, 9074 maps, ALMA gallery); mentor deck refreshed; blog draft (method, results, limits, ALMA) | draft complete | - |
| Oct 25 - 26 | review | Krishan reads; fix numbers against RUNS.md (every number named with its metric, RULES.md #6); push | draft ready | - |
| Oct 27 | deliverable | blog draft to Jason | sent | 1 week before Nov 3 |
| Oct 27 - Nov 1 | mentor round | Jason's comments; any extra test he asks for (one GPU day kept free for it) | comments addressed | - |
| Nov 1 - 3 | submission | final blog published; work-product link (branch / PR choice is Krishan's; no PR is touched without him); checkpoint hosting if Jason agrees (weights trained on private mentor data: ask first) | submitted | Nov 3 is the hard end |
| after Nov 3 | consolidation | archive_all.py, one place for notebooks and checkpoints (0c) | - | - |

Placement (checked 2026-10-16: main, backup, chirag, chirag2, jensi, aditi all see the 4 datasets + speedie-prep-fast; anshika sees none): anshika = R6 s32 + R9 s61 (cold starts, ~13 h each, 26 of its 30 h: watch the quota);
Oct 17 05:30: chirag = R8 w0.003 + R8 w0.03 (chirag owns R6, so no R6 share is needed), main = R6 s33, backup = R9 s62; chirag2 = hub for the Oct 18 final evaluation and the ALMA rerun (it already sees R4n s3 / s11 / s23 and owns R8-from-R6).
Shares still needed: anshika <- 5 items from main (today); chirag2 <- the new-seed kernels from anshika, main, backup, chirag (when those jobs finish, ~Oct 18 morning). The R6 -> main / backup / jensi and R4n -> jensi shares asked for earlier are NOT needed any more.

Fixed rules for the rest: no new model ideas after Oct 18; the final test is run once; ALMA is inference only; every run archived and logged the day it lands.
Risks: (1) Jason's answer changes the headline wiggle measure: write-up only, both are already computed. (2) A Kaggle account is suspended (terms restrict multiple accounts): 7 accounts, every job resumes from its own output, checkpoints copied locally. (3) An ALMA surprise like the line-free bug: Oct 22 is slack for it. (4) The Oct 17 seeds disagree with R6: B stays as it is, already measured. (5) Local 8 GB machine: nothing heavy runs locally.

## 0e. After the Oct 16 mentor meeting: re-prioritised (Jason; see PROGRESS.md 2026-10-16 | meeting)

Jason's verdict: the models are good, expect no qualitative jump, so the remaining time goes to post-inference analysis. Order of work, replacing the model-side priorities where they conflict with 0d:

| # | work | why (Jason) | done when | where it runs |
|---|---|---|---|---|
| 1 | **Position-velocity (PV) curve** of the moment-1 (quadratic) along the minor axis, width 1 px and 4 to 5 px; clean / dirty / Wiener / model, as a figure row like the spine one, plus a number (correlation and rms against clean) | The field standard; Speedie 2024 and Longarini et al. (Elias 2-27) fit the wiggle with it; "we need to be able to replicate that" | PV figure + table for Speedie ladder, 9074 (noise-only and deconvolution), Young held-out, an07 / an08; Krishan promised it for the next meeting | Add to the Oct 18 evaluation kernel (no local runs: 8 GB); unit test on a synthetic Keplerian + sinusoid cube first |
| 2 | **exoALMA with a cropped field of view**: crop around the disc, run, paste back; keep the size fix (beam about 10 px) | The field of view is huge and the beam tiny; the network sees features relative to cube size | MWC 758 + 14 discs rerun cropped; the 256 px crop run in `results/alma/mwc758-13co_r4n_v2_2026-10-13_crop256` is the starting point | Oct 19 to 21 ALMA slot |
| 3 | **No continuum subtraction on exoALMA** (already subtracted); keep the residual-continuum report. DSHARP is back in scope (0f row 16) | Double subtraction causes problems | `alma_unified.py` line 223 made optional, off for exoALMA; test in `tests/` | with 2 |
| 4 | Models: keep the Oct 17 seed jobs running in the background; Oct 18 freeze stays; no model work beyond them | "I would be surprised if we got a qualitative jump" | frozen recipe B | as planned |
| 5 | Reply to Jason in writing: the size fix resamples the pixel grid, so the angular beam is unchanged (the beam in pixels shrinks); and ask the three questions not covered (second GI cube, one checkpoint or two, publishing) | He asked about beam scaling; Q3 to Q5 were not discussed | email sent | - |
| 6 | SNR: report the wiggle curve for peak SNR 12 and above as the main result; SNR 3 and 6 are shown for completeness only | Real ALMA is above about 15 to 20; SNR 6 is not relevant | write-up wording | write-up |

Schedule effect: items 1 to 3 fit the existing Oct 18 to 21 slots; the PV code is written Oct 17 so the Oct 18 evaluation kernel can produce it. The Oct 22 spare day now also covers PV surprises (slice geometry from the position angle, the minor-axis direction on inclined discs).
Risk: the minor axis needs each disc's position angle and inclination; the scorecard already uses them for the deprojection, check that they are exposed to the new code before building on them.

## 0h. PLAN TO NOV 1 (2026-10-18 02:45 IST; THE plan; 0g's checklist B2 stays as the detailed item list, its dates are replaced by these)

**By Sun Nov 1 (Krishan submits) four things are finished:**
1. **The project**: frozen model, final test, PV curves, exoALMA and DSHARP results, every number traced in RUNS.md.
2. **The final blog**, published (the midterm one is on Medium: "Denoising Astronomical Observations of Protoplanetary Disks").
3. **A final submission branch**, built like `midterm-submission`: cut from upstream ML4SCI/EXXA `main`, one self-contained folder `DENOISING_DIFFUSION/Final Submission/` with structured code, results, README; pushed with Krishan's go; PR only if Krishan decides.
4. **A final email to Jason**: what was done, the blog link, the branch link, checkpoints status, thanks.
Jason is reachable only this week (last meeting Fri Oct 23): he sees the blog draft by Thu Oct 22.

### Sun Oct 18
1. 9074 + 9025 SNR ladder (native, 11, 20, 40, 80; deconvolution + noise-only): build, launch; optional in the same run: the share of the model's 9074 error below about 2 beams (turns Jason's "beam noise" into a measurement).
2. Zoom code (crop + upsample, beam to about 10 px) + MWC 758 test at 256 / 512 px crops.
3. Render 6 to 8 more held-out Young GI discs (CPU).
4. R6 s33: collect, archive, compare the three R6 seeds with recipe B; Krishan decides one checkpoint or two.
5. Freeze recipe B (RECIPES.md).
6. One-time final test: Speedie r1.5, SNR 6 / 12 / 24, run once.
7. 09:00 email to Jason: beam answer, PV preview, plan for the week, confirm Friday, ask permission to publish checkpoints; no data requests; the crop / zoom sentence waits for the zoom test (send without it if it has not landed).
8. Push to midterm-prep (Krishan's go), with PROGRESS.md and context.md updated in the same push (CLAUDE.md), as for every push below.

### Mon Oct 19
1. Read the ladder (noise limit or generalisation gap), the zoom test (choose the ALMA setting), the new Young discs.
2. PV: fix the 9074 detrend or show 9074 without PV; final PV figures (Speedie lead, Young M1 R100, an07, MWC 758).
3. RESULTS.md: five sizes, PV table, final test, ladder, invented wiggle (PV way).
4. DSHARP: geometry (Huang et al. 2018) logged in research.md, kernel for HD 163296, IM Lup, AS 209, HD 143006, HD 142666.
6. models/README.md count fixed (says 68).
5. Blog skeleton + figure list; final-branch folder layout written down (what goes in, what stays out).

### Tue Oct 20 to Wed Oct 21
1. exoALMA rerun (MWC 758 + 14 discs) with the chosen setting; S1 (pass marks revised with reason or reported failed), S2, S3 (B vs A vs R6), S4 (scale +-25 %), S5 (HD 135344B, DM Tau, LkCa 15), S6 + PV with published geometry and published kinematics; S7 only if time.
2. DSHARP run: PV per disc, HD 143006 across both surveys, HD 142666 with its faintness caveat.
3. Planet check (run_0002_00560, Young M1 R100).
4. Blog sections written in parallel. Contents (fixed now): problem and data; model and recipe; PV curves as the lead result; five sizes; wiggle against SNR with the main result from SNR 12 upward (SNR 6 a footnote, Jason) and the oracle linear limit; 9074 as the noise-limited case (SNR 9.5, right wavelength, wrong pattern); do no harm with the invented wiggle measured the PV way (about 20 m/s at SNR 12); exoALMA (MWC 758 written as "a whole different thing", not a GI disc); DSHARP (interpolated channels); the crop / zoom finding; what did not work (R7, R8 weights, R9, Young in training, SG v2 in training, flip averaging); limits.
5. Krishan: DS9 look.

### Thu Oct 22
1. ALMA + DSHARP analysis, figure gallery.
2. Every number checked against RUNS.md, metric named.
3. Deck refreshed (recipe B images, PV figures, corrections).
4. **Blog draft + figures to Jason** (Krishan sends).
5. Push.

### Fri Oct 23
1. Last meeting: walk the draft; checkpoints (one or two, publish or not); anything he wants added.
2. One GPU day kept for his request.

### Sat Oct 24 to Tue Oct 27: the final branch
1. Branch `final-submission` (name to confirm) from upstream ML4SCI/EXXA `main`, like `midterm-submission`.
2. Folder `DENOISING_DIFFUSION/Final Submission/`:
   - `README.md`: what the project does, the main results (five sizes, PV, exoALMA, DSHARP), how to run, the recipe, links to the blog and the midterm submission.
   - `src/` (library: data, models, evaluation) and `experiments/` (only the final pipeline: `train_unified.py`, `alma_unified.py`, `pv_curve.py`, `wiggle_scorecard.py`, `compare_models.py`, `speedie_prep.py`, `make_sph_cubes.py`, `make_analytic_gi_cubes.py`, `oracle_limit.py`).
   - `kaggle/`: the kernels that produced the final numbers (training, final evaluation, ALMA, DSHARP) and the account / resume tooling, with no keys.
   - `tests/`: the tests of the final pipeline, all passing in a clean checkout.
   - `results/`: RESULTS.md, RECIPES.md, the final tables and figures, exoALMA and DSHARP reports, a short RUNS index of the runs behind the final numbers, the negative results.
   - `MODELS.md`: every checkpoint, what it is, where it lives (weights not in git; published only with Jason's permission).
   - `requirements.txt`, `.gitignore`.
   - Not in the branch: mentor data (private), exoALMA / DSHARP cubes, checkpoints, Kaggle keys, notebooks of the earlier eras (they are in the midterm submission).
3. Fold in Jason's comments on the blog.
4. Krishan confirms what the GSoC submission form needs (by Mon Oct 26, so there is time to react).

### Wed Oct 28 to Thu Oct 29
1. Clean-checkout check of the branch: install from `requirements.txt`, every test passes, the README commands run.
2. Scan the branch for anything that must not be public: Kaggle keys (`kaggle.json`, tokens), mentor data (line emission, sg_synth, SG v2 cubes), exoALMA / DSHARP cubes, checkpoints, personal paths; nothing larger than a figure.
2. Every number in the blog, README and RESULTS.md checked a second time against RUNS.md.
3. Blog finalised (figures, links to the branch).
4. RUNS.md docs debt (notebook 05 v23 to v29) and Kaggle versions (Krishan supplies).
5. models/README.md and context.md up to date.

### Fri Oct 30
1. Push the final branch (Krishan's go); PR only if Krishan decides.
2. Final push of midterm-prep with all docs.
3. Checkpoints published only with Jason's permission.

### Sat Oct 31
1. Buffer, and a final read of everything (blog, README, RESULTS.md, the branch).
2. Publish the blog.
2. Final email to Jason drafted: summary, blog link, branch link, checkpoints status, thanks.

### Sun Nov 1
1. Krishan submits (blog link + branch link).
2. Krishan sends the final email to Jason.

### Decisions still open (Krishan)
1. Last meeting date (assumed Fri Oct 23; confirm in the email). 2. Branch name (`final-submission`?). 3. Blog platform (Medium, like the midterm?). 4. One checkpoint or two (Oct 18). 5. Push go (today). 6. PR to ML4SCI/EXXA or not (Oct 30).

### Every day
No new model ideas; no GIGGLE / discminer fitting; no own continuum subtraction (auto check only); never train on exoALMA / DSHARP; the final test runs once; every job saves before 12 h; every run logged and archived the same day; pushes only on Krishan's word, no co-author trailer; no data requests to Jason.

### After Nov 1
Consolidation (0c), `archive_all.py`; remove the three stray `~/.kaggle-*` folders if not done.

## 0g. FINAL PLAN (2026-10-18 01:30 IST; supersedes the dates in 0f): Jason reachable only this week, submission by Krishan on Nov 1, work day and night

Constraints (Krishan, Oct 18): the week of Oct 19 to 23 is the last time to talk to Jason (meeting Fri Oct 23, to confirm in the email); no further requests for data from him; Krishan submits on **Sun Nov 1**. So everything Jason should see, including a blog draft, reaches him by **Thu Oct 22**, and his comments are folded in Oct 24 to 31. Kaggle quota resets Sat Oct 24 05:30 IST.

### A. Every open item, with where it is done

| # | item | source | status | done in |
|---|---|---|---|---|
| J1 | 9074 "looks like beam noise": treat as the noise-limited case, not the headline; test whether it recovers at ALMA-like SNR | Jason | numbers in hand (PROGRESS 2026-10-17): right wavelength, wrong pattern, beam-only oracle 0.92 so noise, not resolution | 9074 SNR ladder + 9025 diagnostic, Oct 18 |
| J2 | beam keeps its angular size when rescaled | Jason | true in the code | email Oct 18 |
| J3 | "zoom in" on exoALMA = crop AND enlarge (MWC 758 beam 6 px, below the 8 to 20 px trained); crop-only lost flux (Oct 17), crop + upsample untested | Jason | code to write | zoom test Oct 18, decision Oct 19 |
| J4 | PV curve along the minor axis (quadratic M1, slice or 4 to 5 px slit) as the lead measurement; spine kept as second | Jason | built, run on recipe B (eval-final-b): strong on GI discs | final PV figures Oct 19, preview to Jason Oct 18 / 22 |
| J5 | not to do: GIGGLE fit, discminer fit, own continuum subtraction, new model ideas, SNR 6 | Jason | respected | - |
| J6 | his reads: position angle right, Young / Speedie good, success; MWC 758 "a whole different thing"; planet-like features | Jason | planets not checked | planet check Oct 20; write-up framing |
| P1 | freeze recipe B; the ONE-TIME final test (Speedie r1.5 ladder, SNR 6 / 12 / 24) | plan | recipe B unchanged by every new seed | Oct 18 |
| P2 | five-size table of recipe B | plan | DONE (eval-final-b: 51.2 to 51.8 dB, 62 / 89 / 91 at every size) | RESULTS.md Oct 19 |
| P3 | more held-out GI discs without asking Jason: 6 to 8 more Young et al. simulations | Krishan / plan | to render (CPU) | render Oct 18, evaluate Oct 19 |
| P4 | exoALMA rerun, 14 discs + MWC 758, PV per disc, S1 / S2 / S5 (3 discs) / S6, comparison with published kinematics, DS9 look | plan | void until rerun; config waits for J3 | Oct 20 to 21 |
| P5 | DSHARP: 12CO fiducial images, `--spec-interp 0.1` (native 0.35 km/s fails), bright discs (HD 163296, IM Lup, AS 209, HD 143006) + HD 142666 (Terry 2023 kink), geometry from Huang et al. 2018 | Krishan | spectral test passed | kernel Oct 19, run Oct 20 to 21 |
| P6 | one checkpoint or two (R6 alternative): R6 seeds 31 / 32 / 33 vs B on 9074 | plan | s33 resuming | decide Oct 18 when s33 lands |
| P7 | blog: structure, figures, all numbers checked against RUNS.md (RULES.md 6) | plan | not started | skeleton Oct 19, draft Oct 20 to 22, to Jason Oct 22 |
| P8 | mentor deck refreshed with recipe B and PV figures | plan | 22 slides exist | Oct 22 |
| P9 | records: PROGRESS / RUNS / research / context same day; every run archived; checkpoints kept; no co-author trailer | RULES.md | up to date | daily |
| P10 | push to GitHub (midterm-prep only, no PR), with docs | CLAUDE.md | last push Oct 16; much unpushed | when Krishan says; at least before Oct 22 |
| P11 | email to Jason: beam answer, PV preview, what we are doing, confirm Friday, ask permission to publish checkpoints (private data) | plan | draft ready; data request (question 3) dropped | Oct 18 09:00 |
| P12 | risks: Kaggle terms (seven accounts, Jason: "hopefully they don't catch you"), an ALMA bug, write-up time | plan | - | every job resumable, checkpoints local |

### B. Day by day (IST; GPU accounts in brackets)

| when | work |
|---|---|
| **Sun Oct 18, 01:30 to 09:00** | build and launch: (1) hydro ladder kernel, 9074 and 9025 at SNR native / 11 / 20 / 40 / 80, deconvolution and noise-only, recipe B (jensi); (2) zoom code (`--zoom-beam-px`: upsample so the beam spans about 10 px, inside a crop, then back) + MWC 758 test at 256 / 512 px crops (aditi); (3) render 6 to 8 more held-out Young simulations (CPU, backup); collect R6 s33 when its resume lands, archive, decide P6 |
| **Sun Oct 18, 09:00** | email to Jason (P11), with 2 to 3 PV figures; the crop sentence waits for the zoom test if it has not landed |
| **Sun Oct 18, day** | freeze recipe B formally (RECIPES.md); final test Speedie r1.5 once (chirag2); read ladder + zoom results; log everything |
| **Mon Oct 19** | RESULTS.md (five-size, PV table, final test, ladder); decide the ALMA configuration from the zoom test; DSHARP kernel with Huang et al. geometry; evaluate the extra Young discs (GPU); blog skeleton and figure list |
| **Tue Oct 20 to Wed Oct 21** | ALMA rerun: exoALMA 14 discs + MWC 758 (chirag2 + one more account), DSHARP 5 discs (another account); PV per disc; S5 on 3 discs; published-kinematics comparison; planet check (run_0002_00560, Young M1 R100); blog sections written in parallel (data, model, five-size, PV) |
| **Thu Oct 22** | ALMA analysis and gallery; final PV figures; deck refreshed; **send Jason the blog draft + figures** so he can comment before Friday; push (with Krishan's go) |
| **Fri Oct 23** | last meeting: walk the draft; checkpoints (one or two, publish or not); any last analysis he asks for (one GPU day kept free) |
| **Sat Oct 24 to Thu Oct 29** | fold in Jason's comments; final figures; every number checked against RUNS.md; README for the work product; final push; checkpoint hosting only if he agreed |
| **Fri Oct 30 to Sat Oct 31** | buffer + final read |
| **Sun Nov 1** | Krishan submits |

### B2. Full checklist to submission (2026-10-18 02:10; every source cross-checked: plan 0b / 0b2 / 0d / 0f, the Oct 16 meeting, ALMA_PLAN.md S1 to S7 and its thresholds, RUNS.md gaps, CLAUDE.md rules)

Status: [x] done, [~] in progress, [ ] to do. Owner K = Krishan (website / decisions), C = Claude.

**Models and the frozen result**
- [x] M1 five-size evaluation of recipe B (eval-final-b, Oct 17).
- [~] M2 R6 s33: collect when the resume lands, archive, compare with R6 s31 / s32 (C, Oct 18).
- [ ] M3 decide one checkpoint or two (B alone, or B + R6 as a kinematics checkpoint) from the three R6 seeds; rule: R6 joins only if its 9074 gain repeats (0.43) (C proposes, K decides, Oct 18).
- [ ] M4 freeze recipe B in RECIPES.md (C, Oct 18).
- [ ] M5 the ONE-TIME final test exactly as pre-registered: Speedie r1.5 GI and twin, SNR 6 / 12 / 24 (`--ladder-base GI_r1.5,Kep_r1.5`), run once, never re-run (C, Oct 18).
- [ ] M6 RESULTS.md: five-size table, PV table, final test, ladder, noise-limited 9074, invented-wiggle numbers (PV way: about 20 m/s at SNR 12), all with metric named (C, Oct 19).

**Jason's points**
- [ ] J1 9074 / 9025 SNR ladder (native, 11, 20, 40, 80; deconvolution + noise-only) and the reading "noise limit vs generalisation gap" (C, run Oct 18, read Oct 19).
- [ ] J1b optional: beam-scale share of the model error on 9074 (makes "beam noise" a measurement) (C, only if it fits in the ladder run).
- [ ] J1c PV on 9074: a detrend that the large step does not dominate, or show 9074 without PV (C, Oct 19).
- [ ] J2 tell Jason the beam keeps its angular size (email, K sends, Oct 18 09:00).
- [ ] J3 zoom = crop + upsample (beam to about 10 px), MWC 758 at 256 / 512 px crops; choose the ALMA configuration from it (C, test Oct 18, decide Oct 19).
- [ ] J4 final PV figures for the blog: Speedie GI (lead), Young M1 R100, an07, MWC 758 injected; slice 1 px and slit 5 px (C, Oct 19).
- [ ] J6 planet check: run_0002_00560 (the "2560" he named) and Young M1 R100 M1 maps around 7 to 8 o'clock (C, Oct 20).
- [ ] J7 MWC 758 written as "a whole different thing", not a GI disc (C, blog).

**More test discs (no request to Jason)**
- [ ] T1 render 6 to 8 more held-out Young et al. simulations spanning disc and star mass (CPU) (C, Oct 18).
- [ ] T2 evaluate them with recipe B, PV included (GPU) (C, Oct 19).

**exoALMA (ALMA_PLAN.md S1 to S7)**
- [ ] A1 rerun MWC 758 + 14 discs with the frozen recipe and the configuration from J3; continuum `auto` (C, Oct 20 to 21).
- [ ] A2 S1 noise / flux / M1 shift, against ALMA_PLAN.md's provisional pass marks (median M1 shift <= 20 m/s, p95 <= 50, M0 ratio within 5 %); MWC 758 currently shows 54 to 78 m/s median: say whether the marks are revised (with reason) or failed (C, Oct 21).
- [ ] A3 S2 invented structure on line-free planes, per disc (C, Oct 21).
- [ ] A4 S3 model agreement: recipe B vs recipe A vs R6 on MWC 758 and 3 discs (C, Oct 21).
- [ ] A5 S4 pixel-scale robustness: `--scale 0.8` and `1.25` on MWC 758 and 2 discs (C, Oct 21).
- [ ] A6 S5 injection on HD 135344B, DM Tau, LkCa 15 (C, Oct 20 to 21).
- [ ] A7 S6 rotation-curve residual + PV per disc with the published inclination / PA; compare with published exoALMA kinematics (C, Oct 21).
- [ ] A8 S7 (optional): compare with the exoALMA 0.3 arcsec set, only if time.
- [ ] A9 DS9 look at raw vs denoised for MWC 758 and two discs (K, Oct 21 to 22).

**DSHARP**
- [x] D1 spectral test: native 0.35 km/s fails, `--spec-interp 0.1` passes (Oct 17).
- [ ] D2 geometry (inclination, PA, distance) from Huang et al. 2018 (arXiv:1812.04041) and research.md entry (C, Oct 19).
- [ ] D3 kernel for HD 163296, IM Lup, AS 209, HD 143006 (also in exoALMA), HD 142666 (Terry 2023); blind SNR printed per disc; S5 at SNR 6 / 8 / 12 / 24 (C, Oct 19).
- [ ] D4 run and read, PV per disc, HD 143006 compared across the two surveys, HD 142666 with its faintness caveat (C, Oct 20 to 21).

**Write-up and deliverables**
- [ ] W1 blog skeleton + figure list (C, Oct 19).
- [ ] W2 blog sections: problem and data, model and recipe, PV curves (lead), five sizes, wiggle against SNR from 12 up with oracle, 9074 as noise-limited, do no harm (PV-way invented wiggle), exoALMA, DSHARP, what did not work, limits (C drafts, K edits, Oct 20 to 22).
- [ ] W3 every number checked against RUNS.md with its metric named (RULES.md 6) (C, Oct 22 and again Oct 29).
- [ ] W4 blog draft + figures to Jason (K sends, Thu Oct 22).
- [ ] W5 deck refreshed: recipe B images, PV figures, corrections (9074 SNR 9.5 not 11; "one held-out GI disc" wording; crop finding) (C, Oct 22).
- [ ] W6 last meeting Fri Oct 23 (K): checkpoints, publishing permission, anything to add.
- [ ] W7 fold in Jason's comments, final figures, final blog published (C + K, Oct 24 to 29).
- [ ] W8 work-product README (what is where, how to run, recipe, data and its privacy) (C, Oct 27).
- [ ] W9 confirm what the GSoC submission needs (final evaluation form, work-product link) (K, by Oct 26).
- [ ] W10 checkpoints published only with Jason's permission (private mentor data) (K, after Oct 23).
- [ ] W11 submit (K, Sun Nov 1).

**Records and housekeeping**
- [ ] R1 email to Jason (K sends, Oct 18 09:00): beam answer, what we are doing, PV preview figures, confirm Friday, ask permission to publish checkpoints. No data requests.
- [ ] R2 push to GitHub, midterm-prep only, no co-author trailer, docs in the same commit (needs K's go; at the latest Oct 22, again at the end).
- [ ] R3 context.md section 0 brought up to date (dated 2026-10-15 now) (C, with the next push).
- [ ] R4 models/README.md count (says 68; actual 85+ files) (C, Oct 19).
- [ ] R5 RUNS.md: notebook 05 v23 to v29 docs debt; Kaggle version numbers marked "author to confirm" (C lists, K supplies versions, by Oct 29).
- [ ] R6 every run archived and logged the day it lands; every checkpoint kept (RULES.md 10 to 12) (C, daily).
- [ ] R7 remove the three stray empty `~/.kaggle-*` folders (K, any time).
- [ ] R8 after Nov 1: consolidation (0c), `archive_all.py`.

### C. Rules that do not change

No new model ideas (Jason: no qualitative jump expected); the Speedie r1.5 test runs once; ALMA and DSHARP inference only, never trained on; no GIGGLE / discminer fitting; no own continuum subtraction (auto check only); every job stops and saves before 12 h; every run archived and logged the same day; push only on Krishan's word, no co-author trailer.

### D. Measured on Oct 18 (correction to the deck)

Native peak SNR after the beam, measured as peak(beam (*) clean) / rms(dirty - beam (*) clean): 9074 **9.5** (the deck and the Oct 16 notes said "about 11", from the blind estimate), 9025 7.7, 9015 7.6, 9019 9.3, 9032 33.7, an07 19.7. 9074 is therefore below the SNR 15 to 20 Jason expects of real ALMA data, which supports reading it as noise-limited.

## 0f. MASTER PLAN to submission (2026-10-17: 0d, the Oct 16 mentor meeting (0e) and the GSoC window merged; this is the one to follow)

Dates IST. Submission window Oct 26 to Nov 3 (hard end Tue Nov 3). Mentor meeting assumed Fridays: Oct 23 (confirm with Jason). Kaggle quota resets Saturdays 05:30: Oct 17, Oct 24, Oct 31.
0d is the earlier plan (models, freeze, ALMA, write-up); 0e is Jason's meeting list; everything in both is in the table below. Decision 2026-10-17: all six Oct 17 jobs launch (Krishan's earlier choice stands); the three free accounts (chirag2, jensi, aditi) carry evaluation and the ALMA rerun.

### A. Requirement checklist (nothing is dropped; each row says where it is done)

| # | requirement | source | done in |
|---|---|---|---|
| 1 | Final models best at PSNR and M0/M1/M2, wiggle above dirty and Wiener, five sizes (256 / 300 / 320 / 480 / 600), exoALMA-like noise | project goal (0) | Oct 18 evaluation |
| 2 | PV curve along the minor axis (slice 1 px and slit 4 to 5 px); good-looking curves for clean / dirty / Wiener / model; no wiggle fit, no disc fit | Jason Oct 16 | code Oct 17, figures Oct 18 to 19, preview Oct 22, meeting Oct 23 |
| 3 | exoALMA with a cropped field of view | Jason | TESTED Oct 17 on MWC 758: 128 / 256 px crops lose flux (0.96 / 0.98 vs 1.02) and add false wiggle; the 512 px (12.8") field is kept for the Oct 20 to 21 rerun (PROGRESS.md 2026-10-17) |
| 4 | No second continuum subtraction on exoALMA | Jason | DONE Oct 17: `--cont-sub auto` subtracts only real continuum (both far sides); MWC 758: none found, no effect either way |
| 5 | Size fix keeps the angular beam (pixel grid is resampled only); state it to Jason in writing | Jason | email Oct 17 |
| 6 | Models: no new ideas; seed jobs run in the background | Jason + 0d | Oct 17 launches, Oct 18 freeze |
| 7 | SNR: main result from SNR 12 upward; SNR 3 and 6 for completeness; rescaling exoALMA SNR optional | Jason | write-up wording |
| 8 | Final untouched test: Speedie r1.5 ladder (SNR 6 / 12 / 24), run once | 0d | Oct 18 |
| 9 | exoALMA: MWC 758 + 14 discs, S1 / S2 / S5 / S6 checks; fix the S6 disc mask; S5 injection on 3 discs; compare with published exoALMA I to III kinematics; Krishan's DS9 look; all 14-disc numbers void until this lands | 0d | Oct 20 to 21 |
| 10 | Answer his remarks: "9074 model rows look like beam noise" (use the PV figure and the spine to answer), "planet-like" run_0002 and Young M1 R100 (look at their M1 residual, say what we see) | Jason | Oct 19 + Oct 22 preview |
| 11 | Open questions to Jason: second held-out GI cube; one or two checkpoints (and which of the two models, R6 alternative or B); publishing the checkpoints (private data); PV slit width he prefers; confirm Friday meeting | deck Q3 to Q5 | email Oct 17, meeting Oct 23 |
| 12 | Write-up: blog (leads with PV curves), mentor deck refreshed, every number checked against RUNS.md with its metric named (RULES.md 6) | 0d | Oct 24 to 26 |
| 13 | Deliverables: blog draft Oct 27; submission Nov 1 to 3 (final blog published, work-product link, any portal form: check its requirements by Oct 26) | GSoC | Oct 27, Nov 1 to 3 |
| 14 | Records: every run archived with its notebook and Kaggle version (Krishan supplies the version), RUNS.md, PROGRESS.md, context.md in the same push; every checkpoint kept, indexed in `models/README.md`; research.md for the PV-method reading; no Co-Authored-By; push midterm-prep only, no PRs | RULES.md | each day |
| 16 | **DSHARP CO cubes through the same pipeline** (Krishan, Oct 17; Jason in September: "absolutely, as well", after exoALMA; his Oct 16 note that DSHARP continuum is not subtracted by default is the case `continuum_check` handles). 12CO J=2-1 fiducial images, 20 discs, 0.35 km/s channels, 10 mas pixel, beam about 7.7 x 6.1 px | ALMA_PLAN.md, Jason | DONE Oct 17: spectral test on HD 142666: native 0.35 km/s fails, `--spec-interp 0.1` passes (S5 12 / 8 / 6 m/s, wiggle 0.82 to 0.96). Left: kernel for the chosen discs (bright ones: HD 163296, IM Lup, AS 209, HD 143006; HD 142666 as the scientific case, faint at blind SNR 6.6), geometry from Huang et al. 2018 (Oct 19), run Oct 20 to 21 |
| 15 | Housekeeping: Kaggle shares of the new-seed kernels to chiragbhayal17; remove the three stray empty `~/.kaggle-*` folders (Krishan); deck images from the Oct 18 run (recipe B, not the previous model); after Nov 3 `archive_all.py` consolidation (0c) | misc | below |

### B. Day by day

| day | work | done when |
|---|---|---|
| **Sat Oct 17, 05:30** | Launch all six: chirag = R8 w0.003 + w0.03; main = R6 s33; backup = R9 s62 (orchestrator `submit` from `scratchpad/oct17`, or `make_unified_job.py --oct17`). Smoke-check R9 in its first round. anshika's R6 s32 and R9 s61 finish about now (started Oct 16 15:40, about 13 h). | six jobs running (eight with anshika's two) |
| Sat Oct 17, day | **Collect anshika's two** when they finish: download with the Content-Length and zip check, archive per RULES.md 10, checkpoints into `models/` and `models/README.md`, PROGRESS entry. **Shares** (website only, Krishan): anshika's R6 s32 / R9 s61 and, as they finish, the Oct 17 kernels to chiragbhayal17. | outputs archived, shares done |
| Sat Oct 17, day | **Code:** (a) `alma_unified.py`: continuum subtraction off by default (flag to turn on, residual reported), `--crop-fov` around the disc with paste-back, per-disc `--incl --pa` from the exoALMA tables, fix the S6 disc mask; (b) tests in `tests/`; (c) re-sync the code overlay to every account (`sync_account.py`, includes `pv_curve.py`); (d) `compare_models.py` reads the PV keys when present. **Read** the PV sections of Speedie 2024 and Longarini (Elias 2-27), log in `research.md`, and check slit width and the minor-axis definition against `pv_curve.py` (done and tested already). | tests pass, overlay re-synced, research.md entry |
| Sat Oct 17, evening | **Email Jason:** size fix keeps the angular beam; PV curves coming by Oct 22; three open questions + slit width + Friday. | sent |
| **Sun Oct 18** | **Freeze and final evaluation** (chirag2, jensi, aditi in parallel). Read all finished jobs. Rules: R6 seeds join B only if they repeat R6's 9074 gain (0.43 per pixel); R9 is adopted only if it shows a clear gain on 9074 or the Speedie ladder, else documented; decide checkpoint set (B, plus R6 as documented alternative). One run: untouched final test (Speedie r1.5, once), recipe B at all five sizes, PV figures (Speedie ladder GI and twin, 9074 noise-only and deconvolution, Young held-out, an07 / an08 and `_ai`), alternative checkpoints beside B. | recipe frozen in `RECIPES.md`, evaluation submitted |
| **Mon Oct 19** | Read the evaluation; `RESULTS.md`, `RECIPES.md`, `RUNS.md` (Kaggle versions), archive, PROGRESS. Look at every PV figure; fix slice problems. Replace the deck's previous-model images with recipe B images. Prepare the ALMA kernel (checkpoints, crop settings). Slack day. | numbers and PV figures final |
| **Mon Oct 19 (also)** | **DSHARP prep:** `kernel_alma_dsharp.py` (NRAO bulk server, `images/<disc>_CO.fits`), geometry (inclination, position angle, distance) per disc from Huang et al. 2018 (arXiv:1812.04041), choice of discs (HD 142666 first: the Terry et al. 2023 planet kink; HD 143006, also in exoALMA; MY Lup, GW Lup, WaOph 6, AS 209), a spectral test: the model at 0.35 km/s as it is vs the cube interpolated to 0.1 km/s, judged by S5 injection binned to 0.35 km/s | kernel ready, spectral test decided |
| **Tue Oct 20 to Wed Oct 21** | **ALMA rerun** (exoALMA; DSHARP on a second account): MWC 758 and 14 discs, cropped, no continuum subtraction, size fix inside the crop; PV curve per disc with published geometry; S1, S2, S5 (3 discs), S6; scale and model-agreement checks; compare with published exoALMA kinematics; Krishan's DS9 look. | every cube has report, figures, PV |
| **Thu Oct 22** | Spare day (ALMA surprises, PV fixes). **Preview to Jason:** one PV figure (Speedie GI, 9074) + cropped MWC 758. Deck refreshed. | preview sent |
| **Fri Oct 23** | Mentor meeting: PV curves, cropped exoALMA, frozen numbers, open questions. Log his answers in PROGRESS.md and context.md. Quota resets Sat 05:30: a fresh 30 h per account for any test he asks for. | feedback logged |
| **Sat Oct 24 to Mon Oct 26** | **Write-up.** Blog: (1) problem and data, (2) model and recipe, (3) PV curves clean / dirty / Wiener / model (lead figure), (4) moments and PSNR at five sizes, (5) wiggle against SNR from 12 upward with oracle, 9074, limits, (6) do no harm, (7) exoALMA cropped, (8) what did not work. Numbers checked against RUNS.md; deck refreshed; Krishan reads Oct 25 to 26; push with docs. Check portal requirements. | draft complete, pushed |
| **Tue Oct 27** | Blog draft to Jason. | sent |
| **Wed Oct 28 to Sat Oct 31** | Mentor round; one GPU day kept free (quota resets Oct 31 05:30). | comments addressed |
| **Sun Nov 1 to Tue Nov 3** | Publish the final blog; submit work-product link and any portal form; release checkpoints only if Jason agrees; final push (midterm-prep only); all checkpoints kept and indexed. | submitted by Nov 3 |
| after Nov 3 | Consolidation (0c): `archive_all.py`, one place for notebooks, datasets and checkpoints. | - |

### C. Accounts

anshika: R6 s32 + R9 s61 (running). Oct 17: chirag = R8 x2; main = R6 s33; backup = R9 s62. Free for evaluation and ALMA: chirag2 (hub; sees R4n s3 / s11 / s23 and owns R8-from-R6), jensi, aditi. Every job saves before 12 h and resumes from its own output (RULES.md 14).

### D. Gates and rules

Freeze rules as in row 6 and 8 above. No new model ideas after Oct 18; the final test runs once; ALMA is inference only, never trained on exoALMA; Speedie is test only; mentor data stays private; local runs impossible (8 GB), Kaggle only; no wiggle-fit and no disc-fit routine; DSHARP out of scope.

### E. Risks

(1) PV on noisy cubes looks bad on the dirty row (that is the point; show the 5 px slit as well). (2) Fitted geometry on ALMA discs is poor: use published inclination and position angle. (3) An ALMA bug like the line-free finder: Oct 19 and Oct 22 are slack. (4) Kaggle quota or a suspended account: seven accounts, resumable jobs, checkpoints copied locally. (5) Jason's answers change the headline: write-up only. (6) The Oct 18 evaluation reveals a recipe problem: one GPU day is kept free after Oct 23 and again after Oct 31.

### F. Done means

Frozen recipe with final numbers at five sizes; PV figures for the test discs and for MWC 758 and the 14 exoALMA discs; the untouched final test reported once; blog draft Oct 27; submission by Nov 3; every number traceable in RUNS.md; every checkpoint kept.

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
