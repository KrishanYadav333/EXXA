# Plan to Nov 3

**Rewritten 2026-09-25 around what Jason actually asked for.** The earlier version (2026-09-10) was built on
our own reading of what mattered, and by 2026-09-24 Block 2 had become a simulated degradation sweep he never
requested. This version starts from his words in the 2026-09-12 meeting, quoted below, and from the ML4SCI org
deadlines (email of 2026-09-23). Anything neither asked for is listed under "optional" and gets time only after
the rest is delivered.

Update this file when a result changes what comes next, not on a schedule. `results/PROGRESS.md` still gets every
run; this file gets the decisions.

---

## What Jason asked for (2026-09-12 meeting, his words)

1. *"My first suggestion would just be to see if mean absolute error helps. That would be really nice if such a
   simple thing helped."* On the current 256px.
2. *"Then try just upsampling or well, less downsampling to something like 480 or something like that."* Because
   600 is expensive: *"those 600, those are so big. You could maybe try to compromise and do like a 320, 480."*
3. Other losses if appropriate: *"like wavelet, or starlet, or maybe even like a negative log likelihood"*;
   *"whatever losses that you have read about or heard about that you think would be appropriate here."*
4. Which models: *"focus on your best ones right now ... your two or three best models and change the loss
   function and the pixel sizes, and just see if anything changes."* Other architectures only *"if you feel like
   you want to explore more"*.
5. Real ALMA data: *"if you want to take the time to do inference on actual ALMA data, this is probably the data
   that we're going to start with"*, the exoALMA release (Harvard Dataverse, doi:10.7910/DVN/CFHWNH).
   - *"Do the Fiduciary line images"*; *"don't worry about the PSF or the mask. Just look at the image. So the dot
     image dot fits, those are what you want."*
   - *"13CO is the one you want ... 13CO is kind of like the canonical one, but 12CO is fine. CS is hard because
     it's really dim."*
   - Read the first exoALMA paper (arXiv:2504.18688) *"so you can just sort of get an idea as to which ones are
     interesting"*. *"MWC 758 ... that's one that people like a lot because it's really weird."*
   - DSHARP: *"D-sharp, absolutely, as well"*, but exoALMA first.
6. VLT: *"If we don't get to the VLT data, you know, that's not a big deal at all."*
7. Viewing: DS9.
8. What he would be happiest with: *"if we get a really good ALMA pipeline down and show it can be used and
   actually like give you scientific insights, that's the thing that I would be most happy with."*
9. Pace: *"I don't expect you to get all that done by next week. But I think that's just a good plan for the next
   few weeks."*

## ML4SCI org requirements (email, 2026-09-23)

- **Blog post shared with the mentors by Fri Sep 25, 17:00 US Central (Sat Sep 26, 03:30 IST).** Mentors are told
  not to pass the evaluation until they have checked it. For a long project the org allows a progress post now
  and an update at wrap-up.
- **Code on the ML4SCI GitHub** (`ML4SCI/EXXA`, own folder), PR submitted before the org meeting. As of
  2026-09-24 nothing from this summer is upstream.
- **Lightning talk, 3 minutes, Tue Sep 29, 10:30 US Central (21:00 IST).** Attendance required, strict on time.
- **Final submission: Nov 3.**

---

## Where each ask stands (2026-09-25)

| # | Ask | Status |
|---|---|---|
| 1 | MAE | **Done.** `winner_mae_ft` PSNR 39.78, M0 **+26.3%**, below the `sweep_winner_aug` baseline (+29.2%). MAE alone did not help M0; M1/M2 rose. |
| 3 | wavelet / starlet | **Done.** `winner_starlet_ft` 40.32 dB, M0 +42.6 / M1 +80.5 / M2 +81.1; `winner_wavelet_ft` 40.13, +38.8 / +76.1 / +70.0. One seed, spread across the 5 holdout cubes (v30). |
| 3 | pix2pix-style gradient | Trained (`winner_gradient_ft` 40.14), moments not yet in the v30 table. |
| 3 | NLL | Not tried. |
| 2 | 480 / 320 | **Trained**, winner_aug's recipe from scratch (seed 43): `winner_aug_res480` 40.18, `res320` 37.36. **Moments not scored**, and PSNR does not compare across resolutions, so ask 2 has no answer yet. |
| 4 | best 2-3 models | aug done; the p10 and beam loss arms (8) remain in notebook 05, ~4 Kaggle sessions. |
| 5 | exoALMA inference | Notebook 15 (`15-alma-real-cube-inference.ipynb`, branch `alma-validation`) built, smoke-tested only. |

Done but not asked for: notebook 13 (loss sweep on `kin_gamma0`, `sg_k3_fresh`, `ddpm_seed42`, `ddrm_prior`, 28/28
arms). Built but not asked for: notebook 14 (native 600px, 56 arms), `tools/alma_simobserve.py`.

---

## Timeline

Friday meetings: Sep 25, Oct 2, Oct 9, Oct 16, Oct 23, Oct 30.

### Sep 25 to 29: org deadlines first

- **Sep 25, before 03:30 IST Sep 26:** progress blog post to Jason. Content: the loss sweep (asks 1 and 3, with
  the MAE result stated as it is), resolution arms trained, the SG / wiggle findings, ALMA started.
- **Sep 25 meeting:** report asks 1 and 3, and that ask 2 is trained but not yet scored.
- **By Sep 29:** PR to `ML4SCI/EXXA`; 3-minute talk rehearsed.
- **Sep 29, 21:00 IST:** lightning talk.
- **Kaggle, in the background:** notebook 05's remaining arms; notebook 15 on MWC 758 13CO.

### Oct 1 to 7: answer asks 2 and 4

Notebook 16 (`16-checkpoint-evaluation.ipynb`) does the scoring: every checkpoint, the same checks, the same cubes.

- Score moments for `winner_aug_res480` and `res320` on the 5 holdout cubes, same metric as every other row.
  That is the real answer to ask 2.
- Moments for `winner_gradient_ft`, and for the p10 / beam loss arms as they finish.
- **Pick the final 2-3 models on moments** (M0/M1/M2 and the wiggle), not on PSNR. Current candidates:
  `winner_starlet_ft`, `winner_wavelet_ft`, and `winner_aug_res480` if its moments hold up.
- NLL loss only if a slot is free; it is optional in his words.

### Oct 1 to 20: real ALMA data (asks 5, 7, 8)

The full design (data, domain gap, pipeline, how we will know it worked, gates, risks) is in `ALMA_PLAN.md`.

- **Oct 1-7:** MWC 758 fiducial **13CO**, then 12CO, `.image.fits` only, run with the chosen 2-3 models. Open the
  raw and denoised cubes in DS9.
- **Oct 8-14:** 2-3 more exoALMA disks chosen from the exoALMA I paper, then DSHARP.
- **Oct 15-20:** the scientific reading: what the denoiser does to the kinematics of real disks. MWC 758 has
  known spirals and kinks in the exoALMA papers, so compare against them. This is the "scientific insights" he
  asked for, and the headline of the final blog.

### Oct 23 to 30: final blog and repo

- Write the final blog update (org: update the progress post at wrap-up). Structure: loss and resolution results,
  then real ALMA results, then the negative results told plainly.
- Clean the upstream PR so the notebooks run; archive per RULES.md #10; check every checkpoint cited is in
  `models/` (RULES.md #12).
- Draft to Jason by Oct 27, leaving time for a revision round.

### Oct 31 to Nov 2: buffer. Nov 3: final submission.

---

## Optional: only after everything above

- **Notebook 14** (native 600px, 56 arms). Jason proposed 320/480 *because* 600 is expensive; parked.
- **`tools/alma_simobserve.py` degradation sweep.** Tests our own question (does model value depend on how
  degraded the input is, `headroom_scatter.png`) on simulated ALMA noise. Works end to end; not requested.
- **VLT / PDS 70.** *"Not a big deal at all"* if skipped.
- **NLL loss**, if not already fitted in during Oct 1-7.

---

## History: Block 1, SG-training / wiggle question (Sep 10 to 23, closed)

Kept as the record of how that question was settled. Superseded as a plan by the sections above.

Current state: SG training improves M0/M1/M2 and degrades the actual wiggle diagnostic, shown
at n=5 with a confirmed-not-a-masking-artifact check (PROGRESS.md 2026-09-10). Notebook 12
(spectral context, `k=0/1/2`, scored on the wiggle directly) is running now.

**Decision gate, as soon as 12 lands:**

- **If `resid_r` climbs with `k`** -- spectral context is fixing the actual mechanism (no
  neighbouring-channel information). Sweep `k` further (3, maybe 4) on Kaggle, pick the best,
  and that becomes the SG training recipe going into Block 2. Cheap, no architecture change.
- **If it doesn't move** -- the problem is in the loss, not the input. Build the SG version of
  `KinematicLoss` (already exists and is tested for line emission, `src/utils/losses.py`,
  2026-08-28): switch SG training to the channel-stack architecture notebook 08 uses
  (`n_neighbors=15, stack_target=True, out_channels=31`), which is real rework of
  `synthesize_sg_pairs`'s consumer, budget 2-3 days including a Kaggle run.

**Gate resolved 2026-09-11: fast path, confirmed.** `k=3` (7 input channels) scores 0.681 on
the wiggle against dirty's own 0.594, the first SG-trained arm in this project to EXCEED doing
nothing rather than just approach it, and posts the best PSNR/M1/M2 in the SG thread. Full
sweep: `k=0` 0.366, `k=1` 0.590, `k=2` 0.506, `k=3` 0.681, not cleanly monotonic (`k=2` dips)
but the trend is clearly upward. PROGRESS.md 2026-09-11 (two entries) has the full numbers.
**Block 1's recipe is spectral context, `k=3` as the current best candidate, `k=1` as the
cheaper alternative** (22 min vs 34, a real but smaller wiggle cost). No architecture rebuild
needed, the fast path held. Remaining: pull `sg_k3_fresh.pth` off Kaggle; `k=4` would say
whether the trend keeps climbing but is now lower priority given a working positive result is
already in hand.

**Before committing to the SG rebuild either way, run notebook 08 first** (line-emission
kinematic-gamma sweep, built 2026-08-29, never run). 14 cubes instead of 3-5 disks means a much
cleaner read on whether `KinematicLoss` helps the wiggle-adjacent M1 signal at all, before
spending days building the same thing for a noisier, data-poor regime. If it doesn't help on
14 cubes it is very unlikely to help on 3. This is a days-not-weeks diagnostic and should run
in parallel with notebook 12's result landing, not after.

**Also in this block, lower priority, run if GPU time allows:** seed repeats on the leave-one-
out folds (notebook 11's design confound, named before that run: cube variance and training
variance are still mixed for `fresh` and partly for `finetune`). Two seeds per fold would cost
another ~4 hours of GPU time and would turn "directional" into an actual error bar. Do this
only if it doesn't push into Block 2's start.

**Exit criterion for Block 1:** either a working recipe that keeps the moment gains and does
not lose the wiggle (report it as the fix), or a clearly negative result with the mechanism
named (report it as a real finding, same as the retraction and the wiggle-degradation result
were reported honestly rather than hidden). Either outcome is a legitimate stopping point; do
not let this block run past 2026-09-23 chasing a positive result that isn't there.

---

## Standing practice, unchanged, applies throughout

- `results/PROGRESS.md` and `context.md` updated in the same push as the change (CLAUDE.md,
  "Before every git push").
- Every checkpoint kept until GSoC finishes (RULES.md #12); nothing gets cleaned up as a
  "losing arm" the way `winner_beam` almost was.
- A result gets reported honestly whether it's positive or not -- the retraction and the
  wiggle-degradation finding are both more useful to Jason than a quieter, incomplete story
  would have been, and that doesn't change now.
