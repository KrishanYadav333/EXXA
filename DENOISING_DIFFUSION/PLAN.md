# Plan to Nov 2

53 days, ~7.5 weeks, from 2026-09-10. Written now rather than at the start of the project
because the shape of the remaining work only became clear after the retraction and the
SG-training wiggle result; a plan written in July would have pointed at the wrong problem.

Three blocks, in this order and for this reason: **close the SG-training question first**
because it is already in flight and cheap to finish; **ALMA validation second** because it is
the one item explicitly required for the final blog (Jason, 2026-08-07: "we can get to ALMA
data for the final blog") and currently has zero work against it, the largest structural gap
in the project; **writeup last**, with real days reserved rather than squeezed from whatever
is left.

Update this file the way `context.md` gets updated: when a week's plan changes because of what
a result actually showed, not on a fixed schedule. `results/PROGRESS.md` still gets every run;
this file only gets the decisions that change what's next.

---

## Block 1 -- close the SG-training / wiggle question (Sep 10 - Sep 23, ~2 weeks)

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

## Block 2 -- ALMA validation (Sep 24 - Oct 14, ~3 weeks)

Org task 6. Zero work against it as of today, repeatedly identified as the largest gap,
repeatedly deferred for whichever SG thread was active. This is not optional for the final
blog; Jason named it explicitly.

**Redesigned 2026-09-11, after Phase J closed.** The original plan scored one clean disk
through one real `simobserve` dirty cube -- a single verdict. That design now carries a known
risk it didn't have before: the headroom scatter (`context.md`, Phase J closing summary)
shows every model tested loses to doing nothing whenever the input is already lightly
degraded, and real ALMA/DSHARP dirty data tends to sit in exactly that low-degradation
regime. A single-cube ALMA verdict is likely to just be an eighth instance of that pattern,
not a new finding -- which would waste the block's one shot at real-telescope validation on
a result this project can already predict.

**New design: a degradation-axis experiment, not a single verdict.** Inject the synthetic
kinematic signal into real DSHARP dirty data at several degradation levels (added noise, or
array configuration/integration time via `simobserve`), score U-Net, classical filters
(Gaussian/median, already in `src/baselines.py`) and `tclean` at each level, and plot
recovery against input degradation the same way `headroom_scatter.png` does. This tests the
project's actual central claim -- that model value is a function of input degradation -- on
real telescope noise, instead of re-litigating one more high-headroom cube. If the trend
holds on real data too, that is a substantially stronger final-blog result than one cube
either winning or losing.

**Week 1 (Sep 24-30): setup and CASA fluency.** `simobserve` on the clean SG disks (or DSHARP
data directly), building toward several noise/configuration levels rather than one. Goal for
the week is the CASA pipeline working end-to-end at a single level first -- fluency before
breadth. Budget real time for CASA's own learning curve; this is new tooling for the project.

**Week 2 (Oct 1-7): the degradation sweep itself.** Run the same injected-signal comparison
across at least 3 degradation levels (more if time allows), scoring U-Net / classical
baselines / `tclean` at each with the same wiggle protocol used throughout Phase J. Separately,
still worth checking: does the beam-plus-noise approximation in `synthesize_sg_pairs.py` hold
up against a real `simobserve` dirty cube on the same clean disk -- if it doesn't, that bounds
how much to trust every SG-training result from Block 1.

**Week 3 (Oct 8-14): sweep continuation, not slack.** This redesign needs more `simobserve`
runs than the original single-cube plan, so this week is no longer a free overrun buffer --
it is where the degradation sweep actually gets its 3rd-5th points if Week 2 only managed 2-3.
If Weeks 1-2 finish with a full sweep already, use this week for a second disk. CASA and
interferometric simulation are exactly the kind of new-tooling work that has overrun before
in this project (the four wrong Phase 0 verdicts, the DDRM checkpointing bug); if the sweep
itself is short even at 3 points, that is still a real result and better than forcing a 5th
point into Block 3's time.

**Exit criterion:** at least 3 points on a real-DSHARP-noise degradation-vs-gain curve, for
at least one method (U-Net), scored and reported the same honest way as everything else in
this project, including if the trend doesn't hold on real data -- that would itself be a
finding worth having, since it would mean the synthetic-noise headroom scatter doesn't
generalize, which the final blog needs to know before claiming it does.

---

## Block 3 -- final blog and submission (Oct 15 - Nov 2, ~2.5 weeks)

Deliberately not squeezed from whatever time is left; it is planned as its own block from the
start, the way the midterm blog was.

**Oct 15-21: write.** Structure mirrors the midterm post's (classical baselines to
architecture comparison to line-emission U-Net to self-gravitating pivot), extended with:
the Phase 0 gate and why DDRM/VIREO were struck for line emission and reopened for SG data;
the retraction, told honestly, it is a stronger story than a clean result would have been;
the SG-training result (Block 1's outcome, whichever way it landed); the ALMA validation
(Block 2's outcome). Pull directly from `results/PROGRESS.md` rather than reconstructing the
narrative from memory, it is already the accurate chronological record.

**Oct 22-26: notebook and repo cleanup.** The public notebook link needs to actually run
clean, not just be internally correct. Archive anything not already archived per RULES.md
#10; confirm every checkpoint referenced in the blog is still in `models/` per RULES.md #12.

**Oct 27-30: mentor pass.** Send the draft, leave real time for a response and a revision
round rather than sending it two days before the deadline.

**Oct 31 - Nov 2: submission buffer.** No new work scheduled here on purpose. If everything
above finished on time, this is slack; if something ran long, this is where it lands instead
of the deadline itself.

---

## Standing practice, unchanged, applies through all three blocks

- `results/PROGRESS.md` and `context.md` updated in the same push as the change (CLAUDE.md,
  "Before every git push").
- Every checkpoint kept until GSoC finishes (RULES.md #12); nothing gets cleaned up as a
  "losing arm" the way `winner_beam` almost was.
- A result gets reported honestly whether it's positive or not -- the retraction and the
  wiggle-degradation finding are both more useful to Jason than a quieter, incomplete story
  would have been, and that doesn't change for Blocks 2 and 3.
