# Day-by-day, Sep 10 - Nov 2

Companion to `PLAN.md`, which has the reasoning and the exit criteria; this is the checklist.
54 calendar days.

**Read this note before the table:** Block 1 has one real decision gate (notebook 12's
result), and everything after it shifts depending on which way that gate resolves. Days 1-9
are firm. Days 10 onward are the *faster* path (spectral context works, no rebuild needed).
If the gate goes the other way, Block 1 gains 2-3 days for the `KinematicLoss` rebuild and
every date from there shifts right by that much -- update this file the day the gate resolves,
don't pretend the slower path fits the same slots.

---

## Block 1: close the SG-training / wiggle question

| # | Date | Day | Task |
|---|---|---|---|
| 1 | Sep 10 | Thu | Notebook 12 running on Kaggle (started today). |
| 2 | Sep 11 | Fri | Pull notebook 12's result + checkpoints once it lands. Read `resid_r` vs `k`. **Decision gate**, see note above. In parallel: kick off notebook 08 (line-emission kinematic-gamma sweep) on Kaggle, it's independent of the gate and has been idle 2 weeks. |
| 3 | Sep 12 | Sat | Notebook 08 still running (4 arms, hours). If gate said "sweep k further": start `k=3` on Kaggle. |
| 4 | Sep 13 | Sun | Pull notebook 08's result. Log to PROGRESS.md: does `kinematic_gamma>0` help M1 on 14 cubes at all. This is the evidence that decides whether the SG rebuild (if needed) is worth building. |
| 5 | Sep 14 | Mon | **If gate was "spectral context works":** pick the best `k`, this is now the SG training recipe. Move to day 10's task early. **If gate was "no fix, build KinematicLoss for SG":** start the rework -- `synthesize_sg_pairs.py`'s consumer needs `n_neighbors=15, stack_target=True, out_channels=31` to match notebook 08's shape. |
| 6 | Sep 15 | Tue | (slow path) Build the SG channel-stack dataset + notebook (13), mirroring notebook 08's design. Verify locally at reduced scale per the standing convention. |
| 7 | Sep 16 | Wed | (slow path) Run notebook 13 on Kaggle: `kinematic_gamma` sweep on SG data. |
| 8 | Sep 17 | Thu | (slow path) Pull results, score wiggle directly (reuse `score_sg_wiggle_loo.py`'s pattern). |
| 9 | Sep 18 | Fri | (slow path) Log outcome. Both paths converge here: a recipe that works, or a named negative result. Either way, Block 1's finding is now fixed for the blog. |
| 10 | Sep 19 | Sat | Seed repeats on the leave-one-out folds, IF time allows and Block 1 is otherwise done (PLAN.md: lower priority, don't let it push into Block 2). Start 1-2 seeds on Kaggle. |
| 11 | Sep 20 | Sun | Seed repeats continue / pull results. |
| 12 | Sep 21 | Mon | Write up Block 1's final state in PROGRESS.md and PLAN.md. Store and index every checkpoint produced this block (RULES.md #12). |
| 13 | Sep 22 | Tue | Buffer. Anything from days 1-12 that overran lands here. |
| 14 | Sep 23 | Wed | **Block 1 hard exit.** Whatever state it's in, move to Block 2 tomorrow regardless (PLAN.md is explicit about not chasing a positive result past this date). |

---

## Block 2: ALMA validation

Dates below assume Block 1 exits on schedule (day 14, Sep 23). If Block 1 ran long, shift
everything in this block right by the same number of days.

| # | Date | Day | Task |
|---|---|---|---|
| 15 | Sep 24 | Thu | CASA install/setup, `simobserve` basics. Budget the whole day for tooling, not results. |
| 16 | Sep 25 | Fri | Pick one clean SG disk, work out the antenna config + integration time to simulate. |
| 17 | Sep 26 | Sat | First `simobserve` attempt, expect it to fail once or twice. |
| 18 | Sep 27 | Sun | Debug day if needed, else first real dirty cube out of `simobserve`. |
| 19 | Sep 28 | Mon | Verify the `simobserve` output is sane (inspect it the way `phase0_diagnostics` inspects a suspicious cube -- shapes, BUNIT, a real beam in the header). |
| 20 | Sep 29 | Tue | Log the first real ALMA-simulated pair to PROGRESS.md. Week 1 goal met. |
| 21 | Sep 30 | Wed | Compare `synthesize_sg_pairs.py`'s beam+noise approximation against this real cube, same clean disk. Does the approximation hold? |
| 22 | Oct 1 | Thu | Finish that comparison, log it either way -- it bounds how much to trust every Block 1 result. |
| 23 | Oct 2 | Fri | Score Block 1's chosen model on the `simobserve` cube: moments. |
| 24 | Oct 3 | Sat | Score the same model on the wiggle, same method as `score_sg_wiggle_loo.py`. |
| 25 | Oct 4 | Sun | Log the ALMA-realistic result. This is the number the final blog actually needs. |
| 26 | Oct 5 | Mon | If Weeks 1-2 finished on time: start simulating a second disk for n=2. If not: this is buffer, see Week 3 note in PLAN.md. |
| 27 | Oct 6 | Tue | Second disk continued. |
| 28 | Oct 7 | Wed | Second disk's dirty cube produced. |
| 29 | Oct 8 | Thu | Score the second disk, same as days 23-25. |
| 30 | Oct 9 | Fri | Log n=2 ALMA result. |
| 31 | Oct 10 | Sat | Buffer. |
| 32 | Oct 11 | Sun | Buffer. |
| 33 | Oct 12 | Mon | Buffer. |
| 34 | Oct 13 | Tue | Write up Block 2's final state in PROGRESS.md and PLAN.md. |
| 35 | Oct 14 | Wed | **Block 2 exit.** |

---

## Block 3: final blog and submission

| # | Date | Day | Task |
|---|---|---|---|
| 36 | Oct 15 | Thu | Draft outline from `results/PROGRESS.md` directly (it's already the accurate chronology, don't reconstruct from memory). Sections: baselines, line-emission U-Net, Phase 0 + why DDRM/VIREO were struck then reopened, the retraction (told straight), SG training result, ALMA validation. |
| 37 | Oct 16 | Fri | Write: baselines through line-emission U-Net section. |
| 38 | Oct 17 | Sat | Write: Phase 0 + physics-informed methods section. |
| 39 | Oct 18 | Sun | Write: the retraction section. This is the one that needs the most care, not the least. |
| 40 | Oct 19 | Mon | Write: SG training + wiggle result section. |
| 41 | Oct 20 | Tue | Write: ALMA validation section. |
| 42 | Oct 21 | Wed | First full-draft read-through, fix gaps. |
| 43 | Oct 22 | Thu | Public notebook: make sure it actually runs clean end to end, fresh clone. |
| 44 | Oct 23 | Fri | Repo cleanup: archive anything not yet archived (RULES.md #10), confirm every checkpoint the blog references is still in `models/` (RULES.md #12). |
| 45 | Oct 24 | Sat | Figures pass: regenerate or clean up every figure the blog links to. |
| 46 | Oct 25 | Sun | Second full read-through. |
| 47 | Oct 26 | Mon | Send draft to Jason. |
| 48 | Oct 27 | Tue | Waiting on mentor feedback; keep polishing figures/notebook in the meantime. |
| 49 | Oct 28 | Wed | Revision round once feedback arrives. |
| 50 | Oct 29 | Thu | Finish revisions. |
| 51 | Oct 30 | Fri | Final read-through, links checked, notebook re-run one more time. |
| 52 | Oct 31 | Sat | **Submission buffer.** Nothing new scheduled here on purpose (PLAN.md). |
| 53 | Nov 1 | Sun | Submission buffer. |
| 54 | Nov 2 | Mon | **Deadline.** |
