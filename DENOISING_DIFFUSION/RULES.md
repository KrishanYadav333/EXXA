# Operating rules

Rules earned by losing something. Each one names the incident that produced it, because a
rule without its failure gets argued away the next time it is inconvenient.

---

## 1. Persist a trained model the moment it finishes training

**Never** in a cleanup cell at the end of the notebook.

`CKPT_DIR` (`../results/checkpoints`) lives inside the git clone at
`/kaggle/working/EXXA/DENOISING_DIFFUSION/`. The section-0 bootstrap wipes that with
`git reset --hard`, and it is **not** part of the notebook Output. Only the top level of
`/kaggle/working` survives a session.

> **Incident.** Notebook 06 Kaggle Version 11 (2026-08-11, code `41df52d`) trained four
> sweep arms plus a full 60-epoch model over ~6 h, then died in section 13 at a CSV write.
> Persistence lived in a later cell that was never reached, so all six models were
> destroyed with the container. The numbers survived in the log; the models did not, and
> the entire run had to be repeated from scratch.

Define `persist_ckpt(path, note='')` next to `CKPT_DIR`; make it a no-op off Kaggle or on a
missing file. Call it directly after **every** `.train(...)` — each sweep arm, each seed,
the full run — and after any CSV that is the sole record of a ranking. Keep the
end-of-notebook bulk copy as a backstop, never as the only save.

Stronger still, where the trainer already saves per seed: point `CKPT_DIR` **at**
`/kaggle/working` so the training-time write is itself the durable one. Nothing then depends
on a later cell being reached — there is no call left to skip.

> **Second incident, same rule.** Notebook 08 wrote every seed's checkpoint to
> `../results/checkpoints` and copied them to `/kaggle/working` only in its final cell. It
> survived at 12 runs in one session purely by finishing. Section 1b's resume path reads the
> previous version's Output, which that final cell is what creates — so a timeout would have
> lost the models *and* the ability to resume. Caught while planning the 15-run rebuild
> (2026-08-12), before it cost anything.

A run that dies at hour 5 must cost the hours after the last save, not everything before
it.

---

## 2. The in-notebook git pull updates `src/` only, never the cells

Two different mechanisms, easy to conflate:

| | updates `src/` | updates the cells |
|---|---|---|
| **cell 0b** — `git fetch` + `reset --hard`, inside the kernel | yes | **no** |
| **Kaggle's GitHub integration** — the editor's push/pull | no | **yes** |

Cell 0b runs after Kaggle has already loaded the cells into the kernel, so nothing it does
can rewrite them. It clears `src.*` from the module cache, which is why library changes do
take effect mid-session. Kaggle's own pull is a platform operation and does replace the
cells — use it, or `File -> Import Notebook`.

The danger is the **push**: saving a version sends Kaggle's copy of the notebook into this
repo. A stale Kaggle copy therefore silently reverts committed work, and the push is
automatic where the pull is not.

> **Incidents, all on 2026-08-11.** 06 ran with `src/` at `3397823` but cells from
> `41df52d` — cell 0b reported the new commit, which read as "everything is current" — so it
> ran without the checkpoint fix, printed `no checkpoint found`, and trained from scratch
> instead of resuming. Push `b24c84d` ("Kaggle Notebook | 05_unet_line_emission | Version
> 18") then overwrote `c1f21e1` and `718ca3e` — the D4-augmentation arm and the moment-map
> figures — with a 23-cell copy predating `1d3022b`. That run's results had to be caveated
> in RUNS.md, because the notebook that produced them had no `SEEDS` and no `CONFIGS`.

Before every run: pull the notebook **through Kaggle**, then verify a marker that changed
in the latest commit. Cell 0b printing the newest SHA proves nothing about the cells — that
line is exactly what made the 06 failure look fine. After every run: check whether Kaggle's
push reverted anything before building on top of it.

---

## 3. Never upload a checkpoint as a Kaggle Dataset

A torch checkpoint is internally a zip. Kaggle unpacks archive-like files on **dataset**
upload, so a `.pth` arrives as a *directory* of `data.pkl` / `byteorder` / `version` /
`.data/`, and `torch.load` fails with `[Errno 21] Is a directory`.

Notebook **Outputs** are mounted verbatim and are the route that works: attach the
producing notebook as an input instead.

> **Incident.** Two separate uploads were destroyed this way — `exxa-ddpm-06-v8-epoch99`
> and a later `ddpm_seed42.pth`, the second visibly unpacked into 1956 files.

Save checkpoints as `.pth` (fine for an Output). If one genuinely must become a dataset,
wrap it so the outer container is what Kaggle unpacks.

---

## 4. Compare losses only within one objective and one training view

`best_val_loss` is not a universal quality score. Different objectives regress different
targets and different training views sum over different pixel counts, so a raw minimum
selects by scale, not by quality.

> **Incident.** Cell 6b picked the checkpoint with the lowest `best_val_loss` across all
> candidates. The eps epoch-99 checkpoint reads `13.4706` where the v/cosine run reads
> `67.1754`, so it would have restored the **17.9 dB** model over the **38.4 dB** one — and
> because it predates the `prediction_type` key (added in `750cf38`), nothing would have
> corrected the config, leaving a v-configured trainer decoding eps weights as v. Within
> `v`, the patch arm's loss is summed over 64 px tiles rather than 256 px images, so its
> 12-epoch checkpoint reads `21.97` against the 60-epoch model's `67.18` and wins.

Rank objectives by **PSNR**, computed identically for all of them. When selecting a
checkpoint, filter to the same `prediction_type` and the same training view first, and
refuse rather than fall back to an incompatible one.

---

## 5. A result is not a result until it is attributed

Every run's artifacts go through `collect_outputs()` into
`<notebook>/<UTC>_<sha>/` with a manifest recording the commit, the time, the key config,
and a checksum per file. `results/RUNS.md` maps run folders to Kaggle version numbers,
which the kernel cannot see.

> **Incident.** A flat `results/` directory made
> `moment_map_holdout_summary_ddpm.csv` from the unmasked-M2 era indistinguishable from one
> written after the fix; the only way to tell was to remember. Recovering the v7/v9/v12/v15
> lineage afterwards took a dedicated tool (`src/evaluation/recover_version.py`).

Do not list checkpoints in `collect_outputs` — it writes into `/kaggle/working/outputs`,
so naming them puts a second multi-GB copy inside the same Output. Rule 1 already saved
them.

---

## 6. Never quote a number whose metric you cannot name

Moment improvements are scored **signal-masked and noise-clipped**
(`src/evaluation/moment_maps.moment_improvement`). Numbers from the older whole-map metric
are not comparable and must be labelled, not silently tabulated alongside.

> **Incident.** M2 came out negative for run after run, which read as a model failure. It
> was the metric: averaging over every finite pixel let empty sky dominate, and M2 is a
> ratio whose denominator vanishes exactly there. Separately, an "M0" table in RUNS.md was
> actually M1 — the column had been omitted and the numbers shifted left, making every arm
> look like it beat V12.

Quote the mask, the clip, and the n. State whether a spread is across seeds or across
cubes — RUNS.md carries both and they are not interchangeable.

---

## 7. One notebook, one file — the Kaggle-linked name is the only real one

Kaggle's GitHub integration pushes to the file named after its kernel slug, and to nothing
else. Any second copy of the same notebook under a friendlier name stops receiving runs the
moment the first push lands, and then sits in the repo looking exactly as authoritative as
the file that is actually executing.

> **Incident.** Both `08-seeds-and-augmentation.ipynb` and
> `08-seeds-and-augmentationa369c56f22.ipynb` were at HEAD, 33 identical code cells, same
> title, same section numbers. The slug-named one carried Version 4's outputs — M0 +27.7,
> invented-blob rate 39.0%. The plain-named one carried an earlier session's — M0 +82.0,
> blob rate 0.0% — committed by hand at `d418498` and never re-executed. Reading the wrong
> one first produced a confident, wrong answer about which numbers Version 4 had actually
> printed, and the CSVs it wrote had already been filed as coming from an "unknown session".

Keep exactly one notebook file per Kaggle kernel, at the slug name Kaggle pushes to. If a
copy is downloaded for comparison, it goes under `results/<notebook>/v<N>_.../`, never
beside the live notebook. Before editing, confirm the file you are editing is the one the
last `Kaggle Notebook | ... | Version N` commit touched.

---

## 8. A zero from a diagnostic is a suspect, not a pass

A detector that reports "no problems found" and a detector that cannot fire produce the
same number. The clean-looking result is the one that needs checking hardest, because
nothing about it invites a second look.

> **Incident.** The artifact panel reported `channels with invented blob 0.0% |
> blobs/channel 0.000` for all four arms of 08 Version 2, and that read as four
> hallucination-free models. The threshold was a fraction of the map's **peak**, so on a
> cube with a bright disk nothing in the faint background could ever cross it. `04b3f2c`
> made it floor-relative; the same checkpoints then scored 22.3–39.0% of channels carrying
> an invented blob, concentrated at low SNR (1.58 blobs/channel below SNR 3.9 against 0.213
> above). 05 v18 printed the same `0%` alongside its own warning — *"no channel scored any
> invented structure -- showing the largest overshoot instead, and check n_background_px
> before trusting this"* — and the run was still written into RUNS.md as the best to date.

Any diagnostic reporting exactly zero across every arm is failing until proven otherwise.
Check the denominator: how many pixels were in the mask, how many channels were tested. A
metric that cannot distinguish between arms is not evidence that the arms are equal.

---

## 9. Check which branch the kernel pulled — it decides the metric

`src/` is fetched by cell 0b from a branch named in the bootstrap. Two notebooks pointed at
different branches score with different libraries, and their numbers then get tabulated
side by side as though the difference were the models.

> **Incident.** Notebook 05 clones, fetches and resets to `line-emission`, which is at
> `ef0a37b` and contains none of `bab16d0` (`clip_sigma`), `04b3f2c` (floor-relative
> artifacts) or `8f4da16` (signal-masked moments) — `clip_sigma` does not appear in that
> branch's `moment_maps.py` at all. Notebook 08 pulls `midterm-prep`, which has all three.
> v18's log reads `* branch line-emission -> FETCH_HEAD` / `HEAD is now at 227d2fc`, a
> commit from 2026-07-30, before every one of those fixes. Its M0 +81.2% ±12.6 is therefore
> an **unmasked** number in the same regime as 08 v2's +82.0% ±8.5, not comparable to 08
> v4's +27.7%. RUNS.md nonetheless recorded it as *"the best 05 run to date, and the first
> with every moment positive on every cube"* — a description of the broken metric, since
> the mask is what **creates** the negative M2, as 08 v4 showed on the same models.

Work happens on `midterm-prep`. Every notebook's bootstrap must name that branch, and the
`HEAD is now at ...` line must be read as a claim about `src/`, checked against the fixes
the run depends on. Before comparing two notebooks' numbers, confirm they pulled the same
branch.

## 10. Archive the notebook itself with every run, failures included

Every run gets a folder under `results/<notebook>/v<N>_<date>_<sha>/` containing **the
notebook as it ran, outputs intact**, alongside its log and figures. Not the stripped copy
committed at the repo root: the one Kaggle produced, with its cell outputs still in it.

`<N>` is the **Kaggle version number**, not a local counter. The kernel cannot see it, so it
has to come from the author or from Kaggle's own push commit
(`Kaggle Notebook | <slug> | Version N`). The diagnostics run was filed as `v21` and was
actually Version 22; the folder had to be renamed once the push commit revealed it.

**A run that failed is archived the same way, plus what broke.** The README records the
error verbatim, the cell it came from, how far the run got, and what was salvageable. A
crashed run is evidence: v19 died after four hours and its log is the only record that the
patch arm's epochs were drifting upward before the kernel was reaped.

The reason is that a stripped notebook plus a log is not the run. The outputs carry figures
never written to disk, the exact traceback, the execution counts that show what actually
ran, and the papermill timings. v19 and v20 were archived without their notebooks and those
copies are simply gone, because Kaggle only auto-pushes some saves and the local file was
stripped before anyone noticed.

Checklist per run, before the session's Output expires:

- `05-unet-line-emission.ipynb` — as it ran, outputs intact
- `run_log.txt` — stdout, extracted from the cell outputs if never downloaded separately
- every figure the run produced
- `README.md` — the Kaggle version, what changed since the last run, the headline numbers,
  and for a failure: the error, where it hit, and what survived
- a row in `results/RUNS.md` pointing at the folder

## 11. Log progress the moment a run ends, a notebook lands, or a bug appears

`results/PROGRESS.md` is the chronological record: what was attempted, what happened, what
it cost, and what it changed. `RUNS.md` answers "which run produced this number"; PROGRESS
answers "what state is this project in and how did it get here". They are different
questions and one file cannot do both.

Write an entry on any of three triggers, at the time, not later:

1. **A notebook run finishes**, successfully or not.
2. **A notebook is downloaded into the repo** after a run. This is also the moment to check
   whether Kaggle's copy reverted committed cells (rule 2) and to archive it (rule 10).
3. **A bug is found**, in a notebook, in `src/`, or in a number already published.

An entry is five lines at most: date, trigger, what happened, the evidence, the consequence.
A bug entry names what was wrong, how it was caught, and which published numbers it touches.
The last part is the one that matters. `winner_beam`'s M0 sat in RUNS.md and in the blog as
a finding for days before anyone noticed it had been measured with the model's conditioning
branch dead, and nothing in the repo recorded that the number was ever in question.

The cost of skipping this is not tidiness. Twice now a run has been repeated because the
previous attempt's outcome was in a chat log rather than in the repo, and v19's and v20's
notebooks were lost because nothing prompted anyone to file them while the Output still
existed.

## 12. Keep every checkpoint until GSoC finishes

Nothing gets deleted for being refuted, superseded, or uninteresting. Not the arm that
lost, not the seed that scored worst, not the config the sweep passed over.

`models/` is where they live, split by the notebook that trained them, indexed in
[`models/README.md`](models/README.md) with the numbers that identify each one. Files there
are hardlinks, so a checkpoint is only really gone when the last path to it is removed, and
removing any one path is safe.

The reason is `winner_beam`. Its moment scores went into RUNS.md and the blog as a finding,
and stayed there, before anyone noticed the arm had been scored with its conditioning branch
dead. Correcting it meant re-running inference on that exact checkpoint. M0 moved 105 points
and M2 changed sign. Had the checkpoint been cleaned up as a losing arm, the published number
would have been wrong permanently, with no way back to it short of retraining and no
guarantee of reproducing the same weights.

A result is only as durable as the weights that produced it. Rule 6 says never quote a number
whose metric you cannot name; this is the same rule pointed at the other end, since a metric
you cannot recompute is a number you cannot defend.
