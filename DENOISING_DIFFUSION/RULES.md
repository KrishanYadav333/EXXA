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
