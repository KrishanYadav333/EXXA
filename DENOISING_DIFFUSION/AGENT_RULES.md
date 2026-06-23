# AGENT_RULES.md
## Operating Rules for Agentic IDE — EXXA Denoising Project

---

## RULE 0 — CURRENT PROJECT STATE (read first; pivot of 2026-06-18)

The project pivoted in the 2026-06-18 mentor meeting. Until a later mentor instruction overrides this:

1. **Active dataset = LINE EMISSION cubes** (`data/Line Emission Data/`, FITS, (201,600,600)).
   Continuum (`clean.npy`/`dirty.npy`) is BACKGROUND/foundational — do not make it the focus.
2. **NO patch-based training.** Full images only (downsample to 256×256/300×300 if memory-bound).
   The 64×64 patch approach is deprecated project-wide per mentor.
3. **Active model = Week-3 U-Net on full images.** DDPM is PAUSED (underperformed, SSIM ~0.22).
4. **Channel sampling:** ~50/cube, Gaussian center idx 100, ~75% in [50,150], avoid extremes.
5. **Split at CUBE level:** hold out 3–4 entire cubes inference-only (not train, not val).
6. **Scientific eval:** denoise held-out cubes channel-by-channel → `bettermoments` moment maps →
   compare dirty vs denoised.

Full detail: `context.md` Sections 3A, 6 (2026-06-18), 8. Do not revert to patches or continuum
as the main thread without explicit user/mentor instruction.

---

## RULE 1 — Definition of "Done"

A task is **NOT done** until all of these are true:
1. Code is written
2. Code has been **actually executed** (not just smoke-tested on dummy tensors) with the real dataset (`data/clean.npy`, `data/dirty.npy`) or real Kaggle session
3. Real artifacts exist on disk: loss curves (`.png`), checkpoints (`.pth`), metrics tables (`.csv`), or sample images — not just printed numbers in a chat response
4. You have personally verified the artifact exists and looks reasonable (not NaN, not empty, not all-zero)

If any of these is missing, status must be reported as **"code ready, not yet executed"** or **"partially done"** — never as "Done" or "complete."

**Before declaring anything done, the agent must explicitly answer:**
*"Has this been run on the real dataset, and do the output files exist on disk?"*

---

## RULE 2 — No Silent Deviation from Spec

If the agent decides to deviate from an explicitly stated specification (model size, resolution, loss weights, hyperparameters, file structure, etc.) for any reason — VRAM constraints, simplicity, perceived improvement — it must:
1. **State the deviation explicitly and immediately**, not bury it in a wall of text
2. **State the original spec it deviated from**
3. **State the reason for the deviation**
4. **Ask before proceeding** if the deviation materially changes scope, compute requirements, or timeline — proceed only on lower-risk deviations (e.g. variable naming) without asking

Never let a deviation surface only when directly questioned later.

---

## RULE 3 — Weekly Progress Log

At the end of each working session, the agent must append (not overwrite) an entry to `PROGRESS_LOG.md` in this format:

```
Week N — [Date Range]

Completed (verified with real artifacts)
- [item] — artifact: [path/to/file]

In Progress
- [item] — current blocker or next step

Not Started
- [item from weekly plan, if applicable]

Deviations from plan
- [what changed, why, mentor/contributor approval status]

Open questions for mentor
- [if any]
```

This log is the single source of truth for "what's actually been accomplished" — more reliable than chat history, which can contain inflated claims.

---

## RULE 4 — Mentor Feedback Is Ground Truth, Hierarchically

Priority order when instructions conflict:
1. **Explicit mentor feedback** (Jason Terry, Sergei Gleyzer) from meetings/emails — highest priority, follow literally
2. **Contributor's (Krishan's) own stated decisions** in this session
3. **Agent's own judgment/suggestions** — lowest priority, always flagged as a suggestion, never silently substituted for 1 or 2

When the agent disagrees with mentor feedback or thinks there's a better approach, it should say so explicitly and ask, rather than just doing what it thinks is better.

---

## RULE 5 — Checkpoint and Artifact Discipline

- Kaggle sessions have time limits — checkpoint every 500–1000 training steps, not just at the end
- Never overwrite a previous best checkpoint unless the new one is verified better on validation metrics
- Every checkpoint save must be accompanied by a log line stating: epoch/step, val loss, val SSIM, file path
- Keep old experiment results (e.g. the 64px DDPM baseline) rather than deleting — append new sections, don't replace, per Rule 2 reasoning already established this week

---

## RULE 6 — Reproducibility Requirements

Every training run must log/print at minimum:
- Random seed used
- Exact hyperparameters (batch size, lr, loss weights, epochs, model config)
- Hardware used (GPU type, count)
- Final train/val metrics

This should be saved alongside the checkpoint (e.g. as a sidecar `.json` or in the notebook markdown cell), not just printed and lost in chat.

---

## RULE 7 — Metric Reporting Honesty

- Always report SSIM as the primary metric for this project (per CONTEXT.md Section 7 rationale) — never lead with PSNR/MSE alone since they can mislead (favor blur over structure preservation)
- Never cherry-pick the best epoch without disclosing it — if reporting "best @ epoch 23," also show how stable nearby epochs are
- If a result looks suspiciously good or suspiciously bad, flag it for double-checking rather than reporting at face value

---

## RULE 8 — Ask Before Expensive/Irreversible Actions

The agent must ask for confirmation before:
- Starting a training run estimated to take >30 minutes
- Deleting or overwriting any file in `results/checkpoints/`
- Making changes that would require re-running already-completed work
- Switching frameworks, major library versions, or core architecture choices not already discussed

---

## RULE 9 — Session Continuity

At the **start** of each session, the agent should:
1. Read `CONTEXT.md` and the latest entries in `PROGRESS_LOG.md`
2. State a one-line summary of "where we left off" before doing anything new
3. Confirm whether previous session's claimed-done items actually have verified artifacts (cross-check against Rule 1)

---

## RULE 10 — Communication Style

- No inflated language ("massive breakthrough," "perfect," "fully complete") unless objectively true and verified
- Prefer precise, falsifiable claims: "Trained for 30 epochs, best val SSIM 0.76 at epoch 27, checkpoint saved" over "Great results!"
- When uncertain, say so directly rather than guessing confidently
- Flag anything that looks like a bug, anomaly, or inconsistency immediately rather than working around it silently

---

## RULE 11 — No Emojis

Never use emojis in any communication, documentation, code comments, or output. Use plain text only.

---

## RULE 12 — Protected Files (Never Delete)

The following files must NEVER be deleted without explicit user confirmation:

**Reference Materials:**
- `GSOC_2025_EXXA_Main.ipynb` — Original reference notebook from previous GSoC project
- Any file with "REFERENCE" or "ORIGINAL" in the name

**Core Documentation:**
- `context.md` — Project context for AI agents
- `AGENT_RULES.md` — This file
- `progress.md` — Progress tracking log
- `readme.md` — Project README

**Core Source Code:**
- Anything in `src/` directory
- Anything in `tests/` directory

**Data Files:**
- `data/clean.npy`
- `data/dirty.npy`
- Any `.npy`, `.fits`, or other data files

**Results/Artifacts:**
- Checkpoint files (`*.pth`, `*.pth.tar`)
- Results CSV files
- Training curve images
- Comparison visualizations

**Notebooks:**
- `00_master_research_notebook.ipynb` — Master notebook
- Individual numbered notebooks (01-05)

Before deleting ANY file, the agent must:
1. Verify it is NOT on the protected list above
2. Verify it is a genuinely temporary file (ends in `.tmp`, `_temp`, `_check`, etc.)
3. Ask user confirmation if there's any doubt

---

**Last updated:** June 18, 2026 (added Rule 0 — line-emission pivot)
