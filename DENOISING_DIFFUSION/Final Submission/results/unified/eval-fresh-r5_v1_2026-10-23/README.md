# eval-fresh-r5 v1, 2026-10-23

Kaggle `kaggle-acct-b/eval-fresh-r5` v1 (launched 2026-10-22 ~22:45 IST, complete before 04:40 IST 2026-10-23). Candidate E (results/PLAN.md 0i-3): R5 seed(s) 1,2
(existing checkpoints, `models/unified/unified_r5_k15_s*.pth`), each as a single network through the control's evaluation (`eval-fresh-control`
kernel with only the checkpoints changed). Young TEST3 rendered in-kernel with the same command and seed as the control, which also rendered
in-kernel; Wiener rc_r is identical on every disc, so the cubes match. `eval-fresh-r5.py` = the kernel source as pulled from Kaggle; `run_log.txt` = its log;
`eval_fresh_r5/` = JSONs and per-case figures. Table: `python experiments/cand_step1.py`. Reading: results/PROGRESS.md 2026-10-23.
