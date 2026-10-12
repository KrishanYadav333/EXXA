# wiggle-correction, Kaggle Version 2 (2026-10-09 to 10): Wiener + learned correction, leave-one-hydro-disk-out

Private GPU script kernel `krishanyadav333/wiggle-correction`; code `experiments/train_wiggle_correction.py` + `wiggle_scorecard.py` + `src/data/analytic_gi.py` (NOT committed at launch;
shipped in the private dataset `exxa-wiggle-code` with sha256 sums, staged from local HEAD 5b04466; see PROGRESS.md 2026-10-09 "run"). Version 1 was the 2.7 min smoke test.
5 folds x 9000 steps (base 48, batch 12 x 3 centres, 128 px crops, lr 3e-4 one-cycle, fp16), 605 min total, 120 min per fold. Each fold trains on the 4 other hydro disks (301/600 px
resampled by the cube's own grid) and the 6 analytic disks an01-an06 (the 101-channel version), fresh noise every round, and scores the held-out hydro disk, SG v2, an07, an08.

Files: `run_log.txt` (printed log), `summary.json` (every metric, every fold, every case). Checkpoints: `corr_fold<disk>.pth` in the Kaggle kernel Output (rule #1: written each round).
Headline: rc_r on 9074 0.13 (dirty) / 0.16 (Wiener) / 0.44 (corrected); 9025 0.13 / 0.21 / 0.44. Table in PROGRESS.md 2026-10-10.
