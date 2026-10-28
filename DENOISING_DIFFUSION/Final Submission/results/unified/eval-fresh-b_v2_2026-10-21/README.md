# eval-fresh-b v2, 2026-10-21

Kaggle `kaggle-acct-b/eval-fresh-b` v2 (complete; v1 failed in 1 min, TEST3 input not visible from account B). Fresh test set of the improvement track
(PLAN.md 0i): recipe B (`fresh_B`) and candidate A = B + `--central-fix` (`fresh_A_cfix`) scored by the same kernel on analytic GI an09 to an16
and the no-wiggle twins an09k / an12k / an15k at 600 px, deconvolution and noise-only (`_ai`) regimes. Gate inactive on every case (gate_a 1.0,
blind SNR 14.8 to 33.5). **The Young TEST3 part did not run**: the in-kernel render failed (`sarracen` not installed, all 8 FAILED, the kernel did
not stop); it is scored by `kaggle-acct-b/eval-fresh-young`. Tables: `fresh_tables.md` (from `experiments/fresh_tables.py`). Reading: results/PROGRESS.md 2026-10-21.

Figures (fetched 2026-10-22): `eval_fresh/figs_fresh_B/` and `figs_fresh_A_cfix/`, a moment / wiggle panel and a minor-axis PV plot per case.
