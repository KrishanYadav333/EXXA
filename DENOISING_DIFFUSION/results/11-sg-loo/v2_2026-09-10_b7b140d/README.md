# 11 Kaggle Version 2 (author-reported) -- leave-one-out, and nothing is separable

Version number from the author, not a push commit -- notebook 11 was downloaded manually
rather than auto-pushed through Kaggle's GitHub link, so there is no `push` commit to verify
it against (unlike notebook 10's V4, `ee040ae`). Code commit `b7b140d`, confirmed from cell
0b's own log. 128 minutes total, 5 folds x 2 arms, checkpoints NOT downloaded (10 files,
`loo*_*.pth`, still only on Kaggle).

## Result

5 folds, every cube holding out exactly once, seed fixed at 42. Moment improvement over dirty,
signal-masked at frac=0.05.

| fold | holdout | frozen (M0/M1/M2) | finetune (M0/M1/M2) | fresh (M0/M1/M2) |
|---|---|---|---|---|
| 0 | run_9015 | +23.6 / -0.6 / -22.3 | +35.0 / +26.3 / +9.3 | +37.4 / +17.9 / +7.4 |
| 1 | run_9019 | +23.8 / +9.7 / -76.3 | -24.1 / -7.5 / -44.0 | -6.3 / -20.7 / -139.7 |
| 2 | run_9025 | +4.6 / +12.3 / -9.5 | +13.2 / +22.3 / +7.1 | +14.2 / +19.2 / +7.7 |
| 3 | run_9032 | -111.3 / -284.2 / -112.8 | -68.2 / -92.8 / -36.9 | -288.7 / -351.5 / -246.5 |
| 4 | run_9074 | -10.3 / -0.6 / -43.6 | -15.1 / +20.3 / +6.2 | +10.7 / +2.2 / -48.4 |

| arm | PSNR mean +/- std | M0 mean +/- std | M1 mean +/- std | M2 mean +/- std |
|---|---|---|---|---|
| frozen | 29.53 +/- 2.98 | -13.93 +/- 56.28 | -52.70 +/- 129.56 | -52.91 +/- 41.98 |
| finetune | 30.88 +/- 3.03 | -11.84 +/- 39.27 | -6.30 +/- 50.18 | -11.64 +/- 26.42 |
| fresh | 28.27 +/- 1.36 | -46.55 +/- 136.27 | -66.56 +/- 160.08 | -83.89 +/- 109.02 |

## The notebook's own verdict: not separable

`fresh - finetune` per moment, against the fold-to-fold standard deviation (deliberately the
full spread, not the standard error -- n=5 supports no significance claim either way):

- M0: -34.71 +/- 104.38 pp, fresh wins 4/5 folds, smaller than the spread
- M1: -60.26 +/- 111.05 pp, fresh wins 0/5 folds, smaller than the spread
- M2: -72.25 +/- 86.59 pp, fresh wins 1/5 folds, smaller than the spread

Every difference is smaller than the fold-to-fold noise. This run cannot say which arm is
better.

## What it does say

**Fold 3 (`run_9032` as holdout) is catastrophic for every arm**, `frozen` included (-111.3 /
-284.2 / -112.8). This was predicted in the notebook BEFORE the run, from the cube's own
diagnostics: rmsdiff 0.107 against the other four's 0.46-0.54, and a signal mask covering
98.9% of the field in the earlier validation entry. Written down in advance rather than
explained after, which is the point of writing it down.

**`frozen`'s spread is the only clean cube-variance measurement here.** It never trains, so
its fold-to-fold swing (PSNR std 2.98, M0 std 56.3, M2 std 42.0) is caused by the cube alone,
not by training noise. `finetune` and `fresh` mix cube variance with training variance
(notebook 10 V1 vs V4 already showed `fresh` alone can move 66 pp between identical runs on
the SAME cube), so their spreads in this table are not directly comparable to `frozen`'s or to
each other on equal footing.

**Against that baseline, `finetune` is the most stable and has the best mean on every
moment**: lower std than `frozen` on M0 and M2, and the least negative mean on M0, M1 and M2
of the three arms. That supports "finetune is not worse and is more predictable," not
"finetune is better" -- the spreads are still too large to call that a finding.

## What this does NOT establish

The seed was fixed across folds specifically to isolate cube variance (RULES.md #6). Notebook
10 V4 then showed that was the wrong isolation for `fresh`: a fold-to-fold difference in that
arm cannot be read as a cube effect when the same cube can move 66 pp on its own. This design
limitation was flagged in PROGRESS.md before this run and applies to reading it: `fresh`'s
column here mixes cube and training variance, `frozen`'s and `finetune`'s do not (to the
extent `finetune`, starting from trained weights, has a non-delicate stopping decision, per
notebook 10's log evidence).

**Not yet done:** seed repeats per fold, which would separate the two variance sources
properly. Checkpoints need pulling before the next run overwrites them on Kaggle's side.
