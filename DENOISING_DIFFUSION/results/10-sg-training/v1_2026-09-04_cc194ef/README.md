# 10 Kaggle Version 1 -- training on self-gravitating data closes the domain gap

First real training run on self-gravitating data, answering Jason's suggestion. Ran on the
pairs synthesized in `experiments/synthesize_sg_pairs.py` (his own pairs cannot train a
denoiser, see PROGRESS.md 2026-09-04). Code commit `be616fd`, confirmed from cell 0b's own log.
Dataset `exxa-sg-synth-pairs`. Both arms converged and early-stopped, ~18 min each.

## Result

Holdout cube `run_9074_00025_rt_00`, never trained or validated on. Moment improvement over
the dirty cube, signal-masked at frac=0.05.

| arm | PSNR | SSIM | M0 | M1 | M2 |
|---|---|---|---|---|---|
| `frozen` (winner_aug, no training) | -- | -- | -10.3% | -0.6% | **-43.6%** |
| `finetune` (winner_aug, 0.1x LR) | 30.859 | 0.98343 | -6.5% | **+36.5%** | +15.6% |
| `fresh` (random init) | 30.024 | 0.98055 | **+5.0%** | +21.8% | **+26.0%** |

**Training on SG data works.** `frozen` is negative on all three moments; both trained arms
are positive on M1 and M2. That is the domain gap closing, measured for the first time with a
forward operator that is known exactly rather than estimated.

**`fresh` beats `finetune` 2 of 3.** M0 by 11.5 pp, M2 by 10.4 pp; `finetune` takes M1 by
14.7 pp. This contradicts the prediction written down before the run, which was that
line-emission pretraining would help given only three training disks. It does not obviously
help, and on M0 it appears to hurt.

**PSNR does not track the science, again.** 30.86 against 30.02 is nearly a tie while the
moments differ by 10-15 pp in opposite directions, and `finetune` had the BETTER validation
loss (0.0027 vs 0.0033) while losing on M0 and M2. Consistent with the spectral-context result
(05 v26) and with the reason RULES.md #4 exists.

## What this does not establish

**One holdout cube, three training disks, one seed.** This project has already measured how
unreliable that is: V7/V9 saw M2 swing +18.4% -> +2.5% on the same cube with zero config
change, and v25 saw identical-seed reruns swing 1.73 dB. A 10-15 pp gap on n=1 sits inside
that. `fresh` beating `finetune` is directional, not established, and should not be quoted as
a finding without more disks or more seeds.

**The domain gap is understated here.** `frozen` scores M0 -10.3% on this synthesized holdout
against -86.5% on Jason's real SG pair. The synthesized corruption (beam convolution plus
correlated Gaussian noise) is gentler and cleaner than whatever produced his dirty cubes, so
this run measures a smaller gap than the real one.

**`run_9032` is in the training set** and is an odd cube: its synthesized pair came out at
rmsdiff 0.107 against the others' 0.46-0.54, and its signal mask covered 98.9% of the field.
One third of the training data is therefore easier than intended.

## Follow-ups

- More seeds, and rotate the holdout across all five disks, before quoting any arm as better.
- Score these checkpoints on the GI wiggle metric, not just the moments. The kinematic
  question is the one the SG data exists to answer, and `finetune` winning M1 by 14.7 pp is
  the hint worth chasing there.
- Checkpoints `sg_finetune.pth` / `sg_fresh.pth` are in the Kaggle Output and must be pulled
  down before the next run wipes them (RULES.md #1, #12). Not yet retrieved at time of writing.
