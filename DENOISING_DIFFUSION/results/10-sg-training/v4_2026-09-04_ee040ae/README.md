# 10 Kaggle Version 4 -- V1's headline does not reproduce; `fresh` is unstable

Same code, same dataset, same seed (42), same split as V1. Code commit `340e26c` (V1 ran
`be616fd`; the only change between them is the `val_metrics` patch that fills in `frozen`'s
PSNR, which does not touch training). Re-run specifically to test whether V1's
`fresh` > `finetune` result survives a second draw.

**It does not.**

| arm | PSNR | M0 | M1 | M2 | V1 | reproduced? |
|---|---|---|---|---|---|---|
| `frozen` | 29.032 | -10.3 | -0.6 | -43.6 | -10.3 / -0.6 / -43.6 | **exactly** |
| `finetune` | 30.874 | -7.0 | +36.7 | +15.3 | -6.5 / +36.5 / +15.6 | within 0.5 pp |
| `fresh` | 29.258 | **-61.1** | **-27.3** | **-42.2** | +5.0 / +21.8 / +26.0 | **no, collapsed** |

`frozen` reproducing exactly is the control that makes the rest trustworthy: it is inference
only, so it must be identical, and it is. `finetune` reproducing to 0.5 pp says the pipeline
is not noisy in general. `fresh` swinging 66 pp on M0 -- from beating everything to being
worse than the untrained baseline -- is therefore a property of that arm, not of the setup.

## What this means

**V1's "fresh beats finetune" is withdrawn.** It was one draw of a high-variance arm. The
prediction recorded before V1 (that pretraining would help with only three training disks)
is the one the two draws together support, though n=2 is not a basis for a strong claim
either.

**The dominant variance here is training, not the cube.** Two runs on the *same* split with
the *same* seed produced a 66 pp difference. That is v25's finding recurring: early stopping
on a noisy validation set converts run-to-run nondeterminism into large swings. The evidence
is in the logs -- `fresh` early-stopped at epoch 23 with best epoch 15 here, against epoch 33
with best epoch 25 in V1, so it stopped 10 epochs earlier from a worse point. `finetune`
stopped at epoch 32 with best epoch 24 in both runs, because starting from trained weights it
is already near a good solution and the early-stopping decision is not delicate.

**Consequence for notebook 11.** Its leave-one-out design fixes the seed across folds to
isolate *cube* variance. Given what V4 shows, that isolates the wrong thing for `fresh`: a
fold-to-fold difference in that arm cannot be attributed to the cube when the same cube can
move 66 pp on its own. LOO as designed remains valid for `finetune`; for `fresh` it needs
seed repeats per fold, or its per-fold numbers have to be read as a lower bound on the noise
rather than as a cube effect.

## Checkpoints

V4's weights were NOT downloaded. `models/10-sg/sg_finetune.pth` and `sg_fresh.pth` are V1's
and remain so -- the V1 archive's numbers were measured from exactly those, and overwriting
them would break the ability to re-measure (RULES.md #12, the reason correcting `winner_beam`
was possible at all). If V4's are pulled later they belong beside them under a v4 name, not
on top.
