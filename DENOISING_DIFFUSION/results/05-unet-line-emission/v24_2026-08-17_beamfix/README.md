# 05 Kaggle Version 24 — the beam arm, re-scored with its beam vector

First run after the fix in which `denoise_cube()` passes the beam vector. No training; the
beam arm's five cube-scores were recomputed and `winner_patch`'s PSNR was re-measured on the
common full-image validation set.

## The beam arm was never as bad as reported

```
                        M0        M1        M2
before (no beam vec)  -95.7%    +14.1%    -27.3%
after  (with vector)   +9.6%    +63.2%    +20.9%
```

**A 105 percentage point swing on M0**, and M2 flips sign. The old numbers measured a
beam-conditioned model running with its conditioning branch dead, because `UNet.forward`
ignores `beam=None` silently and section 6 never passed one, while section 4 did.

Per cube, with the vector:

```
run_0002_00560_rt_00     M0   14.3%   M1  54.4%   M2   -7.5%
run_0002_00560_rt_01     M0   45.5%   M1  83.4%   M2   54.1%
run_0002_00560_rt_04     M0   56.7%   M1  92.2%   M2   31.4%
run_0025_01000_rt_04     M0 -133.8%   M1  12.0%   M2   34.0%
run_0026_00005_rt_04     M0   65.2%   M1  74.3%   M2   -7.5%
```

Four of five cubes are strongly positive on M0; one (`run_0025_01000_rt_04`) is −133.8% and
drags the mean down to +9.6%. The failure is one cube, not the arm.

## What this changes

**Beam conditioning is not refuted.** It is positive on all three moments and simply not
better than the winner: M0 +9.6% against augmentation's +29.2% and patience-10's +33.5%, on
one seed. "Neutral to slightly worse, single seed" is the honest statement. The previous
reading, that it took the second-best PSNR while posting the worst M0 in the table, was an
artifact of the bug and should not be repeated.

That reading appears in the submitted midterm blog, which stays as written. It is corrected
here, in `RUNS.md`, and in `PROGRESS.md`.

## Also in this run

`winner_patch` PSNR 33.96 -> **34.98** dB, now measured on 256px full images like every other
arm rather than on 64px crops. Its moments are unchanged (M0 −40.8%), since `denoise_cube`
always ran full cubes. The patch arm remains a clear negative result.
