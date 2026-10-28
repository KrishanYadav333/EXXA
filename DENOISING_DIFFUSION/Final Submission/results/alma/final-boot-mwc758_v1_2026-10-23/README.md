# final-boot-mwc758 v1, 2026-10-23

Kaggle `kaggle-acct-b/final-boot-mwc758` v1 (GPU, about 2 h 35 min). PLAN.md 0i-4 U: empirical M1 error map for MWC 758 13CO. Recipe B with the
exact `alma-exoalma-final-p0` setup (512 px, `--target-beam-px 10`, continuum auto) plus `--boot 4` (`alma_unified.bootstrap_m1`, first run).
The main pass reproduces the Oct 18 run exactly (noise reduction 195.74, M1 shift 72.6 m/s). `alma_boot/MWC_758_13CO/report.json` ("U"),
`*_m1_boot_std.fits` (m/s, NaN outside the S1 mask), `boot_error_map.png` (drawn locally from the FITS). `kernel_boot.py` = the kernel as it ran.
Reading: results/PROGRESS.md 2026-10-23.
