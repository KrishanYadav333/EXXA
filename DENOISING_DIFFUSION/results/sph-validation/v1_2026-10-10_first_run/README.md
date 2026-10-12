# sph-validation v1 (Kaggle CPU): first run, gate NOT passed, cause found

Speedie dump `sg_00005` rendered with `src/data/sph_cube.py` (MCFOST's length scale 1.5, 155.9 pc, i 23.2 deg) against speedie_prep's MCFOST
clean cube, r0.5, GI disc and Keplerian twin, tau0 1 to 100.

Result as run: bulk rotation matches (M1 corr 0.92, slope 0.95 GI / 1.00 twin, amplitude +-1.12 vs +-1.13 km/s) but the wiggle does not
(rc_r -0.05 GI, ~0 twin; rotation-curve residual rms 230 m/s against clean's 69 m/s, same on the twin so it is systematic, not wiggle).
The residual map is a large-scale red / blue dipole. tau0 changes nothing (rc_r -0.047 to -0.055).

Cause, from the saved M1 maps (`results/sph-validation/` v2 and the scan in PROGRESS.md 2026-10-10): the discrete PA set (+-236.7, +180) is
not the sky PA of our projection. Rotating our M1 by -24 deg lifts M1 corr 0.91 to 0.995 and cuts the rms difference to MCFOST 0.242 to
0.061 km/s. v2 of the kernel fits the angle continuously and chooses spin flip / mirror on the high-pass M1, because the bulk rotation
cannot tell the four combinations apart (planted-offset test: all 0.99, wiggle correlation 0.03 for the wrong one, 0.93 for the right one).
