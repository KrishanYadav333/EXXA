# Blog images, in the order they appear in the post

Numbered 1-N to match reading order. Sources are the archived run folders, so each
image still traces to the run that produced it.

## 1.png  (3730 KB)
- **section:** 3. The data, and what it looks like before anything is done to it
- **source:** `results/stats/sample_grid.png`
- **caption:** Sample channels across the dataset. The variety in disk morphology, inclination and brightness across simulations is what the model has to generalise over from six independent training disks.

## 2.png  (464 KB)
- **section:** 3. The data, and what it looks like before anything is done to it
- **source:** `results/stats/noise_difference.png`
- **caption:** Dirty minus clean: the corruption itself, isolated. This is the structured sidelobe field from Section 1, not white noise, note the spatial correlation and the way it concentrates around bright emission.

## 3.png  (31 KB)
- **section:** 3. The data, and what it looks like before anything is done to it
- **source:** `results/stats/pixel_distribution.png`
- **caption:** Pixel intensity distributions, dirty against clean. Clean puts almost all its mass in a single spike at zero: in the truth, empty sky is exactly empty. Dirty smears that background into a broad hump past 0.4. The two bac...

## 4.png  (1406 KB)
- **section:** 3. The data, and what it looks like before anything is done to it
- **source:** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_maps_holdout.png`
- **caption:** Moment maps of one held-out cube: clean truth against the dirty input. The intensity map survives the noise reasonably well; the velocity and dispersion maps are visibly destroyed. That gap is the thing to close.

## 5.png  (43 KB)
- **section:** 4. First architecture comparison, on continuum images
- **source:** `results/stats/baseline_metrics_chart.png`
- **caption:** The same comparison as a chart. The ordering flips completely depending on which bar you read.

## 6.png  (243 KB)
- **section:** 4. First architecture comparison, on continuum images
- **source:** `experiments/unet_vs_all_comparison.png`
- **caption:** U-Net against every alternative on continuum data. The visual ordering is the SSIM ordering, not the PSNR ordering, which is the whole point.

## 7.png  (138 KB)
- **section:** 4. First architecture comparison, on continuum images
- **source:** `experiments/hybrid_vs_mse_comparison.png`
- **caption:** MSE-only against the hybrid MSE+SSIM loss on the same architecture. MSE alone produces the smoother, lower-error, less correct image.

## 8.png  (49 KB)
- **section:** 5. The first DDPM run, and why it failed
- **source:** `results/_archive-continuum-era/superseded_2026-07-25T1305_bf7a819/diffusion_loss.png`
- **caption:** DDPM training loss from the continuum era. It converges; it simply converges to something that is not competitive.

## 9.png  (1654 KB)
- **section:** 7. Continuum subtraction
- **source:** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_maps_continuum_comparison.png`
- **caption:** Moment maps with and without continuum subtraction. The contaminated version is not subtly worse; the intensity map is dominated by the dust pedestal.

## 10.png  (723 KB)
- **section:** 8. The U-Net on line emission: what each run changed
- **source:** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell20.png`
- **caption:** Five validation channels, dirty, U-Net denoised, clean truth, from 05 v20 (`sweep_winner_p10` seed 42, the checkpoint selected on M0 rather than PSNR). The same five channels the DDPM figure in Section 12 shows, so the t...

## 11.png  (70 KB)
- **section:** 8. The U-Net on line emission: what each run changed
- **source:** `results/05-unet-line-emission/v12_2026-07-10T1928_c61d112/moment_map_holdout_summary.png`
- **caption:** V12 across all five held-out cubes. Positive on all three moments on every cube, the result that made V12 the reference every later experiment is measured against.

## 12.png  (35 KB)
- **section:** 8. The U-Net on line emission: what each run changed
- **source:** `results/05-unet-line-emission/v18_2026-08-11_b24c84d/figure_cell16.png`
- **caption:** v18 against the V12 reference, five held-out cubes, dots are individual cubes. M0 and M1 both clear V12 while M2 lands level. This is the run that superseded V12 as reference checkpoint.

## 13.png  (686 KB)
- **section:** 9. Architecture comparison on line-emission data
- **source:** `results/09-architecture-comparison/v7_2026-08-02T1721_ee491fc/architecture_moment_maps.png`
- **caption:** Moment maps by architecture on one held-out cube. The U-Net row is visibly closest to the clean truth on M0 and M1. The autoencoder and VAE rows show the boxy, over-smoothed reconstruction their bottlenecks impose.

## 14.png  (48 KB)
- **section:** 10. Classical baselines on line emission
- **source:** `results/07-classical-baselines/v2_2026-07-31T1904_eb03589/classical_vs_learned_moments.png`
- **caption:** Learned against classical across all three moments, identical protocol.

## 15.png  (72 KB)
- **section:** 11. Seeds, sweeps, augmentation
- **source:** `results/05-unet-line-emission/v20_2026-08-14_c28b860/figure_cell22.png`
- **caption:** All six arms on the full metric, five held-out cubes. Coloured bars span seeds, the hatched bar is V12's published figure on the old unmasked metric and is not comparable, dots are individual cubes. The dots matter more ...

## 16.png  (100 KB)
- **section:** 11. Seeds, sweeps, augmentation
- **source:** `results/08-seeds-and-augmentation/v4_2026-08-02T0421_1ca611f/seed_spread.png`
- **caption:** Per-seed spread for four configurations. Dots are individual seeds. The overlap between the first two groups is the entire "reproduction failure" story.

## 17.png  (1825 KB)
- **section:** 12. The DDPM rerun
- **source:** `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/line_emission_ddpm_comparison.png`
- **caption:** DDPM v13 on validation channels: dirty, denoised, clean truth. Per channel this is a good reconstruction. Row 3 is the whole problem in one frame: the background is a flat mid-scale wash rather than black. Integrated ove...

## 18.png  (846 KB)
- **section:** 12. The DDPM rerun
- **source:** `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/ddpm_moment_maps.png`
- **caption:** DDPM moment maps. The velocity map recovers the rotation dipole cleanly. The intensity map shows the problem: the sky, black in the truth, sits at a raised floor. The dispersion map is saturated across the entire field.

## 19.png  (62 KB)
- **section:** 12. The DDPM rerun
- **source:** `results/06-ddpm-line-emission/v13_2026-08-12T0450_19efd47/moment_map_holdout_summary_ddpm.png`
- **caption:** The same result per cube. This is not a model that is uniformly bad; it is a model that is fine on three cubes and catastrophic on two.
