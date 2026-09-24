# ALMA plan: our denoisers on real exoALMA data

Written 2026-09-25. `PLAN.md` holds the dated schedule and the org deadlines; this file holds the
design: what data, what has to be true of the pipeline, how we will know it worked, and what could go
wrong. It follows what Jason asked for (2026-09-12 meeting). Anything that is our own addition is marked
**[ours]** so it is never mistaken for his request.

---

## 1. Scope, in Jason's words

- *"if you want to take the time to do inference on actual ALMA data, this is probably the data that we're
  going to start with"*: the exoALMA release, Harvard Dataverse doi:10.7910/DVN/CFHWNH.
- *"do the Fiduciary line images"*; *"don't worry about the PSF or the mask. Just look at the image. So the dot
  image dot fits, those are what you want."*
- *"13CO is the one you want ... 13CO is kind of like the canonical one, but 12CO is fine. CS is hard because
  it's really dim."*
- *"MWC 758 ... that's one that people like a lot because it's really weird."* Read the exoALMA I paper to see
  which disks are interesting. DSHARP *"absolutely, as well"*, after exoALMA.
- Models: *"focus on your best ones"*, two or three.
- Success: *"a really good ALMA pipeline ... show it can be used and actually give scientific insights."*
- Viewing: DS9. VLT: *"not a big deal at all"* if skipped.

**Out of scope unless time is left:** VLT and PDS 70; the `simobserve` degradation sweep
(`tools/alma_simobserve.py`, our own design); notebook 14.

---

## 2. Where we are (2026-09-25)

| piece | state |
|---|---|
| `tools/alma_infer.py` | Built. Smoke-tested on 8 noise-only planes only. Never run on a full cube. |
| `15-alma-real-cube-inference.ipynb` | Built for Kaggle (fetches cubes from Dataverse). Never run. Branch `alma-validation` is **not pushed**, so Kaggle cannot clone it yet. |
| Checkpoints | `winner_aug_seed43` available; loss-sweep arms (`starlet_ft`, `wavelet_ft`, `mae_ft`, `gradient_ft`) live in 05's Kaggle Output. |
| Real data | Nothing complete locally. A 540 MB partial MWC 758 12CO cube (74 of 301 planes) exists, git-ignored. |
| Results on real data | None. |

---

## 3. The data

### What a fiducial cube is (exoALMA I, section 4.2.1)

Real Band 7 observations (configurations C43-3 plus C43-6, plus the ACA for seven of the fifteen disks), CLEANed
with a source-specific robust and uv-taper to a **circular 0.15" beam**, **100 m/s** channels for 12CO and 13CO
(200 m/s for CS), about **1.5 K** rms per channel. The images are already continuum-subtracted (Fig. 1 of the paper
shows continuum-subtracted 12CO); this is checked against the data, not assumed (item B2).

Consequence that shapes everything below: **a fiducial cube is an observation, not a clean truth.** It has thermal
noise correlated over the beam and CLEAN artefacts. There is no ground truth on it, so "inference" can only
measure what the model changes, and any claim that it improved the data needs an independent check (section 6).

### File layout and size

`<disk>_<line>_fiducial.image.fits`, one per disk per line, 0.6 to 1.3 GB (13CO/12CO 1.26 GB, CS 0.63 GB).
Download only `.image.fits`; skip `.psf`, `.mask` and `.smoothed_psf` (Jason). Measured header for
MWC 758 12CO: 1024x1024 px at 25 mas, 301 channels, real sky position, Jy/beam, frequency axis.
Locally the link ran at 150 to 250 KB/s, so cubes are fetched **on Kaggle** (notebook 15 does it).

### Disk shortlist

From exoALMA I Tables 1 and 2. Training inclinations span 12 to 70 degrees, so every disk below is in range.
`F13CO` is the integrated 13CO flux, a proxy for how bright the line is.

| order | disk | i (deg) | d (pc) | M* (Msun) | F13CO (Jy km/s) | substructure | why |
|---|---|---|---|---|---|---|---|
| 1 | **MWC 758** | 19.4 | 156 | 1.40 | 5.46 | R, A, I | Jason named it. Low inclination, close to training cubes at 12 to 22 degrees. |
| 2 | HD 135344B | -16.1 | 135 | 1.61 | 6.19 | R, A, C | Bright, low inclination, cavity and asymmetry. |
| 3 | HD 143006 | -16.9 | 167 | 1.56 | 1.84 | R, A, I, W | Faintest 13CO in the shortlist: the low-SNR stress test. Warp. |
| 4 | LkCa 15 | 50.3 | 157 | 1.14 | 9.83 | R, C | Higher inclination, like most training cubes. Large disk, may need a wider field. |
| 5 | V4046 Sgr | -34.1 | 72 | 1.73 | 17.05 | R, I | Brightest, nearest; large angular size. |

Not chosen: RXJ1604 (i = 6 deg, near face-on, below the training range), the highly inclined disks are excluded
by the survey itself (5 to 60 degrees). Order is a proposal; Jason said to read the paper and pick.

---

## 4. The domain gap

The models were trained on 14 MCFOST radiative-transfer simulations. Measured from their `.para` files and
FITS headers:

| property | training cubes | exoALMA fiducial | consequence |
|---|---|---|---|
| Molecule | **13CO J=2-1** (220.4 GHz) | 13CO J=3-2 (330.6 GHz), 12CO J=3-2 | Same molecule. This is a concrete reason for Jason's "13CO first": 12CO is optically thick and can carry foreground absorption the models never saw. |
| Star | one setting: 1.0 Msun, 4282 K | 0.45 to 1.73 Msun | Velocity field scales with sqrt(M). Models saw one mass. |
| Inclination | 11.7 to 70.3 deg | 5 to 60 deg (survey limit) | Overlap is good. MWC 758 (19.4) sits with three training cubes at 11.7, 13.4, 21.7. |
| Disk size | R_out = 300 AU in a 600 AU box, every cube. At 100 to 193 pc that is 1.55" to 3.0", 11 to 22 beams. | gas radii from about 1" to 7" | Training disks always **fill the frame**. A real disk framed differently is a different picture. |
| Beam | 0.10 to 0.185", FWHM **5.8 to 11.2 px at 256 px, median 7.9** | 0.15" circular | See pixel scale below. |
| Pixel | 5.2 to 10.0 mas at 600 px | 25 mas, 1024 px | Resampling needed. |
| Channels | 201 x 0.1 km/s | 301 x 0.1 km/s (CO) | Same width. |
| Noise | simulated dirty-image noise with sidelobe structure | CLEAN-restored, beam-correlated, real | Real domain shift. The models never saw CLEAN noise. |
| Turbulence | 0.02 to 0.20 km/s | real | Line-width range is covered. |

### Pixel-scale and field decisions

Two policies are defensible and they disagree, so we run both and compare (check S4):

- **P1, beam-matched (default).** Resample to about **19 mas/px**, which puts the 0.15" beam at 7.9 px, the
  training median at 256 px. This replaces the earlier 16 mas default (9.4 px, near the top of the training range).
- **P2, frame-filling.** Choose the field as 2 x R_out of the real gas disk, as every training cube has, then
  resample to the model grid. R_out is measured from the cube's own moment 0, not assumed.

The field must be a multiple of 8 px on the model grid (the U-Net has three downsamplings). Only 256px
models are run; the 320/480/600 px arms train at other pixel scales and are excluded for now.

---

## 5. The pipeline

Built: `tools/alma_infer.py`, run by notebook 15.

1. Read the cube, crop the central field, replace NaN (outside the primary beam) with 0.
2. Subtract the cube's own continuum (mean of the first and last 5 channels), as training does.
3. Per-channel min-max by that channel's own range (invertible), bilinear resize to the model grid.
4. Model forward, undo the resize, undo the min-max.
5. Moments 0/1/2 of raw and denoised through the project's `generate_moment_maps` (3 sigma clip, edge-channel rms).
6. Report: off-line noise removed, M0 total ratio, M1 shift in km/s, plus `report.txt`, two figures, one
   denoised FITS per model.

**Models: `winner_aug_seed43`, `winner_starlet_ft`, `winner_wavelet_ft`** as the working set, revised once the
moments for the resolution arms are scored. The beam, kinematic (`kin_gamma0`), spectral-context (`sg_k3`) and
diffusion checkpoints are not run: they need inputs a bare cube does not provide or they are a different method.

### Build list

| id | item | why |
|---|---|---|
| B1 | Change the default pixel scale to 19 mas | beam-in-pixels at the training median |
| B2 | Check the cube is continuum-subtracted (edge-channel mean, header HISTORY) and record it | do not assume |
| B3 | Measure R_out from the raw moment 0 and print beam-in-pixels for each policy | needed for P2 |
| B4 | Write raw and denoised **moment maps and the raw cube as FITS with WCS** | DS9 needs FITS, not PNGs |
| B5 | Identify line-free planes from the spectrum, not just the first and last 5 | off-line rms is only as good as that choice |
| B6 | Keplerian fit and residual map for raw and denoised (`src/evaluation/gi_wiggle.py`) | kinematics is the science |
| B7 | Invented-structure detector on line-free planes, fixed threshold | see S2 |
| B8 | Injection-recovery on real noise **[ours]** | see S5 |
| B9 | Multi-scale runner (P1, P2, plus or minus 25% pixel scale) | see S4 |

---

## 6. How we will know it worked

A real cube has no truth, so no single number says "good". Jason's bar is qualitative: a pipeline that works
and gives insight. These checks make that claim defensible. **S1, S6 and the DS9 inspection are the ask; S2 to S5
are our additions [ours], there to stop us reporting an artefact as science.**

| id | check | what it answers | how |
|---|---|---|---|
| S1 | Noise, flux, M1 shift | Did it remove noise without changing the signal? | Built. Off-line rms, M0 total ratio, M1 shift on the brightest 10% of pixels, against the channel width. |
| S2 **[ours]** | Invented structure | Does it draw structure into pure noise? | Run on line-free planes, count connected pixels above a fixed threshold. Known risk here: 08 v2 measured 22 to 39% invented blobs at a fixed threshold after its own detector wrongly returned 0% (RULES.md #8), so check the denominator. |
| S3 **[ours]** | Model agreement | Which features are robust? | Compare aug43, starlet, wavelet. A feature all three keep is more credible than one only a single model draws. |
| S4 **[ours]** | Pixel-scale robustness | Is the result an artefact of our resampling? | Rerun at P1, P2 and plus or minus 25%. If moments move by more than the model does, the resampling is the story. |
| S5 **[ours]** | Injection-recovery on real noise | The only test with a known answer on real noise | Inject one of our held-out clean cubes (holdout RunIDs 0002, 0025, 0026) into line-free exoALMA planes at several peak SNRs, denoise, score PSNR and M0/M1/M2 with the project's own protocol so numbers compare with RULES.md #6. Doubles as a degradation curve on real noise. Limit: injected signal skips CLEAN, so it tests robustness to real noise statistics, not to CLEAN artefacts. |
| S6 | Kinematics | Are the velocity features kept? | Keplerian residual map of raw against denoised: correlation of the two, and residual amplitude ratio. Then compare with the published exoALMA kinematic analyses (Izquierdo et al. 2024, the discminer analysis cited in exoALMA I, and the companion papers). |
| S7 optional | Independent reference | Is the denoised cube closer to something real? | exoALMA also ships a 0.3" set at about 0.25 K noise. Smooth the fiducial and the denoised cube to 0.3" and compare each with it. Needs that dataset's DOI, which Jason did not point to. |

### Provisional thresholds

Placeholders anchored to exoALMA's quoted velocity precision of about 10 m/s, to be **revised after the first
run**, not treated as pass marks: median |M1 shift| on bright pixels of at most 20 m/s and 95th percentile at most
50 m/s; M0 total ratio within 5% of 1; off-line rms reduced without invented structure (S2 near the raw rate).
If the first run lands far outside these, the first suspect is preprocessing (pixel scale, normalization), not
the model.

---

## 7. Sequence and decision gates

| when | do | gate |
|---|---|---|
| Sep 25 to 26 | Push `alma-validation`. Run notebook 15 on MWC 758 **13CO** with `winner_aug_seed43` only. B1, B4. | **Gate A:** are S1 numbers sane? If not, debug preprocessing before adding anything. |
| Sep 27 to 28 | Fix what Gate A shows. B2, B3, B5. Add 12CO. | |
| Sep 29 | Lightning talk. Use one MWC 758 figure if Gate A passed. | |
| Oct 1 to 3 | Add starlet and wavelet. S2, S3, B7. | **Gate B:** does any model invent structure (S2)? If so, that is a headline negative result. |
| Oct 4 to 7 | S5 injection-recovery (B8) and S4 multi-scale (B9). | **Gate C:** do S5 numbers on real noise match the synthetic-noise tables? If not, the real-noise gap is the finding. |
| Oct 8 to 14 | HD 135344B and HD 143006, then LkCa 15 or V4046 Sgr. DSHARP CO cubes (0.35 km/s channels, so a coarser spectral domain). | |
| Oct 15 to 20 | S6 and the scientific reading. Open raw and denoised cubes in DS9 side by side. | **Gate D:** is there an insight worth a blog section, positive or negative? |
| Oct 23 to 30 | Write up (PLAN.md). | |

Kaggle cost is small: about 5 minutes and a 3 minute download per disk and line at 256 px on a T4 (an estimate,
first run measures it). The constraint is engineering and interpretation, not GPU quota.

---

## 8. Risks

| risk | effect | mitigation |
|---|---|---|
| CLEAN noise is out of distribution | Model treats correlated noise as signal, or ignores it | S2, S5 |
| Resampling artefacts | Results driven by our interpolation | S4 |
| Disk larger than the field | Emission cut off, edge effects in moments | B3, per-disk field |
| Line-free planes assumed, not checked | Wrong noise estimate, wrong flux | B5 |
| 12CO foreground absorption | Negative features the models never saw | 13CO first; 12CO flagged when used |
| Only one stellar mass in training | Velocity field scaling not learned | S6, and say so |
| Per-channel min-max on a noisy real channel | Scale set by noise extremes | Same as training; watch S1 |
| Dataverse fetch fails on Kaggle | Notebook stops | Attach the cubes as a Dataset instead (notebook 15 supports it) |
| Any number quoted without its metric | RULES.md #6 | Name the metric on every figure |

---

## 9. Viewing in DS9

Jason: DS9 is fine, *"it's not the most powerful, but it's probably the most straightforward."* Outputs to open
(after B4): the raw cube, each `<label>_denoised.fits`, and the moment maps. Suggested:

```
ds9 raw.fits aug43_denoised.fits starlet_denoised.fits -tile column -lock frame wcs -lock scale yes -lock slice image
```

Step through channels with the slice control and watch whether structure appears that the raw cube does not
support, and whether the line-free planes stay empty.

---

## 10. Deliverables

- Notebook 15 run archived per RULES.md #10 (`results/15-alma-real-cube-inference/v<N>_.../`), with `report.txt`,
  figures, denoised FITS, README naming what was and was not checked.
- One figure per disk: raw and denoised M0, M1, M2 and the residual.
- S5 table, and the S2 and S4 results, including negative ones.
- A blog section: what the models do to real kinematics, told with the negative results.
- Every entry in `results/PROGRESS.md`, and every number with its metric named (RULES.md #6).

---

## 11. Open questions

1. Confirm the disk order (MWC 758, HD 135344B, HD 143006, then LkCa 15 or V4046 Sgr), or does Jason have others?
2. Which two or three models are final? Depends on the 320/480 moment scoring (PLAN.md, Oct 1 to 7).
3. Is a 0.35 km/s DSHARP CO cube acceptable given the models trained on 0.1 km/s channels?
4. Is the 0.3" exoALMA set worth fetching for S7?
5. Does Jason want P1 or P2 reported as the headline, or both?
