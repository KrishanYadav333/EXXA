# Research log

Every literature search, paper read and the conclusion drawn from it goes here, newest first,
in the same session it happened (rule in `/CLAUDE.md`). `results/PROGRESS.md` records what we
ran; this file records what we read and what it means for the project.

Entry format: **date | topic**, then *Question*, *Read* (linked), *Findings* (what the source
actually says, with numbers), *Measured here* (our own check, if any), *Decision / next step*.
A claim from a paper is marked as the paper's; a claim we measured links to the script or
PROGRESS.md entry.

---

## 2026-10-10 | Speedie et al. 2024 MCFOST settings (`GIdisk_13C16O_mcfost.para`, Zenodo 11668694), read for the SPH-converter validation

**Read.** The release's MCFOST parameter file, `sg.setup`, `sg.discparams`, and the dump `sg_00005` itself (sarracen 1.3.1).
**Findings.** Executed as `mcfost ... -phantom sg_00005 -mol -casa -age 10Myr -keep_particles 0.9999 -scale_length_units 1.5`: the dump's lengths were scaled by 1.5 before ray tracing
(dump gas R 1-99 percentile 84-755 au, so 126-1130 au as imaged). Grid 2048 x 2048 over 6385 au, distance 155.9 pc, PA 236.7 deg; 13CO (`13co@xpol.dat`) at a constant abundance 1e-7;
v_turb 0.05; temperature computed by MCFOST (radiative equilibrium), not taken from the SPH internal energy. Dump: 991,216 gas particles, star 2.406 Msun, disc 0.694 Msun (q = 0.288),
SPH temperature 5-30 K (5-95 %).
**Used.** `validate_sph_speedie.py` applies the 1.5 length scale with the dynamically consistent velocity scaling (sqrt(1/1.5)) and reports the M1 slope against MCFOST, so a
different velocity convention in MCFOST shows up as a slope away from 1 rather than passing silently.

## 2026-10-10 (later) | Second search for public GI hydro data: three more Phantom releases, one with a mass-ratio sweep

**Question.** Mentors cannot supply more GI hydro sims ("find it from somewhere else"). The first search (entry below) found only Speedie et al. 2024. Is there more, searched
directly in Zenodo's API rather than through papers?

**Searched / read.** Zenodo API, `type=dataset`, queries `"gravitational instability" phantom`, `gravitationally unstable disc simulation`, `self-gravitating disc SPH`, `gravito-turbulent`,
`spiral arms self-gravitating protoplanetary simulation data`; file lists of every hit below via `/api/records/<id>`; arXiv abstracts [2608.13058](https://arxiv.org/abs/2608.13058) (Young et al. 2026) and
[2409.10765](https://arxiv.org/abs/2409.10765) (Rowther et al. 2024); search result for Longarini et al. 2026 ([2608.23537](https://arxiv.org/pdf/2608.23537), MNRAS doi 10.1093/mnras/stag1564).

**Findings.** All CC-BY-4.0, all Phantom SPH dumps (raw particles, readable with Splash or the `sarracen` Python package), none with line cubes: we would project them ourselves.

| Zenodo | paper | what | size | relevance |
|---|---|---|---|---|
| [22045440](https://zenodo.org/records/22045440) | Young, Rice, Booth, Meru 2026 (MNRAS accepted) | **63 runs**, M* 0.1-1 Msun, R_out 50/100/200 au, **mu = Md/M* 0.2-1.4**, improved radiative cooling (modified Lombardi) with irradiation | 28.7 GB, 76 MB-1.4 GB per run | **The mass-ratio sweep we lack.** Abstract: discs to mu ~0.4 do not fragment; large-scale arms only for M* <~ 0.3 Msun, otherwise flocculent compact spirals; compact 50 au discs may fragment. Protostellar (young, massive) discs, so the high-mu end is beyond Class II; mu 0.2-0.5 overlaps Speedie (0.29) and Terry (0.15-0.75). |
| [15375096](https://zenodo.org/records/15375096) | Rowther, Price, Pinte, Nealon, Meru, Alexander 2024 | 5 runs, Phantom + live MCFOST radiative transfer, isolated irradiated discs | 3.7 GB (`Simulations.zip` 1.6 GB) | GI short-lived (strong arms < 10 kyr without infall), weak spirals persist: weak-wiggle and decaying-GI examples, useful negatives. |
| [22041928](https://zenodo.org/records/22041928) | Longarini, Clarke, Booth, Leedham 2026 | 7 gravito-turbulent dumps (dust at St 0.01-100) + a hi-res set | 2.3 GB, 112 MB per dump | Gas is the same saturated gravito-turbulent disc in each, so ~1 disc, several snapshots. Longarini wrote GIGGLE, so this is the turbulence the analytic model idealises. |
| [15280731](https://zenodo.org/records/15280731) | Rice et al. 2025 | dust in GI discs | 196 GB single tar | Not practical (one archive, figure data only). Skipped. |

Not useful: [16750300](https://zenodo.org/records/16750300) (broken disc, not GI), [14930509](https://zenodo.org/records/14930509) (pkdgrav3 galaxy-scale). The Terry et al. set is still not public.

**Decision.** Build one SPH-to-cube projector (gas particles to a velocity channel cube on our grid, emission from the upper layer as a proxy for optically thick 12CO/13CO, then our beam and the measured
ALMA-like correlated noise), validate it against Speedie's own MCFOST cube (same dump, same geometry: if our M1 residual matches theirs the proxy is good enough), then run it over Young et al. (mass-ratio sweep),
Rowther et al. and Longarini et al. at many inclinations / PAs / distances. Hold out whole simulations, not geometries, for validation. Speedie stays held out for testing.

**Feasibility check (same day, local CPU, `results/gi-sph-sources/`).** Downloaded the smallest Young run (`M0.3_R50_Mu0.6`, 76 MB, dumps 00000/00010) and one Longarini dump (`st10_00100`, 112 MB); both read with
`sarracen` 1.3.1 (`read_phantom`, units au / Msun / G = 1). Young: 1 sink of 0.306 Msun, particle mass 3.6e-7, gas to R ~55 au, Md/M* = 0.57 at t = 4056 code units. Longarini: 958,829 gas + 193,618 dust particles,
star 1.0 Msun, gas to ~25 au. Upper-layer (z > 1 local scale height) mass-weighted v_los at i = 30 deg minus the projected azimuthal-mean v_phi(R) (`quick_vmap_young_longarini.png`): Young shows large-scale
two-armed spiral residuals well beyond +-0.3 km/s; Longarini shows many flocculent tightly wound arms at ~0.1 km/s. Both are the kinematic GI signal the training set lacks, at two different strengths.
Because these runs use G = 1 code units with no physical scale baked into the dynamics (Young's radiative cooling does set one, Longarini's beta-cooling does not), a dump can be rescaled in radius and
stellar mass (velocities by sqrt(M/R)) to match ALMA disc sizes; the Young runs should be used at or near their native scale.

## 2026-10-10 | Where else to get GI hydro simulations (not from the mentors, not exoALMA)

**Question.** The wiggle gap on real hydro disks is data coverage (PROGRESS.md 2026-10-10). Are there public GI hydro simulations or synthetic cubes beyond Speedie et al. 2024?

**Searched / read.** Zenodo search results; the records [5567886](https://zenodo.org/records/5567886) and [10205110](https://zenodo.org/records/10205110) (file lists via the API); the setup files of
[Zenodo 11668694](https://zenodo.org/records/11668694) (`sg.setup`, `sg.discparams`, `sg.in`, `GIdisk_13C16O_mcfost.para`); the arXiv abstract pages of Andrews et al. 2024 ([2405.19574](https://arxiv.org/abs/2405.19574)) and
Xu et al. 2025 ([2504.18751](https://arxiv.org/abs/2504.18751)); search summaries for Hall et al. 2020 ([2007.15686](https://arxiv.org/abs/2007.15686)) and Terry et al. 2022 ([2108.11282](https://arxiv.org/pdf/2108.11282)).

**Findings.**
- Public GI hydro DATA is scarce. Speedie et al. 2024 is the only release found with a GI disk, its Keplerian twin, synthetic ALMA cubes AND the SPH dump. Zenodo 5567886 ("Warping away gravitational instabilities") holds only a
  poster (10 MB, no simulation data). Andrews et al. 2024 (injection-recovery of self-gravity in MWC 480 kinematics, Md/M* to 0.1) and Xu et al. 2025 (3D GI with full radiation transport, ApJ) show no data release on their arXiv pages.
  For Hall et al. 2020 and Terry et al. 2022 no data-availability statement was found in the search results (not checked in the journals); Terry et al. ran seven SPH discs at q = 0.15-0.75 with MCFOST 13CO cubes, exactly the set
  the project lacks; contact jpterry@uga.edu.
- Zenodo 10205110 is Longarini's GIGGLE (`giggle_functions.py`, version0, 2023-11-25, CC-BY-4.0), the file we already have: there is no newer or fixed public version.
- The Speedie simulation: Phantom, 1,000,000 SPH particles, star 2.4 Msun, disc 0.7 Msun (q = 0.29), R 80-500 au, Sigma ~ R^-1, H/R 0.05 at 80 au, Gammie cooling beta = 10, adiabatic, 30 outer orbits, minimum Toomre Q 0.46.
  MCFOST: 2048 x 2048 px over 6385 au, inclination 23.2 deg, PA 236.7 deg, distance 155.9 pc. The prep script had written 162.9 pc and 23.5 deg into its headers: corrected in `speedie_prep.py`; the first kernel's output headers
  need the same correction (data unaffected; au/px and the fixed inclination used by the wiggle fit are).

**Decision / next step.**
1. Turn the one released SPH dump into many observations: project the particles at other inclinations, position angles and distances (our own line-of-sight projection, no MCFOST), so the model sees the same real GI
   velocity field in many geometries. More geometry, not more physics: still one q and one snapshot.
2. Ask the authors who already have the sets (Terry et al.: seven q values; Hall et al.; Speedie et al.) for their dumps or cubes. Best payoff per hour.
3. Running new SPH ourselves: rerunning the Speedie setup at other disc masses with Phantom (open source) is possible in principle, but 1M particles to 30 outer orbits is far beyond a 4-core Kaggle session; a low-resolution
   (~100-200k particle) campaign is possible but under-resolved for GI (Terry et al. kept h/H < 0.25). Only if 1 and 2 do not deliver.

---

## 2026-10-09 | How to get one denoiser that is best on PSNR, M0/M1/M2 and the GI wiggle (literature behind the training plan)

**Question.** Our models score well on PSNR and moments and lose the wiggle (PROGRESS.md 2026-10-08/09). What do the papers say about
(a) how big the real signal is, (b) architectures for spectral cubes, (c) learned imaging that stays faithful to the data, (d) training
without clean truth, (e) what to expect from learned denoisers on real cubes, (f) losses that keep fine structure?

**Read** (abstracts / search summaries plus the pages fetched; full texts not read unless stated):
- Speedie et al. 2024 *Nature* and its data release ([Zenodo 11668694](https://zenodo.org/records/11668694), README read in full); [ALMA news](https://almaobservatory.org/?p=79620).
- Hall et al. 2020 ([arXiv 2007.15686](https://arxiv.org/abs/2007.15686), abstract only; the fetch returned no resolution or sensitivity numbers, so none are quoted),
  Terry et al. 2022 ([2108.11282](https://arxiv.org/pdf/2108.11282), read 2026-10-08), Longarini et al. ([2108.11387](https://arxiv.org/pdf/2108.11387), summary only).
- QRNN3D, Wei et al. 2021 ([2003.04547](https://arxiv.org/pdf/2003.04547), summary).
- R2D2 / R3D3, Aghabiglou, Chu, Dabbech, Wiaux ([2403.05452](https://arxiv.org/html/2403.05452v3), method section fetched).
- Noise2Self, Batson and Royer 2019 ([1901.11365](https://arxiv.org/pdf/1901.11365), summary).
- Einig et al. 2023 ORION-B line-cube denoising ([2307.13009](https://arxiv.org/pdf/2307.13009), summary); benchmark of PCA / ICA / wavelet IST / 3D U-Net on spectral cubes
  ([2602.10893](https://arxiv.org/abs/2602.10893), abstract fetched).
- Focal / guided frequency losses ([2601.20878](https://arxiv.org/html/2601.20878), [2309.15563](https://ar5iv.labs.arxiv.org/html/2309.15563), summaries).

**Findings (the papers').**
1. *Size of the signal.* The simulated GI disk of Speedie et al. reproduces a southern minor-axis wiggle of 39.1 +- 1.8 m/s, "quantitatively similar" to AB Aur. exoALMA
   quotes line centres to ~10 m/s (research.md 2026-10-08). Terry et al.: A_wiggle is measurable at ALMA's spatial and spectral resolution and beam convolution barely changes it.
2. *Spectral structure.* QRNN3D: 3D convolution captures local spatio-spectral structure, a quasi-recurrent pooling carries global correlation along the spectrum, and the
   network accepts an arbitrary number of bands. Noise2Self: blind denoising needs only noise that is independent across the dimension you mask while the signal is correlated.
3. *Faithful learned imaging.* R2D2 builds the image as a sum of network outputs; every iteration the network sees the current estimate and the RESIDUAL DIRTY image
   (data minus forward model of the estimate), 15 networks (7-8 in R3D3, which interleaves approximate data-fidelity layers with U-Nets). Loss L1. The authors say the
   iteration removes the U-Net's over-smoothing and hallucination, but that is from visual inspection of a few examples, and they list residual structure near bright emission as a limitation.
4. *Real cubes.* On spectral cubes with peak SNR 2.5-8 ([2602.10893](https://arxiv.org/abs/2602.10893)): PCA / ICA barely help, wavelet iterative soft thresholding is
   robust at moderate SNR but suppresses emission at low SNR, a supervised 3D U-Net beats it in distribution but hallucinates at lower SNR and recovers ~80% of the flux on a
   morphology absent from training (IST: better). ORION-B autoencoder: the higher moments (centroid velocity, dispersion) are the ones denoising moves; it returns narrower
   line widths than ROHSA, especially at low SNR, so M2 must be checked separately.
5. *Losses.* Networks trained by gradient descent favour low frequencies (spectral bias); frequency-domain losses (focal, log-focal, guided) improved PSNR/SSIM in
   super-resolution and bioimage restoration. Caveat from the same literature: preserving high frequencies helps only when they are signal.

**Measured here that bears on it (all in PROGRESS.md 2026-10-08/09).**
- Noise, not the beam, kills the wiggle: rc_r noise-free beam 0.885 vs dirty 0.34 on the 5 hydro disks. 3D Wiener is at the LINEAR ceiling (oracle linear 0.923 vs 0.925 on SG v2) and does
  not beat dirty on rc_r on the hydro disks (0.349 vs 0.341). So the missing information has to come from a learned prior over disk structure.
- Line-centre error is 76-135 m/s median on the hydro disks for dirty and every method: larger than the real 39 m/s wiggle. A denoiser that does not get to ~10 m/s cannot
  recover the wiggle of a real disk at all, whatever its PSNR.
- On SG v2 (high SNR, headroom 0.043) every learned model is below dirty; the best learned model on unseen disks (sg_k3_fresh) had spectral context and SG-domain training.
- The earlier kinematic-loss arms (nb08 gamma 1 and 10) collapsed the wiggle (resid_r 0.23 / 0.37): a naive kinematic loss is NOT safe; the derivative losses in train_unified.py are
  different but untested.
- Window width: nb12 (old metric) 0.366 / 0.590 / 0.506 / 0.681 for +-0, 1, 2, 3 channels; kin_gamma0 (31 channels, +-1.5 km/s) leads nb16 on PSNR, M1, M2.

**Decision / next step** (the plan sent to the user 2026-10-09):
1. Keep the classical base (native-grid Wiener) and learn a bounded correction: it is robust where a pure network hallucinates (finding 4) and it starts at the linear ceiling.
2. Give the model the whole line (window +-1.5 km/s) and test it against +-0.3 km/s; consider a QRNN3D-style spectral recurrence if variable cube length matters.
3. Feed the data residual (dirty minus beam (*) estimate) and run 2 unrolled passes, R2D2 style, as a later experiment; use the residual's whiteness as a hallucination check.
4. Put the finger signal in the training data (hydro, Speedie GI disk, ~30 randomised analytic disks) and the Keplerian twin in the evaluation, so false fingers are measured.
5. Condition on the beam size in pixels (the U-Net already has a beam_dim path) because resampling to 256-600 px changes the beam in pixels.
6. Judge by rc_r, line-centre error in m/s, finger-scale coherence and M0/M1/M2; never select on PSNR (RULES.md #4). Test any derivative or frequency loss as an ablation, not as a default.
7. For real ALMA data without truth: a channel-wise blind-spot (Noise2Self) fine-tune is plausible because noise is independent between channels, but real cubes can have correlated adjacent channels,
   so first test it on our synthetic pairs, where the true error is known, to see whether the self-supervised loss tracks it.

---

## 2026-10-09 | Non-exoALMA data for training a wiggle-aware, ALMA-ready denoiser

**Question.** Jason does not want models trained on exoALMA data. What other public data can supply (a) realistic GI-wiggle disks with a
clean truth, and (b) real ALMA noise, for the unified model (PROGRESS.md 2026-10-09)?

**Read / searched.** Zenodo record 11668694 (README read in full); CDS J/A+A/666/A168 ReadMe; Zenodo 3928862 and 7430257 file lists; MAPS data
pages (via search results only; archive not reached).

**Findings (the sources').**
1. **Speedie et al. 2024, *Nature* 633, "Gravitational instability in a planet-forming disk", simulated data** ([Zenodo 11668694](https://zenodo.org/records/11668694),
   CC-BY-4.0, 26.2 GB). Phantom SPH GI disk (dump `sg_00005`, setup files) post-processed with MCFOST in 13CO 2-1 at AB Aur's inclination and PA;
   a **Keplerian counterpart from the same dump** (`-no_vr -no_vz -vphi_Kep`). Noise-free model cubes `GIdisk_13CO_lines(.contsub).fits` and
   `Kepdisk_...` (3.4 GB each), and **synthetic ALMA cubes** from `syndisk.synthetic_ALMA()` at the rms, beam and channel width of ALMA program
   2021.1.00690.S (robust 0.5 and 1.5; ~3.1 GB zip each), plus M0/M1/line-width maps, Keplerian residual maps, spiral spine annotations, and
   minor-axis PV-wiggle line centres (raw and detrended). The paper's simulated GI disk shows a minor-axis wiggle of 39.1 +- 1.8 m/s, similar
   to AB Aur. A 2025 preprint ([arXiv 2510.05601](https://arxiv.org/html/2510.05601)) argues infall can produce AB Aur's kinematics without GI;
   that concerns the real disk, not the simulation's own truth.
2. **Real non-exoALMA ALMA line cubes (no clean truth, usable for real-noise realisations and qualitative tests):**
   MAPS Large Program 2018.1.01055.L (IM Lup, GM Aur, AS 209, HD 163296, MWC 480; 12CO/13CO/C18O 2-1, JvM-corrected recommended)
   ([ALMA MAPS page](https://almascience.eso.org/alma-data/lp/MAPS), [MAPS I](https://arxiv.org/pdf/2109.06268), [MAPS II imaging](https://arxiv.org/pdf/2109.06188));
   Elias 2-27 CO and CN cubes, a GI candidate ([CDS J/A+A/666/A168](https://cdsarc.cds.unistra.fr/ftp/J/A+A/666/A168), Paneque-Carreno et al. 2022);
   RU Lup 12CO/13CO/C18O cubes and measurement sets ([Zenodo 3928862](https://zenodo.org/record/3928862));
   Law et al. CO isotopologue cubes ([Zenodo 7430257](https://zenodo.org/records/7430257)) for LkCa 15, HD 34282, DM Tau, Sz 91.

**Measured here.** Nothing downloaded yet. Note: LkCa 15, DM Tau and HD 34282 are exoALMA targets. Their Law et al. cubes are a different
program and not exoALMA data, but to stay clear of Jason's rule they are excluded unless the user says otherwise; Sz 91 from the same record
is fine.

**Decision / next step.**
1. Add the Speedie GI disk as a fourth training/eval kind ("realistic GI"): clean = model cube convolved with the synthetic cube's beam on its
   grid, dirty = syndisk-style beam-correlated noise at the matched rms, fresh every round. Hold out one of {robust 0.5, 1.5} or a spatial
   half for scoring, and score the **Keplerian counterpart as a false-positive test**: a denoiser must not invent fingers where there are none.
2. Build a real-noise bank from line-free channels of MAPS / Elias 2-27 / RU Lup / Sz 91 cubes, rescaled to each training cube's rms, so the
   model meets real ALMA noise statistics (JvM, channel correlation) without training on exoALMA.
3. Download on Kaggle inside the kernel (internet on), not locally: only the needed files (~13 GB of the 26 GB).

**Follow-up 2026-10-10 (headers read by HTTP range request, no download).**
- Model cubes (`{GI,Kep}disk_13CO_lines.contsub.fits`): 2048 x 2048 x 201, 0.02 arcsec/px (CDELT 5.555e-6 deg), 42 m/s channels centred on 0, JY/PIXEL, BMAJ = 0, i.e. no beam.
- Synthetic ALMA cubes (`lines.contsub.JvMcorr.pbcor.fits` inside the zips, deflate-compressed, 3.36 GB raw): 2048 x 2048 x 200, 42 m/s, first channel at -4179 m/s (HALF a channel off
  the model grid, so truth = the average of adjacent model channels), JY/BEAM, robust 0.5 beam 0.237 x 0.175 arcsec (11.8 x 8.8 px), BPA 1.2 deg. The systemic channel spans more than 600 px
  vertically, so the prep crops 1024, not 600.
- Zenodo throughput from this machine was ~0.2 MB/s, so the fetch runs in a CPU-only Kaggle kernel (`speedie-prep`); `experiments/speedie_prep.py` builds clean = model, cropped, beam-convolved with the
  synthetic cube's own beam, Jy/pixel to Jy/beam, resampled to its velocity grid, and measures the synthetic cube's real noise (spectral covariance, spatial correlation, residual against clean).
  Open until that kernel lands: whether syndisk's channels carry spectral noise correlation (our Wiener and training noise assume independent channels).

---

## 2026-10-09 | Longarini's semi-analytic GI-wiggle model (`giggle_functions.py`) and the mentor-meeting slides

**Question.** Jason/Cristiano's "new GI wiggle function": what does it compute, can we use it?

**Read.** `~/Downloads/giggle_functions.py` (762 lines, C. Longarini; copy in `references/longarini_giggle_functions.py`) and
`~/Downloads/Recovering the GI Wiggle.pdf` (our 2026-10-02 mentor-meeting deck). The model's papers are not read here; the code cites no equation numbers.
Presumably it implements the analytic GI-wiggle theory behind Longarini et al. 2021 and Hall et al. 2020 (not verified).

**Findings (the code's).** Rotation curve of a self-gravitating disk: sqrt(G ms/r + disc term), the disc term an elliptic-integral sum over a power-law
Sigma ~ r^p between rin and rout. Perturbations of an m-armed log spiral of pitch alpha: azimuthal amplitude ~ sqrt(G ms/r) beta^(-1/2) (md/ms) / 2, radial amplitude
2 m chi beta^(-1/2) q(r)^2 Omega r with the local Q = 1 mass ratio q(r); the observed M1 is (u_phi cos(phi) + u_r sin(phi)) sin(i). So the wiggle amplitude
scales with md/ms and beta^(-1/2), the same direction as Terry et al. 2022's A_wiggle(q); the disc term makes M1 non-Keplerian, which is the axisymmetric part our
Keplerian resid_r was scoring (PROGRESS.md 2026-10-08).

**Measured here** (PROGRESS.md 2026-10-09, `tests/test_analytic_gi.py`): runs after two import fixes; the Keplerian limit holds; the quadratic estimator recovers the
exact v_los to 0.1 m/s; with it six test disks were built and scored (3D Wiener rc_r 0.557 vs dirty 0.421 on them, but 0.349 vs 0.341 on hydro disks).

**Decision / next step.**
1. Use the model as a source of unlimited clean disks with exact velocity truth (training volume, unit tests for the metric), not as a replacement for hydro
   validation: its spiral is a single smooth mode.
2. The deck's slides 2, 4 and 6 quote Keplerian resid_r numbers (dirty 0.664, 3D Wiener 0.925 on SG v2, 0.628 vs 0.733 on five disks); these are superseded by the
   rotation-curve metric (PROGRESS.md 2026-10-08) and the deck needs a revision before the next mentor meeting.
3. Ask Cristiano for a fixed version, the paper the amplitudes come from, and the emission model he uses with it.

---

## 2026-10-08 | Terry et al. 2022: how the GI wiggle is measured in channel maps, and what our scorecard takes from it

**Question.** We are scoring GI wiggle recovery with a Keplerian-subtracted M1 correlation (resid_r). What does the paper that defines the
wiggle's amplitude and wavelength actually measure, so the scorecard measures the same thing?

**Read.** Terry, Hall, Longarini, Lodato, Toci, Veronesi, Paneque-Carreno, Pinte, *Constraining protoplanetary disc mass using the GI wiggle*,
MNRAS 2022 ([arXiv:2108.11282](https://arxiv.org/abs/2108.11282)), sections 2-4 and Figs. 1-6, Tables 1-3. (WebFetch returned the PDF as binary;
read through the PDF reader instead.)

**Findings (the paper's).**
- Phantom SPH, q = M_disc/M_* = 0.15-0.75, M_* = 0.6 Msun, R 10-300 au, Sigma ~ R^-1, beta-cooling beta = 15; MCFOST 13CO J=3-2 at **140 pc, i = 30 deg**,
  pixel 1.33 au, Hanning-smoothed to 0.03 km/s, convolved with a **0.11" x 0.07"** beam (Elias 2-27-like). The "GI wiggle" is a zig-zag of the emission
  spine in individual velocity channels, and "interlocking fingers" in Keplerian-subtracted M1 (Hall et al. 2020).
- **Extraction, Eq. 3.** In each row n across a channel's emission spine, take the brightest pixel y and its intensity I_n, then
  y_n = [I_n y + sum_{i=1..5} (I_i (y+i) + I_-i (y-i))] / [I_n + sum (I_i + I_-i)]: an intensity-weighted centroid over +/-5 px. y_n(R) is the signal (au).
- Then |A(lambda)|^2 by Fourier transform of y_n; **A_wiggle = max |A(lambda)|, lambda_wiggle its wavelength.** Only R_res <= R <= R_outer
  (R_outer = 95% of disc mass; R_res from h/H < 0.25), channels |v - v_sys| < 1 km/s, averaged over channels.
- Results: A_wiggle = (50 +/- 8) q - 5.4 au raw, (48 +/- 9) q - 6.6 convolved (R^2 0.88 / 0.87); lambda_wiggle = (129 +/- 20) q + 66 au. Convolution with
  the beam barely changes either. Example q = 0.5, systemic channel: lambda = 74 au, A = 18.7 au. Wavelength scales with H (k H ~ 1), amplitude with q.
- Moment-1 (intensity-weighted, their Eq. 4) with Keplerian subtracted shows the fingers for q = 0.25 and 0.75, much stronger at 0.75.

**Measured here** (PROGRESS.md 2026-10-08, `experiments/wiggle_scorecard.py`, `results/wiggle-scorecard/2026-10-08/`).
- Our SG v2 cube is this paper's configuration: q = 0.75 (`lines_13CO_q075_sg_00019.para`), i = 30 deg, 140 pc, 1.33 au/px; the fitted M_* is 0.59 Msun (truth 0.6).
- Eq. 3 implemented on both halves of the spine in every channel within 1 km/s of v_sys. Self-test: a 12 au, 80 au-wavelength sinusoid returns A = 10.6 au,
  lambda = 73.6 au (the Fourier grid is coarse on a 220 au track).
- The paper reads the wiggle off Keplerian-subtracted M1. On our massive synthetic disks the clean cube's Keplerian residual is dominated by an axisymmetric
  dipole (a q >= 0.25 disk does not rotate in a Keplerian way), which made our resid_r mostly a rotation-curve score. Subtracting the clean cube's own
  fitted v_phi(R) leaves the fingers; that rc_r is now the headline metric.
- Noise, not the beam, is what destroys the wiggle in the synthetic pairs (rc_r: noise-free beam 0.80-0.98, dirty 0.04-0.72). That agrees with the paper's finding
  that beam convolution barely changes A_wiggle and lambda_wiggle.

**Decision / next step.**
1. Score GI recovery with rc_r and rc_coh at 40-160 au (this paper's lambda range), plus Eq. 3 spine agreement (spine_r, A_ratio, lam_ratio); keep Keplerian resid_r
   only as a legacy column.
2. Report A_wiggle per method as a ratio to clean, not as a mass: our detrending and track lengths differ from the paper's, so the A(q) fit does not transfer.
3. Dirty cubes inflate A_wiggle 1.2-4.7x (noise moves the spine peak); any real-data wiggle amplitude needs denoising before Terry's A(q) relation can be used.
   That is a concrete scientific reason for this project's denoiser, stated in the paper's own metric.

---

## 2026-10-08 | exoALMA I (Teague et al. 2025): what the real data are, and a correction to our forward-model claim

**Question.** The user pasted *exoALMA I: Science Goals, Project Design and Data Products* ([arXiv:2504.18688](https://arxiv.org/abs/2504.18688),
ApJL 2025; the pasted text is the April 2025 draft, truncated at section 4.2.4). What does it say about the data we plan to run on, and does it change the plan?

**Read.** Teague et al. 2025 (this paper); it points to Loomis et al. (exoALMA II, imaging), Izquierdo et al. (exoALMA, discminer analysis) and
Zawadzki and Czekala (exoALMA IX, RML imaging), none re-read here.

**Findings (the paper's).**
- 15 disks, 12CO J=3-2, 13CO J=3-2, CS J=7-6 plus 330 GHz continuum. Spectral resolution 30.5 kHz (27 m/s), chosen to resolve the line FWHM about 10
  times. Channel spacing finer than 30.5 kHz gives covariant neighbouring channels (Loomis 2018).
- **Three image sets.** Fiducial: circular 0.15" beam by uv-taper, 100 m/s (12CO, 13CO) or 200 m/s (CS), rms about 1.5 K per channel. High resolution:
  beam 65 to 150 mas, mostly 200 m/s channels, rms about 7.5 K (a factor about 5 worse). Low resolution: circular 0.3", rms about 0.25 K in 100 m/s channels.
  Fiducial peak brightness for HD 135344B: 12CO up to about 80 K, 13CO about 60 K, CS about 20 K (their Fig. 3), so per-channel peak SNR is roughly 13 to 50 for the
  CO lines at 1.5 K, and 3 to 10 in the high-resolution set.
- Kinematics: line centres to about 10 m/s, planet-driven kinks up to about 100 m/s, wiggle-like GI perturbations resolved from moment maps and channel maps.
  Line centres are computed with the bettermoments quadratic fit; moment maps released: zeroth, peak, line centre, line width.
- Selection: inclination 5 to 60 degrees, gas radius at least 1", no envelope emission. **MWC 758**: A3, 156 pc, M* 1.40 Msun, i 19.4 deg, PA 240.3 deg,
  12CO 21.0 and 13CO 5.46 Jy km/s, substructure R, A, I, and a large non-Keplerian arc in the south-west (their Fig. 5). HD 135344B: rings, azimuthal asymmetry, cavity.
- RML imaging (exoALMA IX) is used as an independent check of CLEAN-based features; both are image-plane products built from the visibilities.

**Measured here.** Line-emission cube `run_0002_00560_rt_00` (clean and dirty): header beam 0.185 x 0.126" at 0.01"/px on **both** cubes; the clean channel's spatial power is
band-limited like a beam-smoothed image and the dirty-minus-clean noise is band-limited too (numbers in PROGRESS.md 2026-10-02, correction 2026-10-08). Channel spacing 0.1 km/s,
per-channel peak SNR about 29. SG v2: beam 0.160 x 0.100" in both headers at 0.0095"/px, channel spacing 33 m/s (close to the spectral resolution, which is consistent with
the 0.53 channel-to-channel noise correlation we measured; the part that did not decay with lag was a static pattern).

**Decision / consequence.**
1. **Retracted:** "models trained only on A = I line emission have the wrong forward model for real ALMA, and a CASA simobserve + tclean step is the only route to transfer."
   The line-emission pairs already contain a beam on both sides, a matching channel spacing and a matching SNR range; they are the fiducial-product regime. The remaining mismatches are
   beam shape (elliptical vs circular 0.15"), pixel scale and the synthetic noise. A simulated correction for those (re-smooth to a circular 0.15" beam, rescale the pixel grid) is cheap; a CASA
   forward model is not required for a first MWC 758 run.
2. **Where denoising could matter on real data.** At the fiducial SNR (13 to 50) the headroom is small on bright channels (our own headroom figure: model gain falls as the input gets cleaner);
   it is in the faint outer disk and in the **high-resolution set** (rms 7.5 K, peak SNR 3 to 10, beams down to 65 mas), which is where the exoALMA authors had to give up spectral sampling.
   That makes the high-resolution 12CO/13CO sets the best demonstration target, with MWC 758 first.
3. **Spec in physical units.** Features of interest are 10 to 100 m/s, i.e. 0.1 to 1 channel at 100 m/s spacing. A denoiser must not shift line centres by more than about 10 m/s (0.1 channel);
   the quadratic-centroid bias of each model in m/s is the natural acceptance test on real cubes, and it is cheap to add to nb16.
4. The paper cannot give ground truth: no clean cube exists for real disks, so real-data evaluation stays consistency checks plus injection of synthetic disks into real noise.
5. Not read here (text truncated): Loomis et al. (exoALMA II) for pixel size, the CLEAN stopping rule (6 sigma) and the JvM correction, which changes the noise statistics we would match.

---

## 2026-10-02 | How to recover a crisp, clean-like GI wiggle and disk (full pipeline plan)

**Project goal restated (user, 2026-10-02):** a denoising pipeline for ALMA disk cubes with good
GI-wiggle recovery and the best moment scores. It may be several models. Results should be as
close to the clean cubes as the data allow. Training is approved wherever needed.

**Question.** Why do our U-Nets blur the wiggle, and what combination of methods gets closest
to clean?

**Read.**
- Terry et al. 2022, *Constraining protoplanetary disc mass using the GI wiggle*, MNRAS 510, 1671
  ([paper](https://academic.oup.com/mnras/article/510/2/1671/6449005)).
- Longarini et al. 2021, *Analytical GI wiggle*, ApJL 920, L41
  ([paper](https://iopscience.iop.org/article/10.3847/2041-8213/ac2df6)).
- Speedie et al. 2024, *Gravitational instability in a planet-forming disk*, Nature
  ([paper](https://www.nature.com/articles/s41586-024-07877-0),
  [arXiv](https://arxiv.org/html/2409.02196v1)).
- Izquierdo et al., *The Disc Miner I* ([arXiv](https://arxiv.org/pdf/2104.09596)).
- Aghabiglou et al. 2024, *R2D2 DNN series for radio-interferometric imaging*
  ([arXiv](https://arxiv.org/abs/2403.05452), [robust version](https://arxiv.org/html/2503.02554),
  [code](https://github.com/basp-group/R2D2-RI)).
- Terris, Tang, Jackson, Wiaux, *AIRI plug-and-play*
  ([MNRAS 2025](https://academic.oup.com/mnras/article-pdf/537/2/1608/61622906/staf022.pdf)) and
  [HyperAIRI](https://iopscience.iop.org/article/10.3847/1538-4365/ae3149).
- Loomis et al. 2025, *exoALMA II: calibration and imaging pipeline*
  ([arXiv](https://arxiv.org/pdf/2504.19870)); *exoALMA IX: regularized maximum likelihood imaging
  of non-Keplerian features* ([arXiv](https://arxiv.org/html/2504.19111)).
- Dong, Roth, Schiele 2020, *Deep Wiener Deconvolution* (DWDN)
  ([NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/0b8aff0438617c055eb55f0ba5d226fa-Abstract.html)).
- Zhang, Van Gool, Timofte 2020, *USRNet*
  ([CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.html)).
- Zhang, Zuo, Zhang 2018, *FFDNet* ([paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/FFDNet.pdf)).
- Ohayon, Michaeli, Elad 2024, *Posterior-Mean Rectified Flow* ([arXiv](https://arxiv.org/abs/2410.00418)).
- Spatial-Spectral Transformer for HSI denoising ([arXiv](https://arxiv.org/pdf/2211.14090)),
  SSUMamba ([arXiv](https://arxiv.org/pdf/2405.01726)).
- ALMA spectral-cube denoising benchmark, 2026 ([arXiv](https://arxiv.org/html/2602.10893));
  ORION-B line-cube denoising ([arXiv](https://arxiv.org/pdf/2307.13009));
  3D ALMA source detection trained on simulated cubes ([arXiv](https://arxiv.org/pdf/2211.11462)).
- Revealing fine structure in protoplanetary disks with physics-constrained neural fields
  ([arXiv](https://arxiv.org/html/2509.03623)).

**Findings (from the sources).**
- GI wiggle (Terry 2022): zig-zag in channel maps, "interlocking fingers" in the Keplerian-subtracted
  M1; amplitude A = 50q - 5.4 (R2 0.88), wavelength = 129q + 66 (R2 0.89), q = disc-to-star mass
  ratio; GI perturbs v about 0.4 km/s against about 0.06 for VSI. Extracted from channels with
  |v - vsys| < 1 km/s, averaged; simulated data convolved to a 0.11 x 0.07 arcsec beam and
  0.03 km/s channels. Their conclusion: wavelength and amplitude are measurable from maximum
  resolution ALMA data now.
- Speedie 2024: line centres by the bettermoments quadratic fit; Keplerian background fitted with
  `eddy`; Savitzky-Golay smoothing along velocity (10 channels for 13CO, 42 m/s spacing; 3 for
  C18O, 84 m/s); high-pass filtering with a radius-dependent Gaussian kernel; wiggle shown in
  position-velocity space.
- exoALMA fiducial images: 0.15" restoring beam, 100 m/s channels for 12CO and 13CO, Briggs 0.5,
  CLEAN stopped at 6 sigma, optional JvM correction (Czekala 2021).
- R2D2 / AIRI (radio astronomy): the reconstruction is a series of networks, each taking the
  previous estimate and the data residual, so the output stays tied to the measurement. This is
  the reason these methods do not hallucinate like unconstrained generative samplers.
- DWDN / USRNet / FFDNet (image restoration): with a known blur kernel, explicit Wiener or
  half-quadratic data steps plus a learned refinement beat blind networks; a noise-level map as an
  input lets one model cover many noise levels.
- PMRF: sampling-based restoration reaches perfect perceptual quality at about twice the minimum
  MSE; the MMSE (regression) estimate is the blurry one. Crisp-looking output costs distortion
  unless the data constrain it.
- 2026 ALMA cube benchmark: a 3D U-Net reached RMSE ratios 0.18-0.20 against 0.5-0.6 for wavelet
  IST, but overestimated compact-source flux by about 10% and underperformed IST on real
  out-of-distribution low-SNR data.

**Measured here (all on our data).**
- SG v2 cube, wiggle resid_r vs clean: dirty 0.892, U-Net 0.805, Wiener (2D, beam) 0.900,
  static-noise removal + 3D Wiener 0.925, oracle linear ceiling 0.923, noise-free beam-only 0.920.
  Per-scale: the U-Net is worst exactly at 25-100 px (coherence 0.73-0.77 against dirty 0.85-0.94).
  `experiments/wiggle_wiener_deconv.py`, `experiments/wiggle_wiener3d.py`, PROGRESS.md 2026-10-02,
  figures `results/self-gravitating/wiggle_wiener_vs_unet.png`, `wiggle_3d_wiener.png`.
- SG dirty noise is 0.53 correlated between channels at every lag (a static pattern, 54% of the
  per-channel noise variance); the clean cube has none of it.
- Quantization of line centres onto channel centres, hypothesis refuted (dirty 0.299 in the central
  bin against 0.200 uniform; U-Net 0.274; clean 0.203).
- Isovelocity (channel) structure explains 11.9% of U-Net error variance, 4.0% of dirty's, 2.1% of
  3D Wiener's.
- Line-emission cube `run_0002_00560_rt_00`: noise independent between channels (lag corr
  about 0), no static part, beam-correlated spatially; no beam on the signal (A = I). 3D Wiener with
  a noise spectrum from a different cube: PSNR 29.36 -> 38.75, M1 corr 0.9812 -> 0.9895.
- Resolution: training at native 600 px has a measured ceiling of +0.029 on the wiggle
  (context.md, Phase J).

**Decision / next step.**
1. Stage 0 classical: continuum/static subtraction, then 3D Wiener (deconvolve the beam on SG and
   real data, denoise only on A = I line emission), as the input to every network.
2. Line emission: 3D spatial-spectral network (kin_gamma0 or 3D U-Net / SST / SSUMamba) predicting
   a correction to the Wiener cube, trained on fresh noise each batch.
3. SG and real ALMA: R2D2-style series with the data residual (dirty minus beam-convolved estimate)
   and a noise-level map in every stage; no sampling.
4. Moments: quadratic centroid for M1, noise-aware masks for M0/M2, optional refinement network.
5. Data: fresh noise per batch, physics-preserving augmentation, more clean GI simulations from
   Jason. (An earlier version of this item also asked for a CASA simobserve + tclean forward model
   "without which nothing transfers to MWC 758". Retracted 2026-10-08: our line-emission pairs already
   carry a 0.185 x 0.126" beam on both sides, see the exoALMA I entry above.)
6. Evaluation in nb16: per-scale coherence, Terry-style wavelength/amplitude, channel-map sheets,
   leave-one-out over all disks, selection by finger-scale coherence not PSNR.
7. Stop queuing native-600 work after the current nb14 session; spend the GPU on 2 and 3.

(Superseded 2026-10-08: these r values are Keplerian resid_r, mostly the axisymmetric rotation curve. On the
rotation-curve residual rc_r, Wiener 0.349 vs dirty 0.341. See the Terry et al. 2022 entry and PROGRESS.md 2026-10-08.)
Update (same day, 5 SG synth disks): 3D Wiener alpha=2, no training, mean wiggle r 0.733 vs dirty 0.628
(4/5 wins, 1 tie), PSNR +4 to +14 dB on every disk; alpha=1 fails on all 5; gains are small on the two hard
disks (9025: 0.334 -> 0.657, 9074: 0.245 -> 0.288). So the classical stage is a good input but not the
answer for the wiggle; the learned correction is still needed. PROGRESS.md 2026-10-02.

Open: the 5 line-emission holdouts not run with the wiggle metric yet; alpha selection rule not done. "Identical to clean" is limited below the beam scale (about 6 px on SG, oracle
coherence about 0.35 there); anything crisper than that is prior-driven and must be reported as such.

---
