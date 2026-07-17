# Speaker notes — Line-Emission Denoising (V7 → V9 → V12)

One block per slide in `presentation.html`, in order.

---

## 1. Cover

Quick framing: this is the line-emission U-Net track (separate from the continuum-image
autoencoder/VAE work), covering three Kaggle runs — V7, V9, V12. Goal of the talk: show
what changed each version, what the numbers say, and two open questions I want your call on.

---

## 2. Pipeline recap

- Channel-by-channel, not full-cube: each of the ~201 velocity channels is a separate
  training example. Full-image (no patching), downsampled to 256x256 for memory.
- Shared dirty-scale normalization: both dirty and clean are normalized by the *dirty*
  channel's own (min,max). Say why if asked: at inference time we don't know the clean
  scale — that's what we're predicting — so the clean target has to be posed in a scale
  we can invert later. This was a real bug we hit before V7: normalizing clean by its own
  scale silently destroyed Moment-0 (background floor mismatch, -6400% on M0).
- Linear output head, not sigmoid — shared-scale clean targets can exceed 1.0 since clean
  peaks can be bigger than the dirty max in the same channel.
- Continuum subtraction — this is Jason's suggestion (2026-06-27 mentor note): the
  first/last N channels are line-free (high velocity offset), so their mean is a per-channel
  continuum estimate. Subtract it before normalization -> isolates line emission -> should
  most help Moment-0 since continuum is a pedestal that dominates the raw intensity sum.
- Evaluation contract: always a genuinely held-out cube (never seen in train/val), denoise
  every channel, reassemble the full cube, compare M0/M1/M2 of clean vs dirty vs denoised.

---

## 3. Timeline

Walk it top to bottom, keep it short — this slide is a map, not the content:
- Pre-V7: the normalization fix + continuum subtraction added as an opt-in toggle so the
  original (no-continuum) notebook kept working.
- V7: first side-by-side run — continuum vs no-continuum in the same notebook, same seed.
- V9: doesn't add a feature, it's a controlled ablation — does the continuum window size
  (N) matter?
- V12: consolidation + rigor. Dropped the no-continuum path entirely (continuum wins are
  already established by V7), added the 5-cube holdout summary so we're not eyeballing one
  cube anymore, and fixed an import bug that would've broken Kaggle re-runs.

---

## 4. V7

- Headline: continuum subtraction improves both PSNR and SSIM, not just M0 — that's the
  useful finding since it means the transform isn't just a moment-map bookkeeping trick,
  the model itself reconstructs channels better.
- Checkpoint landed at epoch 28 (early-stopped-ish; loss curve shows it), val_loss 0.0032.
- Point at the moment_comparison image — that's the actual 3-way clean/dirty/denoised
  panel, the one that matters. moment_maps (2-row) is just the clean-vs-dirty sanity check
  before the model is even in the picture.
- Channel-100 diagnostic exists because we got burned once by a normalization bug that
  only showed up when you looked at physical value ranges, not just visual SSIM. Keep
  running it every version as a canary.

---

## 5. V9

- This is a controlled experiment: identical seed, epochs, architecture — only N changes.
  N=1 is Jason's literal original suggestion (just the very first/last channel); N=5 is
  what we'd been defaulting to.
- Result: N=1 wins on every metric (PSNR, SSIM, MSE) and even converges to a lower
  best_val loss.
- **Say this explicitly out loud, don't let it slide by**: this ablation result was never
  folded back into the main pipeline. V7 and V12 both still train with N=5. That's not a
  bug, it's an unresolved decision — ask Jason directly whether N=1 should become the new
  default, or whether there's a reason (robustness? fewer edge channels being noisy at
  N=1?) to prefer N=5 despite the worse numbers here.
- The V9 image panel (loss curve, moment maps, etc.) is from the *main* N=5 run, not the
  ablation's N=1 winner — say this so nobody thinks the pretty pictures are the winning
  config.

---

## 6. V12

- Two separate things happened here, don't conflate them:
  1. Science: Section 11 added — evaluate across all 5 held-out cubes, not just one, with
     mean±std. This is the first time we have an error bar on the improvement numbers.
  2. Plumbing: notebook re-imported `continuum_of` from the src module (previously only
     defined inline in the notebook), and the module didn't have that function — Kaggle
     re-runs would hit `ImportError`. Fixed by adding an array-in `continuum_of(cube, n)` to
     `fits_cube_dataset.py`, refactored the path-based version to call it. Purely a
     packaging fix, no numeric effect — this run's single-cube numbers are consistent with
     V7 modulo the variance discussed on the next slide.
- The Section 11 bar chart is the real payoff of V12: M0 +69.8%±15.2%, M1 +17.5%±7.8%,
  M2 +20.1%±14.3%. Big std on M0 and M2 — flag this, it's the same variance problem as
  slide 8, just visible across 5 cubes instead of 2 runs of 1 cube.

---

## 7. Cross-version PSNR/SSIM

- Two separate charts on purpose — never put PSNR (dB, ~30 scale) and SSIM (0-1 scale) on
  one dual axis, that's a standard chart mistake and actively misleads on magnitude.
- V9's bar is the main N=5 run for consistency with V7/V12, not its N=1 ablation winner —
  say this again, it bears repeating since it's the easiest thing to misread on this slide.
- Second table (same cube, V7 vs V12): the important read is that the *dirty* baseline
  values are bit-for-bit identical across both runs — same cube, same preprocessing — so
  the ~2pp drop in improvement is 100% attributable to training variance, not a data
  pipeline regression. That's the evidence for the next slide's first bullet.

---

## 8. Open items

Treat this as the actual ask-list for the meeting, not a wrap-up slide:
1. Run-to-run variance — same everything, epoch 28 both times, and still a 2pp swing.
   Need Jason's steer: is a multi-seed variance study (say, 3-5 seeds per config) worth
   the compute before we trust any more of these single-run numbers?
2. N=1 vs N=5 — decide and update the main pipeline, or write down why not.
3. Resolution ceiling — 256x256 model, 600x600 native. Not urgent, just flagging it's a
   ceiling on what "denoised" can mean here, in case it matters for the paper framing.
4. HOLDOUT_PICK — a good next step is finding a held-out cube with a visible non-Keplerian
   kink in clean M1 (the planet signature), to pair the quantitative Section-11 story with
   a qualitative "look, we recovered the actual planet signal" figure.

Close by asking directly: which of these 4 does Jason want prioritized before the next run.
