"""
Re-apply the 2026-09-25 fixes to 05-unet-line-emission.ipynb after Kaggle's auto-push overwrites it (RULES.md #2).

Kaggle pushes the whole notebook it ran, so a run started from cells older than the fix writes those old cells back over the fix. This
restores, idempotently (an already-applied edit is skipped, an edit whose anchor is gone raises):

  1. cell 18: `denoise_cube(..., size=)` and `_arm_size(name)`, so the 320/480/600 px arms are scored at the size they were trained at
     (they were scored at 256), and the resolution arms added to STALE_MOMENT_ARMS so their stored rows re-score;
  2. cell 20: the diagnosed arm at its own size;
  3. cell 14: the control arms `winner_hybrid_ft` and `winner_hybrid_p10_ft` (same source and budget as the loss arms, original loss).

Keeps Kaggle's outputs (they are the record of the run just archived) unless --clear-outputs is given.

    python3 DENOISING_DIFFUSION/tools/reapply_05_fixes.py 05-unet-line-emission.ipynb [--clear-outputs]
"""
import json
import sys

EDITS = [
    (18, "def denoise_cube(ho_entry, net):\n",
         "def denoise_cube(ho_entry, net, size=None):\n"),
    (18, "    with fits.open(ho_entry['dirty'], memmap=False) as hdul:\n        dirty_raw = np.ascontiguousarray(hdul[0].data).astype(np.float32)\n        beam_vec = beam_features_of(hdul[0].header)\n",
         "    # `size` is the pixel grid the ARM was trained on (see _arm_size). Until 2026-09-25 this resized every\n"
         "    # arm to TARGET_SIZE=256, so winner_aug_res320/res480 were scored on inputs at a scale they were\n"
         "    # never trained on, and their moment rows said nothing about less downsampling.\n"
         "    size = size or TARGET_SIZE\n"
         "    with fits.open(ho_entry['dirty'], memmap=False) as hdul:\n        dirty_raw = np.ascontiguousarray(hdul[0].data).astype(np.float32)\n        beam_vec = beam_features_of(hdul[0].header)\n"),
    (18, "            t256 = F.interpolate(t, (TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False)\n",
         "            t256 = F.interpolate(t, (size, size), mode='bilinear', align_corners=False)\n"),
    (18, "# clean/dirty moment maps once per cube -- identical for every checkpoint scored below\n",
         "def _arm_size(name):\n"
         "    \"\"\"Pixel grid an arm was trained on: 320/480/600 for the resolution views, else TARGET_SIZE.\"\"\"\n"
         "    return {'res320': RES_SIZES['res320'], 'res480': RES_SIZES['res480'],\n"
         "            'native600': NATIVE_SIZE}.get(CONFIGS[name][1], TARGET_SIZE)\n\n\n"
         "# clean/dirty moment maps once per cube -- identical for every checkpoint scored below\n"),
    (18, "            den, _ = denoise_cube(ho, net)\n            e = cache[ho['folder']]\n",
         "            den, _ = denoise_cube(ho, net, size=_arm_size(name))\n            e = cache[ho['folder']]\n"),
    (18, "STALE_MOMENT_ARMS = {'winner_beam'}\n",
         "# 2026-09-25: the resolution arms were scored at 256 instead of the size they were trained at (see\n"
         "# denoise_cube). Their stored moment rows are not a test of resolution, so they are re-scored too.\n"
         "STALE_MOMENT_ARMS = {'winner_beam', 'winner_aug_res320', 'winner_aug_res480', 'winner_aug_native600'}\n"),
    (20, "    den, _ = denoise_cube(ho, eval_net)\n",
         "    den, _ = denoise_cube(ho, eval_net, size=_arm_size(BEST_CONFIG))\n"),
    (14, "    'winner_gradient_ft':  (dict(WINNER, loss_name='gradient', init_from='sweep_winner_aug', min_epochs=30), 'full'),\n",
         "    'winner_gradient_ft':  (dict(WINNER, loss_name='gradient', init_from='sweep_winner_aug', min_epochs=30), 'full'),\n"
         "    # CONTROLS (2026-09-25). Every *_ft arm changes the loss AND continues a converged model for 30+\n"
         "    # epochs at 0.1x lr, so a gain could be the extra training, not the loss (RULES.md #4: one variable\n"
         "    # at a time). These continue the same source for the same budget with the ORIGINAL hybrid loss.\n"
         "    'winner_hybrid_ft':     (dict(WINNER, loss_name='hybrid', init_from='sweep_winner_aug', min_epochs=30), 'full'),\n"
         "    'winner_hybrid_p10_ft': (dict(WINNER, loss_name='hybrid', init_from='sweep_winner_p10', min_epochs=30), 'full'),\n"),
]


def apply(nb):
    done, skipped = 0, 0
    for ci, old, new in EDITS:
        src = ''.join(nb['cells'][ci]['source'])
        if new in src:
            skipped += 1
        elif src.count(old) == 1:
            nb['cells'][ci]['source'] = src.replace(old, new).splitlines(keepends=True)
            done += 1
        else:
            raise SystemExit(f"cell {ci}: anchor {old[:60]!r} found {src.count(old)}x and the fix is not present; "
                             f"the notebook has changed shape, apply this one by hand")
    return done, skipped


if __name__ == '__main__':
    path = sys.argv[1]
    raw = open(path).read()
    nb = json.loads(raw)
    fmt = (lambda o: json.dumps(o, separators=(',', ':'), ensure_ascii=False))
    assert fmt(nb) == raw, 'not in the minified Kaggle format; refusing to rewrite'
    done, skipped = apply(nb)
    if '--clear-outputs' in sys.argv:
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                c['outputs'] = []; c['execution_count'] = None
            c['metadata'] = {}
    open(path, 'w').write(fmt(nb))
    print(f'{path}: applied {done} edit(s), {skipped} already present')
