"""
`denoise_cube` (notebook 05, section 6) must feed a 2.5D model exactly what the dataset fed
it during training. This runs the notebook's REAL source, not a copy, and compares its input
tensors against `FITSChannelDataset`'s own.

The failure this guards against is silent. `winner_beam` was scored for days with its beam
vector missing, because `UNet.forward` ignores `beam=None`, and the resulting M0 of -95.7%
was published as a finding before anyone noticed it was a missing input rather than a result.
A neighbour stack normalised the wrong way is the same shape of bug: the model still runs,
the numbers still look like numbers, and nothing raises.

Specifically, every neighbour must carry the CENTRE channel's (min, max). `denoise_cube`
computes a per-channel `norm` array for the k=0 path, and slicing that for neighbours would
give each its own scale, erasing the relative amplitude along velocity -- which is the entire
signal the spectral arms exist to use.

    PYTHONPATH=. python3 tests/test_denoise_cube_spectral.py
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits

from src.data.fits_cube_dataset import FITSChannelDataset, continuum_of, beam_features_of

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(os.path.dirname(HERE), "05-unet-line-emission.ipynb")
C, S, BS, CONT_N = 14, 32, 8, 5
failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def extract_denoise_cube():
    """The notebook's own definition, so this cannot drift from what actually runs."""
    nb = json.load(open(NB, encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        for i, line in enumerate(lines):
            if line.startswith("def denoise_cube"):
                j = next((n for n in range(i + 1, len(lines))
                          if lines[n] and not lines[n][0].isspace()), len(lines))
                return "\n".join(lines[i:j])
    return None


class Spy(torch.nn.Module):
    """Records what it is fed and returns the centre channel, so the caller still works."""
    in_channels, beam_dim = 3, 0

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x, t, b=None):
        if self.seen is None:
            self.seen = x.clone()
        return x[:, 1:2]


print("=" * 70)
print("denoise_cube feeds a 2.5D model what the dataset trained it on")
print("=" * 70)

src = extract_denoise_cube()
check("denoise_cube found in the notebook", src is not None)
if src is None:
    sys.exit(1)

with tempfile.TemporaryDirectory() as tmp:
    # Channel c is scaled by (c+1): a 14x amplitude span, so a scale bug cannot hide.
    rng = np.random.default_rng(0)
    clean = (rng.random((C, S, S)).astype(np.float32)
             * np.arange(1, C + 1, dtype=np.float32)[:, None, None])
    dirty = clean + rng.normal(0, 0.05, clean.shape).astype(np.float32)
    hdr = fits.Header({"BMAJ": 4.2e-5, "BMIN": 3.3e-5, "BPA": 16.9, "CDELT2": 0.01 / 3600})
    cp, dp = os.path.join(tmp, "c.fits"), os.path.join(tmp, "d.fits")
    fits.PrimaryHDU(clean, header=hdr).writeto(cp)
    fits.PrimaryHDU(dirty, header=hdr).writeto(dp)

    g = dict(np=np, torch=torch, F=F, fits=fits, continuum_of=continuum_of,
             beam_features_of=beam_features_of, CONTINUUM_N=CONT_N,
             TARGET_SIZE=S, BS=BS, device="cpu")
    exec(src, g)

    spy = Spy()
    g["denoise_cube"]({"folder": "t", "dirty": dp, "clean": cp}, spy)
    got = spy.seen
    check("a k=1 model is fed 3 channels", tuple(got.shape[1:]) == (3, S, S),
          str(tuple(got.shape)))

    ds = FITSChannelDataset(
        [(cp, dp)], channel_sampler_fn=lambda n_channels, seed: list(range(n_channels)),
        target_size=S, verbose=False, subtract_continuum=True, continuum_n=CONT_N,
        n_neighbors=1)

    n = min(BS, C)
    bad = [ch for ch in range(n) if not torch.allclose(got[ch], ds[ch][0], atol=1e-5)]
    check("inference stack matches the training stack on every channel", not bad,
          f"{len(bad)}/{n} channels differ" if bad else f"{n} channels identical")

    # The comparison has to be capable of failing, or it proves nothing (RULES.md #8).
    wrong = got.clone()
    for ch in range(wrong.shape[0]):
        for j in (0, 2):                       # renormalise each neighbour by ITS OWN range
            p = wrong[ch, j]
            span = p.max() - p.min()
            if span > 0:
                wrong[ch, j] = (p - p.min()) / span
    caught = sum(1 for ch in range(n) if not torch.allclose(wrong[ch], ds[ch][0], atol=1e-5))
    check("the check would catch per-neighbour normalisation", caught == n,
          f"{caught}/{n} would be flagged")

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
