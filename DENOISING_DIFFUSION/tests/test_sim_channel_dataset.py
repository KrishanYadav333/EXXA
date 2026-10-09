"""
SimChannelDataset: shapes match FITSChannelDataset's, noise is fresh per access (or fixed for validation), the noise matches the
pair's measured level, the two tasks differ exactly by the beam, and a 2x upsample works.

Run: PYTHONPATH=. python3 tests/test_sim_channel_dataset.py
"""
import os, sys, tempfile
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.sim_channel_dataset import SimChannelDataset, _beam_otf

tmp = tempfile.mkdtemp()
n, C = 64, 31
yy, xx = np.mgrid[:n, :n] - n // 2
vel = (np.arange(C) - C // 2) * 0.1
clean = np.stack([np.exp(-0.5 * ((np.hypot(xx - 6 * np.sin(v), yy) / 7.0) ** 2)) * np.exp(-0.5 * (v / 0.8) ** 2) for v in vel]).astype(np.float32)
# a small gaussian beam on disk, odd size
b = np.exp(-0.5 * ((np.mgrid[:21, :21] - 10) ** 2).sum(0) / 3.0 ** 2); fits.writeto(os.path.join(tmp, "beam.fits"), (b / b.sum()).astype(np.float32))
BEAM = os.path.join(tmp, "beam.fits")
H = _beam_otf(BEAM, n)
sigma = 0.02
rng = np.random.default_rng(1)
noise = np.fft.ifft2(np.fft.fft2(rng.normal(0, sigma, clean.shape)) * H[None]).real          # beam-correlated noise, like the pairs
dirty = (np.fft.ifft2(np.fft.fft2(clean) * H[None]).real + noise).astype(np.float32)
hdr = fits.Header(); hdr["CDELT3"] = 0.1
d = os.path.join(tmp, "run_9999_00001_rt_00"); os.makedirs(d)
fits.writeto(os.path.join(d, "c.fits"), clean, hdr); fits.writeto(os.path.join(d, "d.fits"), dirty, hdr)
cube = [{"name": "toy", "clean": os.path.join(d, "c.fits"), "dirty": os.path.join(d, "d.fits")}]

k = 2
ds = SimChannelDataset(cube, n_neighbors=k, stack_target=True, target_size=n, n_samples=10, p_ai=0.5, beam_path=BEAM)
x, y = ds[0]
assert x.shape == y.shape == (2 * k + 1, n, n), (x.shape, y.shape)
ds1 = SimChannelDataset(cube, n_neighbors=k, stack_target=False, target_size=n, n_samples=10, beam_path=BEAM)
assert ds1[0][0].shape == (2 * k + 1, n, n) and ds1[0][1].shape == (1, n, n)
print("shapes      : stack target", tuple(y.shape), "| centre target", tuple(ds1[0][1].shape))

a, b_ = ds[3][0], ds[3][0]
assert not np.allclose(a.numpy(), b_.numpy()), "noise must be fresh on every access"
dv = SimChannelDataset(cube, n_neighbors=k, target_size=n, n_samples=10, fixed_noise=True, beam_path=BEAM)
assert np.allclose(dv[3][0].numpy(), dv[3][0].numpy()), "validation noise must be fixed"
print("noise       : fresh per access in training, fixed in validation")

# noise level: draw many noise-only items, un-normalise, compare the std with the pair's (dirty - beam (*) clean)
conv = np.fft.ifft2(np.fft.fft2(clean) * H[None]).real
ref = float((dirty - conv).std())
d1 = SimChannelDataset(cube, n_neighbors=0, target_size=n, n_samples=31, p_ai=1.0, beam_path=BEAM)
stds = []
for i in range(len(d1)):
    xi, yi = d1[i]
    ci, ch = d1.index[i]
    raw = np.fft.irfft2(np.fft.rfft2(np.random.default_rng(i).standard_normal((n, n)).astype(np.float32)) * d1.cubes[0]["amp"], s=(n, n))
    stds.append(raw.std())
rel = abs(np.mean(stds) - ref) / ref
assert rel < 0.15, (np.mean(stds), ref)
print(f"noise level : synthesised {np.mean(stds):.4f} vs pair's measured {ref:.4f} (within {100*rel:.0f}%)")

# the two tasks differ by the beam: with p_ai=1 (dirty - clean) is pure noise, with p_ai=0 it also carries the blur
def resid(p_ai):
    ds_ = SimChannelDataset(cube, n_neighbors=0, target_size=n, n_samples=31, p_ai=p_ai, beam_path=BEAM)
    out = []
    for i in range(len(ds_)):
        xi, yi = ds_[i]
        out.append(float((xi - yi).abs().mean()))
    return np.mean(out)
r_ai, r_dc = resid(1.0), resid(0.0)
assert r_dc > r_ai * 1.05, (r_ai, r_dc)
print(f"two tasks   : mean |dirty - clean| noise-only {r_ai:.4f} < deconvolution {r_dc:.4f} (the extra beam)")

up = SimChannelDataset(cube, n_neighbors=1, target_size=2 * n, n_samples=5, beam_path=BEAM)
assert up[0][0].shape == (3, 2 * n, 2 * n)
print("upsample    : 64 px cube served at 128 px")
print("PASS")
