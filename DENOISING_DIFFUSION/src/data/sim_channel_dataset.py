"""
Channel dataset over SIMULATED cubes (hydro GI disks, analytic GI disks) with fresh noise on every access, in the two forms the
project needs (research.md / PROGRESS.md 2026-10-09):

  deconvolution task  dirty = beam (*) clean + noise      (how sg_synth and the analytic pairs were built: a second beam on top)
  noise-only task     dirty = clean + noise               (what a real ALMA CLEAN image is: the beam is already inside clean)

`p_ai` is the probability an item is drawn as the noise-only task. The noise is not Gaussian-white: it is drawn from each cube's OWN
measured noise power spectrum (the residual of its provided dirty/clean pair, after removing beam (*) clean), so it carries the same
beam-correlated texture the pair has. Channels are independent of each other, as in the pairs.

Returns the same (dirty, clean) tensors as `FITSChannelDataset`, with the same shared min-max normalisation (centre dirty channel
sets the scale for the whole stack and for the clean target), so the two can be concatenated.

Spectral window: neighbours sit every `step` channels, step = round(0.1 km/s / |CDELT3|), so a +-k window spans the same velocity
range on a 0.033 km/s cube as on a 0.1 km/s one. Cubes smaller than `target_size` (the 301 px hydro disks) are bilinear-upsampled
AFTER the noise is drawn, so their noise is smoother than a native-size cube's: a known limitation, not a bug.

`fixed_noise=True` (validation): the noise and the task are a deterministic function of the item index, so an early-stopping score is
comparable between epochs.
"""
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch.utils.data import Dataset

_BEAM_CACHE = {}


def _beam_otf(beam_path: str, n: int) -> np.ndarray:
    """FFT of the unit-gain recovered beam centred in an n x n image (complex64)."""
    key = (beam_path, n)
    if key not in _BEAM_CACHE:
        b = fits.getdata(beam_path).astype(np.float64)
        b = b / b.sum()
        pad = np.zeros((n, n))
        h, c = b.shape[0] // 2, n // 2
        pad[c - h:c + h + 1, c - h:c + h + 1] = b
        _BEAM_CACHE[key] = np.fft.fft2(np.fft.ifftshift(pad)).astype(np.complex64)
    return _BEAM_CACHE[key]


def _planes(path: str, chans) -> np.ndarray:
    with fits.open(path, memmap=True) as h:
        d = h[0].data
        return np.stack([np.asarray(d[c], dtype=np.float32) for c in chans])


class SimChannelDataset(Dataset):
    def __init__(self, cubes: List[dict], n_neighbors: int = 0, stack_target: bool = False, target_size: int = 600,
                 n_samples: int = 50, seed: int = 42, p_ai: float = 0.5, fixed_noise: bool = False,
                 beam_path: Optional[str] = None, amp_channels: int = 24):
        self.k, self.stack_target, self.target_size = int(n_neighbors), stack_target, int(target_size)
        self.p_ai, self.fixed_noise = float(p_ai), fixed_noise
        self.beam_path = beam_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                   "results", "self-gravitating", "dirty_beam_recovered_v2.fits")
        self.cubes, self.index = [], []
        rng = np.random.default_rng(seed)
        for ci, c in enumerate(cubes):
            with fits.open(c["clean"], memmap=True) as h:
                hdr, (C, n, _) = h[0].header, h[0].shape
            step = max(1, int(round(0.1 / abs(float(hdr["CDELT3"])))))
            H = _beam_otf(self.beam_path, n)
            idx = np.unique(np.linspace(0, C - 1, min(C, amp_channels), dtype=int))
            cl = _planes(c["clean"], idx)
            dr = _planes(c["dirty"], idx).astype(np.float64)
            conv = np.fft.ifft2(np.fft.fft2(cl) * H[None]).real
            r = dr - conv
            P = np.mean(np.abs(np.fft.rfft2(r - r.mean((1, 2), keepdims=True))) ** 2, axis=0)
            amp = (np.sqrt(P) / n).astype(np.float32)               # white N(0,1) (*) amp reproduces the pair's noise spectrum
            peak = np.array([float(np.max(_planes(c["clean"], [j]))) for j in range(0, C, max(1, C // 64))])
            w = np.interp(np.arange(C), np.arange(0, C, max(1, C // 64))[:len(peak)], peak)
            w = 0.1 + w / max(w.max(), 1e-30)
            chans = rng.choice(C, size=min(n_samples, C), replace=False, p=w / w.sum())
            self.cubes.append({"name": c.get("name", os.path.basename(os.path.dirname(c["clean"]))), "clean": c["clean"],
                               "C": C, "n": n, "step": step, "amp": amp, "H": H})
            self.index += [(ci, int(ch)) for ch in sorted(chans)]

    def __len__(self):
        return len(self.index)

    def _resize(self, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x))[None]
        if t.shape[-1] != self.target_size:
            t = F.interpolate(t, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return t[0].float()

    def __getitem__(self, i: int):
        ci, ch = self.index[i]
        cb = self.cubes[ci]
        rng = np.random.default_rng(i) if self.fixed_noise else np.random.default_rng()
        ai = (i % 2 == 0) if self.fixed_noise else bool(rng.random() < self.p_ai)
        chans = [min(max(ch + j * cb["step"], 0), cb["C"] - 1) for j in range(-self.k, self.k + 1)]
        clean = _planes(cb["clean"], chans)                                        # (2k+1, n, n)
        base = clean if ai else np.fft.ifft2(np.fft.fft2(clean) * cb["H"][None]).real.astype(np.float32)
        white = rng.standard_normal(clean.shape, dtype=np.float32)
        noise = np.fft.irfft2(np.fft.rfft2(white) * cb["amp"][None], s=(cb["n"], cb["n"])).astype(np.float32)
        dirty = base + noise
        tgt = clean if (self.stack_target and self.k > 0) else clean[self.k:self.k + 1]
        lo, hi = float(dirty[self.k].min()), float(dirty[self.k].max())           # shared min-max from the centre dirty channel
        if hi > lo:
            dirty, tgt = (dirty - lo) / (hi - lo), (tgt - lo) / (hi - lo)
        else:
            dirty, tgt = np.zeros_like(dirty), np.zeros_like(tgt)
        return self._resize(dirty), self._resize(tgt)
