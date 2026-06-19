"""
src/training/diffusion.py
=========================
Conditional DDPM training / sampling, adapted from the GSoC main notebook
(`GSOC_2025_EXXA_Main.ipynb`, "Diffusion Algorithm" cell).

Lineage: https://github.com/ermongroup/ddim and https://github.com/bahjat-kawar/ddrm.

Differences from the notebook
-----------------------------
* Single-GPU (the ``nn.DataParallel`` wrapper is dropped).
* Single-channel astronomy data (the notebook's hard-coded 3-channel sampling
  paths are made channel-agnostic via ``config.data.channels``).
* Works directly with this repo's :class:`src.data.dataset.AstroDataset`, whose
  patch batches have shape ``(B, n_patches, 2, H, W)`` -> flattened to
  ``(B*n_patches, 2, H, W)`` with channel layout ``[dirty, clean]``.
* Adds per-epoch loss tracking, best-checkpoint saving, and a DDIM-based
  ``evaluate`` that reports PSNR / SSIM / MSE against the clean ground truth.

The model predicts the noise added to the *clean* channel, conditioned on the
*dirty* channel. Data are mapped to [-1, 1] via :func:`data_transform` before
noising (matching the notebook).
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.diffusion_unet import DiffusionUNet, default_diffusion_config


# --------------------------------------------------------------------------- #
# Data range helpers                                                          #
# --------------------------------------------------------------------------- #
def data_transform(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] -> [-1, 1]."""
    return 2.0 * x - 1.0


def inverse_data_transform(x: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] -> [0, 1] (clamped)."""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Beta schedule                                                              #
# --------------------------------------------------------------------------- #
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Beta schedule for the forward process (ddim/ddrm style)."""
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# --------------------------------------------------------------------------- #
# EMA                                                                        #
# --------------------------------------------------------------------------- #
class EMAHelper:
    """Exponential moving average of model parameters."""

    def __init__(self, mu: float = 0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module: nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module: nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# --------------------------------------------------------------------------- #
# Conditional noise-estimation loss                                          #
# --------------------------------------------------------------------------- #
def noise_estimation_loss(model, x0, t, e, b):
    """
    Conditional DDPM loss.

    Args:
        model: predicts noise from ``cat([cond, x_t], dim=1)`` and ``t``.
        x0: (B, 2, H, W) with channel 0 = dirty (cond), channel 1 = clean (gt), in [-1, 1].
        t:  (B,) timesteps.
        e:  (B, 1, H, W) sampled noise.
        b:  (T,) betas.

    Returns:
        Scalar loss (per-pixel squared error summed over CHW, averaged over batch).
    """
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 1:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()          # noise the clean channel
    output = model(torch.cat([x0[:, :1, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# --------------------------------------------------------------------------- #
# DDIM sampling                                                              #
# --------------------------------------------------------------------------- #
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, x_cond, seq, model, b, eta=0.0):
    """DDIM reverse process conditioned on ``x_cond`` (already in [-1, 1])."""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to("cpu"))
    return xs, x0_preds


# --------------------------------------------------------------------------- #
# Runner                                                                     #
# --------------------------------------------------------------------------- #
def _flatten_patches(x: torch.Tensor) -> torch.Tensor:
    """(B, n, 2, H, W) -> (B*n, 2, H, W); leave (B, 2, H, W) untouched."""
    return x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x


class DenoisingDiffusion:
    """
    Conditional DDPM trainer/sampler.

    Args:
        config: object with ``model``, ``data`` and ``diffusion`` sections
            (see :func:`src.models.diffusion_unet.default_diffusion_config`).
        device: torch device.
        lr: Adam learning rate.
        ema_rate: EMA decay (uses ``config.model.ema_rate`` if not given).
        checkpoint_path: where to save the best checkpoint.
        data_parallel: spread each batch across all visible CUDA devices via
            ``nn.DataParallel``. ``None`` (default) auto-enables it whenever more
            than one GPU is present (e.g. Kaggle's T4 x2); set ``False`` to force
            single-GPU. EMA and checkpoints always operate on the unwrapped model,
            so checkpoints are portable across single-/multi-GPU setups.
    """

    def __init__(
        self,
        config=None,
        device: Optional[str] = None,
        lr: float = 2e-5,
        ema_rate: Optional[float] = None,
        checkpoint_path: str = "results/checkpoints/diffusion_best.pth.tar",
        data_parallel: Optional[bool] = None,
    ):
        self.config = config or default_diffusion_config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path

        # Unwrapped model is the source of truth for params/EMA/checkpoints.
        self._core = DiffusionUNet(self.config).to(self.device)

        n_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if data_parallel is None:
            data_parallel = n_gpu > 1
        self.data_parallel = bool(data_parallel) and n_gpu > 1
        self.num_gpus = n_gpu if self.data_parallel else (1 if self.device.type == "cuda" else 0)

        # self.model is what we call forward on (DataParallel replicates self._core
        # per call from its current params, so EMA updates to _core take effect).
        self.model = nn.DataParallel(self._core) if self.data_parallel else self._core

        self.use_ema = bool(self.config.model.ema)
        if self.use_ema:
            self.ema_helper = EMAHelper(mu=ema_rate if ema_rate is not None else self.config.model.ema_rate)
            self.ema_helper.register(self._core)
        else:
            self.ema_helper = None

        self.optimizer = torch.optim.Adam(self._core.parameters(), lr=lr, eps=1e-8)

        betas = get_beta_schedule(
            beta_schedule=self.config.diffusion.beta_schedule,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        self.step = 0
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    # ----------------------------- training ------------------------------- #
    def _epoch_loss(self, loader, train: bool) -> float:
        self.model.train(train)
        total, count = 0.0, 0
        torch.set_grad_enabled(train)
        for x, _ in loader:
            x = _flatten_patches(x).to(self.device)
            n = x.size(0)
            x = data_transform(x)
            e = torch.randn_like(x[:, 1:, :, :])

            # antithetic timestep sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,), device=self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            loss = noise_estimation_loss(self.model, x, t, e, self.betas)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.step += 1
                if self.use_ema:
                    self.ema_helper.update(self._core)

            total += loss.item()
            count += 1
        torch.set_grad_enabled(True)
        return total / max(count, 1)

    def train(self, train_loader, val_loader=None, n_epochs: int = 30, log_every: int = 1, verbose: bool = True):
        if verbose:
            print("=" * 60)
            print("Conditional DDPM training")
            print(f"  params : {sum(p.numel() for p in self._core.parameters()):,}")
            gpu_note = f"DataParallel x{self.num_gpus}" if self.data_parallel else "single device"
            print(f"  device : {self.device} ({gpu_note}), timesteps: {self.num_timesteps}")
            print("=" * 60)

        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            t0 = time.time()
            train_loss = self._epoch_loss(train_loader, train=True)
            self.train_losses.append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._epoch_loss(val_loader, train=False)
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, best=True)

            if verbose and (epoch % log_every == 0):
                msg = f"[epoch {epoch + 1:3d}/{self.start_epoch + n_epochs}] train {train_loss:.4f}"
                if val_loss is not None:
                    flag = "  *best" if val_loss == self.best_val_loss else ""
                    msg += f" | val {val_loss:.4f}{flag}"
                msg += f"  ({time.time() - t0:.1f}s)"
                print(msg, flush=True)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

    # ----------------------------- sampling ------------------------------- #
    @torch.no_grad()
    def sample(self, x_cond01: torch.Tensor, sampling_timesteps: int = 25, use_ema: bool = True) -> torch.Tensor:
        """
        Denoise dirty observations via DDIM.

        Args:
            x_cond01: (B, 1, H, W) dirty condition in [0, 1].
            sampling_timesteps: number of DDIM steps.
            use_ema: temporarily swap in EMA weights for sampling.

        Returns:
            (B, 1, H, W) denoised estimate in [0, 1].
        """
        backup = None
        if use_ema and self.use_ema:
            backup = {n: p.data.clone() for n, p in self._core.named_parameters() if p.requires_grad}
            self.ema_helper.ema(self._core)

        self.model.eval()
        x_cond = data_transform(x_cond01.to(self.device))
        b, c, h, w = x_cond.shape
        x = torch.randn(b, c, h, w, device=self.device)

        skip = self.num_timesteps // sampling_timesteps
        seq = range(0, self.num_timesteps, skip)
        xs, _ = generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.0)
        out = inverse_data_transform(xs[-1].to(self.device))

        if backup is not None:
            for n, p in self._core.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])
        return out

    @torch.no_grad()
    def evaluate(self, val_loader, sampling_timesteps: int = 25, max_batches: Optional[int] = None,
                 use_ema: bool = True) -> dict:
        """Sample-denoise validation patches and report PSNR / SSIM / MSE in [0, 1]."""
        from pytorch_msssim import ssim as ssim_fn

        psnrs, ssims, mses = [], [], []
        for bi, (x, _) in enumerate(val_loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = _flatten_patches(x)
            dirty = x[:, 0:1, :, :].clamp(0, 1)
            clean = x[:, 1:2, :, :].clamp(0, 1).to(self.device)

            pred = self.sample(dirty, sampling_timesteps=sampling_timesteps, use_ema=use_ema)

            mse = torch.mean((pred - clean) ** 2, dim=(1, 2, 3))
            psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
            s = ssim_fn(pred, clean, data_range=1.0, size_average=False)

            mses.extend(mse.cpu().tolist())
            psnrs.extend(psnr.cpu().tolist())
            ssims.extend(s.cpu().tolist())

        return {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "mse": float(np.mean(mses)),
            "n": len(mses),
        }

    # --------------------------- checkpointing ---------------------------- #
    def save_checkpoint(self, epoch: int, best: bool = False):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "step": self.step,
            "state_dict": self._core.state_dict(),   # unwrapped -> no 'module.' prefix
            "optimizer": self.optimizer.state_dict(),
            "ema_helper": self.ema_helper.state_dict() if self.use_ema else None,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }, self.checkpoint_path)

    def load_checkpoint(self, path: Optional[str] = None, load_ema_into_model: bool = False):
        path = path or self.checkpoint_path
        # weights_only=False: our own checkpoint embeds the DotDict config object.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._core.load_state_dict(ckpt["state_dict"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.use_ema and ckpt.get("ema_helper") is not None:
            self.ema_helper.load_state_dict(ckpt["ema_helper"])
            if load_ema_into_model:
                self.ema_helper.ema(self._core)
        self.start_epoch = ckpt.get("epoch", 0)
        self.step = ckpt.get("step", 0)
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"=> loaded '{path}' (epoch {self.start_epoch}, step {self.step})")
        return self
