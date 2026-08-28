"""
Loss functions for EXXA denoising pipeline.

HybridLoss : alpha * MSE + beta * (1 - SSIM)
VAELoss    : alpha * MSE + beta * (1 - SSIM) + gamma * KL
"""

from typing import Optional

import torch
import torch.nn as nn
from pytorch_msssim import ssim


class HybridLoss(nn.Module):
    """
    Weighted combination of MSE loss and SSIM-based loss.

    Tanmay's ratio: total_loss = 0.8 * MSE + 0.2 * (1 - SSIM)
    MSE anchors pixel-level accuracy; SSIM guides structural fidelity.

    Args:
        alpha (float): Weight for MSE loss.  Default: 0.8
        beta  (float): Weight for SSIM loss. Default: 0.2
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_fn = nn.MSELoss()

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the hybrid loss.

        Args:
            output: Model prediction tensor  (B, C, H, W), range [0, 1]
            target: Ground-truth tensor      (B, C, H, W), range [0, 1]

        Returns:
            total_loss: alpha * mse_loss + beta * ssim_loss
            mse_loss:   Raw MSE component
            ssim_loss:  Raw SSIM component  (1 - SSIM)
        """
        mse_loss = self.mse_fn(output, target)
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)
        total_loss = self.alpha * mse_loss + self.beta * ssim_loss
        return total_loss, mse_loss, ssim_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    pred = torch.rand(2, 1, 64, 64)
    gt   = torch.rand(2, 1, 64, 64)

    # Tanmay's ratio: 0.8 MSE + 0.2 (1-SSIM)
    criterion = HybridLoss(alpha=0.8, beta=0.2)
    total, mse, ssim_l = criterion(pred, gt)

    print(f"Total loss : {total.item():.6f}  (0.8*MSE + 0.2*SSIM_loss)")
    print(f"MSE  loss  : {mse.item():.6f}")
    print(f"SSIM loss  : {ssim_l.item():.6f}  (1 - SSIM score)")


# ---------------------------------------------------------------------------
# VAE Loss
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    Combined loss for Variational Autoencoder denoising.

    total = alpha * MSE + beta * (1 - SSIM) + gamma * KL

    The KL divergence term regularises the latent space toward N(0, I).
    For convolutional latents (B, C, H_z, W_z) the KL is averaged per element.

    Args:
        alpha (float): MSE weight.         Default: 0.8
        beta  (float): SSIM loss weight.   Default: 0.2
        gamma (float): KL weight.          Default: 0.001
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta:  float = 0.2,
        gamma: float = 0.001,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.mse_fn = nn.MSELoss()

    def forward(
        self,
        output:  torch.Tensor,
        target:  torch.Tensor,
        mu:      torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            output  : Reconstructed image  (B, 1, H, W), range [0, 1]
            target  : Clean ground truth   (B, 1, H, W), range [0, 1]
            mu      : Latent mean          (B, C, H_z, W_z)
            log_var : Latent log-variance  (B, C, H_z, W_z)

        Returns:
            total_loss : alpha*MSE + beta*(1-SSIM) + gamma*KL
            mse_loss   : raw MSE component
            ssim_loss  : raw SSIM component  (1 - SSIM score)
            kl_loss    : KL divergence per element, averaged
        """
        mse_loss  = self.mse_fn(output, target)
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)

        # KL(N(mu, var) || N(0, 1)) = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

        total_loss = (
            self.alpha * mse_loss
            + self.beta  * ssim_loss
            + self.gamma * kl_loss
        )
        return total_loss, mse_loss, ssim_loss, kl_loss


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(42)
    pred    = torch.rand(2, 1, 64, 64)
    gt      = torch.rand(2, 1, 64, 64)
    mu_     = torch.randn(2, 128, 8, 8)
    log_v   = torch.randn(2, 128, 8, 8)

    vae_criterion = VAELoss(alpha=0.8, beta=0.2, gamma=0.001)
    total, mse, ssim_l, kl = vae_criterion(pred, gt, mu_, log_v)
    print("VAELoss sanity check:")
    print(f"  Total loss : {total.item():.6f}")
    print(f"  MSE   loss : {mse.item():.6f}")
    print(f"  SSIM  loss : {ssim_l.item():.6f}")
    print(f"  KL    loss : {kl.item():.6f}")



def spectral_moment1(x: torch.Tensor, velax: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Intensity-weighted mean velocity along the channel axis, differentiable.

    ``x`` is (B, C, H, W) with C the spectral axis; ``velax`` is (C,) in km/s. Returns
    (B, H, W).

    Not an unbiased M1 and does not need to be. It is computed identically on the prediction
    and on the target, so a bias from the finite channel window cancels in their difference.
    What it has to be is SENSITIVE to a shift of the line peak, which is exactly the quantity
    a per-channel denoiser perturbs.

    The denominator is clamped rather than masked: a spaxel with no line contributes a
    near-zero numerator and denominator, and clamping keeps the gradient finite there instead
    of producing NaNs that poison the whole batch.
    """
    w = x.clamp(min=0.0)                       # negative "weights" make the mean meaningless
    num = (w * velax.view(1, -1, 1, 1)).sum(dim=1)
    den = w.sum(dim=1).clamp(min=eps)
    return num / den


class KinematicLoss(nn.Module):
    """
    HybridLoss plus a penalty on velocity-field error.

    Motivation, measured (2026-08-28): the U-Net improves pixel metrics and M0 substantially
    while DEGRADING the GI wiggle, the kinematic diagnostic (residual correlation 0.804
    against 0.891 for doing nothing). It is optimising pixel accuracy, and the wiggle is a
    sub-channel velocity perturbation that per-channel smoothing shifts. Nothing in the
    objective ever asked it to preserve velocity structure.

    L = alpha*MSE + beta*(1-SSIM) + gamma*|M1(pred) - M1(target)|

    The M1 term needs several channels at once, which is why this only works with a dataset
    yielding channel stacks (``FITSChannelDataset(n_neighbors=k)``) and a model whose
    out_channels matches its in_channels.

    Args:
        alpha, beta: as HybridLoss.
        gamma: weight on the velocity term. Its scale differs from the pixel terms (km/s
            against normalised intensity), so this needs tuning rather than a default of 1.
        velax: (C,) channel velocities in km/s.
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, gamma: float = 1.0,
                 velax: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.mse_fn = nn.MSELoss()
        self.register_buffer("velax", velax if velax is not None else torch.zeros(1))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse_fn(output, target)
        # SSIM over the channel stack: treat channels as a batch of single-channel images so
        # the window is spatial, not spectral.
        B, C, H, W = output.shape
        ssim_loss = 1.0 - ssim(output.reshape(B * C, 1, H, W).clamp(0, 1),
                               target.reshape(B * C, 1, H, W).clamp(0, 1),
                               data_range=1.0, size_average=True)

        v = self.velax.to(output.device)
        kin_loss = (spectral_moment1(output, v) - spectral_moment1(target, v)).abs().mean()

        total = self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * kin_loss
        return total, mse_loss, ssim_loss, kin_loss
