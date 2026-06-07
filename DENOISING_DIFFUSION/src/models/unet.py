"""
U-Net architecture for diffusion-based denoising - FIXED VERSION.

Standard U-Net encoder-decoder with skip connections, 
conditioned on timestep via sinusoidal embeddings.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timestep conditioning."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timesteps
        
        Returns:
            (B, embedding_dim) sinusoidal embeddings
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        
        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 32):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        # Second conv
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
            t_emb: (B, time_emb_dim)
        
        Returns:
            (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = nn.SiLU()(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_emb_proj(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = nn.SiLU()(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Downsampling block (2x spatial resolution reduction)."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling block (2x spatial resolution increase)."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for diffusion-based denoising.
    
    Architecture:
    - Encoder: Downsampling path with residual blocks
    - Bottleneck: Residual blocks at lowest resolution
    - Decoder: Upsampling path with skip connections from encoder
    - Output: Noise prediction (same shape as input)
    
    Args:
        in_channels: Number of input channels (default 2 for noisy+clean)
        out_channels: Number of output channels (default 1 for noise prediction)
        base_channels: Number of channels in first layer (default 64)
        channel_multipliers: List of channel multipliers per level (default [1, 2, 4, 8])
        time_emb_dim: Dimension of time embeddings (default 128)
        num_res_blocks: Number of residual blocks per level (default 2)
        groups: Number of groups for group normalization (default 32)
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        groups: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Build encoder and decoder explicitly for clarity
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        
        # Encoder
        in_ch = base_channels
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            
            # Residual blocks at this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, groups))
                in_ch = out_ch
            
            self.enc_blocks.append(blocks)
            
            # Downsample to next level (skip on last)
            if mult != channel_multipliers[-1]:
                self.enc_downs.append(Downsample(out_ch))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.bottleneck.append(ResidualBlock(in_ch, in_ch, time_emb_dim, groups))
        
        # Decoder (reverse order)
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            # Skip connection reduces channel count by projecting concatenated features
            skip_in_ch = base_channels * channel_multipliers[len(channel_multipliers) - 1 - i]
            
            # After concatenation: in_ch (from decoder) + skip_in_ch
            concat_ch = in_ch + skip_in_ch
            
            # Residual blocks: first takes concatenated input
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                if j == 0:
                    # First block takes concatenated features
                    blocks.append(ResidualBlock(concat_ch, out_ch, time_emb_dim, groups))
                else:
                    # Subsequent blocks
                    blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim, groups))
            
            self.dec_blocks.append(blocks)
            in_ch = out_ch
            
            # Upsample (skip on last decoder level)
            if i < len(channel_multipliers) - 1:
                self.dec_ups.append(Upsample(out_ch))
        
        # Final output
        self.norm_out = nn.GroupNorm(groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) - input noisy image
            timesteps: (B,) - timestep indices
        
        Returns:
            (B, out_channels, H, W) - predicted noise
        """
        # Time embedding
        t_emb = self.time_emb(timesteps)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skips = []
        for level, (enc_blocks, down) in enumerate(zip(self.enc_blocks, self.enc_downs + [None])):
            # Process residual blocks
            for block in enc_blocks:
                h = block(h, t_emb)
            
            # Store skip connection before downsampling
            skips.append(h)
            
            # Downsample if not last level
            if down is not None:
                h = down(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb)
        
        # Decoder with skip connections (reverse order)
        for level, (dec_blocks, skip_h) in enumerate(zip(self.dec_blocks, reversed(skips))):
            # Upsample if not first decoder level
            if level > 0:
                h = self.dec_ups[level - 1](h)
            
            # Concatenate with skip connection
            h = torch.cat([h, skip_h], dim=1)
            
            # Process residual blocks
            for block in dec_blocks:
                h = block(h, t_emb)
        
        # Final output
        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        
        return h


def create_model(
    in_channels: int = 2,
    out_channels: int = 1,
    base_channels: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> UNet:
    """
    Create a U-Net model for denoising.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        device: Device to place model on
    
    Returns:
        UNet model on specified device
    """
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=[1, 2, 4, 8],
        time_emb_dim=128,
        num_res_blocks=2,
        groups=32,
    )
    
    return model.to(device)
