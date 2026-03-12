"""
EXXA Denoising Package
======================

A machine learning pipeline for denoising astronomical observations 
of protoplanetary disks using diffusion models.

This package implements Denoising Diffusion Probabilistic Models (DDPM)
and Denoising Diffusion Implicit Models (DDIM) for removing noise from
ALMA and VLT observations more efficiently than traditional methods.

Modules:
--------
- data: Data loading, preprocessing, and augmentation utilities
- models: Diffusion model architectures (U-Net, DDPM, DDIM)
- training: Training loops, metrics, and experiment tracking
- utils: Helper functions and utilities

Author: GSoC 2026 Contributors
Project: ML4SCI EXXA - Denoising Astronomical Observations
"""

__version__ = "0.1.0"
__author__ = "EXXA Team"
__license__ = "MIT"

from . import data
from . import models
from . import training
from . import utils

__all__ = ["data", "models", "training", "utils"]
