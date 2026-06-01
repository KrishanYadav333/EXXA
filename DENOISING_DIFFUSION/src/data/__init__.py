"""
Data loading and preprocessing utilities for astronomical observations.

This module provides classes and functions for:
- Loading FITS files from ALMA/VLT observations
- Loading synthetic training data from simulations
- Data augmentation techniques (physics-aware)
- Train/val/test splitting utilities
- PyTorch Dataset and DataLoader creation
"""

from .dataset import AstroDataset, create_dataloaders
from .fits_loader import FITSLoader, load_fits_file
from .preprocessing import normalize_image, denormalize_image
from .augmentation import get_augmentation_transforms

__all__ = [
    "AstroDataset",
    "create_dataloaders",
    "FITSLoader",
    "load_fits_file",
    "normalize_image",
    "denormalize_image",
    "get_augmentation_transforms",
]
