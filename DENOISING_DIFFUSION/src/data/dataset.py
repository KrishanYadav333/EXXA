"""
PyTorch Dataset class for astronomical observations.

Extracted and refactored from GSOC_2025_EXXA_Main.ipynb.
"""

import torch
import torch.utils.data
import numpy as np
import random
from typing import Tuple, List, Optional, Callable


class AstroDataset(torch.utils.data.Dataset):
    """
    Dataset for loading astronomical images with paired noisy/clean observations.
    
    This dataset handles both .npy files and FITS files, supports random patch
    extraction for memory efficiency, and applies transformations for training.
    
    Args:
        input_array: Noisy input images, shape (N, H, W) or (N, C, H, W)
        gt_array: Clean ground truth images, shape (N, H, W) or (N, C, H, W)
        patch_size: Size of random patches to extract (e.g., 64)
        n: Number of random patches to extract per image
        transforms: torchvision transforms to apply
        parse_patches: If True, extract random patches. If False, resize full image.
    
    Returns:
        Tuple of (image_tensor, image_id) where:
        - image_tensor: shape (2, C, H, W) with [input, gt] concatenated on channel dim
        - image_id: string identifier for the image
    
    Example:
        >>> dirty = np.load("dirty.npy")  # Shape: (1000, 64, 64)
        >>> clean = np.load("clean.npy")  # Shape: (1000, 64, 64)
        >>> dataset = AstroDataset(dirty, clean, patch_size=64, n=4)
        >>> img, img_id = dataset[0]
        >>> print(img.shape)  # torch.Size([4, 2, 64, 64])
    """
    
    def __init__(
        self,
        input_array: np.ndarray,
        gt_array: np.ndarray,
        patch_size: int,
        n: int,
        transforms: Optional[Callable] = None,
        parse_patches: bool = True,
    ):
        super().__init__()
        assert input_array.shape == gt_array.shape, (
            f"Input and GT arrays must have same shape. "
            f"Got {input_array.shape} and {gt_array.shape}"
        )
        
        self.input_array = input_array  # Shape: (N, H, W) or (N, C, H, W)
        self.gt_array = gt_array
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        
    @staticmethod
    def get_params(
        h: int, w: int, output_size: Tuple[int, int], n: int
    ) -> Tuple[List[int], List[int], int, int]:
        """
        Generate random crop parameters for patch extraction.
        
        Args:
            h: Height of image
            w: Width of image
            output_size: (patch_height, patch_width) tuple
            n: Number of patches to extract
            
        Returns:
            (i_list, j_list, th, tw) where:
            - i_list: List of top-left y-coordinates
            - j_list: List of top-left x-coordinates
            - th: Patch height
            - tw: Patch width
        """
        th, tw = output_size
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw
    
    @staticmethod
    def n_random_crops(
        img: np.ndarray, i_list: List[int], j_list: List[int], h: int, w: int
    ) -> List[np.ndarray]:
        """
        Extract multiple random crops from an image.
        
        Args:
            img: Input image array
            i_list: List of top-left y-coordinates
            j_list: List of top-left x-coordinates
            h: Crop height
            w: Crop width
            
        Returns:
            List of cropped image arrays
        """
        crops = [img[i : i + h, j : j + w] for i, j in zip(i_list, j_list)]
        return crops
    
    def get_images(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Load and process images at given index.
        
        Args:
            index: Index of image to load
            
        Returns:
            (image_tensor, image_id) tuple
        """
        # Handle both (N, H, W) and (N, C, H, W) formats
        input_img = self.input_array[index]  # Shape: (H, W) or (C, H, W)
        gt_img = self.gt_array[index]
        
        # If 3D (C, H, W), work with 2D (H, W) for now
        # TODO: Support multi-channel data properly
        if input_img.ndim == 3:
            input_img = input_img[0]  # Take first channel
            gt_img = gt_img[0]
            
        img_id = f"{index:05d}"
        
        if self.parse_patches:
            # Extract random patches
            h, w = input_img.shape
            i, j, ph, pw = self.get_params(
                h, w, (self.patch_size, self.patch_size), self.n
            )
            input_crops = self.n_random_crops(input_img, i, j, ph, pw)
            gt_crops = self.n_random_crops(gt_img, i, j, ph, pw)
            
            outputs = []
            for inp, gt in zip(input_crops, gt_crops):
                # Add channel dimension: (H, W) -> (1, H, W)
                inp_tensor = self.transforms(inp[np.newaxis, ...]) if self.transforms else torch.from_numpy(inp[np.newaxis, ...]).float()
                gt_tensor = self.transforms(gt[np.newaxis, ...]) if self.transforms else torch.from_numpy(gt[np.newaxis, ...]).float()
                
                # Concatenate input and gt on channel dim: (2, H, W)
                outputs.append(torch.cat([inp_tensor, gt_tensor], dim=0))
            
            # Stack patches: (n, 2, H, W)
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resize full image to multiples of 16 (required for U-Net)
            h, w = input_img.shape
            
            # Limit maximum size to 1024
            if h > w and h > 1024:
                w = int(np.ceil(w * 1024 / h))
                h = 1024
            elif w >= h and w > 1024:
                h = int(np.ceil(h * 1024 / w))
                w = 1024
            
            # Round to nearest multiple of 16
            h = int(16 * np.ceil(h / 16.0))
            w = int(16 * np.ceil(w / 16.0))
            
            # Resize images
            input_resized = torch.nn.functional.interpolate(
                torch.from_numpy(input_img[np.newaxis, np.newaxis, ...]).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            
            gt_resized = torch.nn.functional.interpolate(
                torch.from_numpy(gt_img[np.newaxis, np.newaxis, ...]).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            
            # Concatenate: (2, H, W)
            return torch.cat([input_resized, gt_resized], dim=0), img_id
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """Get item by index."""
        return self.get_images(index)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.input_array.shape[0]


def create_dataloaders(
    train_input: np.ndarray,
    train_gt: np.ndarray,
    val_input: np.ndarray,
    val_gt: np.ndarray,
    config: dict,
    parse_patches: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_input: Training noisy images
        train_gt: Training clean images
        val_input: Validation noisy images
        val_gt: Validation clean images
        config: Configuration dictionary with keys:
            - training.patch_n: Number of patches per image
            - data.image_size: Patch size
            - training.batch_size: Batch size
            - data.num_workers: Number of data loading workers
        parse_patches: Whether to extract patches or use full images
        
    Returns:
        (train_loader, val_loader) tuple
        
    Example:
        >>> dirty = np.load("dirty.npy")
        >>> clean = np.load("clean.npy")
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_val, y_train, y_val = train_test_split(
        ...     dirty, clean, test_size=0.2, random_state=42
        ... )
        >>> config = {"training": {"patch_n": 4, "batch_size": 4}, 
        ...           "data": {"image_size": 64, "num_workers": 2}}
        >>> train_loader, val_loader = create_dataloaders(
        ...     X_train, y_train, X_val, y_val, config
        ... )
    """
    import torchvision.transforms as transforms
    
    # Basic transforms (just convert to tensor)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float())
    ])
    
    # Create datasets
    train_dataset = AstroDataset(
        input_array=train_input,
        gt_array=train_gt,
        n=config["training"]["patch_n"],
        patch_size=config["data"]["image_size"],
        transforms=transform,
        parse_patches=parse_patches,
    )
    
    val_dataset = AstroDataset(
        input_array=val_input,
        gt_array=val_gt,
        n=config["training"]["patch_n"],
        patch_size=config["data"]["image_size"],
        transforms=transform,
        parse_patches=parse_patches,
    )
    
    # Adjust batch size if not using patches
    batch_size = config["training"]["batch_size"] if parse_patches else 1
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    return train_loader, val_loader
