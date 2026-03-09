#!/usr/bin/env python3
"""
EXXA Installation Test Script
Verifies that all core dependencies are properly installed
"""

import sys
import importlib
from typing import List, Tuple

def check_package(package_name: str, display_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and return its version."""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✓ {display_name}: {version}"
    except ImportError:
        return False, f"✗ {display_name}: NOT FOUND"

def main():
    print("=" * 50)
    print("EXXA Installation Test")
    print("=" * 50)
    print()
    
    # Core packages to test
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('PIL', 'Pillow'),
        ('astropy', 'AstroPy'),
        ('h5py', 'h5py'),
        ('wandb', 'Weights & Biases'),
    ]
    
    results = []
    all_passed = True
    
    print("Checking core packages:")
    print("-" * 50)
    
    for package, display in packages:
        success, message = check_package(package, display)
        results.append((success, message))
        print(message)
        if not success:
            all_passed = False
    
    print()
    print("-" * 50)
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA: Available")
            print(f"  - GPU Count: {gpu_count}")
            print(f"  - GPU 0: {gpu_name}")
        else:
            print(f"⚠ CUDA: Not available (CPU only)")
    except Exception as e:
        print(f"✗ CUDA Check failed: {e}")
    
    print()
    print("=" * 50)
    
    if all_passed:
        print("✓ All core packages installed successfully!")
        print()
        print("You're ready to start working with EXXA!")
        return 0
    else:
        print("✗ Some packages are missing.")
        print()
        print("Please run: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
