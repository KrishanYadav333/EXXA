import numpy as np
import sys
sys.path.insert(0, r'K:\Krishan\GSoc\EXXA')

from DENOISING_DIFFUSION.src.data.preprocessing import pad_to_multiple

# Create a test image
sample_image = np.random.randn(64, 64).astype(np.float32)
img = sample_image[:63, :67]  # (63, 67)

print(f"Input shape: {img.shape}")
print(f"Input dimensions: H={img.shape[0]}, W={img.shape[1]}")

padded, padding = pad_to_multiple(img, multiple=16)

print(f"\nOutput shape: {padded.shape}")
print(f"Expected shape: (64, 80)")
print(f"Padding tuple: {padding}")

# Calculate expected values
h, w = img.shape
pad_h = (16 - h % 16) % 16
pad_w = (16 - w % 16) % 16

print(f"\nExpected padding:")
print(f"  pad_h = (16 - {h} % 16) % 16 = {pad_h}")
print(f"  pad_w = (16 - {w} % 16) % 16 = {pad_w}")
print(f"  Expected H: {h} + {pad_h} = {h + pad_h}")
print(f"  Expected W: {w} + {pad_w} = {w + pad_w}")
