"""
Wavelet Decomposition
=====================

This example demonstrates how to use the Decompose module for
Haar wavelet decomposition (analysis & synthesis) in 2D.
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt

# For downloading and handling the image
import requests
from io import BytesIO
from PIL import Image

# Wavelet classes for 2D
from splineops.decompose.wavelets.haar import HaarWavelets
from splineops.decompose.wavelets.splinewavelets import (
    Spline1Wavelets,
    Spline3Wavelets,
    Spline5Wavelets
)

# %%
# Load and Normalize a 2D Image
# -----------------------------
#
# Here, we load an example image from an online repository. 
# We convert it to grayscale in [0,1].

url = 'https://r0k.us/graphics/kodak/kodak/kodim07.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Convert to numpy float64
image_color = np.array(img, dtype=np.float64)

# Normalize to [0,1]
image_color /= 255.0

# Convert to grayscale using standard weights
image_gray = (
    image_color[:, :, 0] * 0.2989 +
    image_color[:, :, 1] * 0.5870 +
    image_color[:, :, 2] * 0.1140
)

ny, nx = image_gray.shape
print(f"Downloaded image shape = {ny} x {nx}")

# %%
# 2D Wavelet Decomposition
# ------------------------
#
# We demonstrate wavelet decomposition (analysis) and reconstruction (synthesis)
# using 2D Haar wavelets on a grayscale image.

haar2d = HaarWavelets(scales=3)
coeffs = haar2d.analysis(image_gray)
recon_haar = haar2d.synthesis(coeffs)
err_haar = recon_haar - image_gray
max_err_haar = np.abs(err_haar).max()

print("[Wavelets 2D Haar Test]")
print(f"Max error after 3-scale decomposition: {max_err_haar}")

# Helper function for visualization
def pyramid_with_quadrant_embedding_levels(wavelet, inp, num_levels):
    """
    Perform multi-scale wavelet analysis in-place so that at each level the
    new coarse approximation is stored in the quadrant corresponding to the
    previous level's coarse region.
    
    Parameters
    ----------
    wavelet : AbstractWavelets instance
        A wavelet instance (e.g., HaarWavelets) with the desired number of scales.
    inp : np.ndarray
        Input 2D array (e.g., grayscale image).
    num_levels : int
        The number of decomposition levels to perform.
        
    Returns
    -------
    coeffs : np.ndarray
        Final coefficient array (same size as inp) with the pyramid layout.
    """
    out = np.copy(inp)
    ny, nx = out.shape[:2]
    
    for level in range(num_levels):
        # Process the current top-left subarray
        sub = out[:ny, :nx]
        sub_out = wavelet.analysis1(sub)
        out[:ny, :nx] = sub_out
        
        # Update region size for next level (halve each dimension)
        nx = max(1, nx // 2)
        ny = max(1, ny // 2)
        
    return out

# %%
# 1-Level Decomposition
# ~~~~~~~~~~~~~~~~~~~~~

wavelet1 = HaarWavelets(scales=1)
coeffs1 = pyramid_with_quadrant_embedding_levels(wavelet1, image_gray, 1)
absmax1 = np.abs(coeffs1).max()

plt.figure(figsize=(8, 8))
plt.imshow(coeffs1, cmap='gray', vmin=-absmax1, vmax=absmax1, interpolation='nearest')
plt.title("Haar 1-Level Decomposition", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# 2-Level Decomposition
# ~~~~~~~~~~~~~~~~~~~~~

wavelet2 = HaarWavelets(scales=2)
coeffs2 = pyramid_with_quadrant_embedding_levels(wavelet2, image_gray, 2)
absmax2 = np.abs(coeffs2).max()

plt.figure(figsize=(8, 8))
plt.imshow(coeffs2, cmap='gray', vmin=-absmax2, vmax=absmax2, interpolation='nearest')
plt.title("Haar 2-Level Decomposition", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# 3-Level Decomposition
# ~~~~~~~~~~~~~~~~~~~~~

wavelet3 = HaarWavelets(scales=3)
coeffs3 = pyramid_with_quadrant_embedding_levels(wavelet3, image_gray, 3)
absmax3 = np.abs(coeffs3).max()

plt.figure(figsize=(8, 8))
plt.imshow(coeffs3, cmap='gray', vmin=-absmax3, vmax=absmax3, interpolation='nearest')
plt.title("Haar 3-Level Decomposition", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
