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

def imshow_percentile(coeffs, pct=99, ax=None, title=None):
    """
    Symmetric percentile stretch about zero and display.

    Parameters
    ----------
    coeffs : np.ndarray
        2-D array of wavelet coefficients.
    pct : float, optional
        Percentile for clipping magnitude (0–100). 99 keeps the
        largest 1 % saturated; lower pct → more aggressive stretch.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; if None, uses plt.gca().
    title : str, optional
        A title for the axes.
    """
    if ax is None:
        ax = plt.gca()

    lim = np.percentile(np.abs(coeffs), pct)
    im = ax.imshow(
        coeffs,
        cmap="gray",
        vmin=-lim,
        vmax=+lim,
        interpolation="nearest",
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)
    return im

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
coeffs1  = pyramid_with_quadrant_embedding_levels(wavelet1, image_gray, 1)

plt.figure(figsize=(8, 8))
imshow_percentile(coeffs1, pct=98, title="Haar 1-Level Decomposition")
plt.tight_layout()
plt.show()


# %%
# 2-Level Decomposition
# ~~~~~~~~~~~~~~~~~~~~~

wavelet2 = HaarWavelets(scales=2)
coeffs2  = pyramid_with_quadrant_embedding_levels(wavelet2, image_gray, 2)

plt.figure(figsize=(8, 8))
imshow_percentile(coeffs2, pct=98, title="Haar 2-Level Decomposition")
plt.tight_layout()
plt.show()

# %%
# 3-Level Decomposition
# ~~~~~~~~~~~~~~~~~~~~~

wavelet3 = HaarWavelets(scales=3)
coeffs3  = pyramid_with_quadrant_embedding_levels(wavelet3, image_gray, 3)

plt.figure(figsize=(8, 8))
imshow_percentile(coeffs3, pct=99, title="Haar 3-Level Decomposition")
plt.tight_layout()
plt.show()
