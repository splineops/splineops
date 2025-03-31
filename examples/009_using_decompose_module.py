"""
Using decompose module
======================

This example demonstrates how to use the 'decompose' module for:

- Pyramid decomposition (reduce & expand) in 1D and 2D
- Wavelet decomposition (analysis & synthesis) in 2D:
  * Haar wavelets (2D)
  * Spline wavelets (2D) of orders 1,3,5

You can download this example as both a Python script and a Jupyter notebook.
"""

# %%
# Imports
# -------
#
# We import the necessary libraries and modules for demonstrating
# pyramid decomposition and wavelet analysis/synthesis.

import numpy as np
import matplotlib.pyplot as plt

# For downloading and handling the image
import requests
from io import BytesIO
from PIL import Image

# Pyramid decomposition utilities
from splineops.decompose.pyramid import (
    get_pyramid_filter,
    reduce_1d, expand_1d,
    reduce_2d, expand_2d
)

# Wavelet classes for 2D
from splineops.decompose.wavelets.haar import HaarWavelets
from splineops.decompose.wavelets.splinewavelets import (
    Spline1Wavelets,
    Spline3Wavelets,
    Spline5Wavelets
)

# %%
# 1D Pyramid Decomposition
# ------------------------
#
# We'll keep a simple 1D demonstration (length=10). Then we do a pyramid
# reduce-then-expand. This replicates the logic of a reference test.

x = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0, -6.0], 
             dtype=np.float64)

filter_name = "Centered Spline"
order = 3
g, h, is_centered = get_pyramid_filter(filter_name, order)

reduced = reduce_1d(x, g, is_centered)
expanded = expand_1d(reduced, h, is_centered)
error = expanded - x

print("[1D Pyramid Test]")
print(f"Filter: '{filter_name}' (order={order}), is_centered={is_centered}")
print("Input   x:", x)
print("Reduced   :", reduced)
print("Expanded  :", expanded)
print("Error     :", error)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
axs[0].plot(x, 'o-', label='Input')
axs[0].set_title("1D Input Signal")
axs[0].legend()

axs[1].plot(reduced, 'o--', color='r', label='Reduced')
axs[1].set_title("Reduced (Half-Size)")
axs[1].legend()

axs[2].plot(expanded, 'o--', color='g', label='Expanded')
axs[2].plot(x, 'o-', color='k', alpha=0.3, label='Original')
axs[2].set_title(f"Expanded vs Original (Error max={np.abs(error).max():.3g})")
axs[2].legend()

plt.tight_layout()
plt.show()

# %%
# Download & Prepare the 2D Image
# -------------------------------
#
# Instead of a synthetic 2D array, we download an external color image,
# convert it to grayscale, and convert intensities to the [0..1] range.

url = 'https://r0k.us/graphics/kodak/kodak/kodim07.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Convert to numpy float64
image_color = np.array(img, dtype=np.float64)

# Normalize to [0..1]
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
# 2D Pyramid Decomposition
# ------------------------
#
# Demonstrate pyramid reduce->expand on the grayscale image.

filter_name = "Spline"
order = 3
g, h, is_centered = get_pyramid_filter(filter_name, order)

reduced_2d = reduce_2d(image_gray, g, is_centered)
expanded_2d = expand_2d(reduced_2d, h, is_centered)
error_2d = expanded_2d - image_gray
max_err = np.abs(error_2d).max()

print("[2D Pyramid Test]")
print(f"Filter: '{filter_name}' (order={order}), is_centered={is_centered}")
print("Reduced shape:", reduced_2d.shape)
print("Expanded shape:", expanded_2d.shape)
print(f"Max error: {max_err}")

# %%
# Manual Composition of Pyramid Reductions into One Image
# -------------------------------------------------------
#
# We'll create a single large "canvas" array and copy each reduced image
# into a spiral layout. The pattern we use for up to 4 levels is:
#   - Level 0 (original): at the top
#   - Level 1: below Level 0, aligned to the right
#   - Level 2: to the left of Level 1, sharing a bottom edge with it
#   - Level 3: above Level 2, sharing a left edge with it
#
# For more levels, you can repeat this spiral pattern in additional steps.

def create_spiral_canvas(levels):
    """
    Arrange up to 4 images (levels) in a spiral-like layout:

      - Level 0: top-left (0,0)
      - Level 1: below Level 0, right-aligned
      - Level 2: to the left of Level 1, bottom-aligned with it
      - Level 3: above Level 2, left-aligned with it

    Returns the composed float64 canvas.
    """
    if len(levels) == 0:
        raise ValueError("No images given to compose.")
    
    shapes = [img.shape for img in levels]  # List of (height, width)
    coords = [(0, 0)] * len(levels)         # (top_row, left_col) for each level

    # --- Level 0 at (0,0)
    H0, W0 = shapes[0]
    coords[0] = (0, 0)

    # --- Level 1: below L0, right-aligned
    if len(levels) > 1:
        H1, W1 = shapes[1]
        coords[1] = (H0, W0 - W1)

    # --- Level 2: left of L1, bottom-aligned
    if len(levels) > 2:
        H2, W2 = shapes[2]
        top1, left1 = coords[1]
        coords[2] = (top1 + H1 - H2, left1 - W2)

    # --- Level 3: above L2, left-aligned with L2
    if len(levels) > 3:
        H3, W3 = shapes[3]
        top2, left2 = coords[2]
        coords[3] = (top2 - H3, left2)

    # --- Shift so no negative indices ---
    min_row = min(top for (top, left) in coords)
    min_col = min(left for (top, left) in coords)
    shift_r = -min_row if min_row < 0 else 0
    shift_c = -min_col if min_col < 0 else 0

    shifted_coords = []
    for (top, left) in coords:
        shifted_coords.append((top + shift_r, left + shift_c))

    # --- Compute canvas size ---
    max_row = 0
    max_col = 0
    for (top, left), (H, W) in zip(shifted_coords, shapes):
        bottom = top + H - 1
        right  = left + W - 1
        max_row = max(max_row, bottom)
        max_col = max(max_col, right)

    total_height = max_row + 1
    total_width  = max_col + 1

    # --- Create the canvas and copy each level in place ---
    canvas = np.zeros((total_height, total_width), dtype=np.float64)
    for (img, (top, left)) in zip(levels, shifted_coords):
        h, w = img.shape
        canvas[top:top+h, left:left+w] = img

    return canvas

# Let's produce multiple levels via pyramid reductions
# We'll generate up to 3 additional levels (for a total of 4).
num_reductions = 3
levels = [image_gray]
current = image_gray
for _ in range(num_reductions):
    current = reduce_2d(current, g, is_centered)
    levels.append(current)

# Compose them into a single spiral-like layout
spiral_image = create_spiral_canvas(levels)

# Plot the final single image
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(spiral_image, cmap='gray')
ax.set_title("Pyramid Reductions in a Single Spiral Canvas", fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()


# %%
# Haar Wavelets (2D)
# ------------------
#
# Next, demonstrate wavelet decomposition (analysis) and reconstruction (synthesis)
# using 2D Haar wavelets on the same grayscale image.

haar2d = HaarWavelets(scales=3)
coeffs = haar2d.analysis(image_gray)
recon_haar = haar2d.synthesis(coeffs)
err_haar = recon_haar - image_gray
max_err_haar = np.abs(err_haar).max()

print("[Wavelets 2D Haar Test]")
print(f"Max error after 3-scale decomposition: {max_err_haar}")

# %%
# Progressive Approximation Visualization from Haar Wavelet Decomposition
# -----------------------------------------------------------------------
#
# In the multi-scale Haar wavelet decomposition, the top-left region of the
# coefficients array corresponds to the coarse approximation at that level.
# Here we generate a separate plot for each level:
#   - Level 0: Original image.
#   - Level 1: Approximation after 1 level of decomposition.
#   - Level 2: Approximation after 2 levels.
#   - Level 3: Approximation after 3 levels.
#
# This follows the common Mallat decomposition approach used in PyWavelets.

# Assuming 'coeffs' was obtained earlier by:
#    haar2d = HaarWavelets(scales=3)
#    coeffs = haar2d.analysis(image_gray)
ny, nx = image_gray.shape

# Extract the coarse approximation at each level
approx_levels = [
    image_gray,                     # Level 0: Original image
    coeffs[:ny//2, :nx//2],          # Level 1: Approximation (size ny/2 x nx/2)
    coeffs[:ny//4, :nx//4],          # Level 2: Approximation (size ny/4 x nx/4)
    coeffs[:ny//8, :nx//8]           # Level 3: Approximation (size ny/8 x nx/8)
]

titles = [
    "Level 0: Original Image",
    "Level 1: Coarse Approximation",
    "Level 2: Coarse Approximation",
    "Level 3: Coarse Approximation"
]

# Create a separate plot for each level
for level, (im, title) in enumerate(zip(approx_levels, titles)):
    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap='gray', interpolation='nearest')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
