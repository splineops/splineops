# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_04_antialiasing.py
# sphinx_gallery_end_ignore

"""
Antialiasing
============

Interpolate 2D images with least-squares projection.
Compare them to *standard interpolation*. We compute SNR and MSE only on a
central region to exclude boundary artifacts.
"""

# %%
# Imports
# -------

import numpy as np

# sphinx_gallery_thumbnail_number = 4  # show fourth figure as thumbnail
import requests
from io import BytesIO
from PIL import Image

# %%
# Load and Normalize an Image
# ---------------------------
#
# Here, we load an example image from an online repository.
# We convert it to grayscale in [0, 1].

url = 'https://r0k.us/graphics/kodak/kodak/kodim14.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)

# Convert to [0..1]
input_image_normalized = data / 255.0

# Convert to grayscale via simple weighting
input_image_normalized = (
    input_image_normalized[:, :, 0] * 0.2989 +  # Red channel
    input_image_normalized[:, :, 1] * 0.5870 +  # Green channel
    input_image_normalized[:, :, 2] * 0.1140    # Blue channel
)

zoom_factors_2d = (0.25, 0.25)
border_fraction = 0.3

# Face-centered 64×64 ROI
ROI_SIZE_PX = 64
FACE_ROW, FACE_COL = 250, 445  # (row, col)

h_img, w_img = input_image_normalized.shape

# Top-left of the 64×64 box, clipped to stay inside the image
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))
roi_rect = (row_top, col_left, ROI_SIZE_PX, ROI_SIZE_PX)  # (r, c, h, w)

roi_kwargs = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,  # keeps height at 64 px (square ROI)
    grayscale=True,
    roi_xy=(row_top, col_left),           # top-left of the ROI
)

# %%
# Antialiasing check (A/B 2×2 corner pattern)
# -------------------------------------------
#
# Load image B (same resolution as A), convert both to grayscale in [0, 1],
# and build a synthetic image where, for every 2×2 tile starting at (0,0),
# the *top-left* pixel comes from A while the other three pixels come from B.
# This makes naive 2× downsampling-by-picking-corners recover A, whereas
# anti-aliased downsampling averages roughly 25% A and 75% B.

import matplotlib.pyplot as plt

# Reuse A from earlier: 'input_image_normalized'
url_b = 'https://r0k.us/graphics/kodak/kodak/kodim08.png'
response_b = requests.get(url_b)
img_b = Image.open(BytesIO(response_b.content))
data_b = np.array(img_b, dtype=np.float64)

# B to grayscale in [0,1]
img_b_gray = data_b / 255.0
img_b_gray = (
    img_b_gray[:, :, 0] * 0.2989 +  # R
    img_b_gray[:, :, 1] * 0.5870 +  # G
    img_b_gray[:, :, 2] * 0.1140    # B
)

# Sanity: shapes should match exactly
assert img_b_gray.shape == input_image_normalized.shape, "Images A and B must have identical shape."

# Build the A/B mixed image: A at each 2×2 block's top-left pixel
mixed_ab = img_b_gray.copy()
mixed_ab[0::2, 0::2] = input_image_normalized[0::2, 0::2]

# Show the result
plt.figure(figsize=(6, 6 * mixed_ab.shape[0] / mixed_ab.shape[1]))
plt.imshow(mixed_ab, cmap="gray", vmin=0.0, vmax=1.0)
plt.title("A/B 2×2 corner mix (A at block corners, B elsewhere)")
plt.axis("off")
plt.show()

# Optional: use a different corner by changing the strides below:
# top-right:    mixed_ab[0::2, 1::2] = input_image_normalized[0::2, 1::2]
# bottom-left:  mixed_ab[1::2, 0::2] = input_image_normalized[1::2, 0::2]
# bottom-right: mixed_ab[1::2, 1::2] = input_image_normalized[1::2, 1::2]

# %%
# Antialiasing check (B/A 2×2 corner pattern — opposite)
# ------------------------------------------------------
#
# Reuse A ('input_image_normalized') and B ('img_b_gray') from above.
# For every 2×2 tile starting at (0,0), we take the *top-left* pixel from B
# and the other three pixels from A. Naive 2× downsampling-by-picking-corners
# now recovers B, while anti-aliased downsampling blends ~25% B and ~75% A.

# Build the B/A mixed image: B at each 2×2 block's top-left pixel
mixed_ba = input_image_normalized.copy()
mixed_ba[0::2, 0::2] = img_b_gray[0::2, 0::2]

plt.figure(figsize=(6, 6 * mixed_ba.shape[0] / mixed_ba.shape[1]))
plt.imshow(mixed_ba, cmap="gray", vmin=0.0, vmax=1.0)
plt.title("B/A 2×2 corner mix (B at block corners, A elsewhere)")
plt.axis("off")
plt.show()

# Optional: switch which corner uses B by changing the strides:
# top-right:    mixed_ba[0::2, 1::2] = img_b_gray[0::2, 1::2]
# bottom-left:  mixed_ba[1::2, 0::2] = img_b_gray[1::2, 0::2]
# bottom-right: mixed_ba[1::2, 1::2] = img_b_gray[1::2, 1::2]
