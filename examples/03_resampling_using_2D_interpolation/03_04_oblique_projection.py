# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_04_oblique_projection.py
# sphinx_gallery_end_ignore

"""
Oblique Projection
==================

Interpolate 2D images with and oblique projection.
"""

# %%
# Imports
# -------

import numpy as np
import requests
from io import BytesIO
from PIL import Image

from splineops.utils import (
    resize_and_compute_metrics,      # resampling + metrics
    plot_resized_image,              # visual helpers
    plot_recovered_image,
    plot_difference_image,
)

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

# %%
# Oblique Projection
# ------------------
#
# We use the oblique-projection method.

(
    resized_2d_ob,
    recovered_2d_ob,
    snr_2d_ob,
    mse_2d_ob,
    time_2d_ob
) = resize_and_compute_metrics(
    input_image_normalized,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after a reversing of the zoom factors.

plot_recovered_image(recovered_2d_ob)

# %%
# Resized Image
# ~~~~~~~~~~~~~
#
# We plot the resized image with the oblique-projection method.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_ob,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_ob
)

# %%
# Difference Image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ob,
    snr=snr_2d_ob,
    mse=mse_2d_ob
)