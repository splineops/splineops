"""
Resizing basic example
======================

We use SplineOps to resize a 2D image.

You can download this example as both a Python script and as a Jupyter notebook.
"""

# %%
# Import required libraries
# -------------------------
#
# We import the required libraries, including NumPy for numerical computations,
# Matplotlib for plotting, and the custom `resize` function from the `splineops` package.

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # For SciPy's zoom comparison
from splineops.interpolate.resize import resize  # Unified resize function
import requests
from io import BytesIO
from PIL import Image

# %%
# Basic resizing example
# ----------------------
#
# Load a simple 2D image.

# Load the 'kodim19.png' image
url = 'https://r0k.us/graphics/kodak/kodak/kodim19.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)

# Normalize the image to [0,1]
data_normalized = data / 255.0

# Visualize the original data
plt.figure(figsize=(5, 5))
plt.imshow(data_normalized)
plt.title("Original data")
plt.show()

# Resizing function
def resize_image(data, zoom_factor, degree=3, extension_mode="mirror"):
    resized_channels = []
    for channel in range(data.shape[2]):
        resized_channel = resize(
            data[:, :, channel],
            zoom_factors=zoom_factor,
            degree=degree,
            modes=extension_mode,
            method="interpolation"
        )
        resized_channels.append(resized_channel)
    resized_image = np.stack(resized_channels, axis=-1)
    resized_image = np.clip(resized_image, 0.0, 1.0)
    return (resized_image * 255.0).astype(np.uint8)

# Function to create composite visualization
def create_comparison_plot(original, transformed, zoom_factor):

    # Set global font size
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # Label font size
        'xtick.labelsize': 14,  # X-axis tick font size
        'ytick.labelsize': 14   # Y-axis tick font size
    })

    if zoom_factor < 1:  # Shrinking
        # Create a white canvas matching the original size
        canvas = np.ones((*original.shape[:2], 3), dtype=np.uint8) * 255
        canvas[:transformed.shape[0], :transformed.shape[1]] = transformed
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(canvas)
        ax[1].set_title(f"Shrunken Image (zoom={zoom_factor})")
        ax[1].axis("off")
    else:  # Expanding
        # Create a white canvas matching the transformed size
        expanded_canvas_size = (transformed.shape[0], transformed.shape[1], 3)
        canvas = np.ones(expanded_canvas_size, dtype=np.uint8) * 255
        canvas[:original.shape[0], :original.shape[1]] = original
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(canvas)
        ax[0].set_title("Original Image (on white canvas)")
        ax[0].axis("off")
        ax[1].imshow(transformed)
        ax[1].set_title(f"Expanded Image (zoom={zoom_factor})")
        ax[1].axis("off")
    plt.tight_layout()
    plt.show()

# %%
# Shrinking image
# ~~~~~~~~~~~~~~~
#
# Apply the resize function quadratic B-spline interpolation and a contracting zoom factor.

zoom_factor = 0.3
shrunken_image = resize_image(
    data_normalized, 
    zoom_factor=zoom_factor, 
    degree=2, 
    extension_mode="mirror")
create_comparison_plot(
    data.astype(np.uint8), 
    shrunken_image, 
    zoom_factor=zoom_factor)

# %%
# Expanding image
# ~~~~~~~~~~~~~~~
#
# Apply the resize function with for cubic B-spline interpolation and an expanding zoom factor.

zoom_factor = 2.5
shrunken_image = resize_image(
    data_normalized, 
    zoom_factor=zoom_factor, 
    degree=3, 
    extension_mode="mirror")
create_comparison_plot(
    data.astype(np.uint8), 
    shrunken_image, 
    zoom_factor=zoom_factor)