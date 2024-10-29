"""
Resizing Images with TensorSpline
=================================

This example demonstrates how to resize images using the TensorSpline API with custom resizing options.
"""

# %%
# Importing Libraries
# -------------------
#
# Load necessary libraries and sample images for resizing.

import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from splineops.interpolate.resize import resize

# %%
# Load Sample Image
# -----------------
#
# Load a sample grayscale image from SciPy datasets, convert it to float32 for compatibility, and check its dimensions.

image = datasets.ascent()
print("Original image shape:", image.shape)

# Convert image to float32 for compatibility with TensorSpline requirements
image = image.astype(np.float32)

# Define extents for plotting
original_extent = [0, image.shape[1], 0, image.shape[0]]

# %%
# Resizing with `output_size`
# ---------------------------
#
# Resize the image to a specified output size and position it in the top-left corner of a black canvas the size of the original image.

output_size = (image.shape[0] // 2, image.shape[1] * 3 // 4)
resized_image_size = resize(image, output_size=output_size, modes="mirror", degree=3)
print("Resized image shape (using output_size):", resized_image_size.shape)

# Create a black canvas of the original image size and position the resized image in the corner
canvas_size = np.zeros(image.shape, dtype=image.dtype)
canvas_size[:resized_image_size.shape[0], :resized_image_size.shape[1]] = resized_image_size

# %%
# Resizing with `zoom_factors`
# ----------------------------
#
# Resize the image by scaling factors and position it in the top-left corner of a black canvas the size of the original image.

zoom_factors = (0.5, 0.75)
resized_image_zoom = resize(image, zoom_factors=zoom_factors, modes="mirror", degree=3)
print("Resized image shape (using zoom_factors):", resized_image_zoom.shape)

# Create a black canvas and position the resized image in the corner
canvas_zoom = np.zeros(image.shape, dtype=image.dtype)
canvas_zoom[:resized_image_zoom.shape[0], :resized_image_zoom.shape[1]] = resized_image_zoom

# %%
# Visualization
# -------------
#
# Display the original image, the resized image using `output_size`, and the resized image using `zoom_factors`.

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

# Original image
axes[0].imshow(image, cmap="gray", extent=original_extent, aspect='auto')
axes[0].set_title("Original Image")
axes[0].axis("off")

# Resized image on black canvas (output_size)
axes[1].imshow(canvas_size, cmap="gray", extent=original_extent, aspect='auto')
axes[1].set_title(f"Resized Image on Black Canvas (output_size = {output_size})")
axes[1].axis("off")

# Resized image on black canvas (zoom_factors)
axes[2].imshow(canvas_zoom, cmap="gray", extent=original_extent, aspect='auto')
axes[2].set_title(f"Resized Image on Black Canvas (zoom_factors = {zoom_factors})")
axes[2].axis("off")

plt.tight_layout()
plt.show()
