# Example script: resize_image_example_with_black_background.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from splineops.interpolate.resize import resize

# Load a sample image from SciPy datasets
image = datasets.ascent()
print("Original image shape:", image.shape)

# Convert image to float32 for compatibility with TensorSpline requirements
image = image.astype(np.float32)

# Define extents for the original image
original_extent = [0, image.shape[1], 0, image.shape[0]]

# %%
# Resizing the image with `output_size`
# -------------------------------------
output_size = (image.shape[0] // 2, image.shape[1] // 2)  # Resize to half the height and 3/4 width
resized_image_size = resize(image, output_size=output_size, bases="bspline3", modes="mirror", degree=3)
print("Resized image shape (using output_size):", resized_image_size.shape)

# Create a black canvas with the original image size and place the resized image in the corner
canvas_size = np.zeros(image.shape, dtype=image.dtype)
canvas_size[:resized_image_size.shape[0], :resized_image_size.shape[1]] = resized_image_size

# %%
# Resizing the image with `zoom_factors`
# --------------------------------------
zoom_factors = (0.5, 0.5)
resized_image_zoom = resize(image, zoom_factors=zoom_factors, bases="bspline3", modes="mirror", degree=3)
print("Resized image shape (using zoom_factors):", resized_image_zoom.shape)

# Create another black canvas and place the resized image in the corner
canvas_zoom = np.zeros(image.shape, dtype=image.dtype)
canvas_zoom[:resized_image_zoom.shape[0], :resized_image_zoom.shape[1]] = resized_image_zoom

# %%
# Plotting the results
# ---------------------
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
