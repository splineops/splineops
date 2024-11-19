"""
Example of using the resize functio on a 2D image
=================================================

This example demonstrates how to perform resizing on a simple 2D image using different degrees and resizing methods.
"""

# %%
# Data preparation
# ----------------
#
# Create a simple 2D image as a sample data (e.g., a gradient or a checkerboard pattern).

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.resize import resize

# Create a simple 2D gradient image
nx, ny = 50, 50  # Original image dimensions
data = np.linspace(0, 1, nx * ny).reshape((nx, ny))

# Visualize the original data
plt.figure(figsize=(5, 5))
plt.imshow(data, cmap='gray', aspect='equal')
plt.title("Original data")
plt.show()

# %%
# Resizing with degree 1 and fixed output size
# --------------------------------------------
#
# Apply the resize function with degree=1 for linear B-spline interpolation and a fixed output size.

output_size = (100, 100)  # Target output size (doubling the size)

resized_degree_1 = resize(data, output_size=output_size, degree=1, modes="mirror", method="interpolation")

# Visualize the resized data
plt.figure(figsize=(5, 5))
plt.imshow(resized_degree_1, cmap='gray', aspect='equal')
plt.title("Degree 1, output size 100x100")
plt.show()

# %%
# Resizing with degree 2 and zoom factor 0.3
# ------------------------------------------
#
# Apply the resize function with degree=2 for quadratic B-spline interpolation and a zoom factor of 0.3.

zoom_factor_2 = 0.3  # Downscale by a factor of 0.3

resized_degree_2 = resize(data, zoom_factors=zoom_factor_2, degree=2, modes="mirror", method="interpolation")

# Visualize the resized data
plt.figure(figsize=(5, 5))
plt.imshow(resized_degree_2, cmap='gray', aspect='equal')
plt.title("Degree 2, zoom factor 0.3")
plt.show()

# %%
# Resizing with degree 3 and zoom factor 2.5
# ------------------------------------------
#
# Apply the resize function with degree=3 for cubic B-spline interpolation and a zoom factor of 2.5.

zoom_factor_3 = 2.5  # Upscale by a factor of 2.5

resized_degree_3 = resize(data, zoom_factors=zoom_factor_3, degree=3, modes="mirror", method="interpolation")

# Visualize the resized data
plt.figure(figsize=(5, 5))
plt.imshow(resized_degree_3, cmap='gray', aspect='equal')
plt.title("Degree 3, zoom factor 2.5")
plt.show()
