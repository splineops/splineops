"""
First steps with the TensorSpline API
=====================================

This example demonstrates how to create a basic interpolation using the TensorSpline API.
"""

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.rotate import rotate

# Generate a synthetic 3D "L" shape
size = 64  # Size of the 3D volume
volume = np.zeros((size, size, size), dtype=np.float32)

# Create an "L" shape in the middle of the volume
thickness = 5
volume[:, size // 4:size // 4 + thickness, size // 4:size - size // 4] = 1  # Horizontal part of "L"
volume[:, size // 4:size - size // 4, size // 4:size // 4 + thickness] = 1  # Vertical part of "L"

# Define the rotation parameters
angle = 45  # Rotate by 45 degrees
axis = (0, 0, 1)  # Custom axis of rotation
center = (32, 32, 32)  # Center of rotation (middle of the volume)

# Perform the rotation
rotated_volume = rotate(volume, angle=angle, axis=axis, center=center, degree=3)

# Visualize original and rotated slices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original slices
axes[0, 0].imshow(volume[size // 2], cmap="gray")
axes[0, 0].set_title("Original (XY plane)")
axes[0, 1].imshow(volume[:, size // 2, :], cmap="gray")
axes[0, 1].set_title("Original (XZ plane)")
axes[0, 2].imshow(volume[:, :, size // 2], cmap="gray")
axes[0, 2].set_title("Original (YZ plane)")

# Rotated slices
axes[1, 0].imshow(rotated_volume[size // 2], cmap="gray")
axes[1, 0].set_title(f"Rotated (XY plane, {angle}°)")
axes[1, 1].imshow(rotated_volume[:, size // 2, :], cmap="gray")
axes[1, 1].set_title(f"Rotated (XZ plane, {angle}°)")
axes[1, 2].imshow(rotated_volume[:, :, size // 2], cmap="gray")
axes[1, 2].set_title(f"Rotated (YZ plane, {angle}°)")

# Adjust display settings
for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
