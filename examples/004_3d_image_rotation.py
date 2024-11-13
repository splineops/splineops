"""
3D Image Rotation Visualization with "L" Shape
=============================================

This script demonstrates rotating a 3D synthetic "L" shape on a selected 2D plane and visualizing the results.
"""

# %%
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.rotate import rotate

# %%
# Generate a synthetic 3D "L" shape
# ---------------------------------
#
# We'll create a 3D volume with an "L" shape structure.

size = 64  # Size of the 3D volume
volume = np.zeros((size, size, size), dtype=np.float32)

# Create an "L" shape in the middle of the volume
thickness = 5
volume[:, size // 4:size // 4 + thickness, size // 4:size - size // 4] = 1  # Horizontal part of "L"
volume[:, size // 4:size - size // 4, size // 4:size // 4 + thickness] = 1  # Vertical part of "L"

# Display a middle slice to visualize the original 3D data
plt.figure(figsize=(6, 6))
plt.imshow(volume[size // 2], cmap="gray")
plt.title("Original 3D Volume with 'L' Shape (Middle Slice)")
plt.axis("off")
plt.show()

# %%
# Apply rotation on the (1, 2) plane
# ----------------------------------
#
# Rotate the 3D volume along the (1, 2) plane by a specified angle.

angle = 30  # Rotate by 30 degrees
degree = 3  # Spline degree

# Perform the rotation
rotated_volume = rotate(volume, angle=angle, degree=degree, axes=(1, 2))

# %%
# Visualize Rotated Slices
# ------------------------
#
# We'll display the original and rotated middle slices along the three main planes to compare.

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
