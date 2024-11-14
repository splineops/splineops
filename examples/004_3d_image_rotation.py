# Imports
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

# Display a middle slice to visualize the original 3D data
plt.figure(figsize=(6, 6))
plt.imshow(volume[size // 2], cmap="gray")
plt.title("Original 3D Volume with 'L' Shape (Middle Slice)")
plt.axis("off")
plt.show()

# Apply rotation on the XY plane (around the z-axis)
angle = 30  # Rotate by 30 degrees
degree = 3  # Spline degree

# Perform the rotation using the z-axis as default
rotated_volume = rotate(volume, angle=angle, degree=degree, axis=(0, 0, 1))

# Visualize Rotated Slices
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
