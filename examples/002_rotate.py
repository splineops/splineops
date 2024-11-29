"""
Image rotation
=================

This script demonstrates how to rotate a 2D image from 0 to 360 degrees using Tensor Spline Interpolation, 
with each rotation performed on top of the last rotated image to observe error accumulation.
It also allows specifying a center of rotation and visualizes it in the rotated image.
"""

# %%
# Imports
# -------
#
# Import necessary libraries.

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import animation
from splineops.interpolate.rotate import rotate
from splineops.utils.image_loader import load_collagen_image

# %%
# Load and preprocess image
# -------------------------
#
# Load the image and preprocess it for the rotation animation.

# Load and resize the image
image = load_collagen_image()
size = 500
degree = 3
image_resized = ndimage.zoom(
    image, (size / image.shape[0], size / image.shape[1]), order=degree
)

# Convert to float32
image_resized = image_resized.astype(np.float32)

# Define a custom center for rotation
custom_center = (250, 250)  # Example center coordinates (row, column)

# Rotate the image by 45 degrees using spline of degree 3
rotated_image_45 = rotate(image_resized, angle=45, degree=3, center=custom_center)

# Display the original and rotated images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_resized, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(rotated_image_45, cmap="gray")
ax[1].scatter(custom_center[1], custom_center[0], color="red", label="Center of Rotation")
ax[1].set_title("Rotated Image (45 degrees, spline degree 3)")
ax[1].axis("off")
ax[1].legend()

plt.tight_layout()
plt.show()

# %%
# Create animation
# ----------------
#
# Create the animation of the image being rotated from 0 to 360 degrees using different spline degrees
# and visualize the center of rotation only in the rotated images.

def create_combined_animation(images, center):
    """
    Create an animation showing the image being rotated for different spline degrees.
    
    Parameters:
        images (list of np.array): List of images for each spline degree.
        center (tuple): The center of rotation as (row, column).
        
    Returns:
        ani: Matplotlib animation object.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(6, 18), constrained_layout=True
    )  # Adjusted figsize and layout for vertical placement
    for ax, degree in zip(axes, [0, 1, 3]):
        ax.axis("off")
        ax.set_title(f"Degree {degree}")

    image_plots = [ax.imshow(images[i], cmap="gray") for i, ax in enumerate(axes)]

    # Add center marker only to rotated images
    center_markers = [
        ax.scatter(center[1], center[0], color="red", label="Center of Rotation")
        for ax in axes
    ]

    # Animation function
    def animate(frame):
        nonlocal images  # Ensure we modify the images array from the enclosing scope
        for i, degree in enumerate([0, 1, 3]):
            if frame > 0:
                images[i] = rotate(
                    images[i], angle=24, degree=degree, center=center
                )  # Rotate by 24 degrees each frame
            image_plots[i].set_data(images[i])
        return image_plots + center_markers

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=15, interval=250, blit=True
    )  # 15 frames, rotating 24 degrees per frame
    return ani


# Create initial images list and animation
images = [image_resized.copy() for _ in range(3)]
ani = create_combined_animation(images, custom_center)

# Display the animation
ani_html = ani.to_jshtml()

# %%
# 3D image rotation
# -----------------
#
# Demonstrate the rotation of a 3D image around a custom center and axis.
# The image slices are displayed in grayscale, with the center of rotation
# marked only in the rotated data slices.

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.rotate import rotate

# Define the size of the 3D image
N = 128  # Volume size
data_shape = (N, N, N)

# Define a custom center for rotation
custom_center = (64, 64, 64)  # Center of the 3D volume

# Create a 3D sinusoidal volume
k = 0.1  # Spatial frequency
grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")
coords = np.stack([g - c for g, c in zip(grid, custom_center)], axis=0)  # Shape: (3, N, N, N)
coords_flat = coords.reshape(3, -1)  # Shape: (3, N*N*N)
data = (
    np.sin(k * coords_flat[0, :]) +
    np.cos(k * coords_flat[1, :]) +
    np.sin(k * coords_flat[2, :])
)
data = data.reshape(data_shape)

# Define the rotation angle and axis
angle = 45  # Rotation angle in degrees
axis = (1, 1, 1)  # Custom axis of rotation

# Rotate the data using the rotate function
data_rotated = rotate(data, angle=angle, center=custom_center, degree=3, axis=axis)

# Visualize slices of the original and rotated volumes in grayscale
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Slices to visualize (middle slices in each dimension)
slices = [
    (data, "Original Data", False),  # No center in the original data
    (data_rotated, f"Rotated Data (Angle: {angle}°, Axis: {axis})", True),  # Center in rotated data
]

for i, (volume, title, draw_center) in enumerate(slices):
    # XY Plane
    axs[i, 0].imshow(volume[N // 2], cmap="gray", origin="upper")
    if draw_center:
        axs[i, 0].scatter(custom_center[2], custom_center[1], color="red", label="Center")
        axs[i, 0].legend()
    axs[i, 0].set_title(f"{title} (XY plane)")
    
    # XZ Plane
    axs[i, 1].imshow(volume[:, N // 2, :], cmap="gray", origin="upper")
    if draw_center:
        axs[i, 1].scatter(custom_center[2], custom_center[0], color="red", label="Center")
        axs[i, 1].legend()
    axs[i, 1].set_title(f"{title} (XZ plane)")
    
    # YZ Plane
    axs[i, 2].imshow(volume[:, :, N // 2], cmap="gray", origin="upper")
    if draw_center:
        axs[i, 2].scatter(custom_center[1], custom_center[0], color="red", label="Center")
        axs[i, 2].legend()
    axs[i, 2].set_title(f"{title} (YZ plane)")

plt.tight_layout()
plt.show()
