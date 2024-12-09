"""
Image rotation
=================

This script demonstrates how to rotate a 2D image from 0 to 360 degrees using Tensor Spline Interpolation, 
with each rotation performed on top of the last rotated image to observe error accumulation.
It also allows specifying a center of rotation and visualizes it in the rotated image.

You can download this example as both a Python script and as a Jupyter notebook.
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
degree = 3 # spline degree
rotation_angle = 45
custom_center = (size // 2, size // 2)  # Custom center for rotation (row, column)

image_resized = ndimage.zoom(
    image, (size / image.shape[0], size / image.shape[1]), order=degree
)

# Convert to float32
image_resized = image_resized.astype(np.float32)

# Rotate the image
rotated_image = rotate(
    image_resized,
    angle=rotation_angle,
    degree=degree,
    center=custom_center,
)

# Create a circular mask
radius = size // 2
rows, cols = rotated_image.shape
rr, cc = np.ogrid[:rows, :cols]

# Display the original and rotated images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
ax[0].imshow(image_resized, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

# Display the rotated image
ax[1].imshow(rotated_image, cmap="gray")
ax[1].scatter(
    custom_center[1], 
    custom_center[0], 
    color="red", 
    label="Center of Rotation"
)
ax[1].set_title(f"Rotated Image ({rotation_angle}°, spline degree {degree})")
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

def rotate_and_mask(image, angle, degree, center, radius):
    rotated = rotate(image, angle=angle, degree=degree, center=center)
    rows, cols = rotated.shape
    rr, cc = np.ogrid[:rows, :cols]
    mask = (rr - center[0])**2 + (cc - center[1])**2 <= radius**2
    # We'll return both the rotated image and the mask so we can use the mask as alpha.
    return rotated, mask

# Function to create the animation
def create_combined_animation(image, center, radius):
    fig, axes = plt.subplots(3, 1, figsize=(6, 18), constrained_layout=True)
    degrees_list = [0, 1, 3]
    for ax, d in zip(axes, degrees_list):
        ax.axis("off")
        ax.set_title(f"Spline Degree {d}")
    # Initialize the images with just zeros
    image_plots = []
    for ax in axes:
        img_plot = ax.imshow(
            np.zeros((size, size)),
            cmap="gray",
            vmin=0.0,
            vmax=128.0,
        )
        image_plots.append(img_plot)

    def animate(frame):
        angle = frame * 24  # Cumulative angle
        for i, d in enumerate(degrees_list):
            rotated_image, m = rotate_and_mask(image, angle=angle, degree=d, center=center, radius=radius)
            image_plots[i].set_data(rotated_image)
            image_plots[i].set_alpha(m.astype(float))  # Apply mask as transparency
        return image_plots

    ani = animation.FuncAnimation(
        fig, animate, frames=15, interval=250, blit=True
    )
    return ani

# Create the animation
radius = size // 2
ani = create_combined_animation(image_resized, custom_center, radius)
ani_html = ani.to_jshtml()

# %%
# 3D image rotation
# -----------------
#
# Demonstrate the rotation of a 3D image around a custom center and axis.
# The image slices are displayed in grayscale, with the center of rotation
# marked only in the rotated data slices.

# Define the size of the 3D image
N = 128  # Volume size
data_shape = (N, N, N)

# Define a custom center for rotation
custom_center = (64, 64, 64)  # Center of the 3D volume

# Create a 3D sinusoidal volume
k = 0.1  # Spatial frequency
grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")
coords = np.stack([g - c for g, c in zip(grid, custom_center)], axis=0)
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

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,  # Base font size
    'axes.titlesize': 18,  # Title font size
    'legend.fontsize': 14  # Legend font size
})

# Slices to visualize (middle slices in each dimension)
slices = [
    (data, "Original Data", False),
    (data_rotated, f"Rotated Angle {angle}°, Axis {axis})", True),
]

for i, (volume, title, draw_center) in enumerate(slices):
    # XY Plane
    axs[i, 0].imshow(volume[N // 2], cmap="gray", origin="upper")
    axs[i, 0].axis("off")  # Remove x and y axis numbers
    if draw_center:
        axs[i, 0].scatter(
            custom_center[2], 
            custom_center[1], 
            color="red", 
            label="Center"
        )
        axs[i, 0].legend(fontsize=14)
    axs[i, 0].set_title(f"{title} (XY plane)", fontsize=18)
    
    # XZ Plane
    axs[i, 1].imshow(volume[:, N // 2, :], cmap="gray", origin="upper")
    axs[i, 1].axis("off")  # Remove x and y axis numbers
    if draw_center:
        axs[i, 1].scatter(
            custom_center[2], 
            custom_center[0], 
            color="red", 
            label="Center"
        )
        axs[i, 1].legend(fontsize=14)
    axs[i, 1].set_title(f"{title} (XZ plane)", fontsize=18)
    
    # YZ Plane
    axs[i, 2].imshow(volume[:, :, N // 2], cmap="gray", origin="upper")
    axs[i, 2].axis("off")  # Remove x and y axis numbers
    if draw_center:
        axs[i, 2].scatter(
            custom_center[1], 
            custom_center[0], 
            color="red", 
            label="Center"
        )
        axs[i, 2].legend(fontsize=14)
    axs[i, 2].set_title(f"{title} (YZ plane)", fontsize=18)

# Adjust layout
plt.tight_layout()
plt.show()

