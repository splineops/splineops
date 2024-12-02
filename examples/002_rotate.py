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
degree = 3 # spline degree
rotation_angle = 45
custom_center = (250, 250)  # Custom center for rotation (row, column)

image_resized = ndimage.zoom(
    image, (size / image.shape[0], size / image.shape[1]), order=degree
)

# Convert to float32
image_resized = image_resized.astype(np.float32)

# Rotate the image
rotated_image = rotate(image_resized, angle=rotation_angle, degree=degree, center=custom_center)

# Display the original and rotated images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
ax[0].imshow(image_resized, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

# Display the rotated image
ax[1].imshow(rotated_image, cmap="gray")
ax[1].scatter(custom_center[1], custom_center[0], color="red", label="Center of Rotation")
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

    # Animation function
    def animate(frame):
        nonlocal images  # Ensure we modify the images array from the enclosing scope
        for i, degree in enumerate([0, 1, 3]):
            if frame > 0:
                images[i] = rotate(
                    images[i], angle=24, degree=degree, center=center
                )  # Rotate by 24 degrees each frame
            image_plots[i].set_data(images[i])
        return image_plots

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=15, interval=250, blit=True
    )  # 15 frames, rotating 24 degrees per frame
    return ani


# Function to create a circular mask
def apply_circular_mask(image, radius, center=None):
    """
    Apply a circular mask to an image, leaving only the inside of the circle visible.
    
    Parameters:
        image (np.array): Input image.
        radius (int): Radius of the circle.
        center (tuple, optional): Center of the circle (row, column). If None, defaults to the center of the image.
    
    Returns:
        masked_image (np.array): Image with the circular mask applied.
    """
    # Determine the center of the circle
    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)
    
    # Create a grid of coordinates
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create the circular mask
    mask = distance_from_center <= radius
    
    # Apply the mask to the image
    masked_image = np.ones_like(image) * 255  # Create a white background
    masked_image[mask] = image[mask]  # Keep the image data inside the circle
    
    return masked_image

# Mask the image
circle_radius = 200  # Define the radius of the circle
masked_image = apply_circular_mask(image_resized, radius=circle_radius, center=custom_center)

# Display the masked image
plt.figure(figsize=(6, 6))
plt.imshow(masked_image, cmap="gray")
plt.title("Image with Circular Mask Applied")
plt.axis("off")
plt.show()

# Replace `image_resized` with `masked_image` for animation
images = [masked_image.copy() for _ in range(3)]
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
