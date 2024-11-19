"""
2D image rotation
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
from scipy import ndimage, datasets
from matplotlib import animation
from splineops.interpolate.rotate import rotate

# %%
# Load and preprocess image
# -------------------------
#
# Load the image and preprocess it for the rotation animation.

# Load and resize the ascent image
image = datasets.ascent()
size = 500  # Resize image to 500x500 for faster computation
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