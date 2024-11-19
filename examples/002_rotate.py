"""
2D image rotation
=================

This script demonstrates how to rotate an 2D image being from 0 to 360 degrees using the Tensor Spline Interpolation, 
with each rotation performed on top of the last rotated image to observe error accumulation.
"""

# %%
# Imports
# -------
#
# Import necessary libraries.

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, datasets
from IPython.display import HTML, display
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

# Rotate the image by 45 degrees using spline of degree 3
rotated_image_45 = rotate(image_resized, 45, degree=3)  # Use rotate_image from rotate.py

# Display the original and rotated images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_resized, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(rotated_image_45, cmap="gray")
ax[1].set_title("Rotated Image (45 degrees, spline degree 3)")
ax[1].axis("off")
plt.tight_layout()
plt.show()

# %%
# Create animation
# ----------------
#
# Create the animation of the image being rotated from 0 to 360 degrees using different spline degrees.


def create_combined_animation(images):
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
                    images[i], 24, degree=degree
                )  # Rotate by 24 degrees each frame
            image_plots[i].set_data(images[i])
        return image_plots

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=15, interval=250, blit=True
    )  # 15 frames, rotating 24 degrees per frame
    return ani


# Create initial images list and animation
images = [image_resized.copy() for _ in range(3)]
ani = create_combined_animation(images)

# Display the animation
ani_html = ani.to_jshtml()
