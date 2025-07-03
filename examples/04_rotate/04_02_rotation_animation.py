"""
Rotation Animation
==================

We use the rotate module to rotate a 2D image.
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from splineops.rotate.rotate import rotate
from splineops.resize.resize import resize
import requests
from io import BytesIO
from PIL import Image

# %%
# Load and Preprocess the Image
# -----------------------------
#
# Load a Kodak image, convert it to grayscale, normalize it,
# and then resize it by a factor of (0.5, 0.5). After that,
# scale its intensity back to [0, 255] before rotation.

# Load the 'kodim17.png' image
url = 'https://r0k.us/graphics/kodak/kodak/kodim22.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)

# Convert to grayscale using a standard formula
data_gray = (
    data[:, :, 0] * 0.2989 +
    data[:, :, 1] * 0.5870 +
    data[:, :, 2] * 0.1140
)

# Normalize the grayscale image to [0,1]
data_normalized = data_gray / 255.0

# Define zoom factors for resizing
zoom_factors = (0.3, 0.3)
interp_method = "cubic"

# Resize the image using spline interpolation (this returns image in [0,1])
image_resized = resize(
    data_normalized, 
    zoom_factors=zoom_factors, 
    method=interp_method
)

# Bring the resized image back to [0,255]
image_resized = (image_resized * 255.0).astype(np.float32)

# Use the center of the resized image as the custom center of rotation
custom_center = (image_resized.shape[0] // 2, image_resized.shape[1] // 2)
radius = min(image_resized.shape) // 2

# %%
# Create an Animation
# -------------------
#
# Create the animation of the image being rotated from 0 to 360 degrees. Explore the effect of the spline degree.

def rotate_and_mask(image, angle, degree, center, radius):
    rotated = rotate(image, angle=angle, degree=degree, center=center)
    rows, cols = rotated.shape
    rr, cc = np.ogrid[:rows, :cols]
    mask = (rr - center[0])**2 + (cc - center[1])**2 <= radius**2
    # We'll return both the rotated image and the mask so we can use the mask as alpha.
    return rotated, mask

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
            np.zeros((image.shape[0], image.shape[1])),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        image_plots.append(img_plot)

    # Smaller rotation angle per frame
    rotation_step = 10  # Degrees per frame (adjust this value)
    total_frames = 360 // rotation_step  # Number of frames for a full rotation

    def animate(frame):
        angle = frame * rotation_step  # Increment rotation angle
        for i, d in enumerate(degrees_list):
            rotated_img, m = rotate_and_mask(image, angle=angle, degree=d, center=center, radius=radius)
            image_plots[i].set_data(rotated_img)
            image_plots[i].set_alpha(m.astype(float))  # Apply mask as transparency
        return image_plots

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=250, blit=True
    )
    return ani

# Create the animation
ani = create_combined_animation(image_resized, custom_center, radius)
ani_html = ani.to_jshtml()