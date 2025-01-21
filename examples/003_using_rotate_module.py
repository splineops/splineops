"""
Using rotate module
===================

Use the rotate module to rotate a 2D image.

You can download this example at the tab at right, as both a Python script and as a Jupyter notebook.
"""

# %%
# Imports
# -------
#
# Import necessary libraries.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from splineops.interpolate.rotate import rotate
from splineops.interpolate.resize import resize
import requests
from io import BytesIO
from PIL import Image

# %%
# Load and preprocess the image
# -----------------------------
#
# Load a Kodak image, convert it to grayscale, normalize it,
# and then resize it by a factor of (0.5, 0.5). After that,
# scale it back to [0, 255] before rotation.

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
degree = 3  # spline degree

# Resize the image using spline interpolation (this returns image in [0,1])
image_resized = resize(
    data_normalized, 
    zoom_factors=zoom_factors, 
    degree=degree, 
    method="interpolation"
)

# Bring the resized image back to [0,255]
image_resized = (image_resized * 255.0).astype(np.float32)

# Define rotation angle
rotation_angle = 45

# Use the center of the resized image as the custom center of rotation
custom_center = (image_resized.shape[0] // 2, image_resized.shape[1] // 2)

# Rotate the image (now in [0,255])
rotated_image = rotate(
    image_resized,
    angle=rotation_angle,
    degree=degree,
    center=custom_center,
)

# Create a circular mask
radius = min(image_resized.shape) // 2
rows, cols = rotated_image.shape
rr, cc = np.ogrid[:rows, :cols]

# Display the original and rotated images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
ax[0].imshow(image_resized, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original Resized Image")
ax[0].axis("off")

# Display the rotated image
ax[1].imshow(rotated_image, cmap="gray", vmin=0, vmax=255)
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
custom_center_3d = (64, 64, 64)  # Center of the 3D volume

# Create a 3D sinusoidal volume
k = 0.1  # Spatial frequency
grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")
coords = np.stack([g - c for g, c in zip(grid, custom_center_3d)], axis=0)
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
data_rotated = rotate(data, angle=angle, center=custom_center_3d, degree=3, axis=axis)

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
    (data_rotated, f"Rotated Angle {angle}°, Axis {axis}", True),
]

for i, (volume, title, draw_center) in enumerate(slices):
    # XY Plane
    axs[i, 0].imshow(volume[N // 2], cmap="gray", origin="upper")
    axs[i, 0].axis("off")  # Remove x and y axis numbers
    if draw_center:
        axs[i, 0].scatter(
            custom_center_3d[2], 
            custom_center_3d[1], 
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
            custom_center_3d[2], 
            custom_center_3d[0], 
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
            custom_center_3d[1], 
            custom_center_3d[0], 
            color="red", 
            label="Center"
        )
        axs[i, 2].legend(fontsize=14)
    axs[i, 2].set_title(f"{title} (YZ plane)", fontsize=18)

# Adjust layout
plt.tight_layout()
plt.show()
