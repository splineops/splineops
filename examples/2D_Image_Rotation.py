"""
2D Image Rotation
=================

This script demonstrates how to rotate an 2D image being from 0 to 360 degrees using the Tensor Spline Interpolation, with each rotation performed on top of the last rotated image to observe error accumulation.
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

from splineops.interpolate.tensorspline import TensorSpline

# %%
# Helper Functions
# ----------------
#
# Define the helper functions to rotate an image and create the animation.

def rotate_image_splineops(image, angle, degree=3, mode="zero"):
    """
    Rotate an image by a specified angle using SplineOps' TensorSpline method.

    Parameters:
    - image: The input image as a 2D numpy array.
    - angle: The rotation angle in degrees.
    - degree: The degree of the spline (0-7).
    - mode: The mode for handling boundaries (default is "zero").

    Returns:
    - Rotated image as a 2D numpy array.
    """
    dtype = image.dtype
    ny, nx = image.shape
    xx = np.linspace(0, nx - 1, nx, dtype=dtype)
    yy = np.linspace(0, ny - 1, ny, dtype=dtype)
    data = np.ascontiguousarray(image, dtype=dtype)

    degree = max(0, min(degree, 7))
    basis = f"bspline{degree}"

    tensor_spline = TensorSpline(
        data=data, coordinates=(yy, xx), bases=basis, modes=mode
    )
    angle_rad = np.radians(-angle)
    cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
    original_center_x, original_center_y = (nx - 1) / 2.0, (ny - 1) / 2.0
    oy, ox = np.ogrid[0:ny, 0:nx]
    ox = ox - original_center_x
    oy = oy - original_center_y

    nx_coords = cos_angle * ox + sin_angle * oy + original_center_x
    ny_coords = -sin_angle * ox + cos_angle * oy + original_center_y

    eval_coords = ny_coords.flatten(), nx_coords.flatten()
    interpolated_values = tensor_spline(coordinates=eval_coords, grid=False)
    rotated_image = interpolated_values.reshape(ny, nx)

    return rotated_image

# %%
# Load and Preprocess Image
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
rotated_image_45 = rotate_image_splineops(image_resized, 45, degree=3)

# Display the original and rotated images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_resized, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(rotated_image_45, cmap='gray')
ax[1].set_title('Rotated Image (45 degrees, spline degree 3)')
ax[1].axis('off')
plt.tight_layout()
plt.show()

# %%
# Create Animation
# ----------------
#
# Create the animation of the image being rotated from 0 to 360 degrees using different spline degrees.

def create_combined_animation(images):
    fig, axes = plt.subplots(3, 1, figsize=(6, 18), constrained_layout=True)  # Adjusted figsize and layout for vertical placement
    for ax, degree in zip(axes, [0, 1, 3]):
        ax.axis('off')
        ax.set_title(f'Degree {degree}')
    
    image_plots = [ax.imshow(images[i], cmap='gray') for i, ax in enumerate(axes)]
    
    # Animation function
    def animate(frame):
        nonlocal images  # Ensure we modify the images array from the enclosing scope
        for i, degree in enumerate([0, 1, 3]):
            if frame > 0:
                images[i] = rotate_image_splineops(images[i], 24, degree=degree)  # Rotate by 24 degrees each frame
            image_plots[i].set_data(images[i])
        return image_plots

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=15, interval=250, blit=True)  # 15 frames, rotating 24 degrees per frame
    return ani

# Create initial images list and animation
images = [image_resized.copy() for _ in range(3)]
ani = create_combined_animation(images)

# Display the animation
ani_html = ani.to_jshtml()
