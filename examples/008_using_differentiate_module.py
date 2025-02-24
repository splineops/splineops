"""
Using differentiate module
==========================

In this example, we demonstrate how to use the differentiate module to compute various
differential operations on an image. We will perform:

- Gradient Magnitude
- Gradient Direction
- Laplacian
- Largest Hessian Eigenvalue
- Smallest Hessian Eigenvalue
- Hessian Orientation

and visualize the results.

You can download this example at the tab at right, as both a Python script and as a Jupyter notebook.
"""

# %%
# Imports
# -------
#
# We import the necessary libraries and modules:

import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

# Import the Differentials class from your module (adjust import path as needed)
from splineops.differentiate.differentials import differentials

# %%
# Data preparation
# ----------------
#
# We retrieve an example color image, convert it to grayscale,
# and normalize its intensities to the [0..1] range.

url = 'https://r0k.us/graphics/kodak/kodak/kodim15.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
image = np.array(img, dtype=np.float64)

# Convert to [0..1]
image_normalized = image / 255.0

# Convert to grayscale via simple weighting
image_gray = (
    image_normalized[:, :, 0] * 0.2989 +
    image_normalized[:, :, 1] * 0.5870 +
    image_normalized[:, :, 2] * 0.1140
)

# %%
# Create a helper function to visualize results with a colorbar:
def show_result_with_colorbar(title, result, units="Value"):
    """
    Displays a 2D result with a colorbar indicating min/max values.
    
    Parameters
    ----------
    title : str
        Title for the plot.
    result : ndarray
        2D array representing the image or field to display.
    units : str
        Label for the colorbar (e.g., 'Intensity', 'Radians', etc.).
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(result, cmap='gray', aspect='equal')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # Show numeric range in colorbar label
    vmin, vmax = result.min(), result.max()
    cbar.set_label(f"{units} range [{vmin:.3f}, {vmax:.3f}]")
    plt.title(title)
    plt.axis('off')
    plt.show()

# Show the original grayscale image with a colorbar
show_result_with_colorbar("Original Image", image_gray, units="Intensity")

# %%
# Gradient Magnitude
# ------------------
# The gradient magnitude provides an intensity image indicating
# the rate of change of intensity at each pixel. High values
# indicate edges or sharp transitions in the image.

diff = differentials(image_gray.copy())
diff.run(differentials.GRADIENT_MAGNITUDE)
grad_magnitude_result = diff.image

show_result_with_colorbar("Gradient Magnitude", grad_magnitude_result, units="Value")

# %%
# Gradient Direction
# ------------------
# The gradient direction indicates the angle of the local gradient
# vector at each pixel (in radians). This can be useful for edge
# orientation detection and directional filtering.

diff = differentials(image_gray.copy())
diff.run(differentials.GRADIENT_DIRECTION)
grad_direction_result = diff.image

# Note that gradient direction ranges from -π to π (arctan2).
show_result_with_colorbar("Gradient Direction", grad_direction_result, units="Direction (radians)")

# %%
# Laplacian
# ---------
# The Laplacian of an image highlights regions of rapid intensity change.
# It is computed here by adding the second derivatives along
# both the horizontal (x) and vertical (y) directions.

diff = differentials(image_gray.copy())
diff.run(differentials.LAPLACIAN)
laplacian_result = diff.image

show_result_with_colorbar("Laplacian", laplacian_result, units="Value")

# %%
# Largest Hessian
# ---------------
# The Hessian matrix at each pixel contains second-order partial derivatives
# of the image intensity. Its eigenvalues indicate curvature along
# principal directions. Here, we compute the *largest eigenvalue* of
# the Hessian, which often highlights tube-like or ridge-like structures.

diff = differentials(image_gray.copy())
diff.run(differentials.LARGEST_HESSIAN)
largest_hessian_result = diff.image

show_result_with_colorbar("Largest Hessian Eigenvalue", largest_hessian_result, units="Value")

# %%
# Smallest Hessian
# ----------------
# The *smallest eigenvalue* of the Hessian matrix can highlight
# features orthogonal to those emphasized by the largest eigenvalue.
# It can be useful for detecting certain types of structures.

diff = differentials(image_gray.copy())
diff.run(differentials.SMALLEST_HESSIAN)
smallest_hessian_result = diff.image

show_result_with_colorbar("Smallest Hessian Eigenvalue", smallest_hessian_result, units="Value")

# %%
# Hessian Orientation
# -------------------
# The Hessian orientation encodes the principal directions of curvature
# in the neighborhood of each pixel. Formally, it's the orientation of
# the eigenvectors of the Hessian matrix. 
#
# Positive or negative signs (and the values themselves) can be interpreted
# to understand how image structures are oriented locally. Values range
# roughly from -π/2 to +π/2 in this particular definition.

diff = differentials(image_gray.copy())
diff.run(differentials.HESSIAN_ORIENTATION)
hessian_orientation_result = diff.image

# The resulting range is approximately [-π/2, +π/2].
show_result_with_colorbar("Hessian Orientation", hessian_orientation_result, units="Orientation (radians)")
