"""
Using differentiate module
==========================

In this example, we demonstrate how to use the differentiate module to compute various differential operations on an image. We will perform:

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
from splineops.differentiate.differentials import Differentials

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

# Create a helper function to visualize results:
def show_result(title, result):
    """
    Helper function to display an image result in a new figure.
    """
    plt.figure(figsize=(6, 5))
    plt.title(title)
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()

show_result("Original Image", image_gray)

# %%
# Gradient Magnitude
# ------------------
# The gradient magnitude provides an intensity image indicating
# the rate of change of intensity at each pixel. High values
# indicate edges or sharp transitions in the image.

# Create a fresh Differentials object with the grayscale image
diff = Differentials(image_gray.copy())

# Perform the Gradient Magnitude operation
diff.run(Differentials.GRADIENT_MAGNITUDE)

# Retrieve the result from diff.image
grad_magnitude_result = diff.image

# Visualize
show_result("Gradient Magnitude", grad_magnitude_result)

# %%
# Gradient Direction
# ------------------
# The gradient direction indicates the angle of the local gradient
# vector at each pixel (in radians). This can be useful for edge
# orientation detection and directional filtering.

diff = Differentials(image_gray.copy())
diff.run(Differentials.GRADIENT_DIRECTION)
grad_direction_result = diff.image

# Visualize
# Note that gradient direction ranges from -π to π (arctan2).
# We'll display as grayscale just to showcase the variety of angles.
show_result("Gradient Direction", grad_direction_result)

# %%
# Laplacian
# ---------
# The Laplacian of an image highlights regions of rapid intensity change.
# It is computed here by adding the second derivatives along
# both the horizontal (x) and vertical (y) directions.

diff = Differentials(image_gray.copy())
diff.run(Differentials.LAPLACIAN)
laplacian_result = diff.image

# Visualize
show_result("Laplacian", laplacian_result)

# %%
# Largest Hessian
# ---------------
# The Hessian matrix at each pixel contains second-order partial derivatives
# of the image intensity. Its eigenvalues indicate curvature along
# principal directions. Here, we compute the *largest eigenvalue* of
# the Hessian, which often highlights tube-like or ridge-like structures.

diff = Differentials(image_gray.copy())
diff.run(Differentials.LARGEST_HESSIAN)
largest_hessian_result = diff.image

# Visualize
show_result("Largest Hessian Eigenvalue", largest_hessian_result)

# %%
# Smallest Hessian
# ----------------
# The *smallest eigenvalue* of the Hessian matrix can highlight
# features orthogonal to those emphasized by the largest eigenvalue.
# It can be useful for detecting certain types of structures.
    
diff = Differentials(image_gray.copy())
diff.run(Differentials.SMALLEST_HESSIAN)
smallest_hessian_result = diff.image

# Visualize
show_result("Smallest Hessian Eigenvalue", smallest_hessian_result)

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

diff = Differentials(image_gray.copy())
diff.run(Differentials.HESSIAN_ORIENTATION)
hessian_orientation_result = diff.image

# Visualize
# These values may be negative, so a gray colormap helps see the differences.
show_result("Hessian Orientation", hessian_orientation_result)