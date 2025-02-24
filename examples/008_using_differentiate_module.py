import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from splineops.differentiate.differentials import Differentials

# Load the "ascent" image from scipy datasets
url = 'https://r0k.us/graphics/kodak/kodak/kodim15.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
image = np.array(img, dtype=np.float64)

# Convert to [0..1]
image_normalized = image / 255.0

# Convert to grayscale via simple weighting
image_normalized = (
    image_normalized[:, :, 0] * 0.2989 +  # Red channel
    image_normalized[:, :, 1] * 0.5870 +  # Green channel
    image_normalized[:, :, 2] * 0.1140    # Blue channel
)

# Instantiate the Differentials class with the image
differential = Differentials(image_normalized)

# Choose an operation to run, e.g., Hessian Orientation
operation = Differentials.GRADIENT_MAGNITUDE

# Run the selected operation
differential.run(operation)

# The result is stored in differential.image
result = differential.image

# Define a mapping of operations to their names for titles
operation_titles = {
    Differentials.GRADIENT_MAGNITUDE: "Gradient Magnitude",
    Differentials.GRADIENT_DIRECTION: "Gradient Direction",
    Differentials.LAPLACIAN: "Laplacian",
    Differentials.LARGEST_HESSIAN: "Largest Hessian",
    Differentials.SMALLEST_HESSIAN: "Smallest Hessian",
    Differentials.HESSIAN_ORIENTATION: "Hessian Orientation"
}

# Display the original and the processed image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_normalized, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(operation_titles[operation])
plt.imshow(result, cmap='gray')

plt.show()
