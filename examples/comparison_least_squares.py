import numpy as np
from splineops.interpolate.resize import resize  # Assuming resize function

# Create a 10x10 square image
def create_square_image():
    img = np.zeros((10, 10))
    img[3:7, 3:7] = 1.0
    return img

# Resize using least-squares cubic interpolation
def resize_least_squares(input_image, zoom_factor):
    input_image_normalized = (input_image / 255.0).astype(np.float64)
    resized_image = resize(
        data=input_image_normalized,
        zoom_factors=(zoom_factor, zoom_factor),
        degree=3,
        method="least-squares"
    )
    return resized_image

# Initialize image and parameters
input_image = create_square_image()
zoom_factor = 3.0  # Set zoom factor for resizing down to 5x5

# Print the original input image
print("Original Input Image (10x10):")
print(input_image / 255.0)  # Print normalized input image

# Step 1: Resize down to 5x5
downscaled_image = resize_least_squares(input_image, zoom_factor)
print("\nDownscaled Image (5x5):")
print(downscaled_image)  # Print downscaled image

# Step 2: Resize back to 10x10
reverted_image = resize(
    data=downscaled_image,
    output_size=(10, 10),
    degree=3,
    method="least-squares"
)
print("\nReverted Image (10x10):")
print(reverted_image)  # Print reverted image
