import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.rotate import rotate

def test_single_rotation_with_center():
    """
    Test the `rotate` function with a custom center of rotation 
    and visualize the results.
    """

    # Create a test image (100x100 pixels with a white rectangle)
    data = np.zeros((100, 100))
    data[40:60, 40:50] = 1  # Add a simple white rectangle

    # Define the rotation angle and center
    angle = 45  # Degrees
    center = (50, 50)  # Custom center (row, column)

    # Rotate the image with the custom center
    rotated_data = rotate(data, angle=angle, degree=3, center=center)

    # Visualize the original and rotated images with the center marked
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axs[0].imshow(data, cmap="gray", origin="upper")  # Match visualization convention
    axs[0].scatter(center[1], center[0], color="red", label="Center of Rotation")  # Flip to (x, y) for scatter
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[0].legend()

    # Rotated image
    axs[1].imshow(rotated_data, cmap="gray", origin="upper")
    axs[1].scatter(center[1], center[0], color="red", label="Center of Rotation")
    axs[1].set_title(f"Rotated Image (Angle: {angle}°)")
    axs[1].axis("off")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Run the test
test_single_rotation_with_center()
