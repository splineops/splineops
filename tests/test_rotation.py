import numpy as np
from splineops.interpolate.rotate import rotate
import matplotlib.pyplot as plt

def test_rotate():

    N = 100  # Image size
    data_shape = (N, N)
    ndim = 2

    # Create a coordinate grid centered at the image center
    grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")
    center_coords = [(dim - 1) / 2.0 for dim in data_shape]

    # Center the coordinates
    coords = np.stack([g - c for g, c in zip(grid, center_coords)], axis=0)  # Shape: (ndim, N, N)
    coords_flat = coords.reshape(ndim, -1)  # Shape: (ndim, N*N)

    # Define the function f(x, y)
    k = 0.1  # Spatial frequency
    data = np.sin(k * coords_flat[0, :]) + np.cos(k * coords_flat[1, :])
    data = data.reshape(data_shape)

    # Rotate the data using the rotate function
    angle = 30  # Rotation angle in degrees
    data_rotated = rotate(data, angle=angle, degree=3, center=None)

    # Compute the expected data by rotating the coordinates
    angle_rad = np.radians(angle)
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)

    # Rotation matrix as used in the rotate function
    R = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Apply rotation to the coordinates
    rotated_coords_flat = R @ coords_flat

    # Compute the expected data at the rotated coordinates
    data_expected_flat = np.sin(k * rotated_coords_flat[0, :]) + np.cos(k * rotated_coords_flat[1, :])
    data_expected = data_expected_flat.reshape(data_shape)

    # Compute the difference between the rotated data and the expected data
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference))
    print(f"Maximum difference: {max_diff}")

    # Plot the original data, rotated data, expected data, and the difference
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Original image (generated data)
    axs[0].imshow(data, cmap='viridis', origin='lower')
    axs[0].set_title("Original Image")

    # Rotated image
    axs[1].imshow(data_rotated, cmap='viridis', origin='lower')
    axs[1].set_title(f"Rotated Image (Angle: {angle}°)")

    # Expected image (analytically calculated)
    axs[2].imshow(data_expected, cmap='viridis', origin='lower')
    axs[2].set_title("Expected Rotated Image")

    # Difference image
    im = axs[3].imshow(difference, cmap='coolwarm', origin='lower')
    axs[3].set_title("Difference (Rotated - Expected)")
    fig.colorbar(im, ax=axs[3], orientation='vertical', label='Difference')

    plt.tight_layout()
    plt.show()

    # Set a tolerance for the maximum acceptable difference
    tolerance = 1e-5
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

# Run the test
test_rotate()