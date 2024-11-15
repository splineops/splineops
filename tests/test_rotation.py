import numpy as np
from splineops.interpolate.rotate import rotate
import matplotlib.pyplot as plt

def test_rotate():

    N = 500  # Image size
    data_shape = (N, N)
    ndim = 2
    margin = 20  # Margin to exclude around the original boundaries

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

    # Create a mask with margins on the original image
    mask_original = np.zeros(data_shape, dtype=bool)
    mask_original[margin:N-margin, margin:N-margin] = True  # Exclude margins

    # Rotate the data using the rotate function
    angle = 30  # Rotation angle in degrees
    data_rotated = rotate(data, angle=angle, degree=3, center=None)

    # Rotate the mask using the same rotate function
    mask_rotated = rotate(mask_original.astype(float), angle=angle, degree=0, center=None)
    # Since the mask is binary, use degree=0 (nearest neighbor) interpolation

    # Threshold the rotated mask to get back to a binary mask
    mask_rotated = mask_rotated > 0.5  # Convert back to boolean

    # Compute the expected data by rotating the coordinates
    angle_rad = np.radians(angle)
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)

    # Rotation matrix to compute expected data
    R = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Apply rotation to the coordinates to get the rotated coordinates
    rotated_coords_flat = R @ coords_flat

    # Compute the expected data at the rotated coordinates
    data_expected_flat = np.sin(k * rotated_coords_flat[0, :]) + np.cos(k * rotated_coords_flat[1, :])
    data_expected = data_expected_flat.reshape(data_shape)

    # Compute the difference between the rotated data and the expected data within the valid region
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference[mask_rotated]))
    print(f"Maximum difference within the valid region: {max_diff}")

    # Mask the difference array for plotting
    difference_masked = np.copy(difference)
    difference_masked[~mask_rotated] = np.nan  # Exclude invalid pixels from the difference image

    # Plot the original data, rotated data, expected data, and the difference
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    # Original image (generated data)
    axs[0].imshow(data, cmap='viridis', origin='lower')
    axs[0].set_title("Original Image")

    # Original mask
    axs[1].imshow(mask_original, cmap='gray', origin='lower')
    axs[1].set_title("Original Mask")

    # Rotated image
    axs[2].imshow(data_rotated, cmap='viridis', origin='lower')
    axs[2].set_title(f"Rotated Image (Angle: {angle}°)")

    # Expected image (analytically calculated)
    axs[3].imshow(data_expected, cmap='viridis', origin='lower')
    axs[3].set_title("Expected Rotated Image")

    # Difference image within the valid region
    im = axs[4].imshow(difference_masked, cmap='coolwarm', origin='lower')
    axs[4].set_title("Difference (Rotated - Expected)")
    fig.colorbar(im, ax=axs[4], orientation='vertical', label='Difference')

    plt.tight_layout()
    plt.show()

    # Set a tolerance for the maximum acceptable difference
    tolerance = 1e-5
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

# Run the test
test_rotate()
