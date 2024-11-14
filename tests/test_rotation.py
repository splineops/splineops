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

    # Now create the mask
    # Coordinates of the rotated image centered at its center
    coords_rotated = coords.copy()
    coords_rotated_flat = coords_rotated.reshape(ndim, -1)

    # Apply inverse rotation to the rotated image coordinates to map back to the original image coordinates
    # Here, we use the positive angle because we're reversing the rotation
    cos_angle_inv = np.cos(angle_rad)
    sin_angle_inv = np.sin(angle_rad)

    # Inverse rotation matrix
    R_inv = np.array([
        [cos_angle_inv, sin_angle_inv],
        [-sin_angle_inv, cos_angle_inv]
    ])

    coords_original_flat = R_inv @ coords_rotated_flat

    # Shift coordinates back to original image indices
    coords_original_indices = coords_original_flat + np.array(center_coords)[:, np.newaxis]

    # Check which coordinates are within the bounds of the original image
    valid_mask = (
        (coords_original_indices[0, :] >= 0) & (coords_original_indices[0, :] <= N - 1) &
        (coords_original_indices[1, :] >= 0) & (coords_original_indices[1, :] <= N - 1)
    )

    valid_mask_image = valid_mask.reshape(data_shape)

    # Compute the difference between the rotated data and the expected data within the valid region
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference[valid_mask_image]))
    print(f"Maximum difference within the valid region: {max_diff}")

    # Mask the difference array for plotting
    difference_masked = np.copy(difference)
    difference_masked[~valid_mask_image] = np.nan  # Exclude invalid pixels from the difference image

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

    # Difference image within the valid region
    im = axs[3].imshow(difference_masked, cmap='coolwarm', origin='lower')
    axs[3].set_title("Difference (Rotated - Expected)")
    fig.colorbar(im, ax=axs[3], orientation='vertical', label='Difference')

    plt.tight_layout()
    plt.show()

    # Set a tolerance for the maximum acceptable difference
    tolerance = 1e-5
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

# Run the test
test_rotate()
