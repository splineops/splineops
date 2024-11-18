import numpy as np
from splineops.interpolate.rotate import rotate
import matplotlib.pyplot as plt

def test_3d_rotate_with_center():
    """
    Test the 3D `rotate` function with a custom center and axis of rotation.
    """

    N = 128  # Volume size
    data_shape = (N, N, N)
    margin = 20  # Margin to exclude around the original boundaries

    # Define a custom center
    custom_center = (64, 64, 64)  # Example: center of the 3D volume

    # Create a coordinate grid
    grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")

    # Center the coordinates relative to the custom center
    coords = np.stack([g - c for g, c in zip(grid, custom_center)], axis=0)  # Shape: (3, N, N, N)
    coords_flat = coords.reshape(3, -1)  # Shape: (3, N*N*N)

    # Define the function f(x, y, z) relative to the custom center
    k = 0.1  # Spatial frequency
    data = (
        np.sin(k * coords_flat[0, :]) +
        np.cos(k * coords_flat[1, :]) +
        np.sin(k * coords_flat[2, :])
    )
    data = data.reshape(data_shape)

    # Create a mask with margins on the original volume
    mask_original = np.zeros(data_shape, dtype=bool)
    mask_original[
        margin:N-margin, margin:N-margin, margin:N-margin
    ] = True  # Exclude margins

    # Define the rotation angle and axis
    angle = 45  # Rotation angle in degrees
    axis = (1, 1, 1)  # Custom axis of rotation

    # Rotate the data using the rotate function
    data_rotated = rotate(data, angle=angle, center=custom_center, degree=3, axis=axis)

    # Rotate the mask using the same rotate function
    mask_rotated = rotate(
        mask_original.astype(float), angle=angle, axis=axis, center=custom_center, degree=0
    )
    # Since the mask is binary, use degree=0 (nearest neighbor) interpolation

    # Threshold the rotated mask to get back to a binary mask
    mask_rotated = mask_rotated > 0.5  # Convert back to boolean

    # Compute the expected data by rotating the coordinates
    angle_rad = np.radians(angle)
    ux, uy, uz = np.array(axis) / np.linalg.norm(axis)  # Normalize the axis
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)
    one_minus_cos = 1 - cos_angle

    # Rotation matrix using the axis-angle formula
    R = np.array([
        [cos_angle + ux**2 * one_minus_cos,
         ux * uy * one_minus_cos - uz * sin_angle,
         ux * uz * one_minus_cos + uy * sin_angle],
        [uy * ux * one_minus_cos + uz * sin_angle,
         cos_angle + uy**2 * one_minus_cos,
         uy * uz * one_minus_cos - ux * sin_angle],
        [uz * ux * one_minus_cos - uy * sin_angle,
         uz * uy * one_minus_cos + ux * sin_angle,
         cos_angle + uz**2 * one_minus_cos]
    ])

    # Apply rotation to the coordinates to get the rotated coordinates
    rotated_coords_flat = R @ coords_flat

    # Compute the expected data at the rotated coordinates
    data_expected_flat = (
        np.sin(k * rotated_coords_flat[0, :]) +
        np.cos(k * rotated_coords_flat[1, :]) +
        np.sin(k * rotated_coords_flat[2, :])
    )
    data_expected = data_expected_flat.reshape(data_shape)

    # Compute the difference between the rotated data and the expected data within the valid region
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference[mask_rotated]))
    print(f"Maximum difference within the valid region: {max_diff}")

    # Plot slices of the original, rotated, expected, and difference volumes
    fig, axs = plt.subplots(4, 3, figsize=(18, 24))

    # Slices to visualize (middle slices in each dimension)
    slices = [
        (data, "Original Data"),
        (mask_original.astype(float), "Original Mask"),
        (data_rotated, f"Rotated Data (Angle: {angle}°)"),
        (data_expected, "Expected Rotated Data"),
    ]

    for i, (volume, title) in enumerate(slices):
        axs[i, 0].imshow(volume[N // 2], cmap="viridis", origin="upper")
        axs[i, 0].set_title(f"{title} (XY plane)")
        axs[i, 1].imshow(volume[:, N // 2, :], cmap="viridis", origin="upper")
        axs[i, 1].set_title(f"{title} (XZ plane)")
        axs[i, 2].imshow(volume[:, :, N // 2], cmap="viridis", origin="upper")
        axs[i, 2].set_title(f"{title} (YZ plane)")

    # Difference volume
    difference_masked = np.copy(difference)
    difference_masked[~mask_rotated] = np.nan
    axs[3, 0].imshow(difference_masked[N // 2], cmap="coolwarm", origin="upper")
    axs[3, 0].set_title("Difference (XY plane)")
    axs[3, 1].imshow(difference_masked[:, N // 2, :], cmap="coolwarm", origin="upper")
    axs[3, 1].set_title("Difference (XZ plane)")
    axs[3, 2].imshow(difference_masked[:, :, N // 2], cmap="coolwarm", origin="upper")
    axs[3, 2].set_title("Difference (YZ plane)")

    plt.tight_layout()
    plt.show()

    # Set a tolerance for the maximum acceptable difference
    tolerance = 1e-5
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

# Run the test
test_3d_rotate_with_center()
