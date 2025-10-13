# splineops/tests/test_03_01_rotate_2d.py

import numpy as np
import pytest
from splineops.rotate.rotate import rotate

def generate_rotated_data_and_mask(data_shape, custom_center, margin, angle, k):
    ndim = 2

    # Create a coordinate grid
    grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")

    # Center the coordinates relative to the custom center
    coords = np.stack([g - c for g, c in zip(grid, custom_center)], axis=0)
    coords_flat = coords.reshape(ndim, -1)

    # Define the function f(x, y) relative to the custom center
    data = np.sin(k * coords_flat[0, :]) + np.cos(k * coords_flat[1, :])
    data = data.reshape(data_shape)

    # Create a mask with margins on the original image
    mask_original = np.zeros(data_shape, dtype=bool)
    mask_original[margin:data_shape[0] - margin, margin:data_shape[1] - margin] = True

    # Rotate the data
    data_rotated = rotate(data, angle=angle, center=custom_center, degree=3)

    # Rotate the mask
    mask_rotated = rotate(mask_original.astype(float), angle=angle, center=custom_center, degree=0)
    mask_rotated = mask_rotated > 0.5  # Convert back to boolean

    return data, data_rotated, mask_original, mask_rotated, coords_flat

def compute_expected_data(coords_flat, angle, k):
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
    return data_expected_flat

@pytest.mark.parametrize("N, margin, custom_center, angle, k, tolerance", [
    # Basic cases
    (500, 50, (250, 250), 30, 0.1, 1e-6),
    (300, 30, (150, 150), 45, 0.2, 1e-5),

    # Custom center at different locations
    (400, 40, (200, 100), 60, 0.15, 1e-5),
    (600, 50, (300, 300), 90, 0.1, 1e-5),
    (500, 50, (100, 400), 120, 0.05, 1e-5),

    # Large N with varied angles
    (1000, 100, (500, 500), 15, 0.2, 1e-5),
    (1000, 100, (800, 200), 135, 0.3, 5e-5),

    # Small N with fine-grained rotation
    (200, 20, (100, 100), 3, 0.1, 1e-6),
    (200, 20, (50, 150), 273, 0.25, 3e-5),

    # Extreme angles
    (500, 50, (250, 250), 0, 0.1, 1e-6),
    (500, 50, (250, 250), 361, 0.1, 1e-6),
    (500, 50, (250, 250), -44, 0.1, 1e-6),
])
def test_rotate_with_center(N, margin, custom_center, angle, k, tolerance):
    data_shape = (N, N)
    data, data_rotated, mask_original, mask_rotated, coords_flat = generate_rotated_data_and_mask(
        data_shape, custom_center, margin, angle, k
    )
    data_expected_flat = compute_expected_data(coords_flat, angle, k)
    data_expected = data_expected_flat.reshape(data_shape)

    # Compute the difference within the valid region
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference[mask_rotated]))

    # Assert the difference is within the tolerance
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"
