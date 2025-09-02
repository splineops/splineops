# splineops/tests/test_03_02_rotate_3d.py

import numpy as np
import pytest
from splineops.rotate.rotate import rotate

def generate_3d_rotated_data_and_mask(data_shape, custom_center, margin, angle, k, axis):
    ndim = 3

    # Create a coordinate grid
    grid = np.meshgrid(*[np.arange(dim) for dim in data_shape], indexing="ij")

    # Center the coordinates relative to the custom center
    coords = np.stack([g - c for g, c in zip(grid, custom_center)], axis=0)  # Shape: (3, N, N, N)
    coords_flat = coords.reshape(ndim, -1)  # Shape: (3, N*N*N)

    # Define the function f(x, y, z) relative to the custom center
    data = (
        np.sin(k * coords_flat[0, :]) +
        np.cos(k * coords_flat[1, :]) +
        np.sin(k * coords_flat[2, :])
    )
    data = data.reshape(data_shape)

    # Create a mask with margins on the original volume
    mask_original = np.zeros(data_shape, dtype=bool)
    mask_original[
        margin:data_shape[0] - margin,
        margin:data_shape[1] - margin,
        margin:data_shape[2] - margin,
    ] = True

    # Rotate the data
    # We use degree 1 as degree 3 is much more computationally expensive
    data_rotated = rotate(data, angle=angle, center=custom_center, degree=1, axis=axis)

    # Rotate the mask
    mask_rotated = rotate(
        mask_original.astype(float), angle=angle, axis=axis, center=custom_center, degree=0
    )
    mask_rotated = mask_rotated > 0.5  # Convert back to boolean

    return data, data_rotated, mask_original, mask_rotated, coords_flat

def compute_3d_expected_data(coords_flat, angle, k, axis):
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
    return data_expected_flat

@pytest.mark.parametrize("N, margin, custom_center, angle, k, tolerance, axis", [
    # Basic cases
    (128, 20, (64, 64, 64), 45, 0.1, 4e-3, (1, 1, 1)),
    (64, 10, (32, 32, 32), 90, 0.2, 4e-3, (0, 1, 0)),

    # Custom center and varied angles
    (128, 20, (32, 64, 96), 30, 0.15, 6e-3, (1, 0, 0)),
    (128, 20, (96, 96, 96), 60, 0.1, 4e-3, (0, 0, 1)),

    # Large volume and fine-grained rotation
    (256, 30, (128, 128, 128), 5, 0.1, 3e-2, (1, 0, 1)),
    (256, 30, (128, 64, 192), 15, 0.25, 3e-2, (1, 1, 0)),

    # Extreme angles
    (128, 20, (64, 64, 64), 0, 0.1, 4e-3, (1, 1, 1)),
    (128, 20, (64, 64, 64), 360, 0.1, 4e-3, (1, 1, 1)),
    (128, 20, (64, 64, 64), -45, 0.1, 4e-3, (1, 1, 1)),
])
def test_3d_rotate_with_center(N, margin, custom_center, angle, k, tolerance, axis):
    data_shape = (N, N, N)
    data, data_rotated, mask_original, mask_rotated, coords_flat = generate_3d_rotated_data_and_mask(
        data_shape, custom_center, margin, angle, k, axis
    )
    data_expected_flat = compute_3d_expected_data(coords_flat, angle, k, axis)
    data_expected = data_expected_flat.reshape(data_shape)

    # Compute the difference within the valid region
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference[mask_rotated]))

    # Assert the difference is within the tolerance
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"
