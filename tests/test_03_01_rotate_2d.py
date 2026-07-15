# splineops/tests/test_03_01_rotate_2d.py

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import splineops.affine.affine as affine_module
from scipy.ndimage import affine_transform
from splineops.affine.affine import (
    AffineCoefficientField,
    AffinePlan,
    affine_transform as spline_affine,
    rotate,
)


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
    mask_original[margin : data_shape[0] - margin, margin : data_shape[1] - margin] = (
        True
    )

    # Rotate the data
    data_rotated = rotate(data, angle=angle, center=custom_center, degree=3)

    # Rotate the mask
    mask_rotated = rotate(
        mask_original.astype(float), angle=angle, center=custom_center, degree=0
    )
    mask_rotated = mask_rotated > 0.5  # Convert back to boolean

    return data, data_rotated, mask_original, mask_rotated, coords_flat


def compute_expected_data(coords_flat, angle, k):
    angle_rad = np.radians(angle)
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)

    # Rotation matrix to compute expected data
    R = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # Apply rotation to the coordinates to get the rotated coordinates
    rotated_coords_flat = R @ coords_flat

    # Compute the expected data at the rotated coordinates
    data_expected_flat = np.sin(k * rotated_coords_flat[0, :]) + np.cos(
        k * rotated_coords_flat[1, :]
    )
    return data_expected_flat


@pytest.mark.parametrize(
    "N, margin, custom_center, angle, k, tolerance",
    [
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
    ],
)
def test_rotate_with_center(N, margin, custom_center, angle, k, tolerance):
    data_shape = (N, N)
    data, data_rotated, mask_original, mask_rotated, coords_flat = (
        generate_rotated_data_and_mask(data_shape, custom_center, margin, angle, k)
    )
    data_expected_flat = compute_expected_data(coords_flat, angle, k)
    data_expected = data_expected_flat.reshape(data_shape)

    # Compute the difference within the valid region
    difference = data_rotated - data_expected
    max_diff = np.max(np.abs(difference[mask_rotated]))

    # Assert the difference is within the tolerance
    assert (
        max_diff < tolerance
    ), f"Max difference {max_diff} exceeds tolerance {tolerance}"


def test_rotate_promotes_integer_input() -> None:
    result = rotate(np.arange(16).reshape(4, 4), angle=0.0, degree=1)

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.arange(16).reshape(4, 4))


@pytest.mark.parametrize("degree", [-1, 8])
def test_rotate_rejects_out_of_range_degree(degree):
    with pytest.raises(ValueError, match="degree"):
        rotate(np.ones((3, 3)), angle=0.0, degree=degree)


def test_rotate_is_invariant_to_coordinate_tile_size(monkeypatch):
    data = np.arange(63, dtype=np.float64).reshape(7, 9)

    monkeypatch.setattr(affine_module, "_AFFINE_TILE_SIZE", 10_000)
    untiled = rotate(data, angle=23.0, degree=3, mode="mirror")
    monkeypatch.setattr(affine_module, "_AFFINE_TILE_SIZE", 7)
    tiled = rotate(data, angle=23.0, degree=3, mode="mirror")

    np.testing.assert_allclose(tiled, untiled, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("degree", range(6))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotate_2d_matches_equivalent_scipy_transform(degree, dtype):
    rng = np.random.default_rng(20260715 + degree)
    data = rng.standard_normal((23, 19)).astype(dtype)
    angle = 23.0
    center = np.array((7.5, 11.25), dtype=dtype)
    angle_rad = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=dtype,
    )
    offset = center - matrix @ center

    actual = rotate(
        data,
        angle=angle,
        center=tuple(center),
        degree=degree,
        mode="mirror",
    )
    expected = affine_transform(
        data,
        matrix,
        offset=offset,
        output_shape=data.shape,
        order=degree,
        mode="mirror",
        prefilter=True,
    )

    tolerance = 2e-4 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_affine_plan_reuses_fixed_geometry_across_images():
    shape = (13, 17)
    angle = np.radians(-19.0)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = (np.asarray(shape) - 1.0) / 2.0
    plan = AffinePlan(
        shape,
        matrix,
        center - matrix @ center,
        degree=3,
        mode="mirror",
    )
    rng = np.random.default_rng(20260715)

    assert plan.geometry_cached
    assert plan.retained_bytes > 0
    for data in (rng.standard_normal(shape), rng.standard_normal(shape)):
        expected = spline_affine(
            data,
            matrix,
            center - matrix @ center,
            degree=3,
            mode="mirror",
        )
        np.testing.assert_allclose(plan(data), expected, rtol=0.0, atol=0.0)


def test_rotate_supports_explicit_batch_and_channel_axes():
    rng = np.random.default_rng(20260715)
    data = rng.standard_normal((2, 11, 13, 3))

    result = rotate(
        data,
        angle=17.0,
        degree=1,
        mode="mirror",
        spatial_axes=(1, 2),
    )
    expected = np.empty_like(result)
    for batch in range(data.shape[0]):
        for channel in range(data.shape[-1]):
            expected[batch, :, :, channel] = rotate(
                data[batch, :, :, channel],
                angle=17.0,
                degree=1,
                mode="mirror",
            )

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_affine_plan_supports_output_buffer():
    data = np.arange(63, dtype=np.float64).reshape(7, 9)
    plan = AffinePlan(data.shape, np.eye(2), degree=1, mode="mirror")
    expected = plan(data)
    out = np.empty_like(expected)

    returned = plan(data, out=out)

    assert returned is out
    np.testing.assert_equal(out, expected)


def test_affine_plan_prefilters_once_and_applies_precomputed_coefficients():
    rng = np.random.default_rng(20260716)
    data = rng.standard_normal((2, 11, 13, 3))
    angle = np.radians(-11.0)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = (np.asarray((11, 13)) - 1.0) / 2.0
    plan = AffinePlan(
        (11, 13), matrix, center - matrix @ center, degree=3, mode="mirror"
    )

    coefficient_out = np.empty_like(data)
    coefficients = plan.prefilter(data, spatial_axes=(1, 2), out=coefficient_out)
    actual = plan.apply_coefficients(coefficients, spatial_axes=(1, 2))
    expected = plan(data, spatial_axes=(1, 2))

    assert coefficients is coefficient_out
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert plan.configuration["geometry_cached"] is True
    assert plan.retained_bytes == (
        plan.geometry_retained_bytes + plan.template_retained_bytes
    )
    with pytest.raises(TypeError, match="dtype"):
        plan.apply_coefficients(coefficients.astype(np.float32), spatial_axes=(1, 2))


def test_affine_coefficient_field_checks_provenance_and_reuses_across_geometries():
    rng = np.random.default_rng(20260717)
    data = rng.standard_normal((2, 11, 13, 3))
    first = AffinePlan((11, 13), np.eye(2), degree=3, mode="mirror")
    second = AffinePlan(
        (11, 13),
        np.array([[1.0, 0.05], [-0.03, 1.0]]),
        output_shape=(9, 15),
        degree=3,
        mode="mirror",
    )

    field = first.prepare_coefficients(data, spatial_axes=(1, 2))

    assert field.is_compatible(first)
    assert field.is_compatible(second)
    assert field.shape == data.shape
    assert field.dtype == data.dtype
    assert field.nbytes == data.nbytes
    assert field.spatial_axes == (1, 2)
    assert field.configuration == {
        "input_shape": (11, 13),
        "spatial_axes": (1, 2),
        "degree": 3,
        "mode": "mirror",
        "dtype": "<f8",
    }
    np.testing.assert_allclose(
        second.apply_coefficients(field),
        second(data, spatial_axes=(1, 2)),
        rtol=0.0,
        atol=0.0,
    )
    copied_values = field.values
    copied_values.fill(0.0)
    assert np.any(field.values)
    with pytest.raises(AttributeError, match="immutable"):
        field._spatial_axes = (0, 1)

    incompatible = AffinePlan((11, 13), np.eye(2), degree=1, mode="mirror")
    assert not field.is_compatible(incompatible)
    with pytest.raises(ValueError, match="Incompatible coefficient field"):
        incompatible.apply_coefficients(field)
    with pytest.raises(ValueError, match="differs from the tagged"):
        first.apply_coefficients(field, spatial_axes=(0, 1))


def test_affine_coefficient_field_safe_roundtrip_and_concurrent_reuse(tmp_path):
    rng = np.random.default_rng(20260718)
    data = rng.standard_normal((2, 17, 19, 2))
    source = AffinePlan((17, 19), np.eye(2), degree=3, mode="mirror")
    target = AffinePlan(
        (17, 19),
        np.array([[1.0, 0.04], [-0.02, 1.0]]),
        degree=3,
        mode="mirror",
    )
    field = source.prepare_coefficients(data, spatial_axes=(1, 2))
    archive = tmp_path / "coefficients.npz"

    field.save(archive)
    restored = target.load_coefficients(archive)
    expected = target.apply_coefficients(field)

    assert restored.configuration == field.configuration
    assert restored.spatial_axes == field.spatial_axes
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _: target.apply_coefficients(restored), range(8))
        )
    for result in results:
        np.testing.assert_equal(result, expected)

    incompatible = AffinePlan((17, 19), np.eye(2), degree=1, mode="mirror")
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.load_coefficients(archive)
    with pytest.raises(TypeError, match="Construct coefficient fields"):
        AffineCoefficientField(
            np.ones((17, 19)),
            (0, 1),
            object(),
            {},
            {},
        )


def test_affine_plan_enforces_geometry_memory_limit():
    with pytest.raises(MemoryError, match="max_retained_bytes"):
        AffinePlan((20, 20), np.eye(2), max_retained_bytes=1)
