"""Compiled affine evaluation against the independent NumPy spline path."""

import importlib

import numpy as np
import pytest

from splineops.affine import AffinePlan, affine_transform, rotate
from splineops.spline_interpolation.modes.narrow_mirroring import NarrowMirroring

affine_module = importlib.import_module("splineops.affine.affine")


@pytest.mark.parametrize("shape", [(13, 17), (7, 9, 11), (1, 5), (2, 1, 3)])
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_evaluation_preserves_spline_contract(shape, degree, mode, dtype):
    rng = np.random.default_rng(20261028 + degree)
    data = rng.normal(size=shape).astype(dtype)
    matrix = np.eye(len(shape)) + rng.uniform(-0.25, 0.25, (len(shape), len(shape)))
    offset = rng.uniform(-2, 2, len(shape))
    output_shape = tuple(size + 2 for size in shape)
    params = dict(output_shape=output_shape, degree=degree, mode=mode, dtype=dtype)
    reference = AffinePlan(shape, matrix, offset, backend="numpy", **params)
    compiled = AffinePlan(shape, matrix, offset, backend="scipy", **params)
    before = data.copy()
    expected = reference(data)
    actual = compiled(data)
    tolerance = 2e-5 if dtype == np.float32 else 3e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(data, before)
    assert actual.dtype == dtype
    # Both backends deliberately use exactly the same sample-to-coefficient
    # operator, including the finite-support zero-boundary solve.
    np.testing.assert_array_equal(compiled.prefilter(data), reference.prefilter(data))


@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_boundary_knots_and_multiple_extension_periods(mode, dtype):
    data = np.arange(20, dtype=dtype).reshape(4, 5)
    for offset in [(-0.5, 4.5), (-19.5, 22.5), (0.0, 0.0), (3.0, 4.0)]:
        params = dict(output_shape=(3, 4), degree=3, mode=mode)
        actual = affine_transform(data, np.eye(2), offset, **params)
        expected = affine_transform(data, np.eye(2), offset, backend="numpy", **params)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_float32_preserves_rounded_coordinates_far_from_origin():
    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    params = dict(degree=3, mode="periodic")
    matrix = np.array([[0.93, 0.17], [-0.11, 1.07]])
    offset = (1000000.3, -1000000.2)
    actual = affine_transform(data, matrix, offset, **params)
    expected = affine_transform(data, matrix, offset, backend="numpy", **params)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
def test_strided_batches_output_and_cross_backend_coefficient_reuse(
    dtype, mode, tmp_path
):
    rng = np.random.default_rng(13)
    # Two non-spatial axes, reversed strides, and reversed spatial-axis order.
    data = rng.normal(size=(7, 2, 9, 3)).astype(dtype)[::-1, :, ::-1, :]
    axes = (2, 0)
    params = dict(output_shape=(8, 6), degree=3, mode=mode, dtype=dtype)
    matrix = np.array([[1.03, 0.15], [-0.2, 0.95]])
    reference = AffinePlan((9, 7), matrix, (-0.4, 0.8), backend="numpy", **params)
    compiled = AffinePlan((9, 7), matrix, (-0.4, 0.8), **params)
    expected = reference(data, spatial_axes=axes)
    destination = np.empty((6, 2, 8, 6), dtype=dtype)[..., ::2]
    returned = compiled(data, spatial_axes=axes, out=destination)
    assert returned is destination
    tolerance = 3e-6 if dtype == np.float32 else 3e-12
    np.testing.assert_allclose(returned, expected, rtol=tolerance, atol=tolerance)
    field = reference.prepare_coefficients(data, spatial_axes=axes)
    assert field.is_compatible(compiled)
    np.testing.assert_array_equal(compiled.apply_coefficients(field), returned)
    path = tmp_path / "coefficients.npz"
    field.save(path)
    restored = compiled.load_coefficients(path)
    assert restored.is_compatible(reference)
    np.testing.assert_array_equal(reference.apply_coefficients(restored), expected)


@pytest.mark.parametrize("backend", ["numpy", "scipy"])
def test_linear_affine_matches_analytic_polynomial(backend):
    i, j, k = np.indices((9, 11, 13), dtype=np.float64)
    data = 1.5 + 2 * i - 3 * j + 0.25 * k
    matrix = np.array([[0.5, 0.125, 0], [0, 0.5, 0.125], [0.125, 0, 0.5]])
    offset = np.array([1, 2, 3])
    grid = np.indices((5, 6, 7), dtype=np.float64).reshape(3, -1)
    x, y, z = matrix @ grid + offset[:, None]
    expected = (1.5 + 2 * x - 3 * y + 0.25 * z).reshape(5, 6, 7)
    actual = affine_transform(
        data,
        matrix,
        offset,
        output_shape=(5, 6, 7),
        degree=1,
        mode="mirror",
        backend=backend,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("degree", [0, 6, 7])
def test_unsupported_degrees_keep_reference_and_reject_forced_scipy(degree):
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    params = dict(degree=degree, mode="mirror")
    plan = AffinePlan(data.shape, np.eye(2), (-0.5, 0.5), **params)
    assert plan.backend == "numpy"
    reference = AffinePlan(
        data.shape, np.eye(2), (-0.5, 0.5), backend="numpy", **params
    )
    np.testing.assert_array_equal(plan(data), reference(data))
    with pytest.raises(ValueError, match="scipy backend"):
        AffinePlan(data.shape, np.eye(2), backend="scipy", **params)


def test_custom_extension_and_float16_keep_reference():
    assert AffinePlan((5, 7), np.eye(2), mode=NarrowMirroring()).backend == "numpy"
    assert AffinePlan((5, 7), np.eye(2), degree=1, dtype=np.float16).backend == "numpy"
    with pytest.raises(ValueError, match="backend"):
        AffinePlan((5, 7), np.eye(2), backend="invalid")


def test_compiled_plan_does_not_construct_or_cache_unused_spline_template(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Compiled evaluation must not build a zero-valued TensorSpline")

    monkeypatch.setattr(affine_module, "TensorSpline", fail)
    plan = AffinePlan((100, 120, 80), np.eye(3), max_retained_bytes=1)
    assert plan.backend == "scipy"
    assert not plan.geometry_cached
    assert plan.geometry_retained_bytes == plan.template_retained_bytes == 0


def test_rotate_forwards_backend_selection():
    data = np.arange(30, dtype=np.float64).reshape(5, 6)
    np.testing.assert_allclose(
        rotate(data, 23, backend="scipy"),
        rotate(data, 23, backend="numpy"),
        rtol=0,
        atol=1e-12,
    )


def test_compiled_aliased_output_is_safe():
    data = np.arange(63, dtype=np.float64).reshape(7, 9)
    expected = rotate(data, 17)
    assert rotate(data, 17, out=data) is data
    np.testing.assert_array_equal(data, expected)
