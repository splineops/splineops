"""Basis precision contracts across NumPy scalar-promotion rules."""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from splineops import TensorSpline
from splineops.spline_interpolation.bases.utils import asbasis, basis_map


@pytest.mark.parametrize("name", basis_map)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_basis_preserves_floating_array_precision(name, dtype):
    coordinates = np.linspace(-6, 6, 258, dtype=dtype).reshape(6, 43)[:, ::-2]
    before = coordinates.copy()
    coordinates.flags.writeable = False
    values = asbasis(name)(coordinates)
    assert values.dtype == coordinates.dtype
    assert values.shape == coordinates.shape
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(coordinates, before)


@pytest.mark.parametrize("degree", [8, 9])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scalar", [False, True])
def test_high_degree_basis_precision_and_independent_values(degree, dtype, scalar):
    knots = np.arange(degree + 2, dtype=np.float64) - (degree + 1) / 2
    typed_knots = knots.astype(dtype)
    if scalar:
        coordinates = np.array(0.25, dtype=dtype)
    else:
        coordinates = np.unique(
            np.concatenate(
                [
                    np.linspace(knots[0] - 0.5, knots[-1] + 0.5, 4097, dtype=dtype),
                    typed_knots,
                    np.nextafter(typed_knots, dtype(-np.inf)),
                    np.nextafter(typed_knots, dtype(np.inf)),
                ]
            )
        )
    # Exercise eval directly as well as __call__ in the array-contract test.
    values = asbasis(f"bspline{degree}").eval(coordinates)
    expected = BSpline.basis_element(knots, extrapolate=False)(
        coordinates.astype(np.float64)
    )
    expected = np.nan_to_num(expected, nan=0.0)
    assert values.dtype == coordinates.dtype
    assert values.shape == coordinates.shape
    # High-degree float32 polynomials have cancellation near the support edge.
    # This absolute budget checks the existing evaluation, not a new formula.
    tolerance = 8e-5 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(values, expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("degree", [8, 9])
def test_high_degree_integer_coordinates_keep_fractional_basis_values(degree):
    coordinates = np.arange(-5, 6, dtype=np.int64)
    values = asbasis(f"bspline{degree}")(coordinates)
    knots = np.arange(degree + 2, dtype=np.float64) - (degree + 1) / 2
    expected = np.nan_to_num(
        BSpline.basis_element(knots, extrapolate=False)(coordinates), nan=0.0
    )
    assert values.dtype.kind == "f"
    assert 0 < values[5] < 1
    np.testing.assert_allclose(values, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("degree", [8, 9])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
def test_high_degree_interpolation_preserves_precision_and_samples(degree, dtype, mode):
    coordinates = np.arange(32, dtype=dtype)
    data = np.sin(coordinates * 0.2).astype(dtype)
    spline = TensorSpline(data, (coordinates,), bases=f"bspline{degree}", modes=mode)
    reproduced = spline((coordinates,))
    assert reproduced.dtype == data.dtype
    # Match the existing high-degree cardinal-reproduction precision budgets.
    tolerance = 2e-4 if dtype == np.float32 else 1e-8
    np.testing.assert_allclose(reproduced, data, rtol=0, atol=tolerance)
