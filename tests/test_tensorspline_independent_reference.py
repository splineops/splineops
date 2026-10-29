"""Spline evaluation against independent basis functions and dense systems."""

import math

import numpy as np
import pytest
from scipy.interpolate import BSpline

from splineops import TensorSpline
from splineops.spline_interpolation.bases.utils import asbasis


def _bspline(x, degree, *, symmetric=False):
    x = np.asarray(x, dtype=float)
    if degree == 0:
        if symmetric:
            return np.where(np.abs(x) == 0.5, 0.5, (np.abs(x) < 0.5).astype(float))
        return ((x >= -0.5) & (x < 0.5)).astype(float)
    knots = np.arange(degree + 2) - (degree + 1) / 2
    return np.nan_to_num(BSpline.basis_element(knots, extrapolate=False)(x))


def _basis(x, degree, family, *, symmetric=False):
    # MOMS are linear combinations of a B-spline and its even derivatives
    # (Blu, Thevenaz & Unser, IEEE TIP 10(7), 2001, pp. 1069-1080).
    # Derivatives are independently formed as finite differences of lower
    # degree SciPy B-splines, not from SplineOps' piecewise polynomials/poles.
    # https://bigwww.epfl.ch/publications/blu0101.pdf
    weights = {
        0: (),
        1: (),
        2: (1 / 60,),
        3: (1 / 42,),
        4: (1 / 36, 1 / 15120),
        5: (1 / 33, 1 / 7920),
    }
    result = _bspline(x, degree, symmetric=symmetric)
    if family == "omoms":
        for term, weight in enumerate(weights[degree], start=1):
            derivative = 2 * term
            for shift in range(derivative + 1):
                result += (
                    weight
                    * (-1) ** shift
                    * math.comb(derivative, shift)
                    * _bspline(
                        x + derivative / 2 - shift,
                        degree - derivative,
                        symmetric=symmetric,
                    )
                )
    return result


CASES = [("bspline", n) for n in range(10)] + [("omoms", n) for n in range(6)]


@pytest.mark.parametrize("family,degree", CASES)
def test_basis_matches_independent_bspline_derivative_definition(family, degree):
    x = np.r_[np.linspace(-6, 6, 481), np.arange(-6, 6) + 0.137]
    expected = _basis(x, degree, family)
    np.testing.assert_allclose(
        asbasis(f"{family}{degree}")(x), expected, atol=5e-13, rtol=5e-13
    )


@pytest.mark.parametrize("degree", [0, 2, 4])
def test_symmetric_omoms_discontinuities_equal_the_two_sided_mean(degree):
    x = np.arange(-4, 4.25, 0.25)
    expected = _basis(x, degree, "omoms", symmetric=True)
    np.testing.assert_allclose(
        asbasis(f"omoms{degree}-sym")(x), expected, atol=5e-13, rtol=5e-13
    )


def _evaluation_matrix(query, size, degree, family, mode):
    # Explicit integer coefficient extension, independent of the library's
    # support enumeration and boundary index mapping.
    query = np.asarray(query)
    radius = (degree + 1) / 2
    matrix = np.zeros((query.size, size))
    start = math.floor(float(np.min(query)) - radius) - 1
    stop = math.ceil(float(np.max(query)) + radius) + 2
    for index in range(start, stop):
        if mode == "zero":
            if not 0 <= index < size:
                continue
            column = index
        elif mode == "periodic":
            column = index % size
        elif size == 1:
            column = 0
        else:
            column = index % (2 * (size - 1))
            if column >= size:
                column = 2 * (size - 1) - column
        matrix[:, column] += _basis(query - index, degree, family)
    return matrix


@pytest.mark.parametrize("family,degree", CASES)
@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
@pytest.mark.parametrize("size", [1, 2, 9])
def test_interpolant_matches_independent_dense_cardinal_solve(
    family, degree, mode, size
):
    coordinates = np.arange(size, dtype=float)
    samples = np.random.default_rng(size).normal(size=size)
    query = np.linspace(-3.137, size + 2.173, 31)
    cardinal = _evaluation_matrix(coordinates, size, degree, family, mode)
    coefficients = np.linalg.solve(cardinal, samples)
    expected = _evaluation_matrix(query, size, degree, family, mode) @ coefficients
    spline = TensorSpline(samples, coordinates, f"{family}{degree}", mode)
    np.testing.assert_allclose(spline(query), expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_mixed_basis_and_boundary_tensor_uses_independent_axis_systems(dtype):
    shape = (4, 5, 3)
    degrees = (3, 5, 8)
    families = ("omoms", "omoms", "bspline")
    modes = ("zero", "mirror", "periodic")
    rng = np.random.default_rng(40)
    samples = rng.normal(size=shape)
    if np.issubdtype(dtype, np.complexfloating):
        samples = samples + 1j * rng.normal(size=shape)
    samples = samples.astype(dtype)
    real_dtype = samples.real.dtype
    coordinates = tuple(np.arange(n, dtype=real_dtype) for n in shape)
    queries = tuple(np.linspace(-0.37, n + 0.23, 7, dtype=real_dtype) for n in shape)
    expected = samples.astype(np.complex128 if np.iscomplexobj(samples) else float)
    for axis, (size, degree, family, mode) in enumerate(
        zip(shape, degrees, families, modes)
    ):
        cardinal = _evaluation_matrix(np.arange(size), size, degree, family, mode)
        evaluation = _evaluation_matrix(
            queries[axis].astype(float), size, degree, family, mode
        )
        transform = np.linalg.solve(cardinal.T, evaluation.T).T
        expected = np.moveaxis(
            np.tensordot(transform, expected, axes=(1, axis)), 0, axis
        )
    spline = TensorSpline(
        samples, coordinates, tuple(f"{f}{d}" for f, d in zip(families, degrees)), modes
    )
    tolerance = 2e-5 if real_dtype == np.dtype(np.float32) else 2e-11
    np.testing.assert_allclose(
        spline(queries), expected, atol=tolerance, rtol=tolerance
    )
