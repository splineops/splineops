"""Guard input ownership and exact identity paths without relying on solvers."""

import numpy as np
import pytest

from splineops.adaptive_regression_splines import DenoisingPlan, denoise_y
from splineops.adaptive_regression_splines.sparsification import sparsest_interpolant
from splineops.spline_interpolation.bases.utils import asbasis
from splineops.spline_interpolation.modes.finite_support_coefficients import (
    FiniteSupportCoefficients,
)


@pytest.mark.parametrize("basis_name", ["linear", "bspline3", "omoms5"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("strided", [False, True])
def test_zero_boundary_coefficients_preserve_readonly_input(basis_name, dtype, strided):
    rng = np.random.default_rng(731)
    samples = rng.normal(size=(2, 3, 17)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        samples += 1j * rng.normal(size=samples.shape).astype(dtype)
    if strided:
        samples = samples.transpose(1, 0, 2)[..., ::-1]
    original = samples.copy()
    samples.flags.writeable = False
    basis = asbasis(basis_name)

    coefficients = FiniteSupportCoefficients.compute_coefficients(samples, basis)

    # Independently assemble the finite cardinal interpolation equations.
    grid = np.arange(samples.shape[-1], dtype=float)
    matrix = basis(grid[:, None] - grid[None, :])
    reference = np.linalg.solve(matrix, samples.reshape(-1, grid.size).T).T
    tolerance = 2e-6 if samples.real.dtype == np.float32 else 2e-13
    np.testing.assert_allclose(
        coefficients, reference.reshape(samples.shape), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_array_equal(samples, original)
    assert coefficients.dtype == samples.dtype
    assert not np.shares_memory(coefficients, samples)
    coefficients[...] = 0
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("entry_point", ["function", "plan", "path"])
def test_zero_regularization_is_exact_for_large_finite_observations(entry_point):
    x = np.arange(7, dtype=float)
    y = np.array([1, 0.8, -0.9, 0.5, 1, -0.8, 0.7]) * 1e308
    original = y.copy()
    y.flags.writeable = False
    plan = DenoisingPlan(x)
    with np.errstate(all="raise"):
        if entry_point == "function":
            result, diagnostics = denoise_y(x, y, 0, return_diagnostics=True)
        elif entry_point == "plan":
            result, diagnostics = plan.solve(y, 0, return_diagnostics=True)
        else:
            result, reports = plan.solve_path(y, [0, 0], return_diagnostics=True)
            np.testing.assert_array_equal(result[1], original)
            diagnostics = reports[0]
            assert reports[1] == diagnostics
            result = result[0]
    np.testing.assert_array_equal(result, original)
    assert not np.shares_memory(result, y)
    assert diagnostics.converged
    assert diagnostics.iterations == 0
    assert diagnostics.primal_residual == diagnostics.dual_residual == 0


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="Platform has no wider floating-point dtype.",
)
@pytest.mark.parametrize("entry_point", ["function", "plan", "sparse_x", "sparse_y"])
def test_regression_rejects_values_unrepresentable_in_solver_precision(entry_point):
    ordinary = np.arange(3, dtype=float)
    large = ordinary.astype(np.longdouble)
    large[-1] = np.finfo(np.longdouble).max / 2
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite"):
        if entry_point == "function":
            denoise_y(large, ordinary, 0)
        elif entry_point == "plan":
            DenoisingPlan(large)
        elif entry_point == "sparse_x":
            sparsest_interpolant(large, ordinary)
        else:
            sparsest_interpolant(ordinary, large)
