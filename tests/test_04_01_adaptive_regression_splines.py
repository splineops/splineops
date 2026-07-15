# splineops/tests/test_04_01_adaptive_regression_splines.py

import numpy as np
import pytest
from splineops.adaptive_regression_splines import DenoisingDiagnostics, denoise_y
from splineops.adaptive_regression_splines.sparsification import (
    _sparsify_amplitudes,
    linear_spline,
    sparsest_interpolant,
)


@pytest.mark.parametrize(
    "lambda_, expected_mse_upper_bound",
    [
        (0.0, 1e-2),
        (1e-4, 7e-3),
        (1e-2, 9e-4),
        (1e-1, 1e-15),
        (1.0, 1e-20),
    ],
)
def test_denoise_approaches_linear_regression(lambda_, expected_mse_upper_bound):
    """
    Test that as lambda increases, the TV-denoising approaches classical linear regression.

    Under total-variation regularization, larger lambda places more penalty
    on local variations, thus pushing the solution toward the global linear
    regression fit. Hence, as lambda grows, the denoised signal should match
    the direct linear fit more closely.

    We generate synthetic data y = a*x + b + noise with randomly chosen a, b,
    then run denoise_y with various lambda.
    For larger lambda, the MSE between the denoised result and the direct linear fit
    should be smaller.
    """
    np.random.seed(42)  # For reproducibility

    # Generate x-coordinates
    x = np.linspace(0, 1, 50)

    # Pick a random slope and intercept around some nominal values
    slope_true = 2.0 + 0.5 * np.random.randn()
    intercept_true = 1.0 + 0.3 * np.random.randn()

    # Inject more noise than before
    noise = 0.1 * np.random.randn(len(x))
    y_noisy = slope_true * x + intercept_true + noise

    # Reference: direct linear regression on the noisy data
    lin_coeffs = np.polyfit(x, y_noisy, 1)  # [slope, intercept]
    y_linfit = np.polyval(lin_coeffs, x)

    # Denoise using our ADMM-based total-variation module
    # (We also set rho = lambda_ for convenience, but you could choose another value.)
    y_denoised = denoise_y(x, y_noisy, lamb=lambda_, rho=lambda_)

    # Compare via Mean Squared Error (MSE)
    mse = np.mean((y_denoised - y_linfit) ** 2)

    # Assert that the MSE is within an acceptable bound.
    # As lambda grows, we expect the denoised data to approximate a purely linear function,
    # thus shrinking the MSE.
    assert (
        mse < expected_mse_upper_bound
    ), f"MSE={mse:.3e} exceeds tolerance {expected_mse_upper_bound:.3e} for λ={lambda_}"


def test_sparsify_amplitudes_does_not_mutate_and_preserves_sum() -> None:
    amplitudes = np.array([2.0, 1e-8, -2e-8, -1.0])
    original = amplitudes.copy()

    result = _sparsify_amplitudes(amplitudes, sparsity_tol=1e-5)

    np.testing.assert_equal(amplitudes, original)
    np.testing.assert_allclose(np.sum(result), np.sum(original), rtol=0.0, atol=1e-15)
    np.testing.assert_equal(result[1:3], 0.0)


def test_sparsest_interpolant_reproduces_piecewise_linear_samples() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    knots, amplitudes, polynomial = sparsest_interpolant(x, y)

    np.testing.assert_allclose(
        linear_spline(x, knots, amplitudes, polynomial), y, rtol=0.0, atol=1e-12
    )


def test_regression_rejects_unsorted_samples() -> None:
    x = np.array([0.0, 2.0, 1.0])
    y = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        denoise_y(x, y, lamb=0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        sparsest_interpolant(x, y)


def test_denoising_reports_convergence_without_changing_default_return():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x) + 0.05 * np.cos(17.0 * x)

    result, diagnostics = denoise_y(
        x,
        y,
        lamb=1e-4,
        rho=1e-4,
        relative_tol=1e-6,
        return_diagnostics=True,
    )

    assert isinstance(result, np.ndarray)
    assert isinstance(diagnostics, DenoisingDiagnostics)
    assert diagnostics.converged
    assert 0 < diagnostics.iterations < 10_000
    assert diagnostics.primal_residual >= 0.0
    assert diagnostics.dual_residual >= 0.0


def test_zero_regularization_is_identity_with_closed_form_diagnostics():
    x = np.linspace(0.0, 1.0, 8)
    y = 2.0 * x + 1.0

    result, diagnostics = denoise_y(x, y, lamb=0.0, return_diagnostics=True)

    np.testing.assert_equal(result, y)
    assert diagnostics == DenoisingDiagnostics(0, True, 0.0, 0.0)


def test_iteration_limit_is_visible_in_diagnostics():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)

    _, diagnostics = denoise_y(
        x,
        y,
        lamb=1e-4,
        max_iter=1,
        relative_tol=1e-14,
        return_diagnostics=True,
    )

    assert diagnostics.iterations == 1
    assert not diagnostics.converged
