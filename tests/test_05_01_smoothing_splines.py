# splineops/tests/test_05_01_smoothing_splines.py

import numpy as np
import pytest
from splineops.smoothing_splines.smoothingspline import smoothing_spline, smoothing_spline_nd, recursive_smoothing_spline

@pytest.mark.parametrize("lambda_, gamma, tol", [
    (0.0005, 0.6, 1e-5),
    (0.005, 0.6, 1e-4),
    (0.05, 0.8, 1e-2),
    (0.1, 1.0, 1e-2),
    (0.2, 1.2, 1e-1),
    (0.5, 1.5, 1e-1),
])
def test_smoothing_spline(lambda_, gamma, tol):
    """
    Test that the Recursive Smoothing Spline gives results close to the Fractional Smoothing Spline
    within the specified tolerance.
    """

    # Generate a synthetic noisy sine wave
    x = np.linspace(0, 2 * np.pi, 100)
    signal = np.sin(x) + 0.1 * np.random.normal(size=x.shape)

    # Apply Fractional Smoothing Spline (Baseline)
    m = 1  # No upsampling
    _, smoothed_fractional = smoothing_spline(signal, lambda_, m, gamma)

    # Apply Recursive Smoothing Spline
    smoothed_recursive = recursive_smoothing_spline(signal, lamb=lambda_)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((smoothed_recursive - smoothed_fractional) ** 2)

    # Assert that the MSE is within the acceptable tolerance
    assert mse < tol, f"MSE {mse:.6e} exceeds tolerance {tol:.6e} for λ={lambda_}, γ={gamma}"


@pytest.mark.parametrize("lambda_, gamma, tol", [
    (0.00005, 0.6, 1e-6),
    (0.0005, 0.6, 1e-5),
    (0.005, 0.8, 1e-5),
    (0.05, 1.0, 1e-4),
    (0.1, 1.2, 1e-4),
    (0.2, 1.5, 1e-4),
])
def test_smoothing_spline_vs_nd(lambda_, gamma, tol):
    """
    Test that smoothing_spline and smoothing_spline_nd give the same results
    for 1D data within a given tolerance.
    """

    # Generate a synthetic noisy sine wave
    x = np.linspace(0, 2 * np.pi, 100)
    signal = np.sin(x) + 0.1 * np.random.normal(size=x.shape)

    # Apply 1D Fractional Smoothing Spline
    m = 1  # No upsampling
    _, smoothed_1d = smoothing_spline(signal, lambda_, m, gamma)

    # Apply smoothing_spline_nd (which is general for N-D but should match 1D case)
    smoothed_nd = smoothing_spline_nd(signal.reshape(-1, 1), lambda_, gamma).flatten()

    # Compute Mean Squared Error (MSE)
    mse = np.mean((smoothed_1d - smoothed_nd) ** 2)

    # Assert MSE is within the acceptable tolerance
    assert mse < tol, f"MSE {mse:.6e} exceeds tolerance {tol:.6e} for λ={lambda_}, γ={gamma}"
