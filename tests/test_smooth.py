import numpy as np
import pytest
from splineops.smooth.smoothing_spline import smoothing_spline
from splineops.smooth.smoothing_spline import recursive_smoothing_spline

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
