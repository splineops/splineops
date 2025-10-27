# splineops/tests/test_04_01_adaptive_regression_splines.py

import numpy as np
import pytest
from splineops.adaptive_regression_splines.denoising import denoise_y

@pytest.mark.parametrize("lambda_, expected_mse_upper_bound", [
    (0.0,   1e-2),
    (1e-4,  7e-3),
    (1e-2,  9e-4),
    (1e-1,  1e-15),
    (1.0,   1e-20),
])
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
    lin_coeffs = np.polyfit(x, y_noisy, 1)   # [slope, intercept]
    y_linfit = np.polyval(lin_coeffs, x)

    # Denoise using our ADMM-based total-variation module
    # (We also set rho = lambda_ for convenience, but you could choose another value.)
    y_denoised = denoise_y(x, y_noisy, lamb=lambda_, rho=lambda_)

    # Compare via Mean Squared Error (MSE)
    mse = np.mean((y_denoised - y_linfit) ** 2)

    # Assert that the MSE is within an acceptable bound.
    # As lambda grows, we expect the denoised data to approximate a purely linear function,
    # thus shrinking the MSE.
    assert mse < expected_mse_upper_bound, (
        f"MSE={mse:.3e} exceeds tolerance {expected_mse_upper_bound:.3e} for λ={lambda_}"
    )
