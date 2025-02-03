import numpy as np
import pytest
from splineops.regress.denoising import denoise_y

@pytest.mark.parametrize("lambda_, expected_mse_upper_bound", [
    (0.0,   1e-2),
    (1e-4,  5e-3),
    (1e-2,  5e-4),
    (1e-1,  1e-15),
    (1.0,   1e-20),
])
def test_denoise_approaches_linear_regression(lambda_, expected_mse_upper_bound):
    """
    Test that as lambda increases, the TV-denoising approaches classical linear regression.
    
    We generate synthetic data y = a*x + b + noise, run denoise_y with various lambda,
    and then compare the denoised output against a direct linear regression fit.
    
    For larger lambda, the MSE between denoised and linear fit should be smaller.
    """
    np.random.seed(42)  # For reproducibility

    # Generate synthetic (roughly linear) data: y = 2*x + 1 + noise
    x = np.linspace(0, 1, 50)
    slope_true = 2.0
    intercept_true = 1.0
    noise = 0.05 * np.random.randn(len(x))
    y_noisy = slope_true * x + intercept_true + noise

    # Reference: direct linear regression on the noisy data
    lin_coeffs = np.polyfit(x, y_noisy, 1)   # [slope, intercept]
    y_linfit = np.polyval(lin_coeffs, x)

    # Denoise using our ADMM-based total-variation module
    # We also set rho = lambda_ (common choice), but you can fix rho = 1 if you prefer
    y_denoised = denoise_y(x, y_noisy, lamb=lambda_, rho=lambda_)

    # Compare via Mean Squared Error (MSE)
    mse = np.mean((y_denoised - y_linfit) ** 2)

    # For each lambda_ in the param list, we expect MSE < some upper bound
    # (As lambda_ grows, we expect the MSE to drop sharply.)
    assert mse < expected_mse_upper_bound, (
        f"MSE={mse:.3e} exceeds tolerance {expected_mse_upper_bound:.3e} for λ={lambda_}"
    )
