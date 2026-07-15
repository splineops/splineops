# splineops/tests/test_05_01_smoothing_splines.py

import numpy as np
import pytest
from splineops.smoothing_splines.fract_spline_auto_corr import fractsplineautocorr
from splineops.smoothing_splines.smoothing_spline import (
    SmoothingSplinePlan,
    periodize,
    recursive_smoothing_spline,
    smoothing_spline,
    smoothing_spline_nd,
)


@pytest.mark.parametrize(
    "lambda_, gamma, tol",
    [
        (0.0005, 0.6, 1e-5),
        (0.005, 0.6, 1e-4),
        (0.05, 0.8, 1e-2),
        (0.1, 1.0, 1e-2),
        (0.2, 1.2, 1e-1),
        (0.5, 1.5, 1e-1),
    ],
)
def test_smoothing_spline(lambda_, gamma, tol):
    """
    Test that the Recursive Smoothing Spline gives results close to the Fractional Smoothing Spline
    within the specified tolerance.
    """

    # Generate a synthetic noisy sine wave
    x = np.linspace(0, 2 * np.pi, 100)
    signal = np.sin(x) + 0.1 * np.random.default_rng(42).normal(size=x.shape)

    # Apply Fractional Smoothing Spline (Baseline)
    m = 1  # No upsampling
    _, smoothed_fractional = smoothing_spline(signal, lambda_, m, gamma)

    # Apply Recursive Smoothing Spline
    smoothed_recursive = recursive_smoothing_spline(signal, lamb=lambda_)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((smoothed_recursive - smoothed_fractional) ** 2)

    # Assert that the MSE is within the acceptable tolerance
    assert (
        mse < tol
    ), f"MSE {mse:.6e} exceeds tolerance {tol:.6e} for λ={lambda_}, γ={gamma}"


@pytest.mark.parametrize(
    "lambda_, gamma, tol",
    [
        (0.00005, 0.6, 1e-6),
        (0.0005, 0.6, 1e-5),
        (0.005, 0.8, 1e-5),
        (0.05, 1.0, 1e-4),
        (0.1, 1.2, 1e-4),
        (0.2, 1.5, 1e-4),
    ],
)
def test_smoothing_spline_vs_nd(lambda_, gamma, tol):
    """
    Test that smoothing_spline and smoothing_spline_nd give the same results
    for 1D data within a given tolerance.
    """

    # Generate a synthetic noisy sine wave
    x = np.linspace(0, 2 * np.pi, 100)
    signal = np.sin(x) + 0.1 * np.random.default_rng(42).normal(size=x.shape)

    # Apply 1D Fractional Smoothing Spline
    m = 1  # No upsampling
    _, smoothed_1d = smoothing_spline(signal, lambda_, m, gamma)

    # Apply smoothing_spline_nd (which is general for N-D but should match 1D case)
    smoothed_nd = smoothing_spline_nd(signal.reshape(-1, 1), lambda_, gamma).flatten()

    # Compute Mean Squared Error (MSE)
    mse = np.mean((smoothed_1d - smoothed_nd) ** 2)

    # Assert MSE is within the acceptable tolerance
    assert (
        mse < tol
    ), f"MSE {mse:.6e} exceeds tolerance {tol:.6e} for λ={lambda_}, γ={gamma}"


def test_recursive_smoothing_preserves_constants() -> None:
    signal = np.full(31, 3.25)

    np.testing.assert_allclose(
        recursive_smoothing_spline(signal, lamb=2.0), signal, rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize("lamb", [-1.0, np.inf, np.nan])
def test_smoothing_rejects_invalid_lambda(lamb: float) -> None:
    with pytest.raises(ValueError, match="lamb"):
        recursive_smoothing_spline(np.ones(4), lamb=lamb)


def test_periodize_requires_positive_integer() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        periodize(np.arange(3), 0)


def test_fractional_autocorrelation_rejects_invalid_degree() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fractsplineautocorr(-0.5, np.linspace(-0.5, 0.5, 5))


def test_fractional_smoother_zero_lambda_is_identity_without_upsampling():
    signal = np.random.default_rng(20260715).standard_normal(32)

    _, result = smoothing_spline(signal, lamb=0.0, m=1, gamma=1.4)

    np.testing.assert_allclose(result, signal, rtol=0.0, atol=5e-15)


def test_nd_periodic_cosine_has_the_analytical_frequency_response():
    shape = (32, 48)
    mode = (3, 5)
    lamb = 0.35
    gamma = 1.25
    yy, xx = np.indices(shape)
    signal = np.cos(2.0 * np.pi * (mode[0] * yy / shape[0] + mode[1] * xx / shape[1]))
    omega_squared = (2.0 * np.pi * mode[0] / shape[0]) ** 2 + (
        2.0 * np.pi * mode[1] / shape[1]
    ) ** 2
    expected_gain = 1.0 / (1.0 + lamb * omega_squared**gamma)

    result = smoothing_spline_nd(signal, lamb=lamb, gamma=gamma)

    np.testing.assert_allclose(result, expected_gain * signal, rtol=2e-14, atol=2e-14)


def test_recursive_smoother_matches_independent_dense_boundary_system():
    signal = np.random.default_rng(20260715).standard_normal(31)
    lamb = 0.7
    root = np.sqrt(1.0 + 4.0 * lamb)
    pole = (root - 1.0) / (root + 1.0)
    size = signal.size
    system = np.zeros((size, size))
    system[0, 0] = 1.0 - pole
    system[0, 1] = -pole * (1.0 - pole)
    for index in range(1, size - 1):
        system[index, index - 1] = -pole
        system[index, index] = 1.0 + pole**2
        system[index, index + 1] = -pole
    system[-1, -2] = -pole
    system[-1, -1] = 1.0 - pole + pole**2
    expected = (1.0 - pole) ** 2 * np.linalg.solve(system, signal)

    result = recursive_smoothing_spline(signal, lamb=lamb)

    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=2e-15)


def test_nd_smoothing_plan_matches_one_shot_api_across_inputs():
    rng = np.random.default_rng(20260715)
    plan = SmoothingSplinePlan((17, 24), lamb=0.3, gamma=1.4)

    for data in (rng.standard_normal(plan.shape), rng.standard_normal(plan.shape)):
        expected = smoothing_spline_nd(data, lamb=0.3, gamma=1.4)
        np.testing.assert_allclose(plan(data), expected, rtol=0.0, atol=0.0)


def test_nd_smoothing_plan_retains_only_real_fft_half_spectrum():
    plan = SmoothingSplinePlan((10, 12, 14), lamb=0.2, gamma=1.25)

    assert plan.frequency_response.shape == (10, 12, 8)
    assert plan.retained_bytes == plan.frequency_response.nbytes
    assert not plan.frequency_response.flags.writeable


def test_nd_smoothing_plan_supports_output_buffer():
    data = np.arange(48, dtype=np.float64).reshape(6, 8)
    plan = SmoothingSplinePlan(data.shape, lamb=0.5, gamma=1.0)
    expected = plan(data)
    out = np.empty_like(expected)

    returned = plan(data, out=out)

    assert returned is out
    np.testing.assert_equal(out, expected)


def test_nd_smoothing_plan_rejects_incompatible_shape():
    plan = SmoothingSplinePlan((6, 8), lamb=0.5, gamma=1.0)

    with pytest.raises(ValueError, match="shape"):
        plan(np.ones((8, 6)))
