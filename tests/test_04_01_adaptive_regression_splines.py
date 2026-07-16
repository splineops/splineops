# splineops/tests/test_04_01_adaptive_regression_splines.py

import numpy as np
import pytest
import scipy.optimize as optimize
from splineops.adaptive_regression_splines import (
    DenoisingDiagnostics,
    DenoisingPlan,
    denoise_y,
)
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


@pytest.mark.parametrize(
    "knot_indices,amplitudes",
    [
        ((1,), (1.25,)),
        ((2,), (-0.75,)),
        ((1, 3), (1.0, -2.0)),
        ((1, 2, 4), (1.0, -2.0, 3.0)),
    ],
)
def test_sparsest_interpolant_recovers_known_nonuniform_hinge_models(
    knot_indices, amplitudes
):
    # The alternating slope changes have a unique sparse hinge description at
    # these samples.  Constructing the values from that independent formula
    # checks knot location, amplitude, polynomial, and off-sample evaluation.
    x = np.array([0.0, 0.4, 1.1, 1.9, 3.0, 4.5])
    expected_knots = x[np.asarray(knot_indices)]
    expected_amplitudes = np.asarray(amplitudes)
    expected_polynomial = np.array([-0.3, 0.8])

    def hinge_model(locations):
        values = expected_polynomial[0] + expected_polynomial[1] * locations
        for knot, amplitude in zip(expected_knots, expected_amplitudes):
            values = values + amplitude * np.maximum(locations - knot, 0.0)
        return values

    y = hinge_model(x)
    knots, actual_amplitudes, polynomial = sparsest_interpolant(
        x, y, sparsity_tol=1e-10
    )
    query = np.linspace(x[0], x[-1], 51)

    np.testing.assert_allclose(knots, expected_knots, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        actual_amplitudes, expected_amplitudes, rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(polynomial, expected_polynomial, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        linear_spline(query, knots, actual_amplitudes, polynomial),
        hinge_model(query),
        rtol=0.0,
        atol=2e-14,
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


def test_denoising_plan_matches_one_shot_api_and_reuses_geometry():
    x = np.linspace(0.0, 1.0, 48)
    first = np.sin(2.0 * np.pi * x) + 0.03 * np.cos(19.0 * x)
    second = np.cos(3.0 * np.pi * x) - 0.02 * np.sin(13.0 * x)
    plan = DenoisingPlan(x, rho=0.5)

    for signal, lamb in ((first, 1e-3), (second, 2e-3)):
        expected = denoise_y(x, signal, lamb=lamb, rho=0.5)
        result = plan.solve(signal, lamb=lamb)
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)


def test_denoising_plan_keeps_an_immutable_copy_of_sample_locations():
    x = np.linspace(0.0, 1.0, 8)
    plan = DenoisingPlan(x)
    x[:] = -1.0

    np.testing.assert_equal(plan.x, np.linspace(0.0, 1.0, 8))
    assert not plan.x.flags.writeable


def test_denoising_path_matches_independent_solves_without_hidden_state():
    x = np.arange(32, dtype=np.float64)
    y = np.sin(0.3 * x) + 0.05 * np.cos(1.2 * x)
    lambdas = (0.0, 0.02, 0.05, 0.1)
    plan = DenoisingPlan(x, rho=0.5)

    path, diagnostics = plan.solve_path(
        y, lambdas, relative_tol=1e-8, return_diagnostics=True
    )
    repeated = plan.solve_path(y, lambdas, relative_tol=1e-8)
    independent = np.stack([plan.solve(y, lamb, relative_tol=1e-8) for lamb in lambdas])

    np.testing.assert_allclose(path, independent, rtol=2e-8, atol=2e-8)
    np.testing.assert_equal(repeated, path)
    assert len(diagnostics) == len(lambdas)
    assert plan.retained_array_bytes > plan.x.nbytes
    assert plan.configuration == {"sample_count": x.size, "rho": 0.5}
    with pytest.raises(ValueError, match="non-empty"):
        plan.solve_path(y, ())
    with pytest.raises(ValueError, match="non-negative"):
        plan.solve_path(y, (0.1, -0.1))
    with pytest.raises(TypeError, match="sequence"):
        plan.solve_path(y, 0.1)


def test_denoising_matches_independent_constrained_optimizer():
    x = np.array([0.0, 0.2, 0.55, 0.9, 1.4, 2.0, 2.8, 3.7])
    y = np.array([0.1, 0.8, 0.4, 1.2, 1.0, 1.8, 1.6, 2.4])
    lamb = 0.08
    plan = DenoisingPlan(x, rho=1.0)
    regularizer = plan._regularizer.toarray()
    regularized_size = regularizer.shape[0]

    def objective(variable):
        signal = variable[: x.size]
        slack = variable[x.size :]
        return 0.5 * np.sum((signal - y) ** 2) + lamb * np.sum(slack)

    constraint_matrix = np.block(
        [
            [regularizer, -np.eye(regularized_size)],
            [-regularizer, -np.eye(regularized_size)],
        ]
    )
    constraint = optimize.LinearConstraint(constraint_matrix, -np.inf, 0.0)
    initial = np.concatenate((y, np.abs(regularizer @ y) + 1e-4))
    bounds = optimize.Bounds(
        np.concatenate((np.full(x.size, -np.inf), np.zeros(regularized_size))),
        np.full(x.size + regularized_size, np.inf),
    )
    reference = optimize.minimize(
        objective,
        initial,
        method="SLSQP",
        constraints=(constraint,),
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    assert reference.success, reference.message
    actual = plan.solve(y, lamb, relative_tol=1e-10)
    np.testing.assert_allclose(actual, reference.x[: x.size], rtol=2e-6, atol=2e-6)


def test_linear_spline_prefix_evaluator_matches_direct_hinge_sum():
    rng = np.random.default_rng(20260715)
    knots = np.sort(rng.uniform(-2.0, 2.0, 200))
    amplitudes = rng.standard_normal(knots.size)
    polynomial = np.array([0.75, -1.25])
    locations = rng.uniform(-3.0, 3.0, (17, 13))
    expected = polynomial[0] + polynomial[1] * locations
    for knot, amplitude in zip(knots, amplitudes):
        expected += amplitude * np.maximum(locations - knot, 0.0)

    result = linear_spline(locations, knots, amplitudes, polynomial)

    np.testing.assert_allclose(result, expected, rtol=3e-14, atol=3e-13)


def test_linear_spline_rejects_unsorted_knots():
    with pytest.raises(ValueError, match="sorted"):
        linear_spline(0.5, [1.0, 0.0], [2.0, 3.0], [0.0, 1.0])
