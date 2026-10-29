"""Independent numerical and input-domain checks for the stable CPU APIs."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import optimize
from scipy.interpolate import CubicSpline

from splineops import TensorSpline
from splineops.adaptive_regression_splines import (
    DenoisingPlan,
    denoise_y,
)
from splineops.adaptive_regression_splines.sparsification import (
    sparsest_interpolant,
    linear_spline,
)
from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan
from splineops.multiscale.wavelets.haar import HaarWavelets
from splineops.smoothing_splines import recursive_smoothing_spline, SmoothingSplinePlan
from splineops.smoothing_splines.fract_spline_auto_corr import fractsplineautocorr
from splineops.spline_interpolation.bases.utils import asbasis


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-24, 1e-12, 1.0, 1e12])
def test_uniform_grid_validation_is_independent_of_physical_units(dtype, scale):
    samples = np.array([0.0, 2.0, -1.0, 3.0], dtype=dtype)
    uniform = np.arange(4, dtype=dtype) * dtype(scale)
    spline = TensorSpline(samples, uniform, "bspline3", "mirror")
    np.testing.assert_allclose(spline(uniform), samples, atol=2e-6, rtol=2e-6)
    nonuniform = np.array([0.0, 1.0, 2.5, 3.0], dtype=dtype) * dtype(scale)
    with pytest.raises(ValueError, match="uniform"):
        TensorSpline(samples, nonuniform, "bspline3", "mirror")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("grid", [False, True])
def test_tensorspline_empty_queries_and_threaded_geometry(dtype, grid):
    shape = (7, 9)
    coordinates = tuple(np.arange(n, dtype=dtype) for n in shape)
    samples = np.random.default_rng(29).normal(size=shape).astype(dtype)
    spline = TensorSpline(samples, coordinates, "omoms5", "periodic")
    empty = (np.empty(0, dtype=dtype), np.empty(0, dtype=dtype))
    expected_shape = (0, 0) if grid else (0,)
    assert spline(empty, grid=grid).shape == expected_shape
    assert spline.query_plan(empty, grid=grid)().shape == expected_shape
    query = tuple(np.linspace(-1.25, n + 0.75, 13, dtype=dtype) for n in shape)
    plan = spline.query_plan(query, grid=grid).detach()
    fields = [spline.with_data(samples * dtype(i + 1)) for i in range(4)]
    expected = [field(query, grid=grid) for field in fields]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(plan.apply, fields))
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_symmetric_omoms4_reports_the_polynomial_degree():
    assert asbasis("omoms4-sym").degree == 4


@pytest.mark.parametrize("offset", [0.0, 1e9, 1e12])
@pytest.mark.parametrize("lamb", [0.03, 1e6])
def test_denoising_is_invariant_to_sample_coordinate_translation(offset, lamb):
    x = np.arange(12, dtype=np.float64)
    y = np.sin(x) + 0.25 * x
    reference = denoise_y(x, y, lamb, relative_tol=1e-9)
    shifted = denoise_y(x + offset, y, lamb, relative_tol=1e-9)
    planned = DenoisingPlan(x + offset).solve(y, lamb, relative_tol=1e-9)
    np.testing.assert_allclose(shifted, reference, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(planned, reference, atol=2e-10, rtol=2e-10)


def test_integer_regression_coordinates_do_not_overflow_the_linear_fit():
    x = np.arange(12, dtype=np.int64) + 10**12
    y = np.arange(12, dtype=np.int64) ** 2
    expected = denoise_y(np.arange(12, dtype=float), y.astype(float), 1e6)
    np.testing.assert_allclose(denoise_y(x, y, 1e6), expected, atol=1e-12)


@pytest.mark.parametrize("argument", ["x", "y"])
def test_regression_rejects_complex_samples(argument):
    values = {"x": np.arange(6, dtype=float), "y": np.arange(6, dtype=float) ** 2}
    values[argument] = values[argument].astype(complex) + 1j
    with pytest.raises(TypeError, match="real"):
        denoise_y(**values, lamb=0.1)
    with pytest.raises(TypeError, match="real"):
        DenoisingPlan(values["x"]).solve(values["y"], 0.1)


@pytest.mark.parametrize("sample_count", [16, 40])
def test_denoising_matches_independent_dual_optimizer(sample_count):
    rng = np.random.default_rng(sample_count)
    x = np.cumsum(rng.uniform(0.5, 1.5, sample_count))
    y = rng.normal(size=sample_count)
    lamb = 0.17
    # Form slope differences independently, then solve the smooth box-bounded
    # dual: min_p 1/2 ||y - D.T p||^2 with |p| <= lambda.
    slopes = np.diff(np.eye(sample_count), axis=0) / np.diff(x)[:, None]
    difference = np.diff(slopes, axis=0)

    def objective(dual):
        residual = y - difference.T @ dual
        return 0.5 * residual @ residual, -difference @ residual

    reference = optimize.minimize(
        objective,
        np.zeros(sample_count - 2),
        jac=True,
        bounds=[(-lamb, lamb)] * (sample_count - 2),
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
    )
    assert reference.success, reference.message
    actual, diagnostics = DenoisingPlan(x).solve(
        y, lamb, relative_tol=1e-10, return_diagnostics=True
    )
    assert diagnostics.converged
    np.testing.assert_allclose(actual, y - difference.T @ reference.x, atol=2e-7)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
def test_fractional_autocorrelation_is_periodic_far_from_the_origin(alpha):
    frequencies = np.array([-0.375, -0.125, 0.0, 0.25, 0.5])
    expected = fractsplineautocorr(alpha, frequencies)
    for periods in (-1024, 1024):
        np.testing.assert_allclose(
            fractsplineautocorr(alpha, frequencies + periods),
            expected,
            rtol=2e-13,
            atol=2e-13,
        )


@pytest.mark.parametrize("lamb", [1e-20, 1e20, 1e308])
def test_recursive_smoothing_preserves_constants_at_extreme_regularization(lamb):
    signal = np.full(19, 3.25)
    np.testing.assert_allclose(
        recursive_smoothing_spline(signal, lamb), signal, rtol=2e-15, atol=2e-15
    )


def test_recursive_smoothing_retains_small_tail_of_high_dynamic_range_signal():
    # At lambda=1e-20 the pole is 1e-20 to float64 precision. Each causal
    # step attenuates the impulse by that factor; the reverse correction is
    # smaller than float64 precision at all three positions.
    actual = recursive_smoothing_spline(np.array([1e30, 0.0, 0.0]), 1e-20)
    np.testing.assert_allclose(actual, [1e30, 1e10, 1e-10], rtol=1e-15, atol=0)


def test_recursive_smoothing_avoids_overflow_in_opposite_signed_samples():
    # Scaling commutes with this linear operator, including near the largest
    # finite input. Subtracting neighbors before averaging would overflow.
    signal = np.array([-1.0, 1.0, -0.5, 0.25])
    expected = recursive_smoothing_spline(signal, 0.5)
    with np.errstate(over="raise", invalid="raise"):
        actual = recursive_smoothing_spline(signal * 1e308, 0.5) / 1e308
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_sparse_interpolant_does_not_wrap_unsigned_sample_differences():
    x = np.arange(6, dtype=np.uint8)
    y = np.array([200, 50, 100, 10, 250, 150], dtype=np.uint8)
    knots, amplitudes, polynomial = sparsest_interpolant(x, y)
    np.testing.assert_allclose(
        linear_spline(x, knots, amplitudes, polynomial), y, atol=2e-12, rtol=0
    )


def test_linear_spline_promotes_integer_products_before_evaluation():
    query = np.array([2**40], dtype=np.int64)
    result = linear_spline(
        query,
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.array([0, 2**40], dtype=np.int64),
    )
    np.testing.assert_equal(result, [float(2**80)])


@pytest.mark.parametrize("backend", ["numpy", "scipy"])
def test_affine_rejects_empty_batches_consistently(backend):
    plan = AffinePlan((3, 4), np.eye(2), backend=backend)
    with pytest.raises(ValueError, match="non-empty"):
        plan(np.empty((0, 3, 4)), spatial_axes=(1, 2))


@pytest.mark.parametrize("argument", ["matrix", "offset"])
def test_affine_does_not_discard_imaginary_geometry(argument):
    geometry = {"matrix": np.eye(2), "offset": np.zeros(2)}
    geometry[argument] = geometry[argument].astype(complex) + 1j
    with pytest.raises(TypeError, match="real"):
        AffinePlan((3, 4), **geometry)


def test_affine_archive_loader_rejects_plain_npy(tmp_path):
    source = tmp_path / "not-an-archive.npy"
    np.save(source, np.ones((3, 4)))
    with pytest.raises(ValueError, match="archive"):
        AffinePlan((3, 4), np.eye(2)).load_coefficients(source)


@pytest.mark.parametrize("shape", [(3, 5), (4, 3, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_derivatives_match_independent_clamped_cubic_piecewise_polynomials(
    shape, dtype
):
    samples = np.random.default_rng(40).normal(size=shape).astype(dtype)
    spacing = np.linspace(0.7, 1.4, len(shape))

    def derivative(values, axis, order):
        knots = np.arange(shape[axis]) * spacing[axis]
        zeros = np.zeros(values.shape[:axis] + values.shape[axis + 1 :])
        piecewise = CubicSpline(
            knots, values, axis=axis, bc_type=((1, zeros), (1, zeros))
        )
        return piecewise(knots, nu=order)

    result = DifferentialPlan(shape, spacing)(samples, laplacian=True)
    tolerance = 3e-5 if dtype is np.float32 else 3e-12
    for axis, gradient in enumerate(result.gradient):
        np.testing.assert_allclose(
            gradient, derivative(samples, axis, 1), atol=tolerance, rtol=tolerance
        )
    expected_hessian = []
    for first in range(len(shape)):
        for second in range(first, len(shape)):
            expected = (
                derivative(samples, first, 2)
                if first == second
                else derivative(derivative(samples, second, 1), first, 1)
            )
            expected_hessian.append(expected)
    for actual, expected in zip(result.hessian, expected_hessian):
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        result.laplacian,
        sum(derivative(samples, a, 2) for a in range(len(shape))),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_haar_coefficients_match_independent_block_transform_and_preserve_energy(dtype):
    samples = np.random.default_rng(4).normal(size=(6, 10)).astype(dtype)
    a, b, c, d = (
        samples[::2, ::2],
        samples[::2, 1::2],
        samples[1::2, ::2],
        samples[1::2, 1::2],
    )
    expected = np.block(
        [
            [(a + b + c + d) / 2, (a - b + c - d) / 2],
            [(a + b - c - d) / 2, (a - b - c + d) / 2],
        ]
    )
    actual = HaarWavelets(scales=1).analysis(samples)
    tolerance = 5e-7 if dtype is np.float32 else 2e-14
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        np.sum(actual.astype(float) ** 2),
        np.sum(samples.astype(float) ** 2),
        rtol=tolerance,
    )


def test_cached_plans_cannot_change_parameters_without_rebuilding_the_operator():
    denoising = DenoisingPlan(np.arange(8, dtype=float))
    smoothing = SmoothingSplinePlan((8,), lamb=0.2, gamma=1.0)
    affine = AffinePlan((3, 4), np.eye(2), backend="numpy")
    for plan, name, replacement in (
        (denoising, "rho", 2.0),
        (denoising, "x", np.arange(8) * 2),
        (smoothing, "lamb", 2.0),
        (smoothing, "gamma", 2.0),
        (smoothing, "shape", (4,)),
        (smoothing, "frequency_response", np.ones(5)),
        (affine, "matrix", np.eye(2) * 2),
        (affine, "degree", 1),
        (affine, "mode", "mirror"),
    ):
        with pytest.raises(AttributeError):
            setattr(plan, name, replacement)
    for values in (
        denoising.x,
        smoothing.frequency_response,
        affine.matrix,
        affine.offset,
    ):
        with pytest.raises(ValueError):
            values.flags.writeable = True
