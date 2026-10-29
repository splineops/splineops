"""Independent references for numerical ranges and boundary-domain contracts."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from splineops import TensorSpline
from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan
from splineops.multiscale.pyramid import reduce_2d
from splineops.multiscale.wavelets.haar import HaarWavelets
from splineops.smoothing_splines import (
    SmoothingSplinePlan,
    smoothing_spline,
    smoothing_spline_nd,
)
from splineops.smoothing_splines.fBm_per import fBmper
from splineops.utils.metrics import compute_snr_and_mse_region


@pytest.mark.parametrize("scale", [1e-200, 1e-18, 1.0, 1e150, 1e200])
def test_snr_is_invariant_to_signal_units(scale):
    original = np.ones((3, 4)) * scale
    with np.errstate(all="raise"):
        snr, mse = compute_snr_and_mse_region(original, 0.9 * original)
    assert snr == pytest.approx(20.0, abs=2e-12)
    with localcontext() as context:
        context.prec = 60
        expected_mse = float((Decimal(str(scale)) / 10) ** 2)
    assert mse == pytest.approx(expected_mse, rel=2e-12, abs=0)


def test_snr_zero_reference_and_exact_matches_have_explicit_limits():
    zero = np.zeros((3, 4))
    with np.errstate(all="raise"):
        assert compute_snr_and_mse_region(zero, zero) == (np.inf, 0)
        assert compute_snr_and_mse_region(zero, zero + 2) == (-np.inf, 4)
        snr, mse = compute_snr_and_mse_region(zero + 1e308, zero - 1e308)
    assert snr == pytest.approx(-20 * np.log10(2))
    assert mse == np.inf


@pytest.mark.parametrize(
    "selection", [{"roi": (0, 0, 0, 2)}, {"mask": np.zeros((2, 2), dtype=bool)}]
)
def test_image_metrics_reject_empty_selections(selection):
    with pytest.raises(ValueError, match="non-empty"):
        compute_snr_and_mse_region(np.ones((2, 2)), np.ones((2, 2)), **selection)


@pytest.mark.parametrize("generator", ["smoothing", "fractional_brownian"])
@pytest.mark.parametrize("length,factor", [(1, 49), (3, 49), (7, 98)])
def test_upsampled_time_axis_matches_every_output_sample(generator, length, factor):
    if generator == "smoothing":
        time, result = smoothing_spline(
            np.arange(length, dtype=float), 0.1, factor, 1.5
        )
    else:
        # Preserve the application's legacy NumPy random state around this test.
        state = np.random.get_state()
        try:
            np.random.seed(531)
            time, result = fBmper(1.0, 0.7, factor, length)
        finally:
            np.random.set_state(state)
    assert time.shape == result.shape == (length * factor,)
    np.testing.assert_array_equal(time, np.arange(length * factor) / factor)
    assert time[-1] < length


@pytest.mark.parametrize(
    "lamb,gamma", [(0.0, 1000), (1e308, 2), (1e-308, 320), (1e308, 1600)]
)
def test_radial_smoothing_response_avoids_intermediate_overflow(lamb, gamma):
    with np.errstate(all="raise"):
        plan = SmoothingSplinePlan((8,), lamb, gamma)
    expected = []
    with localcontext() as context:
        context.prec = 80
        for frequency in np.fft.rfftfreq(8):
            omega_squared = Decimal(str((2 * np.pi * frequency) ** 2))
            expected.append(float(1 / (1 + Decimal(str(lamb)) * omega_squared**gamma)))
    np.testing.assert_allclose(plan.frequency_response, expected, rtol=2e-12, atol=0)
    assert plan.frequency_response[0] == 1


@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("grid", [False, True])
def test_distant_spline_queries_follow_their_boundary_mode(mode, dtype, grid):
    samples = np.array([2, -1, 3, 5, 1], dtype=dtype)
    spline = TensorSpline(samples, np.arange(5, dtype=dtype), "bspline3", mode)
    distant = np.array([-1e10, 1e10, -1e20, 1e20, -1e30, 1e30], dtype=dtype)
    if mode == "zero":
        expected = np.zeros_like(distant)
    else:
        period = 5 if mode == "periodic" else 8
        wrapped = np.remainder(distant.astype(np.float64), period).astype(dtype)
        indexes = wrapped.astype(int)
        if mode == "mirror":
            indexes = np.where(indexes < 5, indexes, 8 - indexes)
        expected = samples[indexes]
    with np.errstate(all="raise"):
        result = spline(distant, grid=grid)
        planned = spline.query_plan(distant, grid=grid)()
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(planned, result)


@pytest.mark.parametrize(
    "case",
    [
        "smoothing_factor",
        "smoothing_shape",
        "smoothing_axes",
        "smoothing_nd_axes",
        "wavelet_scales",
        "wavelet_axes",
        "pyramid_axes",
        "affine_degree",
        "affine_shape",
        "affine_output_shape",
        "affine_axes",
        "affine_budget",
        "differential_shape",
        "differential_axes",
        "query_budget",
    ],
)
@pytest.mark.parametrize("boolean", [True, np.bool_(True)])
def test_boolean_values_are_not_integer_configuration(case, boolean):
    with pytest.raises((TypeError, ValueError), match="integer"):
        if case == "smoothing_factor":
            smoothing_spline(np.arange(3, dtype=float), 0.1, boolean, 1.5)
        elif case == "wavelet_scales":
            HaarWavelets(scales=boolean)
        elif case == "wavelet_axes":
            HaarWavelets(scales=1).analysis(np.ones((4, 4)), spatial_axes=(0, boolean))
        elif case == "pyramid_axes":
            reduce_2d(np.ones((4, 4)), np.ones(1), False, spatial_axes=(0, boolean))
        elif case == "smoothing_shape":
            SmoothingSplinePlan((boolean, 4), 0.1, 2)
        elif case == "smoothing_axes":
            SmoothingSplinePlan((3, 4), 0.1, 2).apply(
                np.ones((3, 4)), axes=(0, boolean)
            )
        elif case == "smoothing_nd_axes":
            smoothing_spline_nd(np.ones((3, 4)), 0.1, 2, axes=(0, boolean))
        elif case == "affine_degree":
            AffinePlan((3, 4), np.eye(2), degree=boolean)
        elif case == "affine_shape":
            AffinePlan((boolean, 4), np.eye(2))
        elif case == "affine_output_shape":
            AffinePlan((3, 4), np.eye(2), output_shape=(boolean, 4))
        elif case == "affine_axes":
            AffinePlan((3, 4), np.eye(2))(np.ones((3, 4)), spatial_axes=(0, boolean))
        elif case == "affine_budget":
            AffinePlan((3, 4), np.eye(2), max_retained_bytes=boolean)
        elif case == "differential_shape":
            DifferentialPlan((boolean, 4))
        elif case == "differential_axes":
            DifferentialPlan((3, 4)).apply(np.ones((3, 4)), spatial_axes=(0, boolean))
        else:
            spline = TensorSpline(
                np.arange(3, dtype=float), np.arange(3, dtype=float), "linear", "mirror"
            )
            spline.query_plan(np.empty(0), max_retained_bytes=boolean)


@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distant_fractional_queries_preserve_stencil_offsets(mode, dtype):
    spline = TensorSpline(
        np.array([2, -1, 3, 5, 1], dtype=dtype),
        np.arange(5, dtype=dtype),
        "bspline3",
        mode,
    )
    distant = np.array([-1e12 + 0.25, 1e12 + 0.75])
    if mode == "zero":
        expected = np.zeros(2)
    else:
        expected = spline(np.remainder(distant, 5 if mode == "periodic" else 8))
    with np.errstate(all="raise"):
        result = spline(distant)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
def test_finite_physical_queries_can_exceed_the_relative_coordinate_range(mode):
    step = 2.0**-1000
    samples = np.array([2.0, -1.0, 3.0, 5.0, 1.0])
    spline = TensorSpline(samples, np.arange(5) * step, "bspline3", mode)
    distant = np.array([-(2.0**100), 2.0**100])
    # Powers of two are exact: calculate the expected sample indexes with
    # Python's unbounded integers, without dividing floating-point coordinates.
    if mode == "zero":
        expected = np.zeros(2)
    else:
        period = 5 if mode == "periodic" else 8
        indexes = np.array([(-(2**1100)) % period, (2**1100) % period])
        if mode == "mirror":
            indexes = np.where(indexes <= 4, indexes, 8 - indexes)
        expected = samples[indexes]
    with np.errstate(all="raise"):
        result = spline(distant)
        planned = spline.query_plan(distant)()
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(result, planned)


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="Platform has no wider floating-point dtype.",
)
def test_image_metrics_reject_values_unrepresentable_in_float64():
    values = np.array([np.finfo(np.longdouble).max / 2])
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite"):
        compute_snr_and_mse_region(values, values)


@pytest.mark.parametrize("mode", ["zero", "mirror", "periodic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["numpy", "scipy", "auto"])
@pytest.mark.parametrize("degree", [1, 3, 5])
def test_distant_affine_queries_keep_backend_boundary_parity(
    mode, dtype, backend, degree
):
    data = (np.arange(35, dtype=dtype) + 2).reshape(5, 7)
    offset = np.array([1e30, -1e20], dtype=dtype)
    plan = AffinePlan(
        data.shape,
        np.eye(2),
        offset,
        dtype=dtype,
        degree=degree,
        mode=mode,
        backend=backend,
    )
    if mode == "zero":
        expected = 0
    else:
        indexes = []
        for coordinate, length in zip(offset, data.shape):
            period = length if mode == "periodic" else 2 * (length - 1)
            index = int(float(coordinate)) % period
            if mode == "mirror" and index >= length:
                index = period - index
            indexes.append(index)
        expected = data[tuple(indexes)]
    with np.errstate(all="raise"):
        actual = plan(data)
        reused = plan.apply_coefficients(plan.prepare_coefficients(data))
    # At these scales all output positions round to the same integer query.
    tolerance = 2e-5 if dtype == np.float32 else 3e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(reused, actual)
