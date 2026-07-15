# splineops/tests/test_07_01_multiscale.py

import pytest
import numpy as np

# Pyramid/wavelet classes
from splineops.multiscale.pyramid import (
    get_pyramid_filter,
    reduce_1d,
    expand_1d,
    reduce_2d,
    expand_2d,
)
from splineops.multiscale.wavelets.haar import HaarWavelets
from splineops.multiscale.wavelets.spline_wavelets import (
    Spline1Wavelets,
    Spline3Wavelets,
    Spline5Wavelets,
)

##############################################################################
# 1) Ground truth arrays
##############################################################################

# ====== 1D Pyramid ground truth ======
ground_truth_expanded_1d = [
    -0.044597760218083975,
    0.95501393295756,
    2.291301941950471,
    2.7132412475462986,
    2.026027129984787,
    1.0839703535659702,
    -0.010510748908814163,
    -1.9097664990313807,
    -4.284876954145319,
    -5.819748315428886,
]

# ====== 2D Pyramid ground truth ======
ground_truth_expanded_2d = [
    [0.041890937834978104, 1.3589465618133545, 2.676758050918579, 1.3589465618133545],
    [1.2880852222442627, 2.2180628776550293, 3.1492741107940674, 2.2180628776550293],
    [2.5349957942962646, 3.0784125328063965, 3.6235413551330566, 3.0784125328063965],
    [1.2880852222442627, 2.2180628776550293, 3.1492741107940674, 2.2180628776550293],
]

# ====== Haar 2D wavelet reconstruction (8x8) ======
ground_truth_recon_haar_8x8 = [
    [
        -10.000001907348633,
        -9.000001907348633,
        -8.000000953674316,
        -7.000000953674316,
        -6.000000953674316,
        -5.000000953674316,
        -4.000001430511475,
        -3.0000016689300537,
    ],
    [
        -2.000000476837158,
        -1.0000003576278687,
        -7.164964017647435e-07,
        0.9999992251396179,
        1.999998927116394,
        2.9999988079071045,
        3.999999523162842,
        4.999999523162842,
    ],
    [
        6.000002384185791,
        7.000002384185791,
        8.0,
        9.0,
        9.999999046325684,
        10.999999046325684,
        12.000000953674316,
        13.000001907348633,
    ],
    [
        14.000001907348633,
        15.00000286102295,
        16.000001907348633,
        17.000001907348633,
        18.0,
        19.0,
        20.000001907348633,
        21.000001907348633,
    ],
    [
        22.0,
        23.0,
        23.999996185302734,
        24.999998092651367,
        25.999998092651367,
        27.0,
        28.0,
        29.000001907348633,
    ],
    [30.000001907348633, 31.000003814697266, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0],
    [
        38.000003814697266,
        39.000003814697266,
        40.0,
        41.0,
        42.0,
        43.0,
        44.000003814697266,
        45.000003814697266,
    ],
    [
        46.0,
        47.000003814697266,
        47.999996185302734,
        49.0,
        50.0,
        51.0,
        52.000003814697266,
        53.000003814697266,
    ],
]

# ====== Spline3 2D wavelet reconstruction (8x8) ======
ground_truth_recon_spline3_8x8 = [
    [
        -9.999998092651367,
        -8.999999046325684,
        -8.0,
        -7.0,
        -5.999999523162842,
        -4.999999523162842,
        -4.0,
        -3.0,
    ],
    [
        -1.9999996423721313,
        -0.9999998211860657,
        -1.5747635018215078e-07,
        0.9999998211860657,
        2.0,
        3.0,
        4.0,
        5.0,
    ],
    [
        6.0,
        7.0,
        7.999999523162842,
        9.0,
        10.0,
        11.0,
        11.999999046325684,
        12.999999046325684,
    ],
    [
        13.999999046325684,
        15.000000953674316,
        16.000001907348633,
        17.000001907348633,
        18.0,
        19.0,
        20.0,
        21.0,
    ],
    [22.0, 23.0, 24.0, 25.0, 25.999998092651367, 27.0, 28.0, 29.000001907348633],
    [
        30.000001907348633,
        30.999998092651367,
        31.999996185302734,
        32.999996185302734,
        34.0,
        35.0,
        36.0,
        37.0,
    ],
    [
        38.0,
        38.999996185302734,
        39.999996185302734,
        40.999996185302734,
        42.0,
        43.0,
        44.0,
        45.0,
    ],
    [
        46.0,
        47.000003814697266,
        48.000003814697266,
        49.0,
        49.999996185302734,
        51.0,
        52.00000762939453,
        53.0000114440918,
    ],
]

# Tolerances
RTOL = 1e-9
ATOL = 1e-10

##############################################################################
# 2) Tests
##############################################################################


def test_pyramid_1d():
    """
    Check 1D reduce/expand matches known ground-truth result.
    """
    x_1d = np.array(
        [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0, -6.0], dtype=np.float64
    )
    filter_name = "Centered Spline"
    order = 3
    g, h, is_centered = get_pyramid_filter(filter_name, order)
    reduced = reduce_1d(x_1d, g, is_centered)
    expanded = expand_1d(reduced, h, is_centered)

    np.testing.assert_allclose(
        expanded,
        np.array(ground_truth_expanded_1d, dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_pyramid_2d():
    """
    Check 2D reduce/expand matches known ground-truth result.
    """
    arr_2d = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    filter_name = "Spline"
    order = 3
    g2, h2, is_centered_2d = get_pyramid_filter(filter_name, order)
    reduced_2d = reduce_2d(arr_2d, g2, is_centered_2d)
    expanded_2d = expand_2d(reduced_2d, h2, is_centered_2d)

    np.testing.assert_allclose(
        expanded_2d,
        np.array(ground_truth_expanded_2d, dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )


def test_pyramid_supports_explicit_batch_and_channel_axes():
    rng = np.random.default_rng(20260716)
    data = rng.standard_normal((2, 12, 16, 3)).astype(np.float32)
    g, h, centered = get_pyramid_filter("Spline", 3)

    reduced = reduce_2d(data, g, centered, spatial_axes=(1, 2))
    expanded = expand_2d(reduced, h, centered, spatial_axes=(1, 2))

    assert reduced.shape == (2, 6, 8, 3)
    assert expanded.shape == data.shape
    for batch in range(data.shape[0]):
        for channel in range(data.shape[-1]):
            expected_reduced = reduce_2d(data[batch, :, :, channel], g, centered)
            expected_expanded = expand_2d(expected_reduced, h, centered)
            np.testing.assert_equal(reduced[batch, :, :, channel], expected_reduced)
            np.testing.assert_equal(expanded[batch, :, :, channel], expected_expanded)
    with pytest.raises(ValueError, match="required"):
        reduce_2d(data, g, centered)
    with pytest.raises(ValueError, match="distinct"):
        reduce_2d(data, g, centered, spatial_axes=(1, 1))


@pytest.mark.parametrize("wavelet_class", [HaarWavelets, Spline3Wavelets])
def test_wavelets_support_explicit_batch_and_channel_axes(wavelet_class):
    rng = np.random.default_rng(20260716)
    data = rng.standard_normal((2, 16, 24, 3))
    wavelet = wavelet_class(scales=2)

    coefficients = wavelet.analysis(data, spatial_axes=(1, 2))
    reconstructed = wavelet.synthesis(coefficients, spatial_axes=(1, 2))

    for batch in range(data.shape[0]):
        for channel in range(data.shape[-1]):
            expected = wavelet.analysis(data[batch, :, :, channel])
            np.testing.assert_equal(coefficients[batch, :, :, channel], expected)
    tolerance = 2e-12 if wavelet_class is HaarWavelets else 2e-10
    np.testing.assert_allclose(reconstructed, data, rtol=0.0, atol=tolerance)
    with pytest.raises(ValueError, match="required"):
        wavelet.analysis(data)
    with pytest.raises(ValueError, match="distinct"):
        wavelet.analysis(data, spatial_axes=(1, 1))


@pytest.mark.parametrize(
    "wavelet_class,ground_truth",
    [
        (HaarWavelets, ground_truth_recon_haar_8x8),
        (Spline3Wavelets, ground_truth_recon_spline3_8x8),
    ],
)
def test_wavelet_8x8(wavelet_class, ground_truth):
    """
    Parametrized test for HaarWavelets and Spline3Wavelets on an 8x8 input.
    Checks final reconstruction vs ground truth.
    """
    ny, nx = 8, 8
    image = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) - 10.0

    wavelet = wavelet_class(scales=2)
    coeffs = wavelet.analysis(image)
    recon = wavelet.synthesis(coeffs)

    np.testing.assert_allclose(
        recon, np.array(ground_truth, dtype=np.float32), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("centered", [False, True])
def test_pyramid_singleton_is_preserved(centered):
    sample = np.array([3.5], dtype=np.float64)
    filter_ = np.array([1.0, 0.5])

    np.testing.assert_equal(reduce_1d(sample, filter_, centered), sample)
    np.testing.assert_equal(expand_1d(sample, filter_, centered), sample)


def test_pyramid_odd_length_contract_is_floor_reduction():
    signal = np.arange(5, dtype=np.float64)
    reduced = reduce_1d(signal, np.array([1.0]), centered=False)

    assert reduced.shape == (2,)


@pytest.mark.parametrize("wavelet_class", [HaarWavelets, Spline3Wavelets])
def test_wavelets_reject_shapes_that_are_not_reversible(wavelet_class):
    wavelet = wavelet_class(scales=2)

    with pytest.raises(ValueError, match="divisible"):
        wavelet.analysis(np.ones((7, 8), dtype=np.float32))


@pytest.mark.parametrize("wavelet_class", [HaarWavelets, Spline3Wavelets])
@pytest.mark.parametrize("shape", [(4, 12), (8, 16), (12, 20)])
def test_wavelet_rectangular_perfect_reconstruction(wavelet_class, shape):
    rng = np.random.default_rng(20260715)
    image = rng.standard_normal(shape).astype(np.float64)
    wavelet = wavelet_class(scales=2)

    reconstructed = wavelet.synthesis(wavelet.analysis(image))

    np.testing.assert_allclose(reconstructed, image, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "wavelet_class,max_error",
    [
        (Spline1Wavelets, 2e-7),
        (Spline3Wavelets, 3e-12),
        # The inherited order-5 filter is published with only 5-6 significant
        # digits.  Its bounded error is recorded explicitly instead of being
        # described as perfect reconstruction.
        (Spline5Wavelets, 2e-3),
    ],
)
@pytest.mark.parametrize("shape", [(4, 12), (8, 16), (12, 20)])
def test_all_spline_wavelet_orders_have_a_bounded_reconstruction_error(
    wavelet_class, max_error, shape
):
    image = np.random.default_rng(20260715).standard_normal(shape)
    wavelet = wavelet_class(scales=2)

    reconstructed = wavelet.synthesis(wavelet.analysis(image))

    assert np.max(np.abs(reconstructed - image)) < max_error


@pytest.mark.parametrize("operation", [reduce_1d, expand_1d])
def test_pyramid_promotes_integer_samples(operation):
    result = operation(np.arange(8), np.array([1.0, 0.25]), centered=False)

    assert result.dtype == np.float64


@pytest.mark.parametrize("wavelet_class", [HaarWavelets, Spline3Wavelets])
def test_wavelets_promote_integer_samples_and_preserve_float32(wavelet_class):
    wavelet = wavelet_class(scales=2)

    integer_result = wavelet.analysis(np.arange(64).reshape(8, 8))
    float_result = wavelet.analysis(np.arange(64, dtype=np.float32).reshape(8, 8))

    assert integer_result.dtype == np.float64
    assert float_result.dtype == np.float32


def test_multiscale_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        reduce_1d(np.array([0.0, np.nan]), np.array([1.0]), False)
    with pytest.raises(ValueError, match="finite"):
        HaarWavelets(scales=1).analysis(np.full((2, 2), np.inf))
