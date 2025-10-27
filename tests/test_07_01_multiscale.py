# splineops/tests/test_07_01_multiscale.py

import pytest
import numpy as np

# Pyramid/wavelet classes
from splineops.multiscale.pyramid import (
    get_pyramid_filter,
    reduce_1d, expand_1d,
    reduce_2d, expand_2d
)
from splineops.multiscale.wavelets.haar import HaarWavelets
from splineops.multiscale.wavelets.splinewavelets import Spline3Wavelets

##############################################################################
# 1) Ground truth arrays
##############################################################################

# ====== 1D Pyramid ground truth ======
ground_truth_expanded_1d = [
    -0.044597760218083975, 0.95501393295756, 2.291301941950471, 2.7132412475462986,
    2.026027129984787,     1.0839703535659702, -0.010510748908814163,
    -1.9097664990313807,   -4.284876954145319, -5.819748315428886
]

# ====== 2D Pyramid ground truth ======
ground_truth_expanded_2d = [
    [0.041890937834978104, 1.3589465618133545, 2.676758050918579, 1.3589465618133545],
    [1.2880852222442627,   2.2180628776550293, 3.1492741107940674, 2.2180628776550293],
    [2.5349957942962646,   3.0784125328063965, 3.6235413551330566, 3.0784125328063965],
    [1.2880852222442627,   2.2180628776550293, 3.1492741107940674, 2.2180628776550293]
]

# ====== Haar 2D wavelet reconstruction (8x8) ======
ground_truth_recon_haar_8x8 = [
    [-10.000001907348633, -9.000001907348633,  -8.000000953674316, -7.000000953674316,
     -6.000000953674316,  -5.000000953674316,  -4.000001430511475, -3.0000016689300537],
    [-2.000000476837158,  -1.0000003576278687, -7.164964017647435e-07,  0.9999992251396179,
      1.999998927116394,   2.9999988079071045,  3.999999523162842,   4.999999523162842],
    [6.000002384185791,    7.000002384185791,   8.0,                 9.0,
     9.999999046325684,   10.999999046325684,  12.000000953674316,  13.000001907348633],
    [14.000001907348633,  15.00000286102295,   16.000001907348633, 17.000001907348633,
     18.0,                19.0,                20.000001907348633, 21.000001907348633],
    [22.0,                23.0,                23.999996185302734, 24.999998092651367,
     25.999998092651367,  27.0,                28.0,               29.000001907348633],
    [30.000001907348633,  31.000003814697266,  32.0,               33.0,
     34.0,                35.0,                36.0,               37.0],
    [38.000003814697266,  39.000003814697266,  40.0,               41.0,
     42.0,                43.0,                44.000003814697266, 45.000003814697266],
    [46.0,                47.000003814697266,  47.999996185302734, 49.0,
     50.0,                51.0,                52.000003814697266, 53.000003814697266]
]

# ====== Spline3 2D wavelet reconstruction (8x8) ======
ground_truth_recon_spline3_8x8 = [
    [-9.999998092651367,  -8.999999046325684,  -8.0,               -7.0,
     -5.999999523162842,  -4.999999523162842,  -4.0,               -3.0],
    [-1.9999996423721313, -0.9999998211860657, -1.5747635018215078e-07,  0.9999998211860657,
      2.0,                 3.0,                 4.0,                 5.0],
    [6.0,                 7.0,                 7.999999523162842,  9.0,
     10.0,                11.0,                11.999999046325684, 12.999999046325684],
    [13.999999046325684,  15.000000953674316,  16.000001907348633, 17.000001907348633,
     18.0,                19.0,                20.0,               21.0],
    [22.0,                23.0,                24.0,               25.0,
     25.999998092651367,  27.0,                28.0,               29.000001907348633],
    [30.000001907348633,  30.999998092651367,  31.999996185302734, 32.999996185302734,
     34.0,                35.0,                36.0,               37.0],
    [38.0,                38.999996185302734,  39.999996185302734, 40.999996185302734,
     42.0,                43.0,                44.0,               45.0],
    [46.0,                47.000003814697266,  48.000003814697266, 49.0,
     49.999996185302734,  51.0,                52.00000762939453,  53.0000114440918]
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
    x_1d = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0, -6.0], dtype=np.float64)
    filter_name = "Centered Spline"
    order = 3
    g, h, is_centered = get_pyramid_filter(filter_name, order)
    reduced = reduce_1d(x_1d, g, is_centered)
    expanded = expand_1d(reduced, h, is_centered)

    np.testing.assert_allclose(
        expanded,
        np.array(ground_truth_expanded_1d, dtype=np.float64),
        rtol=RTOL,
        atol=ATOL
    )


def test_pyramid_2d():
    """
    Check 2D reduce/expand matches known ground-truth result.
    """
    arr_2d = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 3.0, 2.0]
    ], dtype=np.float32)
    filter_name = "Spline"
    order = 3
    g2, h2, is_centered_2d = get_pyramid_filter(filter_name, order)
    reduced_2d = reduce_2d(arr_2d, g2, is_centered_2d)
    expanded_2d = expand_2d(reduced_2d, h2, is_centered_2d)

    np.testing.assert_allclose(
        expanded_2d,
        np.array(ground_truth_expanded_2d, dtype=np.float32),
        rtol=RTOL,
        atol=ATOL
    )


@pytest.mark.parametrize(
    "wavelet_class,ground_truth",
    [
        (HaarWavelets, ground_truth_recon_haar_8x8),
        (Spline3Wavelets, ground_truth_recon_spline3_8x8),
    ]
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
        recon,
        np.array(ground_truth, dtype=np.float32),
        rtol=RTOL,
        atol=ATOL
    )
