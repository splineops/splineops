# splineops/src/splineops/spline_interpolation/bases/omoms4basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class OMOMS4Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 5
        degree = support - 1
        poles = (
            -0.41054918579562752416839060105906234069704122255321,
            -0.031684909102441435136285669435572248017432995194835,
        )

        # Call super constructor
        super(OMOMS4Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:
        # Pre-computations
        x_abs = np.abs(x)

        # Case 5/2 < |x|
        y = np.zeros_like(x)

        # Note: Use of asymmetric strategy to minimize the support length
        #  (similar to bspline0). The exact symmetric case is provided below.

        # Case -5/2 <= x < -3/2 or 3/2 <= x < 5/2
        # 207383/120960 - (385 x)/144 + (227 x^2)/144 - (5 x^3)/12 + x^4/24
        # x^4/24 - (5 x^3)/12 + (227 x^2)/144 - (385 x)/144 + 207383/120960
        # 1/144 (x - 5) x (6 (x - 5) x + 77) + 207383/120960
        y = np.where(
            np.logical_or(
                np.logical_and(x >= -5 / 2, x < -3 / 2),
                np.logical_and(x >= 3 / 2, x < 5 / 2),
            ),
            1 / 144 * (x_abs - 5) * x_abs * (6 * (x_abs - 5) * x_abs + 77)
            + 207383 / 120960,
            y,
        )

        # Case -3/2 <= x < -1/2 or 1/2 <= x < 3/2
        # 15217/30240 + (25 x)/72 - (47 x^2)/36 + (5 x^3)/6 - x^4/6
        # -x^4/6 + (5 x^3)/6 - (47 x^2)/36 + (25 x)/72 + 15217/30240
        # 1/72 x (25 - 2 x (6 (x - 5) x + 47)) + 15217/30240
        y = np.where(
            np.logical_or(
                np.logical_and(x >= -3 / 2, x < -1 / 2),
                np.logical_and(x >= 1 / 2, x < 3 / 2),
            ),
            1 / 72 * x_abs * (25 - 2 * x_abs * (6 * (x_abs - 5) * x_abs + 47))
            + 15217 / 30240,
            y,
        )

        # Case -1/2 <= x < 1/2
        # 11383 / 20160 - (13 x ^ 2) / 24 + x ^ 4 / 4
        # x^4/4 - (13 x^2)/24 + 11383/20160
        # 1/4 (x^2 - 13/12)^2 + 1367/5040
        y = np.where(
            np.logical_and(x >= -1 / 2, x < 1 / 2),
            1 / 4 * (x_abs * x_abs - 13 / 12) ** 2 + 1367 / 5040,
            y,
        )

        return y


class OMOMS4SymBasis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 6
        degree = support - 1
        poles = (
            -0.41054918579562752416839060105906234069704122255321,
            -0.031684909102441435136285669435572248017432995194835,
        )

        # Call super constructor
        super(OMOMS4SymBasis, self).__init__(
            support=support, degree=degree, poles=poles
        )

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 5/2 < |x|
        y = np.zeros_like(x)

        # Case |x| == 5/2
        # x^4/48 - (5 x^3)/24 + (227 x^2)/288 - (385 x)/288 + 207383/241920
        # 1/288 (x - 5) x (6 (x - 5) x + 77) + 207383/241920
        y = np.where(
            x_abs == 5 / 2,
            1 / 288 * (x_abs - 5) * x_abs * (6 * (x_abs - 5) * x_abs + 77)
            + 207383 / 241920,
            y,
        )

        # Case 3/2 < |x| < 5/2
        # 207383/120960 - (385 x)/144 + (227 x^2)/144 - (5 x^3)/12 + x^4/24
        # x^4/24 - (5 x^3)/12 + (227 x^2)/144 - (385 x)/144 + 207383/120960
        # 1/144 (x - 5) x (6 (x - 5) x + 77) + 207383/120960
        y = np.where(
            np.logical_and(x_abs > 3 / 2, x_abs < 5 / 2),
            1 / 144 * (x_abs - 5) * x_abs * (6 * (x_abs - 5) * x_abs + 77)
            + 207383 / 120960,
            y,
        )

        # Case |x| == 3/2
        # 89417/80640 - (335 x)/288 + (13 x^2)/96 + (5 x^3)/24 - x^4/16
        # -x^4/16 + (5 x^3)/24 + (13 x^2)/96 - (335 x)/288 + 89417/80640
        # 1/288 x (3 x (-6 x^2 + 20 x + 13) - 335) + 89417/80640
        y = np.where(
            x_abs == 3 / 2,
            1 / 288 * x_abs * (3 * x_abs * (-6 * x_abs * x_abs + 20 * x_abs + 13) - 335)
            + 89417 / 80640,
            y,
        )

        # Case 1/2 < |x| < 3/2
        # 15217/30240 + (25 x)/72 - (47 x^2)/36 + (5 x^3)/6 - x^4/6
        # -x^4/6 + (5 x^3)/6 - (47 x^2)/36 + (25 x)/72 + 15217/30240
        # 1/72 x (25 - 2 x (6 (x - 5) x + 47)) + 15217/30240
        y = np.where(
            np.logical_and(x_abs > 1 / 2, x_abs < 3 / 2),
            1 / 72 * x_abs * (25 - 2 * x_abs * (6 * (x_abs - 5) * x_abs + 47))
            + 15217 / 30240,
            y,
        )

        # Case |x| == 1/2
        # 64583/120960 + (25 x)/144 - (133 x^2)/144 + (5 x^3)/12 + x^4/24
        # x^4/24 + (5 x^3)/12 - (133 x^2)/144 + (25 x)/144 + 64583/120960
        # 1/144 x (x (6 x (x + 10) - 133) + 25) + 64583/120960
        y = np.where(
            x_abs == 1 / 2,
            1 / 144 * x_abs * (x_abs * (6 * x_abs * (x_abs + 10) - 133) + 25)
            + 64583 / 120960,
            y,
        )

        # Case |x| < 1/2 (i.e. 0 <= |x| < 1/2)
        # 11383 / 20160 - (13 x ^ 2) / 24 + x ^ 4 / 4
        # x^4/4 - (13 x^2)/24 + 11383/20160
        # 1/4 (x^2 - 13/12)^2 + 1367/5040
        y = np.where(
            x_abs < 1 / 2, 1 / 4 * (x_abs * x_abs - 13 / 12) ** 2 + 1367 / 5040, y
        )

        return y
