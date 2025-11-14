# splineops/src/splineops/spline_interpolation/bases/omoms2basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class OMOMS2Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 3
        degree = support - 1
        poles = (1 / 17 * (-43 + 2 * np.sqrt(390)),)

        # Call super constructor
        super(OMOMS2Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 3/2 < |x|
        y = np.zeros_like(x)

        # Note: Use of asymmetric strategy to minimize the support length
        #  (similar to bspline0). The exact symmetric case is provided below.

        # Case -3/2 < x < 1/2 or 1/2 <= x < 3/2
        y = np.where(
            np.logical_or(
                np.logical_and(x < 3 / 2, x >= 1 / 2),
                np.logical_and(x < -1 / 2, x >= -3 / 2),
            ),
            1 / 2 * (x_abs - 3) * x_abs + 137 / 120,
            y,
        )

        # Case -1/2 <= x < 1/2
        y = np.where(np.logical_and(x < 1 / 2, x >= -1 / 2), 43 / 60 - x_abs * x_abs, y)

        return y


class OMOMS2SymBasis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 4
        degree = 2
        poles = (1 / 17 * (-43 + 2 * np.sqrt(390)),)

        # Call super constructor
        super(OMOMS2SymBasis, self).__init__(
            support=support, degree=degree, poles=poles
        )

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 3/2 < |x|
        y = np.zeros_like(x)

        # Case |x| == 3/2
        # 137 / 240 - (3 x) / 4 + x ^ 2 / 4
        # 1/240 (60 x^2 - 180 x + 137)
        # 1/4 (x - 3) x + 137/240
        y = np.where(x_abs == 3 / 2, 1 / 4 * (x_abs - 3) * x_abs + 137 / 240, y)

        # Case 1/2 < |x| < 3/2
        # 137/120 - (3 x)/2 + x^2/2
        # 1/2 (x - 3) x + 137/120
        y = np.where(
            np.logical_and(x_abs > 1 / 2, x_abs < 3 / 2),
            1 / 2 * (x_abs - 3) * x_abs + 137 / 120,
            y,
        )

        # Case |x| == 1/2
        # 223 / 240 - (3 x) / 4 - x ^ 2 / 4
        # -1/240 (60 x^2 + 180 x - 223)
        # 223/240 - 1/4 x (x + 3)
        y = np.where(x_abs == 1 / 2, 223 / 240 - 1 / 4 * x_abs * (x_abs + 3), y)

        # Case |x| < 1/2 (i.e. -(1/2) < x < 1/2)
        y = np.where(x_abs < 1 / 2, 43 / 60 - x_abs * x_abs, y)

        return y
