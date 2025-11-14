# splineops/src/splineops/spline_interpolation/bases/omoms3basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class OMOMS3Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 4
        degree = support - 1
        poles = (1 / 8 * (-13 + np.sqrt(105)),)

        # Call super constructor
        super(OMOMS3Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)
        x_abs_2 = x_abs * x_abs
        x_abs_3 = x_abs_2 * x_abs

        # Case 2 <= |x|
        y = np.zeros_like(x)

        # Case |x| < 2 (i.e. 1 <= |x| < 2)
        # 1/42 (58 - 85 x + 42 x^2 - 7 x^3)
        # -x^3/6 + x^2 - (85 x)/42 + 29/21
        # -1/42 (x - 2) (7 (x - 4) x + 29)
        y = np.where(
            np.logical_and(x_abs >= 1, x_abs < 2),
            -1 / 6 * x_abs_3 + x_abs_2 - 85 / 42 * x_abs + 29 / 21,
            y,
        )

        # Case |x| < 1 (i.e. 0 <= |x| < 1)
        # 1/42 (26 + 3 x - 42 x^2 + 21 x^3)
        # x^3/2 - x^2 + x/14 + 13/21
        # 1/42 (3 x (7 (x - 2) x + 1) + 26)
        y = np.where(x_abs < 1, 1 / 2 * x_abs_3 - x_abs_2 + 1 / 14 * x_abs + 13 / 21, y)

        return y
