# splineops/src/splineops/spline_interpolation/bases/bspline5basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline5Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 6
        degree = support - 1
        poles = (
            np.sqrt(135 / 2 - np.sqrt(17745 / 4)) + np.sqrt(105 / 4) - 13 / 2,
            np.sqrt(135 / 2 + np.sqrt(17745 / 4)) - np.sqrt(105 / 4) - 13 / 2,
        )

        # Call super constructor
        super(BSpline5Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 3 <= |x|
        y = np.zeros_like(x)

        # Case |x| < 3 (i.e. 2 <= |x| < 3)
        y = np.where(
            np.logical_and(x_abs >= 2, x_abs < 3),
            1
            / 120
            * (
                243
                + x_abs * (-405 + x_abs * (270 + x_abs * (-90 + x_abs * (15 - x_abs))))
            ),
            y,
        )

        # Case |x| < 2 (i.e. 1 <= |x| < 2)
        y = np.where(
            np.logical_and(x_abs >= 1, x_abs < 2),
            1
            / 120
            * (
                51
                + x_abs
                * (75 + x_abs * (-210 + x_abs * (150 + x_abs * (-45 + x_abs * 5))))
            ),
            y,
        )

        # Case |x| < 1 (i.e. 0 <= |x| < 1)
        y = np.where(
            x_abs < 1,
            1 / 60 * (33 + x_abs * x_abs * (-30 + x_abs * x_abs * (15 - 5 * x_abs))),
            y,
        )

        return y
