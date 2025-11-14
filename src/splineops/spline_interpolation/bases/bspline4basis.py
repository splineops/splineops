# splineops/src/splineops/spline_interpolation/bases/bspline4basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline4Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 5
        degree = support - 1
        poles = (
            np.sqrt(664 - np.sqrt(438976)) + np.sqrt(304) - 19,
            np.sqrt(664 + np.sqrt(438976)) - np.sqrt(304) - 19,
        )

        # Call super constructor
        super(BSpline4Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 5/2 <= |x|
        y = np.zeros_like(x)

        # Case |x| < 5/2 (i.e. 3/2 <= |x| < 5/2)
        y = np.where(
            np.logical_and(x_abs >= 3 / 2, x_abs < 5 / 2),
            1
            / 384
            * (625 + x_abs * (-1000 + x_abs * (600 + x_abs * (-160 + 16 * x_abs)))),
            y,
        )

        # Case |x| < 3/2 (i.e. 1/2 <= |x| < 3/2)
        y = np.where(
            np.logical_and(x_abs >= 1 / 2, x_abs < 3 / 2),
            1 / 96 * (55 + x_abs * (20 + x_abs * (-120 + x_abs * (80 - 16 * x_abs)))),
            y,
        )

        # Case |x| < 1/2 (i.e. 0 <= |x| < 1/2)
        y = np.where(
            x_abs < 1 / 2,
            1 / 192 * (115 + x_abs * x_abs * (-120 + 48 * x_abs * x_abs)),
            y,
        )

        return y
