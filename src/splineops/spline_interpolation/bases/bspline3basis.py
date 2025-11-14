# splineops/src/splineops/spline_interpolation/bases/bspline3basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline3Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 4
        degree = support - 1
        poles = (np.sqrt(3) - 2,)

        # Call super constructor
        super(BSpline3Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 2 <= |x|
        y = np.zeros_like(x)

        # Case |x| < 2 (i.e. 1 <= |x| < 2)
        y = np.where(
            np.logical_and(x_abs >= 1, x_abs < 2),
            1 / 6 * (2 - x_abs) * (2 - x_abs) * (2 - x_abs),
            y,
        )

        # Case |x| < 1 (i.e. 0 <= |x| < 1)
        y = np.where(x_abs < 1, 2 / 3 - 1 / 2 * x_abs * x_abs * (2 - x_abs), y)

        return y
