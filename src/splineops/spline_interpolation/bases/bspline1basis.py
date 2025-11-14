# splineops/src/splineops/spline_interpolation/bases/bspline1basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline1Basis(SplineBasis):
    def __init__(self) -> None:

        # Support and degree (no poles)
        support = 2
        degree = support - 1

        # Call super constructor
        super(BSpline1Basis, self).__init__(support=support, degree=degree)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 1 <= |x|
        y = np.zeros_like(x)

        # Case |x| < 1 (i.e. 0 <= |x| < 1)
        y = np.where(x_abs < 1, 1 - x_abs, y)

        return y
