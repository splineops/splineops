# splineops/src/splineops/spline_interpolation/bases/bspline0basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline0Basis(SplineBasis):
    def __init__(self) -> None:

        # Support and degree (no poles)
        support = 1
        degree = support - 1

        # Call super constructor
        super(BSpline0Basis, self).__init__(support=support, degree=degree)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Case 1/2 < |x|
        y = np.zeros_like(x)

        # Case -1/2 <= x < 1/2
        # Note: Asymmetry introduced. This is a standard strategy to compute
        #  the indexes while keeping the smallest support length by avoiding
        #  averaging two values in special cases at interval limits [-1/2, 1/2].
        #  The exact symmetric case with a longer support is provided below.
        y = np.where(np.logical_and(x >= -1 / 2, x < 1 / 2), 1, y)

        return y


class BSpline0SymBasis(SplineBasis):
    def __init__(self) -> None:

        # Support and degree (no poles)
        support = 2
        degree = 0

        # Call super constructor
        super(BSpline0SymBasis, self).__init__(support=support, degree=degree)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 1/2 < |x|
        y = np.zeros_like(x)

        # Case |x| < 1/2 (i.e. 0 < |x| < 1/2)
        y = np.where(x_abs < 1 / 2, 1, y)

        # Case |x| == 1/2
        y = np.where(x_abs == 1 / 2, 1 / 2, y)

        return y
