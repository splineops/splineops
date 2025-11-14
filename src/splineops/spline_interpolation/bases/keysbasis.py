# splineops/src/splineops/spline_interpolation/bases/keysbasis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class KeysBasis(SplineBasis):
    def __init__(self) -> None:

        # Support and degree (no poles)
        support = 4
        degree = 2
        # TODO(dperdios): degree = 2 only for a=-1/2 (fixed for now)

        # Call super constructor
        super(KeysBasis, self).__init__(support=support, degree=degree)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)
        x_abs_2 = x_abs * x_abs
        x_abs_3 = x_abs_2 * x_abs
        a = -0.5  # best interpolation order (i.e., 3) achievable with a = -1/2

        # Case 2 <= |x|
        y = np.zeros_like(x)

        # Case |x| < 2 (i.e. 1 <= |x| < 2)
        y = np.where(
            # x_abs < 2,
            np.logical_and(x_abs >= 1, x_abs < 2),
            a * x_abs_3 - 5 * a * x_abs_2 + 8 * a * x_abs - 4 * a,
            y,
        )

        # Case |x| < 1 (i.e. 0 <= |x| < 1)
        y = np.where(x_abs < 1, (a + 2) * x_abs_3 - (a + 3) * x_abs_2 + 1, y)

        return y
