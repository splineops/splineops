import numpy as np
import numpy.typing as npt

from bssp.bases.splinebasis import SplineBasis


class BSpline1Basis(SplineBasis):

    def __init__(self):

        # Support (no poles)
        support = 2

        # Call super constructor
        super(BSpline1Basis, self).__init__(support=support)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 1 <= |x|
        # TODO(dperdios): specified dtype? output allocated in base class?
        #  Using the dtype of x may not be the smartest idea
        y = np.zeros_like(x)

        # Case |x| < 1 (i.e. 0 <= |x| < 1)
        y = np.where(x_abs < 1, 1 - x_abs, y)

        return y
