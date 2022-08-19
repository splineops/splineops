import numpy as np
import numpy.typing as npt

from bssp.bases.splinebasis import SplineBasis


class BSpline0Basis(SplineBasis):

    def __init__(self):

        # Support (no poles)
        support = 1

        # Call super constructor
        super(BSpline0Basis, self).__init__(support=support)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 1/2 < |x|
        # TODO(dperdios): specified dtype? output allocated in base class?
        #  Using the dtype of x may not be the smartest idea
        y = np.zeros_like(x)

        # Case |x| < 1/2 (i.e. 0 < |x| < 1/2)
        y = np.where(x_abs < 1 / 2, 1, y)

        # Case |x| == 1/2
        y = np.where(x_abs == 1 / 2, 1 / 2, y)

        return y
