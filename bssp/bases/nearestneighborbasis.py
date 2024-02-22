import numpy as np
import numpy.typing as npt

from bssp.bases.splinebasis import SplineBasis


class NearestNeighborBasis(SplineBasis):

    def __init__(self):

        # Support (no poles)
        support = 1

        # Call super constructor
        super(NearestNeighborBasis, self).__init__(support=support)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 1/2 < |x|
        # TODO(dperdios): specified dtype? output allocated in base class?
        #  Using the dtype of x may not be the smartest idea
        y = np.zeros_like(x)

        # Case -1/2 <= x < 1/2
        # Note: Asymmetry introduced. This is a standard strategy to compute
        #  the indexes while keeping the smallest support length by avoiding
        #  averaging two values in special cases at interval limits [-1/2, 1/2].
        #  The exact symmetric case with a longer support is provided below.
        y = np.where(np.logical_and(x >= -1 / 2, x < 1 / 2), 1, y)

        return y


class NearestNeighborSymBasis(SplineBasis):

    def __init__(self):

        # Support (no poles)
        support = 1

        # Call super constructor
        super(NearestNeighborSymBasis, self).__init__(support=support)

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
