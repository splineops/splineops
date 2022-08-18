import numpy as np
import numpy.typing as npt

from bssp.bases.splinebasis import SplineBasis


class OMOMS2Basis(SplineBasis):

    def __init__(self):

        # Support and poles
        support = 3
        poles = (1 / 17 * (-43 + 2 * np.sqrt(390)),)

        # Call super constructor
        super(OMOMS2Basis, self).__init__(support=support, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.ArrayLike) -> npt.NDArray:

        # Pre-computations
        x = np.asarray(x)
        x_abs = np.abs(x)

        # Case 3/2 < |x|
        # TODO(dperdios): specified dtype? output allocated in base class?
        #  Using the dtype of x may not be the smartest idea
        y = np.zeros_like(x)

        # Case |x| == 3/2
        # 137 / 240 - (3 x) / 4 + x ^ 2 / 4
        # 1/240 (60 x^2 - 180 x + 137)
        # 1/4 (x - 3) x + 137/240
        y = np.where(
            x_abs == 3 / 2,
            1 / 4 * (x_abs - 3) * x_abs + 137 / 240,
            y
        )

        # Case 1/2 < |x| < 3/2
        # 137/120 - (3 x)/2 + x^2/2
        # 1/2 (x - 3) x + 137/120
        y = np.where(
            np.logical_and(x_abs > 1 / 2, x_abs < 3 / 2),
            1 / 2 * (x_abs - 3) * x_abs + 137 / 120,
            y
        )

        # Case |x| == 1/2
        # 223 / 240 - (3 x) / 4 - x ^ 2 / 4
        # -1/240 (60 x^2 + 180 x - 223)
        # 223/240 - 1/4 x (x + 3)
        y = np.where(x_abs == 1 / 2, 223 / 240 - 1 / 4 * x_abs * (x_abs + 3), y)

        # Case |x| < 1/2 (i.e. -(1/2) < x < 1/2)
        y = np.where(x_abs < 1 / 2, 43 / 60 - x_abs * x_abs, y)

        return y
