# splineops/src/splineops/spline_interpolation/bases/omoms5basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class OMOMS5Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 6
        degree = support - 1
        poles = (
            -0.47581271000843991544122436278663222058892748537659,
            -0.070925718968685451773973269699832573209336542715216,
        )

        # Call super constructor
        super(OMOMS5Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 3 <= |x|
        y = np.zeros_like(x)

        # Case 2 <= |x| < 3
        # (17121 - 27811 x + 18180 x^2 - 5980 x^3 + 990 x^4 - 66 x^5)/7920
        # -((x - 3) (66 x^4 - 792 x^3 + 3604 x^2 - 7368 x + 5707))/7920
        # -((x - 3) (2 (x - 6) x (33 (x - 6) x + 614) + 5707))/7920
        y = np.where(
            np.logical_and(x_abs >= 2, x_abs < 3),
            -1
            / 7920
            * (
                (x_abs - 3)
                * (2 * (x_abs - 6) * x_abs * (33 * (x_abs - 6) * x_abs + 614) + 5707)
            ),
            y,
        )

        # Case 1 <= |x| < 2
        # (2517 + 6755 x - 14940 x^2 + 10100 x^3 - 2970 x^4 + 330 x^5)/7920
        # (2517 + 6755 x - 14940 x^2 + 10100 x^3 - 2970 x^4 + 330 x^5)/7920
        # x^5/24 - (3 x^4)/8 + (505 x^3)/396 - (83 x^2)/44 + (1351 x)/1584 + 839/2640
        # (5 x (2 x (x (33 (x - 9) x + 1010) - 1494) + 1351) + 2517)/7920
        y = np.where(
            np.logical_and(x_abs >= 1, x_abs < 2),
            1
            / 7920
            * (
                5
                * x_abs
                * (
                    2 * x_abs * (x_abs * (33 * (x_abs - 9) * x_abs + 1010) - 1494)
                    + 1351
                )
                + 2517
            ),
            y,
        )

        # Case |x| < 1
        # (2061 - 5 x - 1620 x^2 - 200 x^3 + 990 x^4 - 330 x^5)/3960
        # -(330 x^5 - 990 x^4 + 200 x^3 + 1620 x^2 + 5 x - 2061)/3960
        # (2061 - 5 x (2 x (x (33 (x - 3) x + 20) + 162) + 1))/3960
        y = np.where(
            x_abs < 1,
            1
            / 3960
            * (
                2061
                - 5
                * x_abs
                * (2 * x_abs * (x_abs * (33 * (x_abs - 3) * x_abs + 20) + 162) + 1)
            ),
            y,
        )

        return y
