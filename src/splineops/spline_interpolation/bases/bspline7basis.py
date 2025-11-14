# splineops/src/splineops/spline_interpolation/bases/bspline7basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline7Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 8
        degree = support - 1
        poles = (
            -0.53528043079643816554240378168164607183392315234269,
            -0.12255461519232669051527226435935734360548654942730,
            -0.0091486948096082769285930216516478534156925639545994,
        )

        # Call super constructor
        super(BSpline7Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 4 <= |x|
        y = np.zeros_like(x)

        # Case 3 <= x < 4
        # -(-4 + x)^7/5040
        y = np.where(
            np.logical_and(x_abs >= 3, x_abs < 4), -1 / 5040 * (-4 + x_abs) ** 7, y
        )

        # Case 2 <= x < 3
        # x^7/720 - x^6/36 + (7 x^5)/30 - (19 x^4)/18 + (49 x^3)/18 - (23 x^2)/6 + (217 x)/90 - 139/630
        # (7 x (x (x (x (x ((x - 20) x + 168) - 760) + 1960) - 2760) + 1736) - 1112)/5040
        y = np.where(
            np.logical_and(x_abs >= 2, x_abs < 3),
            1
            / 5040
            * (
                7
                * x_abs
                * (
                    x_abs
                    * (
                        x_abs
                        * (x_abs * (x_abs * ((x_abs - 20) * x_abs + 168) - 760) + 1960)
                        - 2760
                    )
                    + 1736
                )
                - 1112
            ),
            y,
        )

        # Case 1 <= x < 2
        # -(21 x^7 - 252 x^6 + 1176 x^5 - 2520 x^4 + 1960 x^3 + 504 x^2 + 392 x - 2472)/5040
        # (2472 - 7 x (x (x (3 (x - 6) x ((x - 6) x + 20) + 280) + 72) + 56))/5040
        y = np.where(
            np.logical_and(x_abs >= 1, x_abs < 2),
            1
            / 5040
            * (
                2472
                - 7
                * x_abs
                * (
                    x_abs
                    * (
                        x_abs
                        * (3 * (x_abs - 6) * x_abs * ((x_abs - 6) * x_abs + 20) + 280)
                        + 72
                    )
                    + 56
                )
            ),
            y,
        )

        # Case |x| < 1 (i.e., -1 < x < 1)
        # (35 x^7 - 140 x^6 + 560 x^4 - 1680 x^2 + 2416)/5040
        # 1/144 (x^5 - 4 x^4 + 16 x^2 - 48) x^2 + 151/315
        y = np.where(
            x_abs < 1,
            1 / 144 * (x_abs**5 - 4 * x_abs**4 + 16 * x_abs**2 - 48) * x_abs**2
            + 151 / 315,
            y,
        )

        return y
