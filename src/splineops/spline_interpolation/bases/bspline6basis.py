# splineops/src/splineops/spline_interpolation/bases/bspline6basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline6Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 7
        degree = support - 1
        poles = (
            -0.48829458930304475513011803888378906211227916123938,
            -0.081679271076237512597937765737059080653379610398148,
            -0.0014141518083258177510872439765585925278641690553467,
        )

        # Call super constructor
        super(BSpline6Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 7/2 <= |x|
        y = np.zeros_like(x)

        # Case 5/2 <= |x| < 7/2
        # (7 - 2 x)^6/46080
        # x^6/720 - (7 x^5)/240 + (49 x^4)/192 - (343 x^3)/288 + (2401 x^2)/768 - (16807 x)/3840 + 117649/46080
        y = np.where(
            np.logical_and(x_abs >= 5 / 2, x_abs < 7 / 2),
            1 / 46080 * (7 - 2 * x_abs) ** 6,
            y,
        )

        # Case 3/2 <= |x| < 5/2
        # -(192 x^6 - 2688 x^5 + 15120 x^4 - 42560 x^3 + 59220 x^2 - 30408 x - 4137)/23040
        # (4137 - 4 x (x (4 x (3 x (4 (x - 14) x + 315) - 2660) + 14805) - 7602))/23040
        y = np.where(
            np.logical_and(x_abs >= 3 / 2, x_abs < 5 / 2),
            1
            / 23040
            * (
                4137
                - 4
                * x_abs
                * (
                    x_abs
                    * (
                        4
                        * x_abs
                        * (3 * x_abs * (4 * (x_abs - 14) * x_abs + 315) - 2660)
                        + 14805
                    )
                    - 7602
                )
            ),
            y,
        )

        # Case 1/2 <= |x| < 3/2
        # (23583 - 420 x - 16380 x^2 - 5600 x^3 + 15120 x^4 - 6720 x^5 + 960 x^6)/46080
        # (20 x (x (4 x (3 x (4 (x - 7) x + 63) - 70) - 819) - 21) + 23583)/46080
        y = np.where(
            np.logical_and(x_abs >= 1 / 2, x_abs < 3 / 2),
            1
            / 46080
            * (
                20
                * x_abs
                * (
                    x_abs
                    * (
                        4 * x_abs * (3 * x_abs * (4 * (x_abs - 7) * x_abs + 63) - 70)
                        - 819
                    )
                    - 21
                )
                + 23583
            ),
            y,
        )

        # Case |x| < 1/2 (i.e., -(1/2) < x < 1/2)
        # -(320 x^6 - 1680 x^4 + 4620 x^2 - 5887)/11520
        # (5887 - 20 x^2 (16 x^4 - 84 x^2 + 231))/11520
        y = np.where(
            x_abs < 1 / 2,
            1 / 11520 * (5887 - 20 * x_abs**2 * (16 * x_abs**4 - 84 * x_abs**2 + 231)),
            y,
        )

        return y
