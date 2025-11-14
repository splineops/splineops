# splineops/src/splineops/spline_interpolation/bases/bspline8basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline8Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 9
        degree = support - 1
        poles = (
            -0.57468690924876543053013930412874542429066157804125,
            -0.16303526929728093524055189686073705223476814550830,
            -0.023632294694844850023403919296361320612665920854629,
            -0.00015382131064169091173935253018402160762964054070043,
        )

        # Call super constructor
        super(BSpline8Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:

        # Pre-computations
        x_abs = np.abs(x)

        # Case 9/2 <= |x|
        y = np.zeros_like(x)

        # Case 7/2 <= |x| < 9/2
        # (9 - 2 x)^8/10321920
        # x^8/40320 - x^7/1120 + (9 x^6)/640 - (81 x^5)/640 + (729 x^4)/1024 - (6561 x^3)/2560 + (59049 x^2)/10240 - (531441 x)/71680 + 4782969/1146880
        y = np.where(
            np.logical_and(x_abs >= 7 / 2, x_abs < 9 / 2),
            1 / 10321920 * (9 - 2 * x_abs) ** 8,
            y,
        )

        # Case 5/2 <= |x| < 7/2
        # -(256 x^8 - 6912 x^7 + 80640 x^6 - 528192 x^5 + 2106720 x^4 - 5163984 x^3 + 7383600 x^2 - 5257836 x + 1104561)/1290240
        # -((4 x (2 x (2 (x - 15) x + 171) - 879) + 3441) (8 x (x (2 (x - 12) x + 99) - 150) + 321))/1290240
        y = np.where(
            np.logical_and(x_abs >= 5 / 2, x_abs < 7 / 2),
            -1
            / 1290240
            * (
                (
                    4 * x_abs * (2 * x_abs * (2 * (x_abs - 15) * x_abs + 171) - 879)
                    + 3441
                )
                * (8 * x_abs * (x_abs * (2 * (x_abs - 12) * x_abs + 99) - 150) + 321)
            ),
            y,
        )

        # Case 3/2 <= |x| < 5/2
        # x^8/1440 - x^7/80 + (3 x^6)/32 - (119 x^5)/320 + (207 x^4)/256 - (1127 x^3)/1280 + (195 x^2)/512 - (1457 x)/5120 + 145167/286720
        # (56 x (2 x (2 x (x (4 x (2 x ((x - 18) x + 135) - 1071) + 9315) - 10143) + 8775) - 13113) + 1306503)/2580480
        y = np.where(
            np.logical_and(x_abs >= 3 / 2, x_abs < 5 / 2),
            1
            / 2580480
            * (
                56
                * x_abs
                * (
                    2
                    * x_abs
                    * (
                        2
                        * x_abs
                        * (
                            x_abs
                            * (
                                4
                                * x_abs
                                * (2 * x_abs * ((x_abs - 18) * x_abs + 135) - 1071)
                                + 9315
                            )
                            - 10143
                        )
                        + 8775
                    )
                    - 13113
                )
                + 1306503
            ),
            y,
        )

        # Case 1/2 <= |x| < 3/2
        # -(1792 x^8 - 16128 x^7 + 48384 x^6 - 28224 x^5 - 90720 x^4 - 7056 x^3 + 365904 x^2 - 252 x - 584361)/1290240
        # (584361 - 28 x (4 x (x (2 x (2 x (4 x ((x - 9) x + 27) - 63) - 405) - 63) + 3267) - 9))/1290240
        y = np.where(
            np.logical_and(x_abs >= 1 / 2, x_abs < 3 / 2),
            1
            / 1290240
            * (
                584361
                - 28
                * x_abs
                * (
                    4
                    * x_abs
                    * (
                        x_abs
                        * (
                            2
                            * x_abs
                            * (
                                2
                                * x_abs
                                * (4 * x_abs * ((x_abs - 9) * x_abs + 27) - 63)
                                - 405
                            )
                            - 63
                        )
                        + 3267
                    )
                    - 9
                )
            ),
            y,
        )

        # Case |x| < 1/2 (i.e., -(1/2) < x < 1/2)
        # (8960 x^8 - 80640 x^6 + 433440 x^4 - 1456560 x^2 + 2337507)/5160960
        # ((16 x^6 - 144 x^4 + 774 x^2 - 2601) x^2)/9216 + 259723/573440
        y = np.where(
            x_abs < 1 / 2,
            1
            / 9216
            * ((16 * x_abs**6 - 144 * x_abs**4 + 774 * x_abs**2 - 2601) * x_abs**2)
            + 259723 / 573440,
            y,
        )

        return y
