# splineops/src/splineops/spline_interpolation/bases/bspline9basis.py

import numpy as np
import numpy.typing as npt

from .splinebasis import SplineBasis

class BSpline9Basis(SplineBasis):
    def __init__(self) -> None:

        # Support, degree and poles
        support = 10
        degree = support - 1
        poles = (
            -0.60799738916862577900772082395428976943963471853991,
            -0.20175052019315323879606468505597043468089886575747,
            -0.043222608540481752133321142979429688265852380231497,
            -0.0021213069031808184203048965578486234220548560988624,
        )

        # Call super constructor
        super(BSpline9Basis, self).__init__(support=support, degree=degree, poles=poles)

    # Methods
    @staticmethod
    def eval(x: npt.NDArray) -> npt.NDArray:
        # Pre-computations
        x_abs = np.abs(x)

        # Case 5 <= |x|
        y = np.zeros_like(x)

        # Case 4 <= |x| < 5
        # -(-5 + x)^9/362880
        # -x^9/362880 + x^8/8064 - (5 x^7)/2016 + (25 x^6)/864 - (125 x^5)/576 + (625 x^4)/576 - (3125 x^3)/864 + (15625 x^2)/2016 - (78125 x)/8064 + 390625/72576
        y = np.where(
            np.logical_and(x_abs >= 4, x_abs < 5),
            -1 / 362880 * (-5 + x_abs) ** 9,
            # -1 / 362880 * np.power((-5 + x_abs), 9),
            y,
        )

        # Case 3 <= |x| < 4
        # x^9/40320 - x^8/1152 + (3 x^7)/224 - (103 x^6)/864 + (43 x^5)/64 - (1423 x^4)/576 + (563 x^3)/96 - (2449 x^2)/288 + (5883 x)/896 - 133663/72576
        # (3 x (x (x (x (x (x (3 x ((x - 35) x + 540) - 14420) + 81270) - 298830) + 709380) - 1028580) + 794205) - 668315)/362880
        y = np.where(
            np.logical_and(x_abs >= 3, x_abs < 4),
            1
            / 362880
            * (
                3
                * x_abs
                * (
                    x_abs
                    * (
                        x_abs
                        * (
                            x_abs
                            * (
                                x_abs
                                * (
                                    x_abs
                                    * (3 * x_abs * ((x_abs - 35) * x_abs + 540) - 14420)
                                    + 81270
                                )
                                - 298830
                            )
                            + 709380
                        )
                        - 1028580
                    )
                    + 794205
                )
                - 668315
            ),
            y,
        )

        # Case 2 <= |x| < 3
        # -(18 x^9 - 450 x^8 + 4860 x^7 - 29400 x^6 + 107730 x^5 - 240660 x^4 + 313740 x^3 - 228600 x^2 + 137295 x - 108710)/181440
        # (108710 - 3 x (2 x (x (x (x (x (3 x ((x - 25) x + 270) - 4900) + 17955) - 40110) + 52290) - 38100) + 45765))/181440
        y = np.where(
            np.logical_and(x_abs >= 2, x_abs < 3),
            1
            / 181440
            * (
                108710
                - 3
                * x_abs
                * (
                    2
                    * x_abs
                    * (
                        x_abs
                        * (
                            x_abs
                            * (
                                x_abs
                                * (
                                    x_abs
                                    * (3 * x_abs * ((x_abs - 25) * x_abs + 270) - 4900)
                                    + 17955
                                )
                                - 40110
                            )
                            + 52290
                        )
                        - 38100
                    )
                    + 45765
                )
            ),
            y,
        )

        # Case 1 <= |x| < 2
        # (77990 + 945 x - 47880 x^2 + 8820 x^3 - 1260 x^4 + 13230 x^5 - 10920 x^6 + 3780 x^7 - 630 x^8 + 42 x^9)/181440
        # (21 x (2 x (x (x (x (x (x ((x - 15) x + 90) - 260) + 315) - 30) + 210) - 1140) + 45) + 77990)/181440
        y = np.where(
            np.logical_and(x_abs >= 1, x_abs < 2),
            1
            / 181440
            * (
                21
                * x_abs
                * (
                    2
                    * x_abs
                    * (
                        x_abs
                        * (
                            x_abs
                            * (
                                x_abs
                                * (
                                    x_abs * (x_abs * ((x_abs - 15) * x_abs + 90) - 260)
                                    + 315
                                )
                                - 30
                            )
                            + 210
                        )
                        - 1140
                    )
                    + 45
                )
                + 77990
            ),
            y,
        )

        # Case |x| < 1 (i.e., -1 < x < 1)
        # -(63 x^9 - 315 x^8 + 2100 x^6 - 11970 x^4 + 44100 x^2 - 78095)/181440
        # (78095 - 21 x^2 (3 (x - 5) x^6 + 100 x^4 - 570 x^2 + 2100))/181440
        y = np.where(
            x_abs < 1,
            1
            / 181440
            * (
                78095
                - 21
                * x_abs**2
                * (3 * (x_abs - 5) * x_abs**6 + 100 * x_abs**4 - 570 * x_abs**2 + 2100)
            ),
            y,
        )

        return y
