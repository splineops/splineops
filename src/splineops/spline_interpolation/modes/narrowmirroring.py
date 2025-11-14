# splineops/src/splineops/spline_interpolation/modes/narrowmirroring.py

import numpy as np
from typing import Tuple
import numpy.typing as npt

from ..bases.splinebasis import SplineBasis
from ..modes.extensionmode import ExtensionMode
from ..utils import _data_to_coeffs

class NarrowMirroring(ExtensionMode):

    # Methods
    @staticmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        # Pre-computation
        data_len = length
        len_2 = 2 * data_len - 2
        indexes_ext = np.zeros_like(indexes)  # TODO(dperdios): in-place update?
        weights_ext = np.copy(weights)  # TODO(dperdios): in-place update?

        # Sawtooth function (with special case for K = 0)
        if data_len == 1:
            return indexes_ext, weights_ext
        else:
            indexes_ext[:] = np.round(
                len_2 * np.abs(indexes / len_2 - np.floor(indexes / len_2 + 0.5))
            )
            return indexes_ext, weights_ext

    @staticmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:

        # Local copy
        coeffs = np.copy(data)

        # Check whether `basis` has no `poles` (i.e., no computation needed)
        if basis.poles is None:
            return coeffs

        # Get poles
        poles = basis.poles

        # Compute coefficients (in-place)
        # _compute_coeffs_narrow_mirror_wg(data=coeffs, poles=poles)
        _data_to_coeffs(data=coeffs, poles=poles, boundary="mirror")
        # TODO(dperdios): the batched version has an issue for
        #  a single-sample signal (anti-causal init). There is also
        #  an issue for the boundary condition with single-sample
        #  signals for the mirror case.

        return coeffs
