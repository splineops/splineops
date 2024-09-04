import numpy as np
from typing import Tuple
import numpy.typing as npt

from splineops.bases.splinebasis import SplineBasis
from splineops.modes.extensionmode import ExtensionMode
from splineops.interpolate.utils import _data_to_coeffs


class PeriodicPadding(ExtensionMode):
    # Methods
    @staticmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Periodic signal extension. Indexes that go out of bounds are wrapped around 
        according to the periodic boundary conditions.
        """

        # Alias for the data length
        data_len = length

        # Ensure valid data length
        if data_len <= 0:
            raise ValueError("The length of the signal must be greater than 0.")

        # Wrap indexes into valid range
        indexes_ext = np.copy(indexes)  # TODO(dperdios): in-place update?
        weights_ext = np.copy(weights)  # TODO(dperdios): in-place update?
        
        indexes_ext = np.mod(indexes_ext, data_len)

        return indexes_ext, weights_ext

    @staticmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:
        """
        Computes coefficients for periodic signal extension using the given basis.
        Periodic boundary conditions are applied.
        """
        
        # Local copy of the data
        coeffs = np.copy(data)

        # Check if basis has no poles (if so, no filtering needed)
        if basis.poles is None:
            return coeffs

        # Get poles from the spline basis
        poles = basis.poles

        # Compute coefficients using periodic boundary conditions (in-place)
        _data_to_coeffs(data=coeffs, poles=poles, boundary="periodic")

        return coeffs
