# periodicpadding.py

import numpy as np
from typing import Tuple
import numpy.typing as npt

from splineops.bases.splinebasis import SplineBasis
from splineops.modes.extensionmode import ExtensionMode
from splineops.interpolate.utils import _data_to_coeffs


class PeriodicPadding(ExtensionMode):

    @staticmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: int
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Extends the signal periodically by wrapping indexes around the signal length.

        Parameters
        ----------
        indexes : ndarray
            The indexes of the data points to be used in interpolation.
        weights : ndarray
            The interpolation weights corresponding to the indexes.
        length : int
            The length of the data array.

        Returns
        -------
        indexes_ext : ndarray
            The extended indexes adjusted for periodicity.
        weights_ext : ndarray
            The corresponding weights (unchanged).
        """
        # Wrap indexes around the signal length to achieve periodicity
        indexes_ext = np.mod(indexes, length)
        weights_ext = weights  # Weights remain unchanged

        return indexes_ext, weights_ext

    @staticmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:
        """
        Computes the spline coefficients with periodic boundary conditions.

        Parameters
        ----------
        data : ndarray
            The input data array.
        basis : SplineBasis
            The spline basis used for interpolation.

        Returns
        -------
        coeffs : ndarray
            The computed spline coefficients.
        """
        coeffs = np.copy(data)

        # Check if basis has poles (i.e., if pre-filtering is needed)
        if basis.poles is None:
            return coeffs

        # Get poles
        poles = basis.poles

        # Compute coefficients in-place with periodic boundary conditions
        _data_to_coeffs(data=coeffs, poles=poles, boundary="periodic")

        return coeffs
