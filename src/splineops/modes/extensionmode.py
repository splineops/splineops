from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy.typing as npt

from splineops.bases.splinebasis import SplineBasis


class ExtensionMode(metaclass=ABCMeta):
    """
    Abstract base class for different extension modes in spline-based operations.

    This class defines the interface for extending signals and computing coefficients
    using different extension modes in conjunction with a given spline basis.
    """

    # Abstract methods
    @staticmethod
    @abstractmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Extend the signal based on the provided indexes, weights, and the length of the signal.

        Parameters
        ----------
        indexes : :class:`numpy.typing.NDArray`
            Array of indexes where the extension should occur.
        weights : :class:`numpy.typing.NDArray`
            Weights associated with the extension at the given indexes.
        length : float
            The length of the original signal.

        Returns
        -------
        Tuple[:class:`numpy.typing.NDArray`, :class:`numpy.typing.NDArray`]
            A tuple containing the extended indexes and the corresponding weights.

        Notes
        -----
        This method needs to be implemented by subclasses to define specific extension behavior.
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:
        """
        Compute the spline coefficients for the given data using the specified spline basis.

        Parameters
        ----------
        data : :class:`numpy.typing.NDArray`
            The input data for which to compute the spline coefficients.
        basis : SplineBasis
            The spline basis to be used for computing the coefficients.

        Returns
        -------
        :class:`numpy.typing.NDArray`
            The computed spline coefficients.
        """
        # TODO(dperdios): add `axis` argument? Currently computations need to
        #  be compatible with N-D arrays along the last axis
        pass
