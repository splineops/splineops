# splineops/src/splineops/spline_interpolation/modes/extensionmodes.py

from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy.typing as npt

from ..bases.splinebasis import SplineBasis

class ExtensionMode(metaclass=ABCMeta):
    # Abstract methods
    @staticmethod
    @abstractmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        pass

    @staticmethod
    @abstractmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:
        # TODO(dperdios): add `axis` argument? Currently computations need to
        #  be compatible with N-D arrays along the last axis
        pass
