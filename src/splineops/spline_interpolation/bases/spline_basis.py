# splineops/src/splineops/spline_interpolation/bases/spline_basis.py

from abc import ABCMeta, abstractmethod
from typing import Sequence, Optional
from numbers import Real
from collections import abc
import numpy.typing as npt
import numpy as np


class SplineBasis(metaclass=ABCMeta):
    """Compact interpolation kernel and its coefficient-prefilter metadata."""

    def __init__(
        self, support: int, degree: int, poles: Optional[Sequence[float]] = None
    ) -> None:
        # Stencil size includes both endpoints for symmetric tie-breaking bases.
        if not isinstance(support, int):
            raise ValueError("Must be an integer.")
        self._support = support

        # Degree
        if not isinstance(degree, int):
            raise ValueError("Must be an integer.")
        self._degree = degree

        # Poles
        if poles is not None:
            if not isinstance(poles, abc.Sequence) or not all(
                isinstance(p, Real) for p in poles
            ):
                raise TypeError(f"Must be a sequence of {Real.__name__}")
            poles = tuple(poles)
        self._poles = poles

    # Properties
    @property
    def support(self):
        """Number of coefficient indexes in an evaluation stencil."""
        return self._support

    @property
    def degree(self):
        """Degree metadata supplied by the concrete basis."""
        return self._degree

    @property
    def poles(self):
        """Recursive prefilter poles, or ``None`` for an interpolating kernel."""
        return self._poles

    # Methods
    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        # Check input
        if not np.isrealobj(x):
            raise ValueError("Must be an array of real numbers.")

        return self.eval(x=x)

    # Abstract methods
    @staticmethod
    @abstractmethod
    def eval(x: npt.NDArray) -> npt.NDArray:
        pass

    # Methods
    def compute_support_indexes(self, x: npt.NDArray) -> npt.NDArray:
        """Return the stencil indexes with shape ``(support, *x.shape)``."""
        # Check input
        if not np.isrealobj(x):
            raise ValueError("Must be an array of real numbers.")

        # Span and offset
        support = self.support
        idx_offset = 0.5 if support & 1 else 0.0  # offset for odd support
        idx_span = np.arange(support, like=x)
        idx_span -= (support - 1) // 2

        # Floor rational indexes and convert to integers
        ind_fl = np.array(np.floor(x + idx_offset), dtype=idx_span.dtype, like=idx_span)

        # Evaluation contracts over this leading stencil axis.
        _ns = tuple([support] + ind_fl.ndim * [1])
        idx = ind_fl + np.reshape(idx_span, _ns)

        return idx
