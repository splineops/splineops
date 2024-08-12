from abc import ABCMeta, abstractmethod
from typing import Sequence, Optional
from numbers import Real
from collections import abc
import numpy.typing as npt
import numpy as np

# TODO(dperdios): Naming BSplineBasis? What about OMOMS or others?
class SplineBasis(metaclass=ABCMeta):
    """
    Abstract base class for spline basis functions.

    Parameters
    ----------
    support : int
        The support of the spline basis, representing the smallest interval in which the basis function is non-zero.
    degree : int
        The degree of the spline basis function. This parameter is not currently used in the implementation.
    poles : Optional[Sequence[float]], optional
        Poles of the spline basis function, by default None.

    Raises
    ------
    ValueError
        If `support` or `degree` is not an integer.
    TypeError
        If `poles` is not a sequence of real numbers.
    """

    def __init__(
        self, support: int, degree: int, poles: Optional[Sequence[float]] = None
    ) -> None:

        # Support
        # TODO(dperdios): should consider renaming it? Support is a measure of
        #  the smallest interval in which φ(x) ≠ 0. When implementing the sym
        #  versions, we have a larger "support"
        if not isinstance(support, int):
            raise ValueError("Must be an integer.")
        self._support = support

        # Degree
        # TODO(dperdios): could be removed for now (as not used at all)
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
        """
        int: The support of the spline basis, representing the smallest interval in which the basis function is non-zero.
        """
        return self._support

    @property
    def degree(self):
        """
        int: The degree of the spline basis function.
        """
        return self._degree

    @property
    def poles(self):
        """
        Optional[Tuple[float]]: The poles of the spline basis function.
        """
        return self._poles

    # Methods
    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the spline basis function at given points.

        Parameters
        ----------
        x : npt.NDArray
            Input array of real numbers where the spline basis is evaluated.

        Returns
        -------
        npt.NDArray
            The evaluated spline basis function values at the input points.

        Raises
        ------
        ValueError
            If `x` is not an array of real numbers.
        """

        # Check input
        if not np.isrealobj(x):
            raise ValueError("Must be an array of real numbers.")

        return self.eval(x=x)

    # Abstract methods
    @staticmethod
    @abstractmethod
    def eval(x: npt.NDArray) -> npt.NDArray:
        """
        Abstract method to evaluate the spline basis function.

        Parameters
        ----------
        x : npt.NDArray
            Input array of real numbers where the spline basis is evaluated.

        Returns
        -------
        npt.NDArray
            The evaluated spline basis function values at the input points.
        """
        pass

    # TODO(dperdios): Alternative constructor?
    # @classmethod
    # def from_name(cls, name: str):
    #     from splineops.bases.utils import create_spline_basis
    #     # Note: cannot import this before initializing SplineBasis
    #     return create_spline_basis(name=name)

    # Methods
    def compute_support_indexes(self, x: npt.NDArray) -> npt.NDArray:
        """
        Compute the support indexes for the spline basis at given points.

        Parameters
        ----------
        x : npt.NDArray
            Input array of real numbers for which to compute support indexes.

        Returns
        -------
        npt.NDArray
            The support indexes for the spline basis.

        Raises
        ------
        ValueError
            If `x` is not an array of real numbers.
        """

        # Check input
        if not np.isrealobj(x):
            raise ValueError("Must be an array of real numbers.")

        # Span and offset
        support = self.support
        idx_offset = 0.5 if support & 1 else 0.0  # offset for odd support
        idx_span = np.arange(support, like=x)
        idx_span -= (support - 1) // 2

        # Floor rational indexes and convert to integers
        # ind_fl = np.array(np.floor(ind + self._idx_offset), dtype=int_dtype)
        ind_fl = np.array(np.floor(x + idx_offset), dtype=idx_span.dtype, like=idx_span)

        # TODO(dperdios): check fastest axis for computations
        # First axis
        _ns = tuple([support] + ind_fl.ndim * [1])
        idx = ind_fl + np.reshape(idx_span, _ns)
        # # Last axis
        # idx = ind_fl[..., np.newaxis] + idx_span

        return idx
