from abc import ABCMeta, abstractmethod
from typing import Sequence, Optional
from numbers import Real
from collections import abc
import numpy.typing as npt


# TODO(dperdios): Naming BSplineBasis? What about OMOMS or others?
class SplineBasis(metaclass=ABCMeta):
    def __init__(self, support: int, poles: Optional[Sequence[float]] = None) -> None:
        # Support
        # TODO(dperdios): should consider renaming it? Support is a measure of
        #  the smallest interval in which φ(x) ≠ 0. When implementing the sym
        #  versions, we have a larger "support"
        if not isinstance(support, int):
            raise ValueError("Must be an integer.")
        self._support = support

        # Degree
        # TODO(dperdios): this is not always true (e.g., for Keys or bspline0sym)
        # TODO(dperdios): could be removed for now (as not used at all)
        self._degree = support - 1

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
        return self._support

    @property
    def degree(self):
        return self._degree

    @property
    def poles(self):
        return self._poles

    # Methods
    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        return self.eval(x=x)

    # Abstract methods
    @staticmethod
    @abstractmethod
    def eval(x: npt.NDArray) -> npt.NDArray:
        pass

    # TODO(dperdios): Alternative constructor?
    # @classmethod
    # def from_name(cls, name: str):
    #     from bssp.bases.utils import create_spline_basis
    #     # Note: cannot import this before initializing SplineBasis
    #     return create_spline_basis(name=name)
