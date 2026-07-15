"""Continuous tensor-product spline interpolation."""

from .tensor_spline import TensorSpline
from .query_plan import TensorSplineQueryPlan

__all__ = ["TensorSpline", "TensorSplineQueryPlan"]
