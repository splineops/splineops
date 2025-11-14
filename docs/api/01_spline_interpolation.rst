.. splineops/docs/api/01_spline_interpolation.rst

.. _api-spline_interpolation:

Spline interpolation
====================

The :class:`splineops.spline_interpolation.tensorspline.TensorSpline` class is the core spline-based interpolator.
Users can construct a spline model for their N-dimensional data (with custom boundary extension modes, spline degrees, etc.) 
and then evaluate the model at arbitrary coordinates.

.. automodule:: splineops.spline_interpolation.tensorspline
   :members:
   :show-inheritance:
   :special-members: __init__, __call__, eval
   :member-order: bysource
