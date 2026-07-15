.. splineops/docs/api/01_spline_interpolation.rst

.. _api-spline_interpolation:

Spline Interpolation
====================

The :class:`~splineops.spline_interpolation.tensor_spline.TensorSpline` class is the core spline-based interpolator.
Users can construct a spline model for their N-dimensional data (with custom boundary extension modes, spline degrees, etc.) 
and then evaluate the model at arbitrary coordinates.

``TensorSpline`` is also available as the stable convenience import
``from splineops import TensorSpline``.  Construction grids must be uniform;
"arbitrary coordinates" describes evaluation locations, not nonuniform input
sampling.

Construction templates can refit compatible sample arrays with ``with_data``
or consume explicit cardinal coefficients with ``with_coefficients``.  Query
plans retain only fixed-coordinate geometry and can therefore evaluate either
kind of compatible spline without merging ``TensorSpline`` into another
module.  Keys cubic convolution and cubic O-MOMS also have independent
published-formula references in the stability-soak suite.

.. automodule:: splineops.spline_interpolation.tensor_spline
   :members:
   :show-inheritance:
   :special-members: __init__, __call__, eval
   :member-order: bysource

Experimental query plans
------------------------

.. automodule:: splineops.spline_interpolation.query_plan
   :members:
   :show-inheritance:
   :special-members: __call__
   :member-order: bysource
