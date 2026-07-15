.. splineops/docs/api/03_affine.rst

.. _api-affine:

Affine
======

Functions for affine transforms (currently rotation) on 2D or 3D data
using spline interpolation.  Rotation uses pull-back coordinates,
whole-sample mirror extension, the input shape as output shape, and bounded
coordinate tiles.  Equivalent SciPy comparisons establish numerical parity,
not a performance win.

.. automodule:: splineops.affine.affine
   :members:
   :undoc-members:
   :show-inheritance:

See also
--------
:class:`~splineops.spline_interpolation.tensor_spline.TensorSpline`
   The base class used internally for spline interpolation.
