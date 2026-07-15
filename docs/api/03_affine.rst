.. splineops/docs/api/03_affine.rst

.. _api-affine:

Affine
======

Spline-interpolated pull-back affine transforms on 2-D or 3-D spatial data.
The module provides a general matrix-and-offset function, a rotation
convenience function, and a reusable geometry plan.  Explicit spatial axes
allow remaining dimensions to act as independent batch or channel axes.
One-shot execution uses bounded coordinate tiles; cached plans retain geometry
behind a caller-controlled memory cap.  ``AffinePlan.prepare_coefficients``
returns an immutable, compatibility-checked ``AffineCoefficientField`` for
deliberate sharing when one sampled field is transformed through several
geometries.  The raw-array ``prefilter`` path remains available for explicit
output-buffer workflows.  Equivalent SciPy comparisons establish numerical
parity, not a universal performance win.

.. automodule:: splineops.affine.affine
   :members:
   :undoc-members:
   :show-inheritance:

See also
--------
:class:`~splineops.spline_interpolation.tensor_spline.TensorSpline`
   The independent continuous spline model used internally for interpolation.
