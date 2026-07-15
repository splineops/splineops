.. splineops/docs/api/05_smoothing_splines.rst

.. _api-smoothing_splines:

Smoothing Splines
=================

Fractional and recursive smoothing-spline tools.  ``SmoothingSplinePlan``
retains a real-FFT half-spectrum response for repeated arrays of one spatial
shape.  Explicit ``axes`` allow the remaining dimensions to serve as batch or
channel axes in one batched FFT execution.  Its periodic transfer function is
checked against an independent dense-DFT fixture derived from the published
fractional-spline estimator.

.. automodule:: splineops.smoothing_splines.smoothing_spline
   :members:
   :undoc-members:
   :show-inheritance:
