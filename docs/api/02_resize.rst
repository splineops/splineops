.. splineops/docs/api/02_resize.rst

.. _api-resize:

Resize
======

Functions for resizing N-dimensional data using standard spline interpolation,
or projection-based antialiasing methods.

High-level helper
-----------------

The main entry point is :func:`~splineops.resize.resize`, which selects both
the spline degrees and (optional) antialiasing behavior via a single
``method`` string. For production downsampling, the recommended public presets
are the oblique-projection methods ``"linear-antialiasing"``,
``"quadratic-antialiasing"`` and ``"cubic-antialiasing"``.

.. autofunction:: splineops.resize.resize

Advanced degrees API
--------------------

For full control over the three spline degrees (interpolation, analysis,
synthesis), use :func:`~splineops.resize.resize_degrees`.

This exposes the underlying Muñoz/Unser projection framework directly,
including advanced/reference equal-degree least-squares configurations.

.. autofunction:: splineops.resize.resize_degrees

Reusable plans
--------------

For repeated same-shape workloads, use
:class:`~splineops.resize.ResizePlan` to resolve the target geometry once and
apply it to multiple arrays.

.. autoclass:: splineops.resize.ResizePlan
   :members:


See also
--------

:class:`~splineops.interpolate.tensor_spline.TensorSpline`
   The base class used internally for spline interpolation.
