.. splineops/docs/api/02_resize.rst

.. _api-resize:

Resize
======

Functions for resizing N-dimensional data using standard spline interpolation,
or specialized least-squares / oblique projection methods.

High-level helper
-----------------

The main entry point is :func:`splineops.resize.resize`, which selects both
the algorithm and the spline degree via a single ``method`` string.

.. autofunction:: splineops.resize.resize

Common values for ``method`` are:

- ``"fast"``                         – interpolation, degree 0 (nearest)
- ``"linear"``                       – interpolation, degree 1
- ``"quadratic"``                    – interpolation, degree 2
- ``"cubic"``                        – interpolation, degree 3

- ``"linear-fast_antialiasing"``     – oblique projection, degree 1
- ``"quadratic-fast_antialiasing"``  – oblique projection, degree 2
- ``"cubic-fast_antialiasing"``      – oblique projection, degree 3

- ``"linear-best_antialiasing"``     – least-squares projection, degree 1
- ``"quadratic-best_antialiasing"``  – least-squares projection, degree 2
- ``"cubic-best_antialiasing"``      – least-squares projection, degree 3

Anti-aliasing variants (``*-fast_antialiasing`` and ``*-best_antialiasing``)
are preferred for down-sampling.

Advanced degrees API
--------------------

For full control over the three spline degrees (interpolation, analysis,
synthesis), use :func:`splineops.resize.resize_degrees`. This exposes the
Muñoz/Unser least-squares and oblique projection framework directly.

.. autofunction:: splineops.resize.resize_degrees


Module reference
----------------

For completeness, the full module API is:

.. automodule:: splineops.resize
   :undoc-members:
   :show-inheritance:


See also
--------

:class:`~splineops.interpolate.tensorspline.TensorSpline`
   The base class used internally for spline interpolation.
