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

The output shape is resolved first and defines the single endpoint-aligned
sampling grid. Use ``axes=`` to identify spatial axes in arrays that also carry
batch or channel dimensions; unselected axes are not filtered.

.. autofunction:: splineops.resize.resize

Advanced degrees API
--------------------

For full control over the three spline degrees (interpolation, analysis,
synthesis), use :func:`~splineops.resize.resize_degrees`.

This exposes the underlying Muñoz/Unser projection framework directly,
including advanced equal-degree least-squares configurations. On the public
zero-shift grid, every projection with analysis degree one or greater uses a
stable direct compact cross-Gram operator; analysis degree zero uses the
finite-difference form. These are degree controls, not additional method
presets.

.. autofunction:: splineops.resize.resize_degrees

Reusable plans
--------------

For repeated same-shape workloads, use
:class:`~splineops.resize.ResizePlan` to resolve the target geometry once and
apply it to multiple arrays. Plans are read-only and safe to share across
threads; a compatible output array also avoids the final allocation and copy.
Process-wide one-shot plans and per-plan idle workspaces are byte-bounded by
default, so occasional large shapes or bursts of callers do not create an
unbounded retained-memory cache.

.. autoclass:: splineops.resize.ResizePlan
   :members:


See also
--------

:class:`~splineops.interpolate.tensor_spline.TensorSpline`
   The base class used internally for spline interpolation.
