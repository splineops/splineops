Internal numerical contracts
============================

SplineOps keeps its public modules independent.  Internal reuse is accepted
only when the participating algorithms have the same mathematical contract;
similar vocabulary is not enough.  This page records those decisions so that
future cleanup does not silently change results.

Compatibility matrix
--------------------

.. list-table:: Primitive compatibility
   :header-rows: 1
   :widths: 18 22 22 38

   * - Primitive
     - Current consumers
     - Reuse decision
     - Reason
   * - Cubic mirror prefilter
     - ``TensorSpline``, differentials
     - Shared implementation
     - Both require cardinal cubic coefficients on a whole-sample mirror
       extension.  Differentials uses the batched last-axis implementation and
       independently applies derivative filters.
   * - Basis evaluation
     - ``TensorSpline``, resize, affine
     - Partly shared
     - Affine evaluates through ``TensorSpline``.  Resize projection has a
       different scale-dependent cross-Gram contract and remains specialized.
   * - Boundary mapping
     - Interpolation, resize, pyramids
     - Not unified
     - Mirror names do not imply identical reflection centers, sample grids,
       or projection behavior.  Each public contract keeps its own mapping.
   * - Coordinate generation
     - ``TensorSpline``, affine, resize
     - Shared policy, separate geometry
     - Evaluation is tiled to bound temporaries.  Affine pull-back coordinates
       and endpoint-aligned resize coordinates are purpose-specific.
   * - Array backend detection
     - ``TensorSpline``
     - Centralized inside interpolation
     - NumPy is the package-wide backend.  CuPy remains experimental and must
       not leak into modules that have no GPU contract.
   * - Plan/cache objects
     - Resize, ``TensorSpline``
     - Separate public concepts
     - ``ResizePlan`` represents reusable regular-grid projection work.
       ``TensorSplineQueryPlan`` retains support geometry for one spline and
       one fixed coordinate query.  Neither abstraction is forced onto the
       other module.

Rules for shared internals
--------------------------

* Public classes and functions do not move merely to reduce duplicate code.
* A shared primitive needs parity tests in every consumer before replacement.
* Dtype promotion, complex values, backend ownership, boundaries, singleton
  behavior, and mutation are part of a primitive's contract.
* Internal APIs may change between releases.  Public modules must not expose
  private helper objects as accidental compatibility promises.
* Future legacy ports retain source-level attribution and method records even
  though the maintainer has cleared the current distribution's provenance; see
  :doc:`provenance`.

This conservative structure is intentional.  It lets SplineOps earn reuse
through evidence while preserving ``TensorSpline``, resize, affine,
differentials, smoothing, regression, and multiscale tools as separate public
capabilities.
