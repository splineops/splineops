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
       NumPy point supports contract directly with their weights; CuPy retains
       its existing broadcast-and-reduce path until GPU evidence exists.
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
     - Resize, ``TensorSpline``, affine, smoothing, regression, differentials
     - Separate public concepts
     - ``ResizePlan`` represents reusable regular-grid projection work.
       ``TensorSplineGeometryPlan`` retains fixed query support independently
       of compatible sample values, and ``AffinePlan`` composes detached
       geometry plans.  Smoothing retains a frequency response, regression a
       sparse factorization, and differentials a fixed shape/spacing contract.
       These purpose-specific lifetimes are not forced into one abstraction.

Array-role contract
-------------------

``batch`` and ``channel`` are roles assigned by a public API, not inferred from
array rank:

.. list-table:: Spatial and non-spatial axes
   :header-rows: 1
   :widths: 20 34 46

   * - Module
     - Spatial axes
     - Batch/channel behavior
   * - Resize
     - Explicit ``axes``; all axes when omitted
     - Unselected axes are preserved exactly.
   * - Affine
     - Exactly two or three ``spatial_axes``
     - Remaining slices are mathematically independent and share one batched
       prefilter/support contraction with memory-bounded query tiles.  Tagged
       persistence is atomic numeric-plus-JSON data, not object serialization.
   * - ``TensorSpline``
     - Every construction-data axis is a spline dimension
     - Query coordinates may be batched; sample-value channel axes are not
       inferred.
   * - Smoothing
     - Explicit ``axes``; every axis when omitted
     - Unselected batch/channel slices share one batched real-FFT execution.
   * - Differentials
     - Two or three explicit ``spatial_axes`` through ``DifferentialPlan``
     - Remaining slices are mathematically independent but execute through one
       batched coefficient/derivative workspace; the legacy ``Differentials``
       object remains scalar.  Structured outputs may not alias the source or
       one another.
   * - Multiscale
     - Two explicit ``spatial_axes`` for 2-D pyramids and wavelets
     - Small planes use cache-sized vectorized groups; cache-filling planes are
       dispatched directly to avoid full-array transpose copies.  Axes are
       never inferred from rank.

Rules for shared internals
--------------------------

* Public classes and functions do not move merely to reduce duplicate code.
* A shared primitive needs parity tests in every consumer before replacement.
* Dtype promotion, complex values, backend ownership, boundaries, singleton
  behavior, and mutation are part of a primitive's contract.
* Internal APIs may change between releases.  Public modules must not expose
  private helper objects as accidental compatibility promises.
* Interpolation prefiltering is shared only by ``TensorSpline`` and
  differentials, which use the same cardinal-spline coefficient contract.
  Resize's scale-dependent projection filters remain specialized.
* Future legacy ports retain source-level attribution and method records even
  though the maintainer has cleared the current distribution's provenance; see
  :doc:`provenance`.

This conservative structure is intentional.  It lets SplineOps earn reuse
through evidence while preserving ``TensorSpline``, resize, affine,
differentials, smoothing, regression, and multiscale tools as separate public
capabilities.
