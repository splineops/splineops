Affine and Differentials graduation audits
===========================================

These audits record evidence as of 2026-07-16.  They are deliberately stricter
than an implementation checklist: a module can be capable and well tested
without yet carrying a stable compatibility promise.  Neither module is
promoted by this review.

Affine audit
------------

.. list-table:: ``AffinePlan`` graduation gates
   :header-rows: 1
   :widths: 24 14 42 20

   * - Gate
     - State
     - Evidence
     - Remaining work
   * - Geometry and numerical contract
     - Pass
     - 2-D/3-D pull-back matrices, output geometry, degrees 0--7, mirror/zero
       modes, analytical cases, and SciPy parity for matched degrees 0--5.
     - Keep higher-degree claims SplineOps-specific.
   * - Axes, dtype, mutation, and buffers
     - Pass
     - Explicit batch/channel axes, float32/float64 execution, source
       preservation, exact and strided output buffers, and randomized scalar
       dispatch parity.
     - None for the documented surface.
   * - Bounded resources
     - Pass
     - Cached plans enforce ``max_retained_bytes``; one-shot tiled execution
       and retained-byte inspection are tested and benchmarked.
     - Continue platform monitoring.
   * - Coefficient reuse and persistence
     - Pass
     - Immutable compatibility fields, atomic schema-2 archives, permanent
       schema-1/schema-2 byte fixtures, checked migration, corruption rejection,
       and endian/metadata validation.
     - Preserve every historical fixture when a future schema is added.
   * - Concurrency and restart behavior
     - Evidence collecting
     - Thread reuse and spawned-process loads pass unit tests and the repeated
       public-API soak; the cross-platform soak workflow publishes JSON.
     - Accumulate real workflow runs without contract changes.
   * - Performance claims
     - Pass, narrowly scoped
     - Complete-workflow gains exist for coefficient sharing, but matched
       one-shot SciPy affine execution remains faster.  Profile evidence, not
       a universal speed claim, governs implementation changes.
     - Re-profile before any further optimization.
   * - Compatibility duration and migration
     - Open
     - The public contract and failure modes are documented.
     - Complete an API-soak period and prepare migration guidance if names or
       buffer rules change.

Decision: ``AffinePlan`` is technically strong but remains **experimental**.
The open gate is compatibility evidence over time, not a missing attempt to
win every one-shot timing comparison.

Differentials audit
-------------------

.. list-table:: ``DifferentialPlan`` graduation gates
   :header-rows: 1
   :widths: 24 14 42 20

   * - Gate
     - State
     - Evidence
     - Remaining work
   * - Derivative convention
     - Pass for current scope
     - Increasing-coordinate gradients, packed upper-triangular Hessians,
       Laplacians, physical spacing, analytical polynomial/trigonometric
       invariants, and whole-sample mirror boundary fixtures.
     - Do not imply boundaries other than the documented cubic mirror model.
   * - Axes, dtype, mutation, and buffers
     - Pass
     - 2-D/3-D explicit spatial axes, integer promotion, float32/float64
       preservation, selective families, prevalidated exact/strided structured
       buffers, non-aliasing, and randomized scalar dispatch parity.
     - None for the planned API's documented surface.
   * - Resource behavior
     - Pass
     - One lazy workspace shares coefficient and derivative intermediates;
       Laplacian-only calls skip mixed Hessians and the plan retains no arrays.
     - Continue batch-size monitoring.
   * - Concurrency and repeated buffers
     - Evidence collecting
     - The repeated volume-feature soak preserves source data and destination
       identities with exact scalar parity.
     - Accumulate cross-platform soak artifacts without contract changes.
   * - Legacy relationship
     - Open
     - ``DifferentialPlan`` provides the general planned numerical API while
       ``Differentials`` preserves scalar convenience and 2-D angular maps.
     - Publish migration guidance before changing or deprecating either name.
   * - Compatibility duration
     - Open
     - Errors and output selection are explicit and covered.
     - Complete the API-soak period; keep angular behavior outside a general
       N-D claim unless it receives its own contract.

Decision: ``DifferentialPlan`` also remains **experimental**.  It is ready for
serious evaluation, but stabilizing it today would turn a short, internally
reviewed history into a compatibility promise prematurely.

Next review trigger
-------------------

Review these tables after a meaningful period of unchanged downstream use and
cross-platform soak artifacts—not merely after another green unit-test run.
Promotion can then happen module by module.  ``TensorSpline``, affine,
differentials, resize, and the research modules remain independent APIs
throughout that process.
