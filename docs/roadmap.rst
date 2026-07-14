Development roadmap
===================

SplineOps 2.0 established projection-based resizing as a fast, documented,
and tested part of the library. The next development cycle will use that work
as the foundation for a reliable shared interpolation engine that can also
power affine transforms, differential operators, and future geometric
operations.

This roadmap records current priorities, not compatibility promises. Its order
is intentional: correctness and explicit mathematical contracts come before
new acceleration work.

Guiding principles
------------------

* Maintain one recommended behavior for each operation. Historical behavior is
  not exposed merely because it existed in an earlier implementation.
* Test mathematical invariants, including sample reproduction and perfect
  reconstruction, instead of relying only on fixed reference arrays.
* Optimize complete workloads, including coefficient construction, memory
  movement, and repeated evaluation—not just isolated kernels.
* Keep accelerated and reference implementations numerically consistent.
* Treat a backend as supported only when it has explicit dispatch, tests, and
  continuous integration coverage.

2.0.x: interpolation reliability
--------------------------------

The first priority is to make the existing interpolation contract dependable
for every supported basis, boundary mode, shape, and dtype.

Planned work
~~~~~~~~~~~~

* Add cardinal-reproduction tests across all bases and boundary modes.
* Cover singleton arrays, signals shorter than a basis support, and short
  periodic signals.
* Reject nonuniform coordinate grids explicitly until true nonuniform spline
  interpolation is implemented.
* Correct periodic coefficient construction when the spline support wraps
  around a signal more than once.
* Define predictable input and output dtype rules, including integer input to
  affine operations.
* Replace backend detection based on type-name strings with explicit NumPy and
  CuPy dispatch.
* Validate parameters consistently and replace silent clamping or failure with
  informative exceptions.

Completion criteria
~~~~~~~~~~~~~~~~~~~

* Interpolation reproduces its samples within a documented tolerance for every
  supported basis and mode.
* Degenerate and short inputs either work correctly or fail with a documented,
  actionable exception.
* NumPy behavior is covered across supported Python versions and platforms.

2.1: shared interpolation platform
----------------------------------

The next milestone will consolidate common spline machinery so that resize,
tensor interpolation, affine transforms, and differential operators do not
maintain separate versions of the same ideas.

Planned work
~~~~~~~~~~~~

* Establish shared coefficient-prefilter and boundary-indexing components.
* Introduce an explicit array-backend abstraction and dtype policy.
* Provide stable public imports for the principal operations.
* Evaluate ``TensorSpline`` in bounded-memory tiles instead of materializing
  unnecessarily large tensor-product temporaries.
* Reduce coefficient-generation transposes, copies, and intermediate arrays.
* Add reusable interpolation plans for workloads that evaluate many arrays or
  coordinate sets with the same geometry.
* Add benchmark coverage for construction, one-shot evaluation, repeated
  evaluation, two-dimensional images, and three-dimensional volumes.

Completion criteria
~~~~~~~~~~~~~~~~~~~

* Benchmarks track runtime and peak memory for representative end-to-end
  workloads.
* Repeated evaluation avoids repeating invariant setup work.
* Public imports and dtype behavior are documented and tested.
* Performance changes remain numerically equivalent to the reference path.

2.2: acceleration across modules
--------------------------------

Once the shared implementation is stable, native acceleration can benefit more
than resizing alone.

Planned work
~~~~~~~~~~~~

* Add optimized coefficient-prefilter and evaluation kernels where benchmarks
  show a material benefit.
* Generate affine coordinates in tiles rather than allocating a full meshgrid.
* Rebuild gradients, Laplacians, and Hessians on the shared spline engine,
  eliminating Python row and column loops.
* Preserve input precision and return results instead of relying on mutation,
  console output, or implicit normalization.
* Add performance-regression reporting for both runtime and memory use.

Acceleration will be accepted when it improves realistic workloads, not only a
microbenchmark, and when numerical agreement is maintained across platforms.

Multiscale reliability
----------------------

Wavelet and pyramid development will proceed behind an explicit reconstruction
contract.

Planned work
~~~~~~~~~~~~

* Add randomized perfect-reconstruction tests across shapes, scales, orders,
  and floating-point dtypes.
* Define and implement reversible handling for odd and singleton dimensions, or
  reject unsupported shapes clearly until it is available.
* Audit higher-order spline-wavelet reconstruction accuracy.
* Replace demonstration-only pyramid definitions with validated filters and
  documented mathematical conventions.
* Optimize row and column processing only after reconstruction invariants pass.

Backend and tooling maturity
----------------------------

The surrounding development infrastructure will evolve with these milestones:

* Add supported Python versions, including Python 3.13, to tests and wheels.
* Run formatting, static typing, documentation, and coverage checks in
  continuous integration.
* Make stochastic tests deterministic and add property-based invariant tests.
* Either test CuPy in continuous integration as a supported backend or narrow
  the public GPU-support claim until that testing exists.
* Publish benchmark results in a form that makes regressions visible without
  making noisy timing thresholds block unrelated contributions.

Longer-term opportunities
-------------------------

After the shared spline platform and multiscale contracts are established,
SplineOps can expand naturally into arbitrary affine matrices, displacement
fields, image warping, and spline-derived Jacobian and Hessian calculations.
These features should reuse the same coefficient, boundary, backend, and
evaluation machinery rather than introduce parallel implementations.

