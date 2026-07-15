Development roadmap
===================

Execution status (2026-07-15)
-----------------------------

Six repository-wide implementation passes are complete.  The first established
positioning, provenance, numerical contracts, bounded ``TensorSpline``
evaluation, conservative internal sharing, reproducible benchmarks, and the
packaging and quality workflows.  The second added reusable geometry plans,
separable tensor-grid contraction, a general affine API with batch/channel
axes, 3-D and multi-output differentials, cached smoothing and denoising plans,
vectorized multiscale axis passes, and a measured native resize scheduler
policy.  The third added explicit coefficient reuse across compatible geometry,
batch/channel axes for smoothing, planned differentials, pyramids, and
wavelets, controlled denoising paths, plan inspection, constrained numerical
references, end-to-end workflow benchmarks, stored regression thresholds, and
a manual cross-platform resize benchmark matrix.  These capabilities remain in
their independent public modules; this work did not fold ``TensorSpline`` or
the research tools into resize.

The fourth pass removed higher-level per-slice dispatch from affine and
differential plans, added cache-bounded batch groups for wavelet transforms,
and introduced immutable compatibility tags for reusable affine coefficients.
Complete-workflow benchmarks now measure those explicit-axis paths directly
and enforce machine-relative regression floors.  The macOS FFT test uses a
tight numerical tolerance, and CI annotations preserve the actual pytest
failure section when a matrix job fails.

The fifth pass profiles and directly contracts NumPy point supports, adapts
wavelet batching to plane working-set size, adds selective and buffered
differential outputs, and hardens affine coefficient persistence and concurrent
reuse.  A batch-size/spatial-size memory sweep now accompanies the complete
workflow benchmark, and the development benchmark runs smoke evidence for
relevant publication-branch changes or a manually selected profile as a
Linux/macOS/Windows matrix.  These additions strengthen evidence without
merging the independent public modules.

The sixth pass adds two realistic stability-soak pipelines, prevalidates every
differential destination before numerical work, and makes affine coefficient
persistence atomic, endian-portable, schema-compatible, and process-tested.
Independent published-formula references now cover Keys, cubic O-MOMS, and the
fractional smoothing transfer function.  The inherited order-5 wavelet table
was verified against its DeconvolutionLab2 source and remains explicitly
approximate rather than receiving invented taps.  Instrumented affine phases
justify a narrower 3-D NumPy contraction change, while the public modules stay
separate.

The detailed work packages below remain the long-term graduation criteria,
not a claim that every experimental module is now stable.  In particular,
implementation completeness and API stability are different promises.

One evidence gate intentionally remains open: GPU-backed CuPy CI.  The
Linux/macOS/Windows development smoke matrix passed for commit ``32b9d8c`` and
the `standard matrix 29429054363
<https://github.com/splineops/splineops/actions/runs/29429054363>`_ passed and was
reviewed during the sixth pass.  Standard results
confirmed numerical portability but not universal explicit-axis speedups, so
the policy and performance claims were narrowed instead of tuning thresholds to
one machine.  The manual resize and development matrices provide deeper
evidence when requested.
The maintainer
cleared the current distribution's provenance on
2026-07-15.  Higher-order research modules remain experimental where a
published numerical reference or exact reconstruction property is incomplete;
the order-5 spline wavelet is one explicit example.  These limits are recorded in
:doc:`project-status` and :doc:`provenance`.

The stabilization pass is intentionally unreleased.  The next decision point
is evidence review rather than another broad implementation wave: review the
standard platform profile and let the two checked-in workloads exercise
coefficient persistence, output selection, and explicit-axis APIs through an
API-soak period.  Promote modules only when their individual graduation gates
are met.  A release is explicitly not required at this stage.

Purpose
-------

SplineOps will become a trustworthy collection of independent spline-based
tools for data sampled on regular Cartesian grids.  Its strongest current
component is projection-based N-D resizing.  The next development cycles will
make ``TensorSpline`` equally dependable, establish consistent numerical
contracts across the package, and then improve the other modules without
erasing their separate identities.

The intended product position is deliberately narrow:

   Precise N-D spline interpolation and resampling, with specialized tools for
   geometric transforms, differentials, smoothing, regression, and multiscale
   analysis.

This is an execution roadmap rather than a list of aspirations.  Work packages
are ordered by dependency, have explicit completion gates, and identify
conditions under which work must stop or change direction.

Module boundaries
-----------------

Public modules remain independent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TensorSpline`` is not a wrapper around resize, and resize is not implemented
by constructing a ``TensorSpline``.  Each public module keeps a purpose-specific
API and may keep a specialized implementation when its workload or mathematics
requires one.

The intended public structure is:

.. code-block:: text

   splineops
   |-- TensorSpline             continuous tensor-product spline model
   |-- resize / ResizePlan      regular-grid scaling and projection
   |-- affine                   geometric coordinate transforms
   |-- differentials           spline-derived image derivatives
   |-- smoothing_splines       regularized signal and image smoothing
   |-- adaptive_regression     sparse piecewise-linear regression
   `-- multiscale              pyramids and wavelet transforms

Internal sharing is allowed only below those APIs:

.. code-block:: text

   internal numerical components
   |-- basis definitions and derivative evaluations
   |-- coefficient prefilters
   |-- boundary and index mapping
   |-- dtype and precision policy
   |-- array-backend dispatch
   `-- bounded-memory evaluation utilities

Sharing is justified only when two modules have the same mathematical
contract.  Similar-looking code with different coordinate, projection, or
boundary semantics must remain separate and be documented as such.

Stability levels
~~~~~~~~~~~~~~~~

The package will communicate module maturity explicitly:

* **Stable:** resize and ``ResizePlan`` after the 2.0 contract.
* **Stabilizing:** ``TensorSpline`` and its basis and extension-mode APIs.
* **Experimental:** affine, differentials, smoothing splines, adaptive
  regression splines, pyramids, and wavelets until their graduation gates are
  met.

An experimental label does not imply removal.  It means that correctness,
edge-case behavior, public imports, or performance contracts are not complete.

Guiding principles
------------------

* Preserve independent, comprehensible public modules.
* Prefer explicit mathematical contracts over historical compatibility.
* Test invariants such as sample reproduction, constant preservation, and
  perfect reconstruction rather than relying only on saved output arrays.
* Maintain a readable NumPy reference path for numerical work that also has an
  accelerated implementation.
* Optimize complete workloads, including construction, coefficient creation,
  allocation, memory movement, and repeated evaluation.
* Require numerical equivalence or a documented accuracy policy for every
  accelerated path.
* Treat a backend as supported only when dispatch, tests, packaging, and
  continuous integration all cover it.
* Do not use native-versus-Python-fallback speedup as evidence of competitive
  advantage.  Compare against external libraries with equivalent semantics
  whenever possible.
* Do not port more legacy code until provenance, licensing, purpose, and a
  maintenance owner are known.

Execution protocol
------------------

The following procedure applies to every work package and is intended to make
the roadmap safe to execute incrementally.

1. Record the current behavior with focused tests and, for performance work,
   a saved benchmark artifact.
2. State the mathematical and API contract before changing an implementation.
3. Make the smallest coherent change that advances one work package.
4. Run focused tests while developing and the full suite before completing the
   package.
5. Compare accelerated results with the reference implementation across
   shapes, dtypes, boundaries, and degenerate cases.
6. Update user documentation, API documentation, examples, and changelog when
   observable behavior changes.
7. Save benchmark data as machine-readable CSV or JSON and generate summaries
   from those artifacts.
8. Remove unsuccessful experiments rather than leaving inactive code paths or
   environment switches behind.
9. Mark a work package complete only when every listed completion gate passes.

Work must pause for review when a proposed change:

* changes a stable public API or the resize sampling grid;
* changes default numerical results beyond documented tolerances;
* introduces a new required dependency or binary backend;
* cannot be explained from a cited method and maintained reference path;
* depends on legacy source whose redistribution or derivation rights are not
  clear; or
* improves a microbenchmark while making representative end-to-end workloads
  slower or substantially more memory-hungry.

Milestone 0: establish the trustworthy baseline
-----------------------------------------------

This milestone creates the evidence and project boundaries needed for all
later work.  It does not redesign numerical algorithms.

WP0.1: product and capability statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Rewrite the README and documentation landing page around precise N-D spline
  interpolation and resampling.
* Replace unqualified claims such as "ground-breaking", "optimized", and GPU
  support with specific, tested capabilities.
* Publish a module-status table with stable, stabilizing, and experimental
  labels.
* Document where SplineOps differs semantically from SciPy, OpenCV,
  scikit-image, and PyTorch.
* Provide one minimal example for ``resize`` and one for ``TensorSpline`` on
  the landing page.

Completion gates
^^^^^^^^^^^^^^^^

* A new user can identify the primary use cases and non-goals from the README.
* Every performance claim links to a reproducible benchmark definition.
* GPU wording is limited to the operations and modes tested in CI.

WP0.2: provenance and licensing inventory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Inventory each implementation derived from ``legacy_code``.
* Record original author, source, publication, original license notice, porting
  history, and current maintainer.
* Reconcile repository-level licensing with embedded research-only, GPL, or
  other notices before distributing derived implementations.
* Add acknowledgements and citations at module level where appropriate.
* Remove or quarantine code whose provenance cannot be resolved.

Completion gates
^^^^^^^^^^^^^^^^

* Every shipped legacy-derived module has a provenance record.
* The distribution contains no known contradictory licensing statements.
* New legacy ports have an explicit review checklist.

WP0.3: reproducible baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Record supported Python, NumPy, SciPy, compiler, and operating-system
  versions.
* Establish deterministic random seeds and eliminate order-dependent tests.
* Capture full test, build, documentation, and resize benchmark baselines.
* Add peak-memory measurement to the benchmark infrastructure.
* Separate exact or near-equivalent cross-library rows from contextual image
  resize comparisons.

Completion gates
^^^^^^^^^^^^^^^^

* One documented command reproduces each baseline suite.
* Benchmark artifacts include environment metadata, runtime, peak memory,
  output shape, dtype, and numerical differences.
* No timing threshold is used as a flaky unit-test assertion.

Milestone 1: stabilize TensorSpline without absorbing it
--------------------------------------------------------

``TensorSpline`` remains a standalone continuous-model class.  This milestone
defines what it accepts, what it computes, and how it behaves at boundaries.

WP1.1: public API and coordinate contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Provide a stable top-level import for ``TensorSpline`` while preserving the
  module import path.
* Decide whether construction coordinates are required; if retained, validate
  that they form a uniform grid within a documented tolerance.
* Explicitly reject nonuniform grids until a true nonuniform algorithm exists.
* Define scalar, point-list, meshgrid, tensor-product-grid, and batched query
  shapes.
* Define whether query coordinates must be sorted for each evaluation mode.
* Correct obsolete import paths in docstrings and API documentation.
* Document object immutability and whether source data or coefficients are
  copied or retained.

Completion gates
^^^^^^^^^^^^^^^^

* Accepted coordinate forms and output shapes are documented with examples.
* Invalid coordinate forms fail with stable, actionable exceptions.
* Public imports are covered by tests and documentation builds.

WP1.2: basis and boundary correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Add cardinal sample-reproduction tests for every advertised basis and mode.
* Cover zero, mirror, and periodic extension across multiple extension
  periods.
* Test signals shorter than basis support and periodic signals around which a
  basis wraps more than once.
* Cover singleton dimensions and clearly reject empty dimensions.
* Test real and complex coefficient paths where complex values are supported.
* Define tolerances by dtype, basis family, and operation instead of using one
  package-wide tolerance.
* Add analytical tests for constants, ramps, polynomials, impulses, and
  periodic trigonometric signals.

Completion gates
^^^^^^^^^^^^^^^^

* Every advertised basis/mode pair either reproduces samples within its
  documented tolerance or is removed from the advertised set.
* Short and singleton cases work or fail through documented validation.
* Periodic evaluation agrees across equivalent coordinate periods.

WP1.3: dtype and backend contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Define accepted input, coordinate, coefficient, accumulator, and output
  dtypes.
* Replace backend detection based on type-name strings with explicit dispatch.
* Prevent accidental host-device transfers in CuPy paths.
* Define which combinations of complex data and real coordinates are
  supported.
* Test NumPy first; retain CuPy support only for paths covered by dedicated CI.

Completion gates
^^^^^^^^^^^^^^^^

* Dtype behavior is predictable and tested for integer, float32, float64,
  complex64, and complex128 inputs as applicable.
* Unsupported array types fail explicitly rather than being misdetected.
* GPU support claims match CI coverage exactly.

WP1.4: correctness defects and maintenance cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Remove duplicate and shadowed function definitions.
* Replace console output in library functions with returned values, logging,
  warnings, or exceptions as appropriate.
* Remove silent parameter clamping.
* Fix mutation of caller-owned arrays unless mutation is explicitly part of an
  API.
* Add formatting, linting, static typing, and coverage reporting to CI in
  non-flaky configurations.

Completion gates
^^^^^^^^^^^^^^^^

* Static checks pass on the supported Python versions.
* Public functions do not print or silently rewrite parameters.
* Known defects identified in the initial audit have regression tests.

Milestone 2: define reusable internal numerical components
----------------------------------------------------------

This milestone reduces duplicated fragile machinery while preserving public
module independence.  Extraction proceeds one component at a time; it is not a
whole-package rewrite.

WP2.1: contract inventory before extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Compare basis definitions used by ``TensorSpline``, resize, affine, and
  differentials.
* Compare coordinate origins, endpoint rules, boundary rules, coefficient
  normalization, and derivative conventions.
* Identify components with genuinely identical contracts.
* Record deliberate differences that must not be unified.
* Design internal protocols that do not expose backend implementation details
  in public APIs.

Completion gates
^^^^^^^^^^^^^^^^

* A written compatibility matrix exists for every proposed shared component.
* No extraction begins where coordinate or boundary semantics remain
  ambiguous.

WP2.2: basis and boundary primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Centralize basis metadata such as support, poles, parity, and derivative
  evaluation where definitions are mathematically identical.
* Provide tested boundary-index mapping primitives for zero, mirror, and
  periodic behavior.
* Keep resize-specific anti-mirror or projection boundary behavior local when
  it does not match interpolation.
* Remove duplicated implementations only after old and new paths pass the same
  invariant suite.

Completion gates
^^^^^^^^^^^^^^^^

* Consumer modules retain their existing public APIs.
* Shared primitives have focused unit tests independent of any consumer.
* Numerical comparisons show no unexplained changes.

WP2.3: coefficient-prefilter interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Define an internal coefficient-construction interface with explicit basis,
  boundary, axis, dtype, and backend arguments.
* Retain specialized resize projection filters separately from cardinal
  interpolation prefilters.
* Support separable axis processing without unnecessary full-array copies.
* Establish a reference NumPy implementation before adding or reusing native
  kernels.

Completion gates
^^^^^^^^^^^^^^^^

* ``TensorSpline`` uses the interface without changing its public behavior.
* Resize uses shared code only where parity and performance both justify it.
* Reference/native comparisons cover short, long, contiguous, and strided
  axes.

WP2.4: bounded-memory evaluation utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Introduce internal iterators or plans that evaluate coordinates in bounded
  tiles.
* Make tile size configurable internally and choose a conservative default.
* Avoid materializing broadcast tensor-product weights for complete large
  outputs.
* Preserve deterministic accumulation order where precision requires it.
* Support caller-provided output arrays when safe.

Completion gates
^^^^^^^^^^^^^^^^

* Peak temporary memory is bounded by tile size rather than full query size.
* Tiled and untiled reference results agree within documented tolerances.
* Small workloads do not regress materially because of tiling overhead.

Milestone 3: make TensorSpline efficient as TensorSpline
--------------------------------------------------------

Optimization here is specific to continuous spline construction and arbitrary
coordinate evaluation.  It does not route through the resize API.

WP3.1: benchmark construction and evaluation separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark dimensions include:

* one-dimensional signals, two-dimensional images, and three-dimensional
  volumes;
* linear, cubic B-spline, higher-degree B-spline, and O-MOMS representatives;
* zero, mirror, and periodic modes;
* one-shot construction/evaluation and repeated evaluations;
* tensor-product grids, point lists, meshgrids, and batched coordinates; and
* runtime, peak memory, coefficient memory, and numerical error.

Completion gates
^^^^^^^^^^^^^^^^

* Reports distinguish coefficient construction from evaluation.
* Repeated-evaluation benchmarks do not reconstruct invariant coefficients.
* Comparisons with SciPy are restricted to semantically comparable cases.

WP3.2: tiled evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Apply the bounded-memory utilities to point-list and grid evaluation.
* Avoid stacking all broadcast weights when separable contractions or tiled
  gathers are possible.
* Choose different internal strategies for tensor-product and arbitrary-point
  queries when benchmarks justify them.
* Add output-buffer support without weakening normal allocation semantics.

Completion gates
^^^^^^^^^^^^^^^^

* Representative 3-D queries complete within a documented memory envelope.
* Runtime and peak memory improve on large workloads without changing results.
* Small query overhead remains acceptable.

WP3.3: reusable evaluation plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Separate reusable coordinate geometry from data-dependent coefficients.
* Cache support indexes and weights for repeated fixed-coordinate evaluation.
* Bound all process-level caches by entry count and retained bytes, or keep
  reuse explicit through a plan object.
* Define thread-safety, immutability, and fork behavior before exposing a plan
  publicly.

Completion gates
^^^^^^^^^^^^^^^^

* Repeated workloads avoid invariant setup work.
* Cache and workspace memory are bounded and tested.
* Plan and ordinary evaluation return equivalent results.

Milestone 4: improve independent consumer modules
-------------------------------------------------

These modules consume tested internal primitives where appropriate, but remain
separate public tools.

WP4.1: affine transforms
~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Define geometry, center, axis, output shape, coordinate direction, boundary,
  and dtype contracts for rotation.
* Replace full-volume meshgrid construction with tiled coordinate generation.
* Preserve ``rotate`` as a simple public API.
* Add a general affine-matrix API only after rotation is stable.
* Compare against analytical functions and semantically configured SciPy
  transforms.

Completion gates
^^^^^^^^^^^^^^^^

* Affine transforms do not allocate complete stacked coordinate volumes.
* Integer input and output dtype behavior is explicit.
* Two-dimensional and three-dimensional analytical accuracy tests pass.

WP4.2: differentials
~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Define whether outputs are raw physical derivatives or visualization-ready
  normalized images; stable numerical APIs must return raw derivatives.
* Preserve input arrays and return results rather than mutating object state.
* Express gradients, Hessians, Laplacians, and derivative bases through tested
  coefficient and basis primitives where contracts match.
* Remove Python row and column loops after a reference implementation is
  established.
* Add pixel-spacing parameters before claiming physical derivative units.

Completion gates
^^^^^^^^^^^^^^^^

* Polynomial and trigonometric analytical derivative tests pass.
* No numerical operation performs implicit visualization normalization.
* Runtime and memory benchmarks cover 2-D images and 3-D volumes if supported.

WP4.3: resize maintenance and targeted profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resize remains specialized.  The v2 direct cross-Gram projection pipeline must
be profiled afresh; June 2026 finite-difference projection profiles are
historical for the replaced paths.

Tasks
^^^^^

* Re-run native phase profiles and cross-library benchmarks on the current v2
  implementation.
* Add hardware-counter evidence where the execution environment permits it.
* Validate a narrow serial-execution gate for small 3-D workloads.
* Consider cubic gather or layout changes only when profiles identify a
  specific end-to-end bottleneck.
* Track runtime and peak memory regressions without imposing noisy CI timing
  failures.

Completion gates
^^^^^^^^^^^^^^^^

* Current reports no longer mix superseded and current projection algorithms.
* Scheduler changes win across the stated workload class on more than one
  hardware configuration.
* Generic 2-D image-resize speed is not used as the primary product claim.

Milestone 5: assess the remaining research modules
--------------------------------------------------

Each module graduates independently.  Sharing a repository does not require a
shared release maturity level.

WP5.1: smoothing splines
~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Remove duplicate definitions and distinguish exact spline methods from
  Butterworth-like approximations.
* Validate parameter domains, empty and singleton signals, and constant
  preservation.
* Compare recursive and FFT formulations where the mathematics predicts
  equivalence.
* Document boundary assumptions and output sampling grids.

Graduation gate
^^^^^^^^^^^^^^^

* Every advertised method is tied to a documented formulation and analytical
  or independent numerical reference.

WP5.2: adaptive regression splines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Fix amplitude sparsification and caller-array mutation.
* Validate sorted, unique sample coordinates and degenerate inputs.
* Test knot recovery on synthetic piecewise-linear signals.
* Benchmark against an appropriate reference only after correctness is
  established.

Graduation gate
^^^^^^^^^^^^^^^

* Recovered functions satisfy interpolation or optimization invariants across
  deterministic synthetic cases.

WP5.3: pyramids and wavelets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Define supported shapes, scales, filters, normalization, and boundary rules.
* Add randomized perfect-reconstruction tests across even, odd, singleton,
  rectangular, and small dimensions.
* Audit higher-order spline-wavelet reconstruction accuracy.
* Reject unsupported shapes clearly until reversible behavior exists.
* Optimize row and column processing only after reconstruction contracts pass.

Graduation gate
^^^^^^^^^^^^^^^

* Perfect reconstruction holds within dtype-specific tolerance for every
  advertised filter, scale, and shape class.

Milestone 6: backend, packaging, and release maturity
-----------------------------------------------------

WP6.1: continuous integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Test every supported Python version, including Python 3.13 when dependencies
  and wheels permit it.
* Run unit tests, native-required tests, formatting, static typing,
  documentation, package builds, and clean-wheel smoke tests.
* Add coverage reporting to reveal untested modules without treating a single
  percentage as proof of correctness.
* Add CuPy CI only when a reliable GPU runner and supported dependency matrix
  exist.

Completion gates
^^^^^^^^^^^^^^^^

* Supported platforms install wheels that include and exercise the native
  extension.
* Documentation examples use valid public imports.
* The advertised backend matrix is generated from tested configurations.

WP6.2: benchmark publication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks
^^^^^

* Generate Markdown reports from versioned benchmark schemas.
* Publish exact-semantics comparisons separately from contextual comparisons.
* Include hardware, threads, library versions, coordinates, boundaries,
  antialiasing semantics, runtime, peak memory, and numerical differences.
* Preserve historical reports but label superseded algorithms and results.

Completion gates
^^^^^^^^^^^^^^^^

* Every headline performance statement can be regenerated from an artifact.
* Benchmark reports state both wins and losses.
* Historical reports cannot be mistaken for the current implementation.

WP6.3: stable-module graduation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A module is marked stable only when it has:

* a clear purpose distinct from neighboring modules;
* a documented mathematical, shape, dtype, and boundary contract;
* stable public imports and examples;
* invariant and edge-case tests;
* a reference or independently validated implementation;
* runtime and memory benchmarks appropriate to its workload; and
* an identified maintenance and provenance record.

Modules that do not meet these gates remain experimental.  Modules without a
clear purpose or maintainable contract may move to a separate research archive
rather than delaying stable modules.

Release sequence
----------------

2.0.x: trustworthy surface
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Complete Milestone 0.
* Stabilize ``TensorSpline`` contracts and public imports from Milestone 1.
* Fix clear correctness and documentation defects.
* Narrow unsupported GPU and performance claims.
* Avoid intentional resize output changes.

2.1: dependable TensorSpline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Complete the ``TensorSpline`` correctness, dtype, and backend gates.
* Introduce only the internal primitives proven compatible in Milestone 2.
* Deliver bounded-memory ``TensorSpline`` evaluation and end-to-end benchmarks.
* Preserve ``TensorSpline`` as its own public class and conceptual model.

2.2: geometric and differential tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Deliver tiled affine evaluation.
* Rebuild differentials around raw numerical outputs and analytical tests.
* Re-profile current resize and accept only evidence-backed targeted changes.

2.3: research-module graduation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Graduate smoothing, adaptive regression, and multiscale modules
  independently as their contracts pass.
* Keep incomplete modules explicitly experimental.

3.0: coherent stable API
~~~~~~~~~~~~~~~~~~~~~~~~

Version 3.0 is warranted only if stable-module graduation requires deliberate
public API cleanup.  It is not a deadline for declaring every module stable.
The main API should contain only operations whose contracts the project is
prepared to maintain.

Dependency order
----------------

The default execution order is:

.. code-block:: text

   product/provenance/baselines
               |
               v
      TensorSpline contracts
               |
               v
   compatible internal primitives
          /           \
         v             v
   tiled TensorSpline  backend policy
         |             |
         +------.------+
                v
       affine and differentials
                |
                v
   independent research-module gates
                |
                v
       stable-module graduation

Resize profiling can proceed alongside ``TensorSpline`` work once the baseline
schema exists, but speculative resize kernel changes do not take priority over
correctness and bounded-memory evaluation.

Autonomous work queue
---------------------

Unless a completion gate or pause condition requires a decision, implementation
should take the first unfinished item in this queue:

#. Publish the capability and module-status table.
#. Create the legacy provenance inventory.
#. Capture deterministic test, build, documentation, runtime, and memory
   baselines.
#. Fix obsolete imports, duplicate definitions, console output, silent
   clamping, and unintended mutation with regression tests.
#. Specify and test the ``TensorSpline`` coordinate and output-shape contract.
#. Complete basis/mode reproduction and degenerate-input tests.
#. Define dtype and explicit backend dispatch.
#. Produce the internal contract compatibility matrix.
#. Extract boundary and basis primitives that pass the compatibility gate.
#. Introduce the coefficient-prefilter interface for cardinal interpolation.
#. Benchmark ``TensorSpline`` construction and evaluation separately.
#. Implement bounded-memory tiled ``TensorSpline`` evaluation.
#. Evaluate explicit reusable coordinate plans.
#. Tile affine coordinate generation and stabilize affine dtype behavior.
#. Rework differentials to return raw, non-mutating numerical results.
#. Re-profile current v2 resize and test the narrow 3-D scheduling hypothesis.
#. Audit and graduate smoothing splines.
#. Audit and graduate adaptive regression splines.
#. Establish multiscale perfect-reconstruction contracts.
#. Complete packaging, backend, and benchmark-publication gates.
#. Review which modules qualify for a coherent stable API.

Measures of success
-------------------

The roadmap succeeds when:

* users understand why they would choose SplineOps and where they should not;
* ``TensorSpline`` remains an independent, dependable continuous spline model;
* resize remains a specialized projection resampler with honest performance
  claims;
* large interpolation and affine workloads have bounded temporary memory;
* accelerated and reference paths agree under explicit numerical policies;
* experimental modules cannot be mistaken for stable ones;
* every shipped legacy-derived implementation has clear provenance; and
* performance reports expose memory, accuracy, semantic differences, wins,
  and losses rather than a single favorable speedup number.
