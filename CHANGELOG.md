# Changelog

All notable changes to SplineOps are documented here.

## Unreleased

### Contracts and correctness

- Added stable top-level imports for `TensorSpline`, `ResizePlan`, `resize`,
  and `resize_degrees` while keeping their public modules independent.
- Defined uniform construction-grid, arbitrary evaluation-coordinate, batch,
  dtype, finiteness, backend, singleton, short-periodic, and output-shape
  contracts for `TensorSpline`.
- Added bounded-memory tiled `TensorSpline` evaluation with a direct fast path
  for small grids, plus a reproducible runtime/peak-memory benchmark.
- Corrected N-D coefficient prefiltering for non-contiguous intermediate axes;
  B-spline degrees 0--5 now have SciPy parity coverage through four dimensions.
- Added experimental, memory-capped `TensorSplineQueryPlan` support for repeated
  fixed-coordinate evaluation, with measured construction break-even evidence.
  Query plans now represent geometry independently of one data array, can be
  reused across compatible `TensorSpline` instances, and support exact output
  buffers.  Separable tensor grids use axis-wise contraction to reduce runtime
  and peak memory.
- Replaced affine full-volume coordinate meshgrids with tiled pull-back
  evaluation, made integer promotion and degree validation explicit, and added
  matched SciPy comparisons across degrees 0--5.  Added a general pull-back
  `affine_transform`, reusable memory-capped `AffinePlan`, output buffers, and
  explicit spatial axes for independent batch and channel transforms.
- Made differential operations return raw results without replacing the source
  image, removed implicit normalization and console output, and vectorized
  row/column spline prefiltering. Added physical spacing, standard increasing-
  coordinate directions, direct gradient/Hessian components, 3-D support, and
  analytical polynomial and trigonometric tests.  `DifferentialPlan` computes
  requested gradient, packed Hessian, and Laplacian outputs through one cached
  workspace.
- Corrected adaptive-regression amplitude sparsification, eliminated caller
  mutation, added opt-in convergence diagnostics, removed a duplicate smoothing
  implementation, and tightened research-module parameter validation.  Added a
  fixed-geometry `DenoisingPlan`, prefix-sum linear-spline evaluation, and a
  real-FFT `SmoothingSplinePlan` with a reusable half-spectrum response.
- Defined reversible wavelet shape requirements, singleton/odd pyramid
  behavior, and rectangular reconstruction audits. Haar and cubic spline
  transforms meet tight reconstruction bounds; order 5 is explicitly documented
  as approximate because the inherited taps have limited precision.  Pyramid,
  Haar, and spline-wavelet axis passes now operate on whole arrays instead of
  dispatching one Python call per row or column.

### Project maturity

- Published module maturity, provenance, internal compatibility, performance,
  and execution-roadmap documentation.
- Narrowed project-wide GPU and universal speed claims; CuPy remains an
  experimental TensorSpline interoperability path.
- Added Python 3.13 to the declared CI and wheel matrix, strict documentation,
  formatting, scoped static typing, coverage, package-build, and clean-wheel
  quality gates.
- Added machine-readable TensorSpline memory/query-plan, affine, and
  differentials benchmark paths, plus a multiscale vectorization benchmark.
  Equivalent affine comparisons report the current SciPy performance advantage
  instead of implying a SplineOps win.
- Re-profiled the native resize scheduler after the v2 numerical rewrite.  A
  measured small-3-D automatic participation cap avoids excessive default
  worker fan-out while preserving explicit `LSRESIZE_NUM_THREADS` overrides.

## 2.0.0 - 2026-07-14

Version 2.0 establishes one resize contract and replaces the numerically
fragile high-order projection implementation. Existing applications should
review the migration notes because output geometry is intentionally no longer
selected through a legacy compatibility switch.

### Breaking changes

- Removed the `inversable` argument from the public, Python fallback, and
  native resize APIs.
- Standardized output lengths from requested zooms as
  `max(1, floor(input_length * zoom + 0.5))`, followed by one
  endpoint-aligned sampling grid for the realized input and output lengths.
- Added explicit `axes` semantics: `axes=None` selects every axis, while
  `axes=()` selects none. Unselected batch or channel axes remain exact
  identity axes.
- Defined canonical behavior for singleton inputs and outputs, same-grid
  operations, output arrays, and real numeric dtypes. Invalid degrees,
  shapes, zooms, axes, and output buffers are rejected consistently.

### Numerical changes

- Replaced repeated high-order integration and differencing with compact
  direct cross-Gram projection for every public zero-shift projection whose
  analysis degree is at least one.
- Retained the finite-difference realization for analysis degree zero.
- Stabilized equal-degree least-squares projection on long signals and added
  high-precision fixtures covering all supported direct-projection degree
  triples.
- Made whole-sample symmetric boundary mapping exact over multiple mirror
  periods and removed the hidden output tail from public zero-shift
  projection.

### Performance and resource usage

- Added immutable, thread-safe `ResizePlan` execution with reusable native
  plans and per-invocation workspace leases.
- Added direct output for compatible arrays through `apply_into` and the
  native `resize_nd_into` path.
- Added a persistent native line scheduler with bounded participation,
  exception propagation, nested-call handling, and fork recovery.
- Bounded process-wide plan caches by both entry count and retained bytes;
  bounded reusable-plan workspace retention independently.
- Reduced plan metadata and profiling-registry overhead, including compact
  sign storage and a fixed atomic profiling registry.

### Migration

Resize spatial axes explicitly when arrays include batch or channel
dimensions:

```python
from splineops import resize

small = resize(
    image_batch,
    output_size=(256, 256),
    axes=(-3, -2),
    method="cubic-antialiasing",
)
```

Remove `inversable=` from existing calls. If an application depended on the
previous size or sample-placement policy, choose the desired integer
`output_size` explicitly and validate the new endpoint-aligned result.

For repeated same-shape workloads, construct `splineops.resize.ResizePlan`
once and reuse it. This avoids repeated direct-projection plan construction
and enables allocation-free final output when a compatible array is supplied.

## 1.3.0 - 2026-06-19

- Previous stable release. See the Git history and the `v1.3.0` tag for its
  complete contents.
