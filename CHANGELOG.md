# Changelog

All notable changes to SplineOps are documented here.

## Unreleased

## 2.2.0 - 2026-07-17

### Documentation

- Added an installable `splineops-selma3d-demo` napari comparison with a
  checksum-pinned single-patch workflow, grid-aware expert masks, local versus
  frozen evidence, and an explicit post-study presentation-sample disclosure.
- Added a canonical claims registry separating the supported SELMA3D
  quality-at-speed statement, internal acceleration evidence, controlled-field
  results, and explicit nonclaims.
- Reorganized the documentation header around visible Getting started,
  Modules, Examples, Evidence, Project, and API destinations so module and
  gallery navigation no longer falls into the generic `More` menu.
- Split the interactive demo dependencies into a lightweight default extra and
  a `selma3d-demo-all` extra for the optional PyTorch-area comparison.
- Added a checksum-pinned, reproducible 3-D microscopy downsampling study using
  the public CC0 ``cells3d`` volume, with cross-library timings, explicitly
  limited real-data proxies, and a separate known-target calibration.
- Added a frozen, embryo-level BBBC050 segmentation validation with manual
  ground truth, leave-one-embryo-out threshold fitting, an acquisition-shifted
  external test, and an explicit report that the superiority criterion failed.
- Added a frozen 72-case controlled 3-D spectral-coarsening validation. Its
  generic-resize NRMSE criterion passed against six alternatives, while a
  post-hoc SciPy polyphase FIR audit decisively rejected broader scientific-
  resampling superiority. Both results and runtime trade-offs are published.
- Added checksum-pinned SELMA3D nuclei and microvessel confirmations. The
  nuclei study retained its failed 10x speed decision. The later 18-patch
  anisotropic microvessel study passed a predeclared family-wise quality rule
  against named SciPy, scikit-image, and PyTorch pipelines and its frozen local
  speed thresholds, with explicit limits on segmentation, specimen-level,
  licensing, and broad-resampling claims.
- Added a frozen 62-volume MiniVess 8x vessel-overview confirmation with
  method-native label grids and paired family-wise inference. All five quality
  margins passed, but the joint quality-at-speed claim failed because the
  PyTorch and SciPy-polyphase speed thresholds did not pass. The complete
  negative decision, per-volume scores, timings, and source checksums are
  retained.

### Tests

- Made the mixed-dtype workspace-cache concurrency test assert the documented
  retention cap rather than a scheduler-dependent exact cache fill.

## 2.1.0 - 2026-07-16

Version 2.1 makes the stable resize path easier to evaluate and adopt while
keeping the broader research modules explicit about their maturity.

### Documentation

- Reworked the README and documentation home page around one clear use case:
  projection-based antialiased resizing of N-D scientific arrays.
- Added a short method-selection quickstart and a practical 3-D volume
  downsampling tutorial with explicit spatial-axis examples.
- Added a public maturity table, an evidence scorecard, and clearer limits on
  performance and accuracy claims.
- Updated the interpolation gallery example to follow ``TensorSpline``'s
  uniform-grid and matching-precision construction contract.
- Moved internal audits and execution records behind the user guides in the
  navigation without removing the underlying evidence.

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
  and peak memory.  Geometry templates can now refit new data or accept
  explicitly precomputed coefficients without hidden mutation.
  NumPy point-support evaluation now contracts gathered coefficients and
  per-axis weights directly, avoiding a broadcast weight product; the existing
  CuPy path is unchanged pending dedicated GPU evidence.
- Replaced affine full-volume coordinate meshgrids with tiled pull-back
  evaluation, made integer promotion and degree validation explicit, and added
  matched SciPy comparisons across degrees 0--5.  Added a general pull-back
  `affine_transform`, reusable memory-capped `AffinePlan`, output buffers, and
  explicit spatial axes for independent batch and channel transforms.  Affine
  plans can prefilter once and share one coefficient field across compatible
  transform geometries, with retained-memory and configuration introspection.
  Added immutable `AffineCoefficientField` tags that reject reuse across an
  incompatible input grid, degree, boundary mode, or precision while allowing
  different matrices and output shapes. Batched coefficient filtering and
  support evaluation now avoid per-slice plan dispatch. Tagged fields support
  an atomic validated JSON-plus-numeric NPZ round trip and concurrent read-only
  reuse by threads or independently loading processes. Schema-2 values use a
  portable byte order, schema-1 archives remain readable, and direct untagged
  construction is rejected.
- Made differential operations return raw results without replacing the source
  image, removed implicit normalization and console output, and vectorized
  row/column spline prefiltering. Added physical spacing, standard increasing-
  coordinate directions, direct gradient/Hessian components, 3-D support, and
  analytical polynomial and trigonometric tests.  `DifferentialPlan` computes
  requested gradient, packed Hessian, and Laplacian outputs through one cached
  workspace and now accepts explicit batch/channel spatial axes.
  Explicit-axis plans now execute all slices through one batched workspace
  instead of constructing one workspace per slice. Gradient, Hessian, and
  Laplacian families can be selected independently; Laplacian-only requests
  skip mixed Hessians, and exact structured output buffers are supported.
  Buffers are now prevalidated for shape, dtype, writability, and non-overlap
  before computation so invalid requests cannot partially write outputs.
- Corrected adaptive-regression amplitude sparsification, eliminated caller
  mutation, added opt-in convergence diagnostics, removed a duplicate smoothing
  implementation, and tightened research-module parameter validation.  Added a
  fixed-geometry `DenoisingPlan`, prefix-sum linear-spline evaluation, and a
  real-FFT `SmoothingSplinePlan` with a reusable half-spectrum response.  Added
  stateless warm-start lambda paths, an independent constrained-optimizer
  check, and explicit smoothing axes for batched arrays.
- Defined reversible wavelet shape requirements, singleton/odd pyramid
  behavior, and rectangular reconstruction audits. Haar and cubic spline
  transforms meet tight reconstruction bounds; order 5 is explicitly documented
  as approximate because the inherited taps have limited precision.  Pyramid,
  Haar, and spline-wavelet axis passes now operate on whole arrays instead of
  dispatching one Python call per row or column, with explicit spatial axes for
  batch and channel arrays. Explicit-axis wavelets now process independent
  planes in cache-bounded vectorized groups, and Haar split/merge writes into
  preallocated destinations to reduce temporary arrays. Large cache-filling
  planes now avoid whole-array transpose copies, removing the measured
  large-plane batch regression while preserving small-plane vectorization.
  The inherited order-5 tap provenance was rechecked against DeconvolutionLab2;
  it now exposes an explicit bounded-approximation reconstruction contract.

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
- Added complete-workflow benchmarks, stored machine-relative regression
  thresholds, plan lifecycle recipes, and a manual Linux/macOS/Windows resize
  benchmark matrix. These are evidence gates, not portable performance
  promises.
- Added targeted explicit-axis affine, differential, and wavelet workflow
  measurements and regression floors. CI failure annotations now select the
  pytest failure section, and cross-platform FFT equivalence uses a tight
  floating tolerance rather than requiring bitwise identity.
- Added explicit batch-count and spatial-size runtime/memory sweeps with
  machine-readable growth and equivalence summaries. The development benchmark
  now runs publication-branch smoke evidence, or a manually selected profile,
  on Linux, macOS, and Windows and publishes per-runner artifacts. These remain
  evidence and soak gates, not portable performance promises.
- Added persisted registration and buffered 3-D feature soak workloads,
  independent Keys/O-MOMS/fractional-smoothing references, and an instrumented
  affine phase profile. A `[standard-bench]` commit deliberately requests the
  standard Linux/macOS/Windows evidence matrix; ordinary relevant pushes remain
  smoke runs.
- Added randomized affine, differential, TensorSpline-geometry, and exact
  wavelet reconstruction contracts; immutable schema-1/schema-2 affine archive
  specimens; and broader independent 2-D Fourier and sparse-hinge references.
  A repeated correctness soak now covers atomic replacement, thread sharing,
  fresh spawned-process restoration, corrupt copies, and caller-buffer reuse
  across smoke/standard/extended profiles with cross-platform JSON artifacts.
- Published the exact backend boundary and separate Affine/Differentials
  graduation audits. CuPy remains experimental TensorSpline interoperability,
  including a documented CPU-transfer path, and no additional module is
  promoted by this release.
- Added a dated, consolidated progress snapshot covering all seven improvement
  passes, current module maturity, local and cross-platform evidence, measured
  performance conclusions, deliberate non-decisions, and the remaining
  practical gates.

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
