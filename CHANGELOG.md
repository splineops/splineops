# Changelog

All notable changes to SplineOps are documented here.

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
