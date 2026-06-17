# Resize PR Readiness

Date: 2026-06-17

This note is the compact upstream-PR checklist for the resize optimization
work. The detailed engineering ledger remains in
`scripts/resize_optimization_notes.md`; the shorter project summary remains in
`scripts/resize_optimization_progress.md`. The PR-facing short brief is
`scripts/resize_oblique_pr_brief.md`, and the ready-to-paste PR draft is
`scripts/resize_upstream_pr_description.md`. The close-out session handoff is
`scripts/resize_session_handoff.md`.

## Objective

Prepare the optimized resize backend for a focused upstream contribution that
keeps the Muñoz/Blu/Unser spline projection framework intact by default while
reducing runtime, memory movement, and repeated setup cost.

The PR narrative should be narrow:

- same public resize semantics by default
- same separable spline projection/interpolation model
- oblique antialiasing presets as the practical default downsampling methods
- faster native CPU implementation for common 2-D and 3-D workloads
- explicit benchmark and correctness artifacts
- experimental or opt-in numerical shortcuts kept out of the default path

## Method Boundary

The exact default path keeps the method-level contract:

- finite-difference integration for projection/antialiasing paths
- spline prefiltering with the existing pole recursion and boundary rules
- mirror/anti-mirror extension behavior matching the Python fallback
- separable axis passes with the same coordinate mapping and row plans
- native/Python parity as the primary correctness target

Optimizations should be described as implementation changes around that
projection framework: plan reuse, batched axis execution, cache-friendly
gather/scatter, fixed-support specialization, direct linear kernels, and
conservative dispatch policy.

The production/downsampling methods are the oblique antialiasing presets:

- `linear-antialiasing`: `(interp=1, analy=0, synthe=1)`
- `quadratic-antialiasing`: `(interp=2, analy=1, synthe=2)`
- `cubic-antialiasing`: `(interp=3, analy=1, synthe=3)`

Equal-degree least-squares configurations, for example
`(interp=3, analy=3, synthe=3)`, remain useful as advanced/reference
configurations through `resize_degrees`, but they should not be presented as
the recommended default. Cubic equal-degree least-squares needs fourth-order
integration in this finite-difference framework; that is slower and can amplify
roundoff on long lines. The oblique presets keep the same synthesis model with
a lower-degree analysis space, which makes the prefilter shorter, faster, and
more numerically robust while staying close to the ideal least-squares result
in practice.

Do not present OpenCV, skimage, or PyTorch image-resize rows as exact algorithm
equivalents. They are useful context, but their coordinate mappings, boundary
rules, kernels, or antialiasing differ from splineops.

## Accepted Optimization Targets

These are the strongest PR candidates because they preserve default semantics
and have measured wins:

- Native batched N-D axis pipeline, especially the oblique antialiasing presets.
- Automatic batched projection routing for 3-D oblique antialiasing presets;
  this is now the primary native optimization target for volumetric
  downsampling.
- Cached immutable 1-D/axis plans and reusable `ResizePlan` execution.
- Row-major batched gather and exact boundary source/sign mapping.
- Strided-offset gather for routed pure interpolation cases.
- Gather-prefilter normalization fusion for supported exact paths.
- Projection output-prefilter scale fusion only under the conservative
  explicit-single-thread policy.
- Fixed-support accumulator specializations for common presets.
- Exact direct linear interpolation kernels.
- Fused exact 2-D and selected 3-D linear paths.
- AVX2/FMA dispatch for selected 2-D linear rows on supported x86 builds.
- Workload-aware thread and batch policies that avoid expensive worker startup
  for cheap passes.
- Auto float32 internals only for pure `float32` interpolation rows where the
  default numerical contract remains acceptable; projection and antialiasing
  float32 internals remain opt-in.

## Current Optimization Map

Fresh full bundle after the 3-D oblique batching pass:

- Report:
  `/tmp/splineops_resize_pr_oblique_batched_20260617/resize_pr_report_oblique_batched_20260617.md`
- Native/Python:
  `/tmp/splineops_resize_pr_oblique_batched_20260617/resize_native_full_oblique_batched_20260617.csv`
- Libraries:
  `/tmp/splineops_resize_pr_oblique_batched_20260617/resize_libraries_full_oblique_batched_20260617.csv`
- Plan:
  `/tmp/splineops_resize_pr_oblique_batched_20260617/resize_plan_standard_oblique_batched_20260617.csv`
- Projection methods:
  `/tmp/splineops_resize_pr_oblique_batched_20260617/resize_projection_methods_standard_oblique_batched_20260617.csv`

Benchmark-derived priorities:

1. Keep oblique antialiasing as the headline resize method. The projection
   method sweep has oblique faster in `36/36` degree-1 rows and `36/36`
   degree-3 rows, with median oblique speedups of `1.24x` and `1.42x`
   respectively versus equal-degree least-squares.
2. Treat 3-D oblique as the main native hotspot. Automatic batched projection
   routing raises the full-report 3-D oblique native/Python bucket to
   `3` cases, median `12.61x`, mean `11.53x`.
3. Treat `ResizePlan` reuse as a secondary ergonomic win rather than the main
   performance claim. The current report shows modest medians:
   `1.026x` for plan reuse and `1.030x` with caller-owned output.
4. Keep external library rows split by semantics. Exact-ish SciPy rows remain
   slower in `0/16` cases, while OpenCV/Torch can be faster on non-equivalent
   image-resize semantics.

## Rejected Or Deferred Ideas

These should not be in the initial upstream PR unless new evidence changes the
tradeoff:

- Compact composite projection operator as a replacement for the existing
  separable projection path. It was mathematically interesting but too risky and
  not a clear implementation win.
- Default float32 internals for projection or antialiasing. The speed is useful,
  but random-output drift versus the conservative path is still measurable.
- Making equal-degree least-squares projection the public default for
  antialiasing. It is theoretically optimal for the orthogonal projection
  criterion, but the oblique presets are the better production tradeoff for
  speed, robustness, and visual downsampling behavior.
- Fused `nb == 2` projection input integration. It was exact in focused checks
  but mixed or negative in timing, so the code was removed and the rejection was
  documented.
- Broad projection strided-gather routing. Prior experiments did not show a
  stable enough end-to-end gain.
- More aggressive blocked scatter, tiled gather, or run-length fixed-window
  rewrites. These remain possible future work, but they increase complexity and
  need a clearer cache/memory model before being PR material.

## Benchmark Artifact Set

Use the wrapper when preparing a PR evidence bundle:

```shell
python scripts/benchmark_resize_pr.py \
  --profile oblique-pr \
  --output-dir /tmp/splineops_resize_pr_oblique_pr
```

For a quick wrapper smoke test:

```shell
python scripts/benchmark_resize_pr.py \
  --profile smoke \
  --output-dir /tmp/splineops_resize_pr_smoke
```

The wrapper emits:

- native versus Python timing CSV/JSON
- splineops versus external-library timing and delta CSV/JSON
- repeated `ResizePlan` timing CSV/JSON
- equal-degree least-squares versus oblique projection timing/quality CSV/JSON
- combined Markdown report
- manifest and exact command list

For focused method-positioning evidence:

```shell
python scripts/benchmark_resize_projection_methods.py \
  --profile standard \
  --output-csv /tmp/splineops_projection_methods.csv
python scripts/benchmark_resize_projection_methods.py \
  --profile stability \
  --degrees 3 \
  --dtypes float64 \
  --output-csv /tmp/splineops_projection_methods_stability.csv
```

## Interpreting Results

Use these categories in PR discussion:

- Native versus Python fallback: proves speedup for the same splineops
  operation and is the most important exact comparison.
- Projection-method comparison: proves why oblique antialiasing is the
  production default even though equal-degree least-squares remains the
  orthogonal-projection reference.
- `ResizePlan`: proves repeated fixed-geometry workloads benefit from cached
  setup and reusable buffers.
- SciPy close-output rows: useful like-for-like context for many interpolation
  cases, but still verify per-row deltas.
- OpenCV, skimage, and PyTorch image-resize rows: contextual performance
  baselines, not exact correctness evidence.

Report relative-L2 deltas next to speed ratios. Rows with large deltas should
be discussed as different operations, even when they are faster.

## Validation Checklist

Before opening or updating an upstream PR:

```shell
python -m py_compile \
  scripts/benchmark_resize_pr.py \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_plan.py \
  scripts/benchmark_resize_projection_methods.py \
  scripts/summarize_resize_benchmarks.py
python -m pytest -q
git diff --check
```

Then generate a fresh artifact bundle with the wrapper and attach or summarize
the generated report.

## PR Shape

Recommended sequencing for a major-library PR:

1. Submit the smallest exact implementation slice that improves common rows.
2. Include targeted parity tests before benchmark claims.
3. Lead with oblique antialiasing benchmark rows, including 3-D coverage, when
   discussing projection/downsampling performance.
4. Include benchmark scripts or reproducible commands in the PR description.
5. Hide local experiment knobs from the public API, or keep them clearly
   internal and undocumented when they are only for A/B validation.
6. Keep approximate/numerically different modes opt-in and separate from the
   exact default path.

The best first PR is not the largest possible optimization bundle. It is the
smallest exact bundle with a strong correctness story, clear benchmark evidence,
and a credible maintenance cost.
