# Resize PR Readiness

Date: 2026-06-17

This note is the compact upstream-PR checklist for the resize optimization
work. The detailed engineering ledger remains in
`scripts/resize_optimization_notes.md`; the shorter project summary remains in
`scripts/resize_optimization_progress.md`.

## Objective

Prepare the optimized resize backend for a focused upstream contribution that
keeps Arrate's least-squares spline resize method intact by default while
reducing runtime, memory movement, and repeated setup cost.

The PR narrative should be narrow:

- same public resize semantics by default
- same separable spline projection/interpolation model
- faster native CPU implementation for common 2-D and 3-D workloads
- explicit benchmark and correctness artifacts
- experimental or opt-in numerical shortcuts kept out of the default path

## Algorithm Boundary

The exact default path keeps the method-level contract:

- finite-difference integration for projection/antialiasing paths
- spline prefiltering with the existing pole recursion and boundary rules
- mirror/anti-mirror extension behavior matching the Python fallback
- separable axis passes with the same coordinate mapping and row plans
- native/Python parity as the primary correctness target

Optimizations should be described as implementation changes around that method:
plan reuse, batched axis execution, cache-friendly gather/scatter, fixed-support
specialization, direct linear kernels, and conservative dispatch policy.

Do not present OpenCV, skimage, or PyTorch image-resize rows as exact algorithm
equivalents. They are useful context, but their coordinate mappings, boundary
rules, kernels, or antialiasing differ from splineops.

## Accepted Optimization Targets

These are the strongest PR candidates because they preserve default semantics
and have measured wins:

- Native batched N-D axis pipeline, including projection and antialiasing.
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

## Rejected Or Deferred Ideas

These should not be in the initial upstream PR unless new evidence changes the
tradeoff:

- Compact composite projection operator as a replacement for the existing
  separable projection path. It was mathematically interesting but too risky and
  not a clear implementation win.
- Default float32 internals for projection or antialiasing. The speed is useful,
  but random-output drift versus the conservative path is still measurable.
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
python scripts/benchmark_resize_pr.py --output-dir /tmp/splineops_resize_pr_<tag>
```

For a quick wrapper smoke test:

```shell
python scripts/benchmark_resize_pr.py \
  --native-profile smoke \
  --library-profile smoke \
  --plan-profile smoke \
  --native-repeats 1 \
  --library-repeats 1 \
  --plan-repeats 1 \
  --plan-frames 2 \
  --threads 1,default \
  --output-dir /tmp/splineops_resize_pr_smoke
```

The wrapper emits:

- native versus Python timing CSV/JSON
- splineops versus external-library timing and delta CSV/JSON
- repeated `ResizePlan` timing CSV/JSON
- combined Markdown report
- manifest and exact command list

## Interpreting Results

Use these categories in PR discussion:

- Native versus Python fallback: proves speedup for the same splineops
  operation and is the most important exact comparison.
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
3. Include benchmark scripts or reproducible commands in the PR description.
4. Hide local experiment knobs from the public API, or keep them clearly
   internal and undocumented when they are only for A/B validation.
5. Keep approximate/numerically different modes opt-in and separate from the
   exact default path.

The best first PR is not the largest possible optimization bundle. It is the
smallest exact bundle with a strong correctness story, clear benchmark evidence,
and a credible maintenance cost.
