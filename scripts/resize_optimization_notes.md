# Resize Optimization Notes

Date: 2026-06-11

This note tracks optimization ideas for `splineops.resize`, especially the native
`_lsresize` backend. The goal is to preserve the exact algorithmic behavior while
reducing runtime, memory movement, and repeated setup cost.

## Handoff: 2026-06-11

Work completed today:

- Fixed the finite-size constant-preservation bug in the spline IIR prefilter.
  The fix uses the exact mirror-boundary finite-length causal initializer with
  the correct reflected term `z^(2N-2-n)`.
- Added regressions for short constant signals and end-to-end constant resizing.
- Added a native scheduling optimization:
  - process shrinking axes before non-shrinking axes
  - process stronger shrink factors earlier
  - skip pure-interpolation identity axes at the binding layer
- Added a reproducible benchmark harness:
  `scripts/benchmark_resize_native.py`.
- Added the first batched native axis-kernel prototype behind a feature flag:
  `LSRESIZE_BATCHED_AXIS=1`.

Current changed files to expect in the working tree:

- `cpp/lsresize/src/resize_nd.cpp`
  - contains the feature-flagged batched pure-interpolation axis path
  - current scope is `analy_degree < 0` only
  - projection and antialiasing still use the existing line-by-line path
- `tests/test_02_03_resize_cpp.py`
  - includes parity tests comparing default native output against
    `LSRESIZE_BATCHED_AXIS=1`
- `scripts/benchmark_resize_native.py`
  - new benchmark script
- `scripts/resize_optimization_notes.md`
  - this handoff and roadmap

Validation already run:

```bash
python -m py_compile scripts/benchmark_resize_native.py
git diff --check
pytest -q
LSRESIZE_BATCHED_AXIS=1 pytest -q
LSRESIZE_BATCHED_AXIS=1 pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
```

Observed results:

- Full suite default: `387 passed`
- Full suite with `LSRESIZE_BATCHED_AXIS=1`: `387 passed`
- Focused suite with `LSRESIZE_BATCHED_AXIS=1`: `133 passed`
- `git diff --check`: clean
- `py_compile`: clean

Benchmark artifacts written under `/tmp`:

- `/tmp/splineops_resize_baseline_standard.json`
- `/tmp/splineops_resize_baseline_standard.csv`
- `/tmp/splineops_resize_batched_standard.json`
- `/tmp/splineops_resize_batched_standard.csv`

Important caveat:

- The batched path is promising for pure cubic interpolation, especially larger
  2-D cases, but it is disabled by default. Do not make it default yet.
- Antialiasing/projection is not batched yet. Most `*-antialiasing` timings in
  batched runs are still measuring the old path, apart from noise and scheduling
  effects.

Suggested next steps tomorrow:

1. Re-run the standard benchmark once to re-establish local timing stability:

   ```bash
   python scripts/benchmark_resize_native.py \
     --profile standard \
     --threads 1,2,4,8,16,default \
     --repeats 5 \
     --warmups 2 \
     --batched-axis off

   python scripts/benchmark_resize_native.py \
     --profile standard \
     --threads 1,2,4,8,16,default \
     --repeats 5 \
     --warmups 2 \
     --batched-axis on \
     --batch-lines 32
   ```

2. Tune `LSRESIZE_BATCH_LINES` for pure interpolation. Early runs suggested 32
   is good, but 8/16/64 should be compared systematically.
3. If pure interpolation remains consistently faster, add an automatic routing
   condition for safe pure-interpolation cases, or keep it behind the flag until
   projection support is ready.
4. Extend the batched design to projection/antialiasing:
   - batched `do_integ`
   - batched `do_diff`
   - batched output prefilter and sampling FIR
   - parity against default native path and Python fallback
5. After projection parity passes, benchmark `cubic-antialiasing` specifically;
   that is the method most relevant to the publication story.

## Current Baseline

The current native implementation is a separable N-D resize:

1. Build a 1-D plan for one axis.
2. Process each independent 1-D line.
3. Ping-pong intermediate arrays between axis passes.
4. Use double precision internally, including for float32 storage.

Useful existing strengths:

- Per-axis weights are precomputed once per axis pass.
- Work buffers are reused per worker.
- Small spline dot products are already unrolled.
- C++ and Python fallback paths agree closely.
- Constant-preservation regressions now cover short finite signals.

Recent safe native scheduling change:

- Shrinking axes are processed before non-shrinking axes.
- Stronger shrink factors are processed earlier.
- Pure interpolation identity axes are skipped at the binding layer.

Local rough timing deltas, default threading:

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| float32, 1024x1024, cubic, zoom 0.37 | 9.76 ms | 7.32 ms | 1.33x |
| float64, 1024x1024, cubic, zoom 0.37 | 9.69 ms | 6.11 ms | 1.59x |
| float64, 1024x1024, cubic-antialiasing, zoom 0.37 | 13.70 ms | 10.32 ms | 1.33x |
| float32, 512x512, cubic-antialiasing, zoom 0.37 | 4.32 ms | 3.91 ms | 1.10x |

These numbers are local best-of timings, not a formal benchmark suite.

## Rejected Experiment

A 2-D transpose fast path was tested:

1. Resize rows contiguously.
2. Transpose the intermediate.
3. Resize original columns as contiguous rows.
4. Transpose back.

This was not kept. It helped some single-thread antialiasing cases but was not
consistently better under default threading. The extra memory movement became a
bottleneck.

## Suggested Next Steps

### 1. Formal Benchmark Harness

Create a small reproducible benchmark script before deeper optimization.

Initial script: `scripts/benchmark_resize_native.py`.

Useful commands:

```bash
python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,2,4,8,16,default \
  --repeats 5 \
  --warmups 2 \
  --output-json /tmp/splineops_resize_baseline_standard.json \
  --output-csv /tmp/splineops_resize_baseline_standard.csv

python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --skip-checks \
  --batched-axis on \
  --batch-lines 32 \
  --output-json /tmp/splineops_resize_batched_standard.json \
  --output-csv /tmp/splineops_resize_batched_standard.csv
```

Requirements:

- Fixed inputs and random seeds.
- Shapes: 512x512, 1024x1024, 2048x2048, and representative 3-D volumes.
- Methods: cubic, cubic-antialiasing, linear-antialiasing.
- Dtypes: float32 and float64.
- Zooms: uniform downsampling, anisotropic downsampling, upsampling, identity on
  one axis.
- Thread counts: 1, 2, 4, 8, 16, default.
- Correctness checks against the Python fallback for selected cases.

This avoids optimizing against noisy ad hoc timings.

### 2. Native Plan Cache

The C++ binding currently rebuilds per-axis plans for each call. Repeated
same-shape/same-zoom workloads, such as video frames or registration loops, can
reuse the same plans.

Options:

- Add a small process-local LRU cache keyed by:
  `(N, zoom, interp_degree, analy_degree, synthe_degree, shift, inversable)`.
- Or expose a Python-facing `ResizePlan` object:
  `plan = ResizePlan(input_shape, output_shape, method); plan.apply(frame)`.

The explicit plan object is better long-term. A private LRU cache is easier to
add first.

### 3. Threading Heuristic

Local measurements showed oversubscription:

- 1024x1024 cubic-antialiasing, float32:
  - 1 thread: about 119 ms
  - 8 threads: about 17 ms
  - 16 threads: about 11 ms
  - 32 requested threads: slower on the test machine

Possible improvements:

- Cap default worker count more conservatively.
- Use workload size to choose fewer threads for small images.
- Add a small environment-controlled autotune mode.
- Eventually replace per-axis thread creation with a persistent thread pool.

The environment override `LSRESIZE_NUM_THREADS` should remain available.

### 4. Batched Native Axis Kernel

This is the most important fundamental redesign.

Current native code processes one 1-D line at a time. But all lines for an axis
share the same plan and weights. A batched C++ kernel could process a block of
lines together:

```text
for output index l:
    y[:, l] = w0 * x[:, k0] + w1 * x[:, k1] + ...
```

Benefits:

- Reuses weights across many lines.
- Vectorizes across lines instead of only across the tiny spline support.
- Reduces per-line overhead.
- More closely matches the vectorized strategy already used in the Python
  fallback, but without NumPy gather overhead.

This should be exact, not an approximation.

Prototype status:

- Feature flag: `LSRESIZE_BATCHED_AXIS=1`.
- Batch size override: `LSRESIZE_BATCH_LINES=<positive integer>`.
- Current scope: pure interpolation only, i.e. `analy_degree < 0`.
- Projection and antialiasing still use the existing line-by-line kernel.
- Tests added compare flagged native output against default native output for
  `linear`, `quadratic`, and `cubic` interpolation on 2-D and 3-D cases.

Early local timings with `LSRESIZE_BATCHED_AXIS=1` and batch size 32:

| Case | Default path | Batched path | Notes |
| --- | ---: | ---: | --- |
| float32, 1024x1024, cubic, 1 thread | 51.1 ms | 30.4 ms | side-by-side best-of |
| float32, 1024x1024, cubic, default threads | 7.1 ms | 5.3 ms | side-by-side best-of |
| float64, 512x512, cubic anisotropic, 1 thread | 6.6 ms | 3.3 ms | side-by-side best-of |
| float64, 512x512, cubic anisotropic, default threads | 1.5 ms | 1.2 ms | side-by-side best-of |

### 5. Interior/Boundary Split

Most output samples are far from mirrored boundaries. The hot loop should not
pay boundary-extension costs for those samples.

Proposed structure:

- Fast interior kernel:
  - no mirror handling
  - no extension buffer
  - direct weighted reads
  - branchless or nearly branchless
- Slow boundary kernel:
  - exact mirror or antisymmetric boundary behavior
  - current generic machinery is acceptable here

For large images, most rows hit the fast interior path.

### 6. Template-Specialized Presets

The generic degree triple is useful, but common presets can be specialized:

- `linear`
- `cubic`
- `linear-antialiasing`
- `cubic-antialiasing`

Compile-time specialization can remove degree branches, fix support sizes, and
make the compiler more aggressive in unrolling and inlining.

This should come after the benchmark harness, since specialized kernels are easy
to make faster in one case and slower in another.

### 7. Optional Float32 Internal Mode

The current native path uses double internally even for float32 storage. That is
robust, but image workloads may not need double precision.

Possible API:

```python
resize(x, zoom_factors=z, method="cubic-antialiasing", precision="float32")
resize(x, zoom_factors=z, method="cubic-antialiasing", precision="float64")
```

Risks:

- Different numerical results.
- Potentially worse constant preservation if not implemented carefully.
- More testing required for small images and high-frequency inputs.

This has high performance upside but should be opt-in at first.

### 8. Output and Temporary Memory Strategy

Current ping-pong intermediates are simple and safe. For large N-D data, memory
traffic may dominate.

Ideas:

- Process smaller tiles for large arrays.
- Fuse adjacent axis passes for small output tiles where practical.
- Reuse native temporary buffers through a plan object.
- Avoid reallocating ping-pong buffers in repeated calls.

This is probably more valuable for video, volumes, and batch processing than for
single small 2-D images.

## Recommended Priority

1. Add a reproducible benchmark harness.
2. Add native plan cache or `ResizePlan`.
3. Improve thread-count heuristic.
4. Implement batched native axis kernel.
5. Split interior and boundary kernels.
6. Add specialized preset kernels.
7. Add opt-in float32 internal mode.
8. Explore tiled/fused memory strategies for large N-D workloads.

The highest-upside exact redesign is the batched native axis kernel combined
with an interior/boundary split. The lowest-risk next engineering step is a
benchmark harness plus native plan reuse.
