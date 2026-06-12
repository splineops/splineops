# Resize Optimization Notes

Date: 2026-06-12

This note tracks optimization ideas for `splineops.resize`, especially the native
`_lsresize` backend. The goal is to preserve the exact algorithmic behavior while
reducing runtime, memory movement, and repeated setup cost.

## Handoff: 2026-06-12

Work completed today:

- Extended the native batched axis kernel behind `LSRESIZE_BATCHED_AXIS=1` from
  pure interpolation to projection and antialiasing.
- Promoted conservative automatic routing to the default when
  `LSRESIZE_BATCHED_AXIS` is unset. Use `LSRESIZE_BATCHED_AXIS=off` to
  force the original line-by-line native path, and `LSRESIZE_BATCHED_AXIS=1`
  to force the batched path everywhere.
- Reworked the batched axis accumulator to skip materializing the full
  `[left pad | line | right pad]` extension buffer. Interior rows now read
  coefficient columns directly, and boundary/tail rows use a precomputed
  per-weight source/sign table.
- Split pure interpolation into its own batched native kernel. It skips
  projection scratch, correction filtering, and projection-only branches.
- Added a pure-interpolation direct-write path for non-contiguous output axes,
  avoiding the intermediate `y` buffer plus strided scatter on those passes.
- Moved the col-major batched IIR prefilter into `cpp/lsresize/src/filters.*`
  so the finite-size mirror initializer is shared instead of duplicated in
  `resize_nd.cpp`.
- Added batched native helpers for:
  - `do_integ`
  - `do_diff`
  - output interpolation prefilter
  - output sampling FIR
- Added native parity regressions comparing the batched path against the
  default native path for:
  - pure `linear`, `quadratic`, and `cubic`
  - `linear-antialiasing`, `quadratic-antialiasing`, and `cubic-antialiasing`
  - equal-degree projection (`interp = analy = synthe`) for degrees 1, 2, 3
- Added an `auto`-mode parity regression covering large 2-D cases and a 3-D
  fallback case.
- Extended `scripts/benchmark_resize_native.py` with:
  - `--batched-axis {env,off,on,auto}`
  - `--batch-lines-sweep`
  - per-result batched mode and batch-size metadata
  - per-case batch-size sweep summaries

Current changed files to expect in the working tree:

- `cpp/lsresize/src/resize_nd.cpp`
- `scripts/benchmark_resize_native.py`
- `scripts/resize_optimization_notes.md`
- `tests/test_02_03_resize_cpp.py`

Validation run:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_pure_interpolation \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_antialiasing \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_equal_degree_projection \
  tests/test_02_03_resize_cpp.py::test_batched_axis_auto_matches_default
.venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
LSRESIZE_BATCHED_AXIS=off .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
LSRESIZE_BATCHED_AXIS=1 .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile scripts/benchmark_resize_native.py
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis env \
  --skip-checks
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis off \
  --skip-checks
git diff --check
```

Observed results:

- After promoting auto routing to the unset default:
  - `py_compile`: clean
  - new batched/default-auto/off parity tests: `60 passed`
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=off`: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=1`: `175 passed`
  - full suite default/unset: `429 passed`
  - explicit `auto` is covered by `test_batched_axis_auto_matches_default`
  - smoke benchmark reports unset/env mode as `batched_axis=<default:auto>`
- `git diff --check`: clean

Fresh local benchmark command:

```bash
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis off \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis on \
  --batch-lines 32 \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 2 \
  --warmups 1 \
  --batched-axis on \
  --batch-lines-sweep 8,16,32,64 \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis auto \
  --skip-checks
```

Selected best-of timings from the forced batched run:

| Case | Default path | Batched path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic-antialiasing, 1 thread | 31.42 ms | 13.22 ms | 2.38x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 108.73 ms | 52.82 ms | 2.06x |
| float64, 1024x1024, cubic-antialiasing, 1 thread | 110.56 ms | 49.70 ms | 2.22x |
| float32, 1024x1024, cubic-antialiasing, default threads | 14.67 ms | 8.22 ms | 1.78x |
| float64, 1024x1024, cubic-antialiasing, default threads | 11.29 ms | 9.17 ms | 1.23x |
| float32, 3-D cubic down, default threads | 6.15 ms | 7.21 ms | 0.85x |

Selected best-of timings from the auto-routing run:

| Case | Default path | Auto path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic down, 1 thread | 12.38 ms | 4.91 ms | 2.52x |
| float32, 512x512, cubic-antialiasing, 1 thread | 26.63 ms | 14.70 ms | 1.81x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 119.30 ms | 41.60 ms | 2.87x |
| float64, 1024x1024, cubic-antialiasing, 1 thread | 105.72 ms | 42.56 ms | 2.48x |
| float32, 1024x1024, cubic-antialiasing, default threads | 10.76 ms | 7.83 ms | 1.37x |
| float32, 3-D cubic down, default threads | 6.19 ms | 5.82 ms | skipped by auto, timing noise |

Selected best-of timings after the interior/boundary split and precomputed
source/sign table:

| Case | Default path | Auto path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic down, 1 thread | 12.03 ms | 7.06 ms | 1.70x |
| float32, 1024x1024, cubic down, 1 thread | 25.33 ms | 21.74 ms | 1.17x |
| float32, 1024x1024, cubic down, default threads | 6.98 ms | 4.63 ms | 1.51x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 80.82 ms | 42.71 ms | 1.89x |
| float32, 1024x1024, cubic-antialiasing, default threads | 11.08 ms | 6.78 ms | 1.63x |
| float64, 1024x1024, cubic-antialiasing, 1 thread | 110.87 ms | 36.20 ms | 3.06x |

The focused repeat benchmark showed high best-time noise for the default path,
so median times should be checked before overfitting the auto heuristic.

Selected best-of timings after the pure-interpolation specialization and
non-contiguous direct-write path:

| Case | Default path | Auto path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic down, 1 thread | 14.00 ms | 6.49 ms | 2.16x |
| float32, 1024x1024, cubic down, 1 thread | 54.71 ms | 24.28 ms | 2.25x |
| float32, 1024x1024, cubic down, default threads | 7.12 ms | 5.57 ms | 1.28x |
| float64, 1024x1024, cubic down, 1 thread | 51.40 ms | 23.27 ms | 2.21x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 119.39 ms | 41.86 ms | 2.85x |
| float32, 1024x1024, cubic-antialiasing, default threads | 12.41 ms | 7.44 ms | 1.67x |
| float64, 1024x1024, cubic-antialiasing, default threads | 14.48 ms | 7.50 ms | 1.93x |

Median-focused repeat timings from the same code path showed:

- float32 1024x1024 cubic down, 1 thread: `54.39 ms -> 23.77 ms`
  (`2.31x` median speedup).
- float64 1024x1024 cubic down, 1 thread: `50.92 ms -> 22.27 ms`
  (`2.29x` median speedup).
- float32 1024x1024 cubic-antialiasing, default threads:
  `13.96 ms -> 8.26 ms` (`1.69x` median speedup).

Smoke `--batch-lines-sweep 8,16,32,64` observations:

- Best batch size is workload/thread dependent.
- After the no-extension-buffer split, the latest smoke sweep picked different
  winners across cases: 8 for small single-thread pure cubic, 16 for some
  default-thread pure/anisotropic cases, 32 for single-thread
  cubic-antialiasing/anisotropic cases, and 64 for default-thread
  cubic-antialiasing.
- The current default of 32 remains a reasonable conservative default. Do not
  change it without a saved standard/full sweep on the target CPU.

Important caveat:

- `LSRESIZE_BATCHED_AXIS` is now default-auto. Unset and `auto` both route only
  sufficiently large 2-D axis passes through batching and leave 3-D on the
  original path because 3-D forced-batched timings remain mixed.
- Use `LSRESIZE_BATCHED_AXIS=off` to force the original line-by-line path for
  debugging, comparisons, and conservative deployments.
- `LSRESIZE_BATCHED_AXIS=1` remains useful for stress/parity testing the
  batched implementation directly.

Suggested next steps:

1. Run a saved standard/full `--batch-lines-sweep` artifact on the target CPU
   before changing the default batch size from 32.
2. Run default-auto versus `LSRESIZE_BATCHED_AXIS=off` saved artifacts on the
   target CPU before tuning the auto heuristic further.
3. Consider fused resize/permutation writes for N-D cases where strided passes
   dominate.

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
  - note: this 2026-06-11 scope was superseded on 2026-06-12 with projection
    and antialiasing support
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

Important caveat as of 2026-06-11:

- The batched path is promising for pure cubic interpolation, especially larger
  2-D cases, but it is disabled by default. Do not make it default yet.
- This caveat was superseded on 2026-06-12: antialiasing/projection is now
  covered by the feature-flagged batched path, but the path remains disabled by
  default pending routing/tuning.

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

2. Tune `LSRESIZE_BATCH_LINES`; early runs suggested 32 is good, but 8/16/64
   should be compared systematically.
3. If batched interpolation and antialiasing remain consistently faster in a
   subset of cases, add an automatic routing condition for those cases.

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
- Current scope: pure interpolation plus projection/antialiasing.
- Tests compare flagged native output against default native output for
  interpolation, antialiasing presets, and equal-degree projection on 2-D and
  3-D cases.

Early local timings with `LSRESIZE_BATCHED_AXIS=1` and batch size 32:

| Case | Default path | Batched path | Notes |
| --- | ---: | ---: | --- |
| float32, 1024x1024, cubic, 1 thread | 51.1 ms | 30.4 ms | side-by-side best-of |
| float32, 1024x1024, cubic, default threads | 7.1 ms | 5.3 ms | side-by-side best-of |
| float64, 512x512, cubic anisotropic, 1 thread | 6.6 ms | 3.3 ms | side-by-side best-of |
| float64, 512x512, cubic anisotropic, default threads | 1.5 ms | 1.2 ms | side-by-side best-of |

Later local timings after projection support was added are recorded in the
2026-06-12 handoff above.

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

1. Add conservative automatic routing for the feature-flagged batched path.
2. Add native plan cache or `ResizePlan`.
3. Improve thread-count heuristic.
4. Split interior and boundary kernels.
5. Add specialized preset kernels.
6. Add opt-in float32 internal mode.
7. Explore tiled/fused memory strategies for large N-D workloads.

The highest-upside exact redesign is the batched native axis kernel combined
with an interior/boundary split. The lowest-risk next engineering step is a
batch-size/thread sweep followed by conservative routing.
