# Resize Optimization Notes

Date: 2026-06-12

This note tracks optimization ideas for `splineops.resize`, especially the native
`_lsresize` backend. The goal is to preserve the exact algorithmic behavior while
reducing runtime, memory movement, and repeated setup cost.

## Weekend Wrap-Up: 2026-06-12

The resize optimization pass now has three solid pillars:

1. **Correctness baseline is stronger.**
   - Fixed the finite-size spline IIR mirror initializer for short signals and
     constant preservation.
   - Added regressions for constants, native/Python parity, batched/native parity,
     forced batched routing, and default-auto routing.
   - Latest full-suite validation: `429 passed`.

2. **Native default path is substantially faster for the target 2-D workloads.**
   - Axis scheduling now processes shrinking axes first and skips pure
     interpolation identity axes.
   - Default thread scheduling is more conservative on SMT machines and avoids
     launching a single worker thread for one-worker decisions.
   - The native batched axis kernel now covers pure interpolation, projection, and
     antialiasing.
   - `LSRESIZE_BATCHED_AXIS` is default-auto when unset. Use
     `LSRESIZE_BATCHED_AXIS=off` for the original line-by-line path and
     `LSRESIZE_BATCHED_AXIS=1` to force batching.
   - The native batched path skips the full extension buffer, uses direct
     coefficient reads for interior rows, and uses exact source/sign mapping for
     boundary rows.
   - Native plans are cached by default with capacity 32. Set
     `LSRESIZE_PLAN_CACHE_SIZE=0` to measure cold-plan behavior.
   - Batched row-run metadata is now precomputed inside `Plan1D`, so repeated
     same-plan calls reuse interior/boundary runs and boundary source/sign maps.

3. **Python fallback is also faster and easier to benchmark.**
   - `scripts/benchmark_resize_native.py` now supports
     `--backend {native,python,both}`.
   - Python fallback defaults are now `SPLINEOPS_BLOCK=256` and
     `SPLINEOPS_ACCUM=support`.
   - Python plan cache capacity defaults to 32 via
     `SPLINEOPS_PLAN_CACHE_SIZE`, with compatibility for
     `LSRESIZE_PLAN_CACHE_SIZE`.

Current default knobs:

| Area | Default | Override |
| --- | --- | --- |
| Native acceleration | auto via `SPLINEOPS_ACCEL=auto` | `always`, `never` |
| Native batched axis | `auto` when `LSRESIZE_BATCHED_AXIS` is unset | `off`, `1`, `auto` |
| Native batch lines | `64` | `LSRESIZE_BATCH_LINES=<n>` |
| Native plan cache | enabled, capacity `32` | `LSRESIZE_PLAN_CACHE_SIZE=<n>` |
| Native threads | workload-aware default | `LSRESIZE_NUM_THREADS=<n>` |
| Python block size | `256` | `SPLINEOPS_BLOCK=<n>` |
| Python accumulator | `support` | `SPLINEOPS_ACCUM=einsum` or `mulsum` |
| Python plan cache | enabled, capacity `32` | `SPLINEOPS_PLAN_CACHE_SIZE=<n>` |

Measured progress so far:

- Native default-auto versus explicit native off:
  - saved sequential artifacts:
    `/tmp/splineops_resize_auto_default_b64_seq.{json,csv}` and
    `/tmp/splineops_resize_off_default_b64_seq.{json,csv}`
  - default-auto with batch size 64 was faster than explicit off on every 2-D
    standard case by median
  - 2-D median speedup averaged about `2.15x`; the weakest 2-D standard case was
    still about `1.57x`
- Native cached row-run/interior-boundary split versus the committed baseline:
  - baseline artifact:
    `/tmp/splineops_resize_native_baseline_head_r8.{json,csv}`
  - final artifact:
    `/tmp/splineops_resize_native_after_plan_row_runs_final_r8.{json,csv}`
  - default-thread medians improved in `15/18` standard cases overall and
    `14/16` 2-D cases
  - 2-D median speedup was about `1.12x` mean / `1.12x` median
  - best-of timings improved in `15/18` cases with about `1.16x` mean /
    `1.12x` median speedup
- Native plan cache:
  - mostly helps small/repeated workloads
  - observed wins include about `1.17x` for 1-D length 64 cubic, `1.08x` for
    16x16 cubic, `1.06x` for 32x32 cubic, and `1.02x` for 64x64
    cubic-antialiasing on the local CPU
- Python fallback:
  - `support`/256 versus previous `mulsum`/64 won `18/18` standard cases
  - mean speedup about `1.85x`, median speedup about `1.82x`
  - saved artifact:
    `/tmp/splineops_resize_python_support_block256.{json,csv}`

Validation status:

- Native editable rebuild: clean.
- Direct batched-axis parity tests: `60 passed`.
- Focused resize suite: `175 passed`.
- Forced Python fallback focused suite: `175 passed`.
- Full suite: `429 passed`.
- `git diff --check`: clean on the latest implementation pass.

### Full Optimization Roadmap

The algorithm is now in a good exact, default-on state. The remaining high-upside
work is mostly specialization, precision policy, and repeated-workload API
design.

1. **Specialized native preset kernels.**
   - Target methods: `linear`, `cubic`, `linear-antialiasing`,
     `cubic-antialiasing`.
   - Replace generic degree-dependent loops in the hottest native paths with
     compile-time support sizes and branch-free preset kernels.
   - Keep the current generic path as the fallback for uncommon degree triples.
   - Introduce behind a temporary feature flag first, then promote only cases
     that win across saved standard/full benchmark artifacts.

2. **Opt-in float32 internal mode.**
   - Current native computation uses double internally even for float32 arrays.
   - Add an experimental precision policy, for example an API option or an env
     flag, that keeps float32 workloads in float32 scratch/accumulation where
     accuracy is acceptable.
   - Validate constant preservation, native/Python agreement thresholds, and
     high-frequency inputs before making any default change.
   - This likely has the largest upside for image workloads, but it changes
     numerical behavior, so it must remain opt-in initially.

3. **Public reusable `ResizePlan`.**
   - The private native cache is useful but implicit. A public plan object would
     make repeated same-shape workloads explicit:
     `plan = ResizePlan(input_shape, zoom, method); out = plan.apply(x)`.
   - Reuse axis order, per-axis `Plan1D`, row-run metadata, output shapes, and
     possibly per-thread scratch.
   - Best fit: video frames, registration loops, batch processing, and repeated
     augmentation with fixed geometry.

4. **Memory and temporary-buffer strategy.**
   - Reuse ping-pong intermediate arrays across repeated calls through a plan.
   - Explore fused final-axis writes or permutation-aware scheduling for
     workloads where strided passes dominate.
   - Consider tiling for large volumes where memory traffic, not arithmetic,
     dominates.

5. **Routing and scheduler calibration on more hardware.**
   - Re-run standard/full artifacts on machines with different core counts,
     cache sizes, and SMT behavior.
   - Revisit `LSRESIZE_BATCH_LINES=64`, default-auto thresholds, and default
     thread caps only after cross-machine artifacts show a consistent better
     choice.

6. **Benchmark hygiene before every default change.**
   - Always save before/after JSON+CSV artifacts.
   - Compare medians and best-of timings; short sub-3 ms cases can be noisy.
   - Keep parity checks enabled for smoke/small profiles, and use
     `--skip-checks` only for large timing sweeps.

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
- Added workload-aware default scheduling in `parallel_utils.h`:
  - explicit `LSRESIZE_NUM_THREADS` remains a hard override
  - unset/default scheduling caps larger pools near a physical-core estimate
    instead of always using every logical CPU
  - one-worker decisions now run directly in the current thread instead of
    launching a single `std::thread`
- Added a bounded process-local native `Plan1D` cache:
  - keyed exactly by line length, degrees, zoom bits, shift bits, and
    `inversable`
  - plans are immutable/read-only after construction and shared by axis workers
  - `LSRESIZE_PLAN_CACHE_SIZE=0` disables the cache for measurements/debugging
- Moved the native batched row map into cached `Plan1D` metadata:
  - contiguous interior/boundary row runs are precomputed once per plan
  - boundary rows use cached per-weight coefficient source/sign mapping
  - batched ND kernels now execute branch-free direct interior runs and fall
    back to the exact mapped boundary runs
  - this removes per-call row-map allocation/setup from repeated same-plan
    workloads
- Promoted the batched-axis default block size from 32 to 64 after a saved
  standard `--batch-lines-sweep` on the local target CPU.
- Extended `scripts/benchmark_resize_native.py` beyond native-only measurements:
  - `--backend {native,python,both}` selects the implementation under test
  - Python rows report `<n/a>` for native thread/batch-line controls
  - smoke parity can now compare native against Python and Python against native
  - JSON metadata records the active Python fallback knobs
- Tuned the pure-Python fallback defaults:
  - `SPLINEOPS_BLOCK` default is now 256 instead of 64
  - `SPLINEOPS_ACCUM` default is now `support`
  - `SPLINEOPS_ACCUM=einsum` and `SPLINEOPS_ACCUM=mulsum` remain available for
    comparison/debugging
  - Python plan cache capacity now defaults to 32 via
    `SPLINEOPS_PLAN_CACHE_SIZE`, with fallback compatibility for
    `LSRESIZE_PLAN_CACHE_SIZE`
  - `SPLINEOPS_PLAN_CACHE=0` or `SPLINEOPS_PLAN_CACHE_SIZE=0` disables the
    Python plan cache

Current changed files to expect in the working tree:

- `cpp/lsresize/src/resize_1d.cpp`
- `cpp/lsresize/src/resize_1d.h`
- `cpp/lsresize/src/resize_nd.cpp`
- `scripts/resize_optimization_notes.md`

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
LSRESIZE_PLAN_CACHE_SIZE=0 .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
SPLINEOPS_ACCEL=never .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  src/splineops/resize/_pycore/resize_nd.py \
  src/splineops/utils/specs.py
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --backend both \
  --threads default \
  --repeats 2 \
  --warmups 1
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
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,2,4,8,16,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis env \
  --skip-checks \
  --output-json /tmp/splineops_resize_scheduler_before_auto.json \
  --output-csv /tmp/splineops_resize_scheduler_before_auto.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 8,16,default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis env \
  --skip-checks \
  --output-json /tmp/splineops_resize_scheduler_after_physicalcap.json \
  --output-csv /tmp/splineops_resize_scheduler_after_physicalcap.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --backend python \
  --threads default \
  --repeats 3 \
  --warmups 1 \
  --skip-checks \
  --output-json /tmp/splineops_resize_python_support_block256.json \
  --output-csv /tmp/splineops_resize_python_support_block256.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --backend native \
  --threads default \
  --repeats 8 \
  --warmups 2 \
  --batched-axis auto \
  --skip-checks \
  --output-json /tmp/splineops_resize_native_after_plan_row_runs_final_r8.json \
  --output-csv /tmp/splineops_resize_native_after_plan_row_runs_final_r8.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis auto \
  --batch-lines-sweep 8,16,32,64,128 \
  --skip-checks \
  --output-json /tmp/splineops_resize_batch_sweep_plan_cache.json \
  --output-csv /tmp/splineops_resize_batch_sweep_plan_cache.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis env \
  --skip-checks \
  --output-json /tmp/splineops_resize_auto_default_b64_seq.json \
  --output-csv /tmp/splineops_resize_auto_default_b64_seq.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis off \
  --skip-checks \
  --output-json /tmp/splineops_resize_off_default_b64_seq.json \
  --output-csv /tmp/splineops_resize_off_default_b64_seq.csv
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
- After the scheduler default-thread update:
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=off`: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=1`: `175 passed`
  - full suite default/unset: `429 passed`
  - final standard thread sweep saved to
    `/tmp/splineops_resize_scheduler_after_physicalcap.{json,csv}`
  - on the local 16-logical/8-core CPU, default scheduling now tracks the
    8-thread/physical-core-style setting instead of the previous all-logical
    default, while explicit `LSRESIZE_NUM_THREADS=16` remains available
- After native plan caching and batch-size tuning:
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `LSRESIZE_PLAN_CACHE_SIZE=0`: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=1`: `175 passed`
  - full suite default/unset: `429 passed`
  - repeated-call microbenchmarks show cache wins mostly on small workloads:
    about `1.17x` for 1-D length 64 cubic, `1.08x` for 16x16 cubic,
    `1.06x` for 32x32 cubic, and `1.02x` for 64x64 cubic-antialiasing
    on this CPU; 64-256 squared cases were closer to `1.01x-1.04x`
  - standard default-thread batch sweep saved to
    `/tmp/splineops_resize_batch_sweep_plan_cache.{json,csv}`
  - batch size 64 won the most median-time cases in the standard sweep and
    gave about `1.07x` median / `1.10x` mean speedup versus batch size 32
    across standard cases
  - sequential standard auto-vs-off artifacts saved to
    `/tmp/splineops_resize_auto_default_b64_seq.{json,csv}` and
    `/tmp/splineops_resize_off_default_b64_seq.{json,csv}`
  - default-auto with batch size 64 was faster than explicit off on every 2-D
    standard case by median; 2-D median speedup averaged about `2.15x`, with
    the weakest 2-D case still about `1.57x`
- After Python fallback tuning:
  - `py_compile`: clean for the benchmark script, Python core, and runtime specs
  - smoke `--backend both`: native/Python parity passed for the smoke profile;
    max absolute differences stayed at `1.19e-7` for float32 cubic and
    `5.55e-16` for float64 cubic
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `SPLINEOPS_ACCEL=never`: `175 passed`
  - full suite default/unset: `429 passed`
  - standard Python benchmark saved to
    `/tmp/splineops_resize_python_support_block256.{json,csv}`
- After cached native row-run/interior-boundary split:
  - editable native rebuild: clean
  - direct batched-axis parity tests: `60 passed`
  - focused resize suite default/unset: `175 passed`
  - fair baseline from committed `HEAD` saved to
    `/tmp/splineops_resize_native_baseline_head_r8.{json,csv}`
  - final default-thread standard artifact saved to
    `/tmp/splineops_resize_native_after_plan_row_runs_final_r8.{json,csv}`
  - versus the committed baseline, standard default-thread medians improved in
    `15/18` cases overall and `14/16` 2-D cases
  - 2-D median speedup: `1.12x` mean / `1.12x` median, with noise/regression
    on small anisotropic float32/large anisotropic float64 medians but best-of
    timings still mostly improved
  - best-of timings improved in `15/18` cases overall, with `1.16x` mean /
    `1.12x` median speedup
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
  --batch-lines 64 \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 2 \
  --warmups 1 \
  --batched-axis on \
  --batch-lines-sweep 8,16,32,64,128 \
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

Smoke/standard `--batch-lines-sweep` observations:

- Best batch size is workload/thread dependent.
- After the no-extension-buffer split, the latest smoke sweep picked different
  winners across cases: 8 for small single-thread pure cubic, 16 for some
  default-thread pure/anisotropic cases, 32 for single-thread
  cubic-antialiasing/anisotropic cases, and 64 for default-thread
  cubic-antialiasing.
- After plan caching and the default-thread scheduler update, a saved standard
  sweep over 8, 16, 32, 64, and 128 picked 64 most often by median time.
  The native default is now 64. Re-run a saved target-CPU sweep before changing
  it again.

Python fallback observations:

- `SPLINEOPS_ACCUM=einsum` at the old block size 64 beat the previous
  `mulsum`/64 default in 17/18 standard-profile cases, with `1.06x` mean
  median speedup.
- Raising the Python block size to 256 was the larger win. `einsum`/256 beat
  `mulsum`/64 in 18/18 cases, with `1.67x` mean and `1.55x` median speedup.
- Block size 512 was only marginally faster on average than 256 and regressed
  several 2-D float32/linear cases, so 256 is the safer default.
- The support-wise streaming accumulator beat `einsum`/256 in 18/18 cases,
  with `1.12x` mean, `1.07x` median, `1.04x` minimum, and `1.32x` maximum
  speedup.
- Against the previous `mulsum`/64 default, `support`/256 beat 18/18 cases,
  with `1.85x` mean and `1.82x` median speedup.
- Saved Python artifacts:
  - `/tmp/splineops_resize_python_mulsum_block64.{json,csv}`
  - `/tmp/splineops_resize_python_einsum_block256.{json,csv}`
  - `/tmp/splineops_resize_python_support_block256.{json,csv}`

Important caveat:

- `LSRESIZE_BATCHED_AXIS` is now default-auto. Unset and `auto` both route only
  sufficiently large 2-D axis passes through batching and leave 3-D on the
  original path because 3-D forced-batched timings remain mixed.
- Use `LSRESIZE_BATCHED_AXIS=off` to force the original line-by-line path for
  debugging, comparisons, and conservative deployments.
- `LSRESIZE_BATCHED_AXIS=1` remains useful for stress/parity testing the
  batched implementation directly.
- Default thread scheduling is now conservative on SMT-style machines. Use
  `LSRESIZE_NUM_THREADS=<n>` to force a particular thread count for benchmark
  sweeps or deployments that benefit from all logical CPUs.
- Native plan caching is on by default with capacity 32. Use
  `LSRESIZE_PLAN_CACHE_SIZE=0` to disable it when measuring cold-plan behavior.

Suggested next steps:

1. Run a saved standard/full `--batch-lines-sweep` artifact on additional target
   CPUs before changing the default batch size from 64.
2. Run default-auto versus `LSRESIZE_BATCHED_AXIS=off` saved artifacts on
   additional target CPUs before tuning the auto heuristic further.
3. Run a standard/full saved thread sweep on target hardware before changing
   the default scheduler constants.
4. Add template-specialized native preset kernels for the common methods
   (`linear`, `cubic`, `linear-antialiasing`, `cubic-antialiasing`).
5. Consider exposing an explicit reusable native `ResizePlan` object if repeated
   same-shape workloads remain important; the private cache is the low-risk
   first step.
6. Consider fused resize/permutation writes for N-D cases where strided passes
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

Status: implemented for pure interpolation, projection, and antialiasing.

The original native code processed one 1-D line at a time. But all lines for an
axis share the same plan and weights. The batched C++ kernel now processes a
block of lines together:

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

Implementation status:

- Feature flag: `LSRESIZE_BATCHED_AXIS=1`.
- Batch size override: `LSRESIZE_BATCH_LINES=<positive integer>`.
- Default routing: `LSRESIZE_BATCHED_AXIS` unset means conservative `auto`.
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

Status: implemented inside the batched native path.

Most output samples are far from mirrored boundaries. The hot loop should not
pay boundary-extension costs for those samples.

Implemented structure:

- Fast interior kernel:
  - no mirror handling
  - no extension buffer
  - direct weighted reads
  - contiguous row runs precomputed in `Plan1D`
- Slow boundary kernel:
  - exact mirror or antisymmetric boundary behavior
  - cached per-weight source/sign mapping in `Plan1D`

For large images, most rows hit the fast interior path. The current
implementation keeps the row-run metadata in the private plan cache; an explicit
public `ResizePlan` could reuse the same metadata across user-managed workloads.

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

1. Add specialized native preset kernels.
2. Add opt-in float32 internal mode.
3. Expose an explicit reusable native `ResizePlan` if repeated same-shape
   workloads are a priority.
4. Explore tiled/fused memory strategies for large N-D workloads.
5. Revisit the auto-routing/thread heuristics with artifacts from additional
   target CPUs.

The batched native axis kernel and interior/boundary split are now implemented.
The next highest-upside exact redesign is method-specialized native kernels,
followed by an opt-in float32 internal mode for image-oriented workloads.
