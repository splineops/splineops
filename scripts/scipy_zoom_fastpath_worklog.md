# SciPy `ndimage.zoom` Fast-Path Work Log

Date: 2026-06-19

This note records the SciPy-side investigation and prototype work started from
the splineops resize audit. It is intentionally practical: what was measured,
what was changed locally, and what remains before an upstream SciPy PR.

## Local Checkouts

- splineops branch: `feature/publication`
- SciPy checkout: `/home/pablo/Documents/scipy`
- SciPy branch: `feature/ndimage-zoom-fastpaths`
- SciPy base commit: `fd362f891a`
- SciPy benchmark commit: `dda79c011c BENCH: add ndimage zoom benchmark cases`
- SciPy implementation commit: `338af31deb ENH: add ndimage zoom fast paths`
- Draft SciPy PR: https://github.com/scipy/scipy/pull/25441

## Current PR Status

Draft PR 25441 is open against SciPy `main`. The PR body includes the full ASV
comparison table, validation commands and the proposed review split. At the
time this note was updated, there were no maintainer comments yet and CircleCI
was still pending.

## Why SciPy Was Slower

The gap was not Python overhead. SciPy's public `ndimage.zoom` already calls
compiled C code:

- Python validation and shape handling: `scipy/ndimage/_interpolation.py`
- geometric interpolation loop: `scipy/ndimage/src/ni_interpolation.c`
- spline prefilter recursion: `scipy/ndimage/src/ni_splines.c`

The main generic costs found in SciPy were:

- per-axis offset and weight allocations,
- generic output-point iterator bookkeeping,
- edge pointer checks inside every output element,
- per-output generic `filter_size = (order + 1) ** rank` accumulation,
- line-buffer copy overhead in cubic `spline_filter1d`.

The first upstreamable target is therefore not a new API. It is a set of
guarded fast paths that preserve existing semantics and fall back to the old
generic code whenever a case is unsupported.

## SciPy Prototype Scope

Implemented in the local SciPy branch:

1. ASV benchmark cases added to `benchmarks/benchmarks/ndimage_interpolation.py`.
2. `ndimage.zoom` fast path for:
   - order 1 and order 3,
   - rank 1D, 2D and 3D,
   - C-contiguous arrays,
   - `float32` and `float64` input/output combinations covered by the path,
   - `mode='mirror'` and `mode='reflect'`,
   - `grid_mode=False`,
   - no shift.
3. Cubic `spline_filter1d` fast path for:
   - order 3,
   - rank 1D, 2D and 3D,
   - C-contiguous arrays,
   - `float32` or `float64` input,
   - `float64` output,
   - `mode='mirror'` and `mode='reflect'`.
4. Parity tests comparing contiguous fast paths against equivalent strided
   inputs, which force SciPy's existing generic code path.

Unsupported modes, dtypes, layouts, orders and `grid_mode=True` continue to
use SciPy's existing implementation.

## Validation So Far

On the SciPy branch after the prototype:

- build passed with `spin build`
- focused zoom fast-path tests passed
- focused spline-filter fast-path tests passed
- `scipy/ndimage/tests/test_splines.py` plus
  `scipy/ndimage/tests/test_interpolation.py`: `1085 passed`
- full `scipy/ndimage`: `5439 passed, 12 skipped, 1 xpassed`
- `git diff --check`: clean

## Measured Impact

Initial splineops audit against SciPy `main` showed:

- 35 rows audited
- 10 same-semantics candidate rows
- candidate median potential speedup: about `5.65x`
- candidate mean potential speedup: about `7.07x`

Representative SciPy ASV timings before the prototype:

| Case | SciPy `main` |
| --- | ---: |
| `2d_linear_aniso_random_f32` | `1.27 ms` |
| `2d_linear_up_sinusoid_f64` | `1.69 ms` |
| `3d_linear_down_random_f32` | `1.20 ms` |
| `2d_linear_down_random_f32` | `218 us` |
| `2d_cubic_up_sinusoid_f32` | `5.89 ms` |
| `2d_cubic_down_random_f32` | `1.50 ms` |
| `2d_cubic_down_random_f64` | `1.94 ms` |
| `3d_cubic_down_random_f32` | `6.75 ms` |

Representative SciPy branch ASV timings after the zoom and prefilter fast
paths:

| Case | SciPy branch |
| --- | ---: |
| `2d_linear_aniso_random_f32` | `180 us` |
| `2d_linear_up_sinusoid_f64` | `232 us` |
| `3d_linear_down_random_f32` | `173 us` |
| `2d_linear_down_random_f32` | `46.3 us` |
| `2d_cubic_up_sinusoid_f32` | `1.53 ms` |
| `2d_cubic_down_random_f32` | `632 us` |
| `2d_cubic_down_random_f64` | `1.30 ms` |
| `3d_cubic_down_random_f32` | `2.87 ms` |

ASV `continuous` comparison against SciPy `main` after cleanup:

| Case | Before `main` | After branch | Ratio |
| --- | ---: | ---: | ---: |
| `2d_cubic_down_random_f64` | `1.80 ms` | `1.22 ms` | `0.68` |
| `2d_cubic_down_random_f32` | `1.09 ms` | `578 us` | `0.53` |
| `3d_cubic_down_random_f32` | `6.28 ms` | `2.50 ms` | `0.40` |
| `2d_linear_down_random_f32` | `198 us` | `45.3 us` | `0.23` |
| `2d_cubic_up_sinusoid_f32` | `5.11 ms` | `1.08 ms` | `0.21` |
| `2d_linear_aniso_random_f32` | `1.06 ms` | `183 us` | `0.17` |
| `3d_linear_down_random_f32` | `1.05 ms` | `171 us` | `0.16` |
| `2d_linear_up_sinusoid_f64` | `1.65 ms` | `247 us` | `0.15` |

ASV reported: `SOME BENCHMARKS HAVE CHANGED SIGNIFICANTLY. PERFORMANCE
INCREASED.`

Latest splineops audit against the patched SciPy build:

- candidate median potential speedup: about `2.60x`
- candidate mean potential speedup: about `2.93x`
- latest report:
  `/tmp/splineops_scipy_zoom_audit_scipy_fastpath_zoom_prefilter_20260619_155659/scipy_zoom_audit_report_scipy_fastpath_zoom_prefilter_20260619_155659.md`

The remaining gap is mostly in cubic cases, especially 3D cubic. Split timings
show that after the interpolation fast path, cubic prefiltering and memory
traffic are significant remaining costs.

## Upstream PR Shape

Recommended SciPy PR framing:

- no public API change,
- no change in output shape, coordinate mapping or boundary behavior,
- benchmarks first,
- guarded fast paths second,
- explicit fallback for all unsupported cases,
- parity tests against the generic implementation.

The SciPy branch has been split into two local commits to make review easier:

1. benchmark cases,
2. implementation plus tests.

## Useful Commands

From `/home/pablo/Documents/scipy`:

```bash
PATH=/home/pablo/Documents/scipy/venv/bin:$PATH venv/bin/spin build
PATH=/home/pablo/Documents/scipy/venv/bin:$PATH venv/bin/spin test --no-build scipy/ndimage -- -q
PATH=/home/pablo/Documents/scipy/venv/bin:$PATH venv/bin/spin bench -t ndimage_interpolation.NdimageZoom --no-build
PATH=/home/pablo/Documents/scipy/venv/bin:$PATH venv/bin/spin bench -t ndimage_interpolation.NdimageZoom --compare --no-dry-run
```

The plain `--compare` form failed locally because `spin` passed `--dry-run` to
`asv continuous`, and the installed ASV version rejected that argument. The
working compare command is the `--no-dry-run` form above.

## Remaining Work

- Monitor PR feedback and CircleCI.
- Decide whether maintainers prefer one PR or a sequence:
  benchmark-only, order-1 zoom, then order-3 zoom/prefilter.
- Consider a deeper `spline_filter1d` redesign only after the first PR is
  discussed, because that is a separate review surface.
- Consider extending the fast paths to `grid_mode=True`, more modes and more
  layouts after the narrow path is accepted.
