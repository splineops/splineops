# Upstream PR Description: Oblique Antialiasing Resize

## Title

Optimize N-D spline resize and promote oblique-projection antialiasing presets

## Summary

This PR optimizes the N-D resize backend and documents the public downsampling
path as the oblique-projection antialiasing method family:

- `linear-antialiasing`
- `quadratic-antialiasing`
- `cubic-antialiasing`

The implementation keeps the existing resize API and spline model, but makes
the native backend substantially faster for interpolation and projection-based
antialiasing workloads. Equal-degree least-squares projection remains available
through `resize_degrees` for advanced/reference use, but it is not promoted as
the routine production downsampling preset.

## Why Oblique Projection

The antialiasing presets use a lower-degree analysis spline and a matching
synthesis spline degree. This preserves the public spline model while avoiding
the higher-order finite-difference integration required by equal-degree
least-squares configurations. In practice this makes the production
downsampling path faster and more robust on long lines, especially in N-D and
3-D workloads.

## Implementation Highlights

- Native batched axis execution for common interpolation and projection rows.
- Cached native resize plans and reusable `ResizePlan` execution buffers.
- Fixed-support accumulator specializations for common presets.
- Direct/fused exact linear paths, including selected 2-D/3-D kernels.
- Workload-aware thread and batch policies.
- Conservative precision policy:
  - automatic float32 internals for validated 2-D/3-D float32 pure
    quadratic/cubic interpolation rows,
  - automatic float32 internals for public 3-D downsampling antialiasing
    presets,
  - conservative float64 internals for 2-D projection/antialiasing, mixed
    projection cases, and equal-degree least-squares by default.

## Fresh Benchmark Evidence

Fresh evidence bundle from the current branch:

```shell
python scripts/benchmark_resize_pr.py \
  --profile oblique-pr \
  --output-dir /tmp/splineops_resize_pr_oblique_pr_20260619 \
  --tag oblique_pr_20260619
```

Report:

```text
/tmp/splineops_resize_pr_oblique_pr_20260619/resize_pr_report_oblique_pr_20260619.md
```

Key rows from that bundle:

| Evidence | Result |
| --- | --- |
| Native/Python full report | 46 overlaps, median `24.34x` speedup |
| Oblique antialiasing native/Python bucket | 13 cases, median `19.14x` |
| 3-D oblique antialiasing bucket | 3 cases, median `19.66x` |
| Oblique vs equal-degree LS, degree 1 | faster in `36/36`, median `1.22x` |
| Oblique vs equal-degree LS, degree 3 | faster in `36/36`, median `1.39x` |
| Exact-ish SciPy rows | SciPy faster in `0/16`, median speed `0.08x` |
| Exact-ish PyTorch rows | PyTorch faster in `0/8`, median speed `0.61x` |
| ResizePlan reuse | 11 cases, median `1.083x` speedup |

The clearest library comparison is 3-D cubic antialiasing:

| Case | splineops | SciPy | skimage | OpenCV | Torch |
| --- | ---: | ---: | ---: | --- | --- |
| `3d_cubic_aa_down_random_f32` | `1.926 ms` | `8.291 ms` | `10.341 ms` | skipped, 2-D only | skipped, no 3-D AA |
| `3d_cubic_aa_down_random_f32_large` | `3.007 ms` | `23.750 ms` | `28.773 ms` | skipped, 2-D only | skipped, no 3-D AA |

## Correctness And Compatibility

- Native/Python parity is checked in the test suite and benchmark suite for
  representative interpolation and oblique antialiasing cases.
- Long-line boundedness is covered for cubic oblique projection to guard
  against the drift seen in high-order equal-degree least-squares projection.
- `ResizePlan` reuse preserves bitwise output equality in the plan benchmark.
- The public `resize(..., method="*-antialiasing")` API remains unchanged.
- Explicit `LSRESIZE_PRECISION=float64` keeps the strict float64-internal path
  available for A/B checks.

## Non-Claims

This PR should not claim that splineops is the fastest generic 2-D image resize
library. OpenCV and Torch are faster on some 2-D contextual rows, but those
rows use different coordinate rules, antialiasing filters, boundary behavior,
or dtype handling.

This PR should also not claim that oblique projection has higher PSNR than
equal-degree least-squares in every controlled round-trip benchmark.
Least-squares remains the orthogonal projection reference and wins some small
PSNR rows. The production argument for oblique projection is speed, lower
integration order, long-line robustness, and N-D coverage.

## Validation Checklist

Current local validation:

```text
python -m pytest -q
561 passed in 58.99s

python -m sphinx -b html docs docs/_build/html_noplot -W --keep-going \
  -D sphinx_gallery_conf.plot_gallery=0
build succeeded

python scripts/benchmark_resize_pr.py --profile smoke \
  --output-dir /tmp/splineops_resize_pr_smoke_recheck_20260619
passed

python scripts/benchmark_resize_pr.py --profile oblique-pr \
  --output-dir /tmp/splineops_resize_pr_oblique_pr_20260619 \
  --tag oblique_pr_20260619
passed

git diff --check
clean
```

## Reviewer Notes

The main API story should stay simple: the `*-antialiasing` presets are the
recommended production downsampling methods. `resize_degrees` is the escape
hatch for users who need explicit projection degrees or want to study
least-squares configurations directly.
