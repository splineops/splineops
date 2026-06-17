# Upstream PR Description: Oblique Antialiasing Resize

## Title

Optimize N-D spline resize and promote oblique-projection antialiasing presets

## Summary

This PR optimizes the N-D resize backend and documents the public downsampling
path as the oblique-projection antialiasing method family:

- `linear-antialiasing`
- `quadratic-antialiasing`
- `cubic-antialiasing`

The implementation keeps the existing spline API, but makes the native backend
substantially faster for interpolation and projection-based antialiasing cases.
Equal-degree least-squares projection remains available through
`resize_degrees` for advanced/reference use, but it is not the promoted routine
downsampling preset.

## Why Oblique Projection

The antialiasing presets use a lower-degree analysis spline and a matching
synthesis spline degree. This preserves the public spline model while avoiding
the higher-order finite-difference integration required by equal-degree
least-squares configurations. In practice this makes the production
downsampling path faster and more robust on long lines, especially in N-D and
3-D workloads.

## Fresh Benchmark Evidence

Fresh evidence bundle:

```shell
python scripts/benchmark_resize_pr.py \
  --profile oblique-pr \
  --output-dir /tmp/splineops_resize_pr_oblique_pr_6411817_20260617 \
  --tag oblique_pr_6411817_20260617
```

Report:

```text
/tmp/splineops_resize_pr_oblique_pr_6411817_20260617/resize_pr_report_oblique_pr_6411817_20260617.md
```

Key rows from that bundle:

| Evidence | Result |
| --- | --- |
| Native/Python full report | 46 overlaps, median `22.95x` speedup |
| Oblique antialiasing native/Python bucket | 13 cases, median `14.07x` |
| 3-D oblique antialiasing bucket | 3 cases, median `13.01x` |
| Oblique vs equal-degree LS, degree 1 | faster in `36/36`, median `1.25x` |
| Oblique vs equal-degree LS, degree 3 | faster in `36/36`, median `1.41x` |
| Exact-ish SciPy rows | SciPy faster in `0/16`, median speed `0.08x` |

The clearest library comparison is 3-D cubic antialiasing:

| Case | splineops | SciPy | skimage | OpenCV | Torch |
| --- | ---: | ---: | ---: | --- | --- |
| `3d_cubic_aa_down_random_f32` | `2.454 ms` | `8.070 ms` | `10.186 ms` | skipped, 2-D only | skipped, no 3-D AA |
| `3d_cubic_aa_down_random_f32_large` | `4.179 ms` | `23.610 ms` | `28.675 ms` | skipped, 2-D only | skipped, no 3-D AA |

## Correctness And Compatibility

- Native/Python parity is checked in the benchmark suite for representative
  interpolation and oblique antialiasing cases.
- Long-line boundedness is covered for cubic oblique projection to guard
  against the drift seen in high-order equal-degree least-squares projection.
- `ResizePlan` reuse preserves bitwise output equality in the plan benchmark.
- The public `resize(..., method="*-antialiasing")` API remains unchanged.

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

- `python -m py_compile` on benchmark/report scripts
- `python scripts/benchmark_resize_pr.py --profile smoke`
- `python scripts/benchmark_resize_pr.py --profile oblique-pr`
- `python -m pytest -q`
- `python -m build --outdir <tmp>`
- Clean-install smoke from the built wheel
- Docs build, if the docs extra is installed

## Reviewer Notes

The main API story should stay simple: the `*-antialiasing` presets are the
recommended production downsampling methods. `resize_degrees` is the escape
hatch for users who need explicit projection degrees or want to study
least-squares configurations directly.
