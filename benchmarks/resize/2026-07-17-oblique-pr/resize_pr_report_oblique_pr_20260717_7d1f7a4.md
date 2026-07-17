# Oblique Antialiasing Resize PR Benchmark Report

## Artifacts

| Kind | Path |
| --- | --- |
| Native/Python | `benchmarks/resize/2026-07-17-oblique-pr/resize_native_full_oblique_pr_20260717_7d1f7a4.csv` |
| Libraries | `benchmarks/resize/2026-07-17-oblique-pr/resize_libraries_full_oblique_pr_20260717_7d1f7a4.csv` |
| ResizePlan | `benchmarks/resize/2026-07-17-oblique-pr/resize_plan_standard_oblique_pr_20260717_7d1f7a4.csv` |
| Projection methods | `benchmarks/resize/2026-07-17-oblique-pr/resize_projection_methods_standard_oblique_pr_20260717_7d1f7a4.csv` |

## Native vs Python Fallback

Artifact: `benchmarks/resize/2026-07-17-oblique-pr/resize_native_full_oblique_pr_20260717_7d1f7a4.csv`

Speedup is computed as Python fallback median divided by the best native median for each case.

| Scope | Cases | Median | Mean | Min | Max |
| --- | --- | --- | --- | --- | --- |
| All native/Python overlaps | 46 | `53.63x` | `62.78x` | `9.32x` | `148.37x` |
| Linear interpolation | 17 | `62.13x` | `61.56x` | `9.32x` | `148.37x` |
| Cubic interpolation | 16 | `72.21x` | `78.74x` | `31.80x` | `139.76x` |
| Oblique antialiasing presets | 13 | `47.42x` | `44.72x` | `14.96x` | `81.72x` |
| 3-D | 12 | `66.08x` | `68.38x` | `23.72x` | `139.76x` |
| 3-D oblique antialiasing | 3 | `50.72x` | `48.38x` | `23.72x` | `70.71x` |

Best native thread setting by case:

| Thread setting | Winning cases |
| --- | --- |
| `1` | 1 |
| `8` | 33 |
| `default` | 12 |

## Cross-Library Comparison

Artifact: `benchmarks/resize/2026-07-17-oblique-pr/resize_libraries_full_oblique_pr_20260717_7d1f7a4.csv`

Here, speed values greater than `1.0x` mean the other backend was faster than splineops.

| Backend | Comparable cases | Faster than splineops | Median speed | Mean speed | Median rel-L2 |
| --- | --- | --- | --- | --- | --- |
| scipy | 23 | 0/23 | `0.13x` | `0.18x` | `6.70e-08` |
| skimage | 23 | 0/23 | `0.11x` | `0.13x` | `2.37e-01` |
| opencv | 18 | 18/18 | `1.30x` | `1.87x` | `2.60e-01` |
| torch | 19 | 7/19 | `0.80x` | `0.94x` | `1.59e-05` |

Exact-ish rows use `rel_l2_diff < 1e-05`.

| Backend | Exact-ish cases | Faster than splineops | Median speed | Mean speed |
| --- | --- | --- | --- | --- |
| scipy | 16 | 0/16 | `0.10x` | `0.13x` |
| torch | 8 | 0/8 | `0.67x` | `0.61x` |

Oblique antialiasing rows use splineops `*-antialiasing` presets. Other libraries are contextual baselines here, not exact oblique projection implementations.

| Backend | Oblique cases | Faster than splineops | Median speed | Mean speed | Median rel-L2 |
| --- | --- | --- | --- | --- | --- |
| scipy | 7 | 0/7 | `0.17x` | `0.31x` | `3.71e-01` |
| skimage | 7 | 0/7 | `0.12x` | `0.14x` | `1.69e-01` |
| opencv | 5 | 5/5 | `2.14x` | `2.48x` | `1.50e-01` |
| torch | 5 | 5/5 | `1.34x` | `1.46x` | `1.93e-01` |

## ResizePlan Reuse

Artifact: `benchmarks/resize/2026-07-17-oblique-pr/resize_plan_standard_oblique_pr_20260717_7d1f7a4.csv`

| Mode | Cases | Median speedup | Mean speedup |
| --- | --- | --- | --- |
| Plan | 11 | `1.201x` | `1.284x` |
| Plan with output | 11 | `1.071x` | `1.301x` |

Per case:

| Case | Method | Dtype | Plan speedup | Plan output speedup | Max abs diff |
| --- | --- | --- | --- | --- | --- |
| 2d_cubic_down_f32 | `cubic` | `float32` | `0.871x` | `0.954x` | `0.00e+00` |
| 2d_cubic_aa_down_f32 | `cubic-antialiasing` | `float32` | `1.054x` | `1.063x` | `0.00e+00` |
| 2d_linear_down_f32 | `linear` | `float32` | `1.937x` | `2.327x` | `0.00e+00` |
| 2d_linear_aniso_f32 | `linear` | `float32` | `2.107x` | `2.228x` | `0.00e+00` |
| 2d_cubic_aniso_f32 | `cubic` | `float32` | `1.247x` | `1.278x` | `0.00e+00` |
| 2d_linear_aa_down_f32 | `linear-antialiasing` | `float32` | `1.098x` | `1.071x` | `0.00e+00` |
| 2d_cubic_down_f64 | `cubic` | `float64` | `1.055x` | `0.915x` | `0.00e+00` |
| 3d_cubic_aniso_f32 | `cubic` | `float32` | `1.044x` | `1.054x` | `0.00e+00` |
| 3d_linear_down_f32 | `linear` | `float32` | `1.261x` | `0.845x` | `0.00e+00` |
| 3d_linear_two_axis01_f32 | `linear` | `float32` | `1.201x` | `1.261x` | `0.00e+00` |
| 3d_cubic_aa_down_f32 | `cubic-antialiasing` | `float32` | `1.244x` | `1.309x` | `0.00e+00` |

## Projection Method Comparison

Artifact: `benchmarks/resize/2026-07-17-oblique-pr/resize_projection_methods_standard_oblique_pr_20260717_7d1f7a4.csv`

This compares splineops equal-degree least-squares projection against the oblique `*-antialiasing` method family.

| Degree | Cases | Oblique faster | Median oblique speedup | PSNR wins LS/Oblique/Interp | SSIM oblique wins |
| --- | --- | --- | --- | --- | --- |
| `1` | 36 | 0/36 | `0.82x` | 24/12/0 | 20/36 |
| `3` | 36 | 32/36 | `1.20x` | 30/6/0 | 19/36 |

Interpretation: this artifact reports measured relative speed and round-trip quality; it does not establish a numerical-stability hierarchy. Equal-degree least-squares is the orthogonal projection control, while oblique is the public quality-cost preset family. On the public zero-shift grid, analysis degree one or greater uses stable direct compact cross-Gram rows and analysis degree zero retains the finite-difference form.

## PR Interpretation

- Use the native/Python section to justify same-algorithm acceleration.
- Treat splineops `*-antialiasing` rows as oblique projection presets; do not describe them as equal-degree least-squares defaults.
- Use exact-ish SciPy or PyTorch rows from the library comparison when arguing about like-for-like semantics.
- Treat OpenCV, scikit-image and non-exact PyTorch rows as contextual image-resize baselines, because they can use different coordinate, boundary, antialiasing and dtype behavior.
- Use A/B sections to defend individual default-on knobs and to identify rows that need another pass before an upstream PR.
