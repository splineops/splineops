# SplineOps

**Projection-based antialiased resizing for N-D scientific arrays.**

SplineOps is designed for precise, repeatable downsampling of regular-grid
2-D images and 3-D volumes. It combines explicit coordinate semantics, a
readable Python reference implementation, a native CPU backend, and reusable
plans for fixed-geometry workloads.

## Installation

SplineOps requires Python 3.11 or newer:

```shell
python -m pip install splineops
```

Published wheels include the native resize extension on supported platforms.

## Quick start

```python
import numpy as np
from splineops import resize

volume = np.random.default_rng(0).random((64, 192, 192), dtype=np.float32)
coarse = resize(
    volume,
    output_size=(32, 96, 96),
    method="cubic-antialiasing",
)
```

The `*-antialiasing` methods project the input spline onto a coarser spline
space instead of treating downsampling as interpolation alone. For repeated
arrays with the same geometry, `ResizePlan` reuses setup and workspace state.

## When to use SplineOps

SplineOps is a good fit when:

- continuous-valued 2-D or 3-D NumPy data must be downsampled;
- coordinate, boundary, and output-shape semantics must be explicit;
- aliasing matters; or
- many arrays share one resize geometry.

Another tool may be a better fit when:

- differentiable GPU execution is required;
- physical-space image metadata must be managed automatically;
- ordinary display-image scaling is the only goal; or
- categorical labels are being resampled, where nearest-neighbour semantics
  are normally appropriate.

## Public maturity

| Module | Status | Public position |
| --- | --- | --- |
| `resize`, `ResizePlan` | Stable | Native N-D interpolation and projection antialiasing with a Python reference path |
| `TensorSpline` | Stabilizing | Continuous tensor-product models evaluated at arbitrary coordinates |
| Affine, differentials, smoothing, regression, pyramids, wavelets | Experimental | Available for research while their contracts and reference coverage mature |

Experimental describes API and validation maturity, not the importance of the
underlying methods. See the [project status](https://splineops.github.io/project-status.html)
for exact evidence and remaining graduation work.

## Numerical and performance position

SplineOps does not claim to be the fastest generic 2-D image resizer. OpenCV,
PyTorch, or other libraries can be faster when their coordinate, boundary,
kernel, and antialiasing conventions are acceptable.

SplineOps is strongest when the spline model matters: defined grids and
boundaries, N-D projection antialiasing, native/reference parity, and repeated
fixed-geometry execution. Cross-library reports include semantic differences
as well as timings; numerical difference from SplineOps is not treated as an
independent accuracy metric.

The flagship [SELMA3D microvessel validation](https://splineops.github.io/selma3d-vessels-study.html)
uses 18 held-out expert-labelled 3-D microscopy patches. For the tested lateral
2x coarsening, SplineOps had higher mean vessel-ranking ROC AUC than the named
SciPy, scikit-image, and PyTorch pipelines under a predeclared family-wise rule.
It was 12.49x, 15.17x, and 1.36x faster, respectively, on the recorded
one-thread CPU run. This is narrow preprocessing superiority—not segmentation
or universal resampling superiority.

Explore that result interactively with the
[installed SELMA3D napari comparison demo](https://splineops.github.io/selma3d-demo.html).
It downloads one
checksum-verified public patch, runs the named methods locally, and provides a
curtain slider over their outputs and expert vessel labels.
The default demo extra avoids PyTorch; install `selma3d-demo-all` when the
PyTorch-area comparison is required.

The exact approved public wording and its limits are maintained in
[Claims and evidence](https://splineops.github.io/claims.html).

The [3-D microscopy case study](https://splineops.github.io/cells3d-study.html)
tests that position on a public two-channel volume. It publishes the input
checksum, code, raw results, local timings, and the limits of each metric.

The stricter [BBBC050 segmentation validation](https://splineops.github.io/bbbc050-study.html)
uses manual labels and embryo-level held-out evaluation. SplineOps was fast and
competitive, but the predeclared quality-superiority criterion was not met.

The [controlled 3-D spectral-coarsening validation](https://splineops.github.io/wavefield3d-study.html)
uses analytical targets rather than a library output. Its frozen criterion
passed against six generic resize alternatives: projection reduced mean NRMSE
by 54% versus the nearest frozen baseline over 72 smooth field cases. A post-hoc
polyphase FIR audit was far more accurate, so this is not evidence of broad
scientific-resampling or downstream-task superiority.

## Backends

- NumPy is supported across the package.
- The native C++ backend accelerates resize on CPU.
- CuPy interoperability is experimental and limited to parts of `TensorSpline`.

Set `SPLINEOPS_ACCEL=never` to force the Python resize reference or
`SPLINEOPS_ACCEL=always` to require the native extension. See the
[backend contract](https://splineops.github.io/backend-support.html) for exact
scope.

## Documentation

- [Getting started](https://splineops.github.io/getting-started.html)
- [Modules](https://splineops.github.io/modules.html)
- [Examples](https://splineops.github.io/examples.html)
- [Claims and evidence](https://splineops.github.io/claims.html)
- [Resize guide](https://splineops.github.io/user-guide/02_resize.html)
- [Volume downsampling tutorial](https://splineops.github.io/volume-downsampling.html)
- [SELMA3D microvessel validation](https://splineops.github.io/selma3d-vessels-study.html)
- [Interactive SELMA3D napari demo](https://splineops.github.io/selma3d-demo.html)
- [Controlled 3-D spectral-coarsening validation](https://splineops.github.io/wavefield3d-study.html)
- [3-D microscopy case study](https://splineops.github.io/cells3d-study.html)
- [BBBC050 segmentation validation](https://splineops.github.io/bbbc050-study.html)
- [Performance evidence](https://splineops.github.io/performance.html)
- [API reference](https://splineops.github.io/api/index.html)
- [Provenance](https://splineops.github.io/provenance.html)
- [Changelog](https://github.com/splineops/splineops/blob/main/CHANGELOG.md)

SplineOps modernizes spline methods developed across the Biomedical Imaging
Group at EPFL and its collaborators. Method citations, implementation history,
and source provenance are recorded explicitly.

## Development

```shell
git clone https://github.com/splineops/splineops.git
cd splineops
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Reproducible resize evidence commands include:

```shell
python scripts/benchmark_resize_pr.py --profile smoke --output-dir /tmp/splineops-smoke
python scripts/benchmark_resize_native.py --backend both --output-csv /tmp/splineops-native.csv
python scripts/benchmark_resize_libraries.py --output-csv /tmp/splineops-libraries.csv
```
