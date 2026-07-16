# SplineOps

SplineOps is a Python and C++ library for precise spline interpolation and
projection-based resizing of data sampled on regular N-dimensional grids. Its
native resize backend is designed for mathematically explicit, repeatable 2-D
and 3-D workloads—including antialiased volumetric downsampling—not merely for
generic image scaling.

The project also develops independent spline-based tools for affine transforms,
differentials, smoothing, sparse regression, pyramids, and wavelets. Those
modules remain available while their numerical contracts and edge cases are
being strengthened; they are not folded into the resize API.

## What is ready today

| Module | Status | Strength |
| --- | --- | --- |
| `resize`, `ResizePlan` | Stable | Native N-D interpolation and projection-based antialiasing with a Python reference path |
| `TensorSpline` | Stabilizing | Continuous tensor-product models with B-spline, O-MOMS, and other bases |
| Affine and differentials | Experimental | General spline-evaluated transforms and 2-D/3-D spline derivatives |
| Smoothing and adaptive regression | Experimental | Fractional smoothing and sparse piecewise-linear models with reusable fixed-geometry plans |
| Pyramids and wavelets | Experimental | Vectorized spline multiscale analysis and reconstruction |

"Experimental" describes API and validation maturity, not the importance of
the underlying methods. See the
[project status](https://splineops.github.io/project-status.html) and
[development roadmap](https://splineops.github.io/roadmap.html) for the exact
graduation criteria.

## Installation

SplineOps requires Python 3.11 or newer:

```shell
python -m pip install splineops
```

The published wheels include the native resize extension on supported
platforms. A pure-Python resize reference implementation remains available for
parity checks and unsupported build environments.

## Resize example

```python
import numpy as np
from splineops import resize

volume = np.random.default_rng(0).random((64, 192, 192), dtype=np.float32)
smaller = resize(
    volume,
    output_size=(32, 96, 96),
    axes=(0, 1, 2),
    method="cubic-antialiasing",
)
```

The `*-antialiasing` methods use spline projection rather than treating
downsampling as interpolation alone. For repeated fixed-geometry workloads,
`splineops.resize.ResizePlan` reuses geometry and workspace state.

## TensorSpline example

`TensorSpline` remains a separate continuous-model abstraction:

```python
import numpy as np
from splineops.spline_interpolation.tensor_spline import TensorSpline

samples = np.array([0.0, 1.0, 0.0, -1.0])
grid = np.arange(samples.size, dtype=np.float64)
spline = TensorSpline(
    data=samples,
    coordinates=(grid,),
    bases="bspline3",
    modes="mirror",
)

values = spline(coordinates=(np.linspace(0.0, 3.0, 31),))
```

For repeated fixed coordinates, `spline.query_plan(...)` can retain support
geometry behind an explicit memory cap and apply it to compatible
`TensorSpline` instances with changing sample values. Resize and `TensorSpline`
share only carefully validated internals where their mathematical contracts
match; they retain distinct APIs, coordinate contracts, and optimized execution
paths. Construction templates can also refit new data or consume explicitly
precomputed coefficients without mutating the original spline.

## Performance position

SplineOps does not claim to be the fastest generic 2-D image resizer. OpenCV or
PyTorch can be faster when their different coordinate, boundary, kernel, and
antialiasing conventions are acceptable.

SplineOps is strongest when the spline model itself matters: explicit degrees,
defined boundaries and sampling grids, N-D projection antialiasing, native and
reference parity, and repeated fixed-geometry workloads. Separable
`TensorSpline` grids, reusable query/affine/smoothing/denoising plans,
multi-output differentials, and whole-axis multiscale operations now avoid
substantial repeated setup or Python dispatch. The benchmark tools report both
runtime and numerical differences so contextual comparisons are not presented
as equivalent algorithms.

Reusable affine coefficients can carry an immutable compatibility tag, and
explicit-axis affine and differential plans execute batch/channel planes
together without changing `TensorSpline`'s independent public model.
Coefficient tags have a validated JSON-plus-numeric persistence path;
`DifferentialPlan` can select only the requested output families and reuse
exact structured output buffers.
The stability-soak examples combine those contracts in persisted registration
fan-out and buffered 3-D feature pipelines while keeping every module public and
independent.

See the
[reusable workflow recipes](https://splineops.github.io/consolidation-recipes.html)
for coefficient sharing across affine geometries, explicit batch/channel axes,
controlled denoising paths, and plan memory inspection.

## Backend support

- NumPy is the supported array backend across the package.
- The native C++ backend accelerates resize on CPU.
- CuPy interoperability exists only in parts of `TensorSpline`. No CuPy
  configuration is currently advertised as supported: zero-boundary
  coefficient filtering can transfer through a CPU solve, and the basis/mode
  matrix has no dedicated GPU continuous-integration coverage. See the
  [exact backend contract](https://splineops.github.io/backend-support.html).

Set `SPLINEOPS_ACCEL=never` to force the Python resize reference path or
`SPLINEOPS_ACCEL=always` to require the native extension.

## Development

```shell
git clone https://github.com/splineops/splineops.git
cd splineops
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Useful resize evidence commands include:

```shell
python scripts/benchmark_resize_pr.py --profile smoke --output-dir /tmp/splineops-smoke
python scripts/benchmark_resize_native.py --backend both --output-csv /tmp/splineops-native.csv
python scripts/benchmark_resize_libraries.py --output-csv /tmp/splineops-libraries.csv
```

Historical optimization and upstream-PR notes under `scripts/` are retained as
engineering records. Files that describe the pre-2.0 projection pipeline are
not the current numerical contract; consult the resize API guide, changelog,
and newly generated benchmark artifacts for current behavior.

## Documentation and provenance

- [Documentation](https://splineops.github.io/)
- [Resize API](https://splineops.github.io/api/02_resize.html)
- [Project status](https://splineops.github.io/project-status.html)
- [Roadmap](https://splineops.github.io/roadmap.html)
- [Provenance inventory](https://splineops.github.io/provenance.html)
- [Changelog](https://github.com/splineops/splineops/blob/main/CHANGELOG.md)

SplineOps builds on decades of spline research and software associated with the
Biomedical Imaging Group at EPFL and its collaborators. The project records
method citations and source provenance explicitly so that this lineage is a
strength users can inspect, reproduce, and credit.
