<!-- splineops/README.md -->

# SplineOps: Spline Operations

`SplineOps` is a Python and C++-based N-dimensional signal-processing library with support for GPU computing.

## Installation

You need at least `Python 3.11` to install `SplineOps`.

Create and activate your Python virtual environment (on Unix or MacOS)

```shell
python -m venv splineops-env
source splineops-env/bin/activate
```

On Windows,

```shell
python -m venv splineops-env
./splineops-env/Scripts/Activate
```

To deactivate the environment use

```shell
deactivate
```

Minimal requirement:

```shell
pip install numpy scipy matplotlib
```

Simply install `SplineOps` using `pip`

```shell
pip install splineops
```

## GPU Compatibility

You can benefit of `cupy` to deploy the `Spline Interpolation` module in `SplineOps`. If a specific CUDA version is required, do

```shell
pip install cupy cuda-version=12.3
```

Install cupy development environment in editable mode

```shell
pip install -e .[dev_cupy]
```

Potential other CuPy libraries
([CuPy from Conda-Forge](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge))

```shell
pip install cupy cutensor cudnn nccl
```

## Development Environment

Install development environment in editable mode

```shell
pip install -e .[dev]
```

## Resize Native Backend

The resize module uses the native CPU `_lsresize` extension when it is available.
Set `SPLINEOPS_ACCEL=never` to force the Python fallback, or
`SPLINEOPS_ACCEL=always` to require the native extension during benchmarking.

Useful benchmark entry points:

```shell
python scripts/benchmark_resize_pr.py --profile oblique-pr --output-dir /tmp/splineops_resize_pr_oblique_pr
python scripts/benchmark_resize_pr.py --output-dir /tmp/splineops_resize_pr_current
python scripts/benchmark_resize_projection_methods.py --profile standard --output-csv /tmp/splineops_projection_methods.csv
python scripts/benchmark_resize_native.py --backend both --output-csv /tmp/splineops_native.csv
python scripts/benchmark_resize_libraries.py --output-csv /tmp/splineops_libraries.csv
python scripts/summarize_resize_benchmarks.py report \
  --native /tmp/splineops_native.csv \
  --libraries /tmp/splineops_libraries.csv \
  --output /tmp/splineops_resize_report.md
```

Current resize behavior and numerical policy are documented in the
[resize API guide](https://splineops.github.io/api/02_resize.html) and
[resize user guide](https://splineops.github.io/user-guide/02_resize.html).
Breaking changes and migration guidance are recorded in the
[changelog](https://github.com/splineops/splineops/blob/main/CHANGELOG.md).

The dated PR preparation notes in `scripts/resize_pr_readiness.md`,
`scripts/resize_oblique_pr_brief.md`, and
`scripts/resize_upstream_pr_description.md`, together with the close-out
`scripts/resize_session_handoff.md`, are historical records. They describe the
implementation and benchmark evidence available at the time, not the current
resize contract.
