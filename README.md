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