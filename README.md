<!-- splineops/README.md -->

# SplineOps: Spline Operations

`splineops` is a Python and C++-based N-dimensional signal-processing library with support for GPU computing.

## Installation

You need at least `Python 3.10` to install `splineops` (ideally `Python 3.12`). `Python 3.11` is also compatible.

Create and activate your Python virtual environment (on Unix or MacOS)

```shell
python -m venv splineops-env
source splineops-env/bin/activate
```

On Windows,

```shell
python -m venv splineops-env
.splineops-env/Scripts/Activate
```

To deactivate the environment use

```shell
deactivate
```

Minimal requirement:

```shell
pip install numpy scipy matplotlib
```

Simply install `splineops` using `pip`

```shell
pip install splineops
```

## Formatting, Type Checking, and Testing

Formatting and type checking is performed as

```shell
tox -e format
tox -e type
```

The testing requires a valid environment with a supported Python version and `tox`
installed. The tests are run with the following command (automatic pick of the
Python version)

```shell
tox
```

The tests can also be launched for a specific Python version (must match the one
installed in the active environment)

```shell
tox -e py310
tox -e py311
tox -e py312
```

*IMPORTANT:* Since CI is not implemented, make sure to run, pass, and/or fix
`tox -e format`, `tox -e type`, and `tox`.

## Packaging

Using `tox` (preferred)

```shell
tox -e build
```

Using `hatch`

```shell
hatch build -t wheel
```

## Development Environment

Install `splineops` development environment in editable mode

```shell
pip install -e .[dev]
```

## GPU Compatibility

You can benefit of `cupy` to deploy `splineops`. If a specific CUDA version is required, do

```shell
pip install cupy cuda-version=12.3
```

Install `splineops` cupy development environment in editable mode

```shell
pip install -e .[dev_cupy]
```

Potential other CuPy libraries
([CuPy from Conda-Forge](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge))

```shell
pip install cupy cutensor cudnn nccl
```

## Building of the Documentation

To build the Sphinx documentation, install `splineops` doc dependencies

```shell
pip install -e .[docs]
```

Navigate to the `docs` directory and run the `make html` command

```shell
cd docs
make html
```

Then, go to `docs/_build/html` and open `index.html` to navigate the
documentation locally.

### Troubleshooting

If you want to make a "clean" build, go to `docs` and manually delete the folders `_build`, `auto_examples`, `gen_modules`, and the file `sg_execution_times.rst`.
Why isn't this done automatically? Because Sphinx optimizes speed and removes redundant tasks, by not re-creating the examples notebooks if they have already been created.
If you, for example, modify the name of the examples files, you will have to delete at least the folder `auto_examples`. Otherwise, the old examples files will not have disappeared automatically and Sphinx will raise an internal warning referring to a toctree.