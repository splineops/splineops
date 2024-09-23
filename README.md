# SplineOps: Spline Operations

`splineops` is a Python-based N-dimensional signal processing library with
support for GPU computing.

## Installation

You need at least `Python 3.10` to install `splineops`, and ideally `Python 3.12`. `Python 3.11` is also compatible.

Install minimal dependencies in a dedicated environment
(shown here using [Mamba](https://mamba.readthedocs.io/en/latest/)).

Create and activate your environment

```shell
mamba create -n myenv
mamba activate myenv
```

Make sure you have the conda-forge channel added to your conda configuration.
If not, you can add it using

```shell
conda config --add channels conda-forge
```

Minimal requirements:

```shell
mamba install numpy scipy
```

Simply install `splineops` using `pip`.

```shell
pip install splineops
```

To run the examples, `matplotlib`, `pooch` (for built-in image datasets)
and `IPython` (for Python UI widgets) will also be required.

```shell
mamba install matplotlib pooch IPython
```

## Formatting, type checking, and testing

Formatting and type checking is performed using the following commands

```shell
tox -e format
tox -e type
```

Testing requires a valid environment with a supported Python version and `tox`
installed. Tests can be run with the following command (automatic pick of the
Python version).

```shell
tox
```

Tests can also be launched for a specific Python version (must match the one
installed in the active environment)

```shell
tox -e py310
tox -e py311
tox -e py312
```

*IMPORTANT:* Since CI is not implemented, make sure to run, pass and/or fix
`tox -e format`, `tox -e type` and `tox`.

## Packaging

Using `tox` (preferred)

```shell
tox -e build
```

Using `hatch`

```shell
hatch build -t wheel
```

## Development environment

Easiest way to install dev dependencies

```shell
mamba install numpy scipy matplotlib pooch IPython black mypy tox hatch pytest
```

Install `splineops` development environment in editable mode

```shell
pip install -e .[dev]
```

## GPU compatibility

You can use `splineops` with `cupy`. If a specific CUDA version is required do

```shell
mamba install cupy cuda-version=12.3
```

Install `splineops` cupy development environment in editable mode

```shell
pip install -e .[dev_cupy]
```

Potential other CuPy libraries
([CuPy from Conda-Forge](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge))

```shell
mamba install cupy cutensor cudnn nccl
```

## Building the documentation

To build the Sphinx documentation, install `splineops` doc dependencies

```shell
mamba install numpy scipy matplotlib pooch IPython sphinx sphinx-gallery sphinx-prompt sphinx-copybutton sphinx-remove-toctrees pydata-sphinx-theme sphinx-design myst-parser jupyterlite-sphinx jupyterlite-pyodide-kernel
```

Install `splineops` doc environment in editable mode

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

If you want to make a "clean" build, go to `docs` and manually delete the folders `_build`, `auto_examples`, `gen_modules`, `notebooks_jupyterlite` and the file `sg_execution_times.rst`.
Why isn't this done automatically? Because Sphinx optimizes speed and removes redundant tasks, by not re-creating the examples' notebooks if they have already been created.
If you for example modify the name of the examples' files, you will have to delete at least the folder `auto_examples`. Otherwise, the old examples' files will not have disappeared automatically, and Sphinx will raise an internal warning referring to a toctree.