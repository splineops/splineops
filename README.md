# splineops: Spline signal processing

## Description

Spline signal processing in N-D with support for GPU computing.

## Installation

Install minimal dependencies in a dedicated environment
(shown here using [Mamba](https://mamba.readthedocs.io/en/latest/)).

First activate you environment

```shell
mamba activate <env-name>
```

Minimal requirements:

```shell
mamba install numpy scipy
```

Simply install `splineops` from its wheel using `pip`.
*IMPORTANT:*
Not yet uploaded on pypi or anaconda/mamba.
A wheel is needed and can be obtained from the source (see Packaging below).

```shell
pip install splineops
```

To run the examples, `matplotlib` will also be required.

```shell
mamba install matplotlib 
```

## Formatting, type checking, and testing

Formatting and type checking is performed using the following commands.

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
mamba install cupy numpy scipy black mypy tox hatch pytest matplotlib
```

Install `splineops` in editable mode

```shell
pip install -e .
```

If a specific CUDA version is required

```shell
mamba install cupy cuda-version=12.3
```

Potential other CuPy libraries
([CuPy from Conda-Forge](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge))

```shell
mamba install cupy cutensor cudnn nccl
```
