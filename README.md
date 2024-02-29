# bssp: B-spline signal processing

## Description

B-spline signal processing in N-D with support for GPU computing.

## Formatting, type checking, and testing

Formatting and type checking is performed using the following commands.

```shell
tox -e format
tox -e type
```

Testing with the following one.

```shell
tox
```

*IMPORTANT:* Since not CI is implemented for now, make sure to run and pass/fix
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

Install `bssp` in editable mode

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
