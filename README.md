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

## Packaging

Using `tox` (preferred)

```shell
tox -e build
```

Using `hatch`

```shell
hatch build -t wheel
```

## Dependencies (dev)

Fastest way to install dependencies for dev.
**TODO:** would probably be simpler to also use `tox` with `mamba`
(using `tox-conda`).

```shell
mamba install cupy numpy scipy black mypy tox hatch pytest
```

If a specific CUDA version is required
```shell
mamba install cupy cuda-version=12.3
```

Other CuPy libraries
[CuPy from Conda-Force](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

```shell
conda install -c conda-forge cupy cutensor cudnn nccl
```
