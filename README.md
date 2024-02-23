# bssp: B-spline signal processing

## Description

B-spline signal processing in N-D with support for GPU computing.

## Packaging

```shell
hatch build -t wheel 
```

## Dependencies (dev)

```shell
conda install -c conda-forge cupy numpy scipy matplotlib black hatch
conda install -c conda-forge cupy cuda-version=12.2
```

[CuPy from Conda-Force](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

```shell
conda install -c conda-forge cupy cutensor cudnn nccl
```
