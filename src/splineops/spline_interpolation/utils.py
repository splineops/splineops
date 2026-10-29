# splineops/src/splineops/spline_interpolation/utils.py

import numpy as np

# TODO(dperdios): use scipy-stubs (https://github.com/microsoft/python-type-stubs)
import scipy  # type: ignore
from functools import lru_cache
from numpy import typing as npt
from typing import Optional, Tuple


def _compute_ck_zero_matrix_banded_v1(
    bk: npt.ArrayLike, fk: npt.ArrayLike
) -> npt.NDArray:
    # Note: fk must be of shape (M,) or (M, K)
    fk = np.asarray(fk)
    dtype = fk.dtype
    bk = np.asarray(bk)  # TODO(dperdios): case in fk.dtype?

    # Extract sizes
    # m: (support - 1) // 2
    m = bk.size // 2
    n = fk.shape[0]

    # Pad data
    # Note: always returns a newly created array (even for m=0)
    pad_width: Tuple[Tuple[int, int], ...]
    if fk.ndim == 1:
        pad_width = ((m, m),)
    elif fk.ndim == 2:
        pad_width = (m, m), (0, 0)
    else:
        raise ValueError("Cannot be more than 2-D array.")
    fk_pad = np.pad(fk, pad_width=pad_width)

    # Create banded system
    diag_size = n + 2 * m
    diag_width = 2 * m + 1
    ab_shape = diag_width, diag_size
    ab = np.zeros(ab_shape, dtype=dtype)
    #   Initialize main diagonal (before filling basis coefficients)
    ab[m] = -1
    #   Fill basis coefficients. Note: upper diagonal must be pre-padded for
    #   `scipy.linalg.solve_banded`.
    for a, b in zip(ab, np.flip(bk)):
        a[m : m + n] = b

    # Solve banded system
    # cc = scipy.linalg.solve_banded((m, m), ab, data_pad)
    ck = scipy.linalg.solve_banded(
        l_and_u=(m, m),
        ab=ab,
        b=fk_pad,
        overwrite_ab=True,
        overwrite_b=True,
        check_finite=False,
    )

    return np.copy(ck[m:-m])


def _data_to_coeffs(
    data: np.ndarray,  # N-D array with last dimension to filter
    poles: np.ndarray,  # 1D array
    boundary: str,
    tol: Optional[float] = None,
) -> np.ndarray:
    """In-place pre-filtering"""

    # TODO: this one is only va

    # Tolerance (if not provided, defaults to data-type tolerance)
    if tol is None:
        # Note: not safe to check data.dtype directly as it may be complex
        tol = np.spacing(1, dtype=data.real.dtype)

    # Flatten data-view except last dimension (on which interpolation occurs)
    data_len = data.shape[-1]
    data_it = data.reshape(-1, data_len)

    # Make sure `poles` is a numpy array
    poles = np.asarray(poles)

    # Compute and apply overall gain
    gain = np.prod((1 - poles) * (1 - 1 / poles))
    data_it *= gain

    for pole in poles:
        # Causal initialization
        data_it[:, 0] = _init_causal_coeff(
            data=data_it, pole=pole, boundary=boundary, tol=tol
        )

        # Causal recursion
        for n in range(1, data_len):
            data_it[:, n] += pole * data_it[:, n - 1]

        # Anti-causal initialization
        data_it[:, -1] = _init_anticausal_coeff(
            data=data_it, pole=pole, boundary=boundary
        )

        # Anti-causal recursion
        for n in reversed(range(0, data_len - 1)):
            data_it[:, n] = pole * (data_it[:, n + 1] - data_it[:, n])  # w/ gain
            # data_it[:, n] = pole * data_it[:, n + 1] + (1 - pole) ** 2 * data_it[:, n]  # w/o gain

    return data


def _init_causal_coeff(
    data: np.ndarray, pole: float, boundary: str, tol: float
) -> np.ndarray:
    """First coefficient of the causal filter"""

    # Pre-computations
    data_len = data.shape[-1]
    horizon = data_len
    if tol > 0:
        horizon = int(np.ceil(np.log(tol)) / np.log(np.abs(pole)))

    if boundary.lower() == "mirror":
        zn = float(pole)  # copy to hold power (not sure float() is required)
        if horizon < data_len:
            # Approximation (accelerated loop)
            c0 = data[:, 0]
            for n in range(1, horizon):
                c0 += zn * data[:, n]
                zn *= pole
        else:
            # Exact expression (full loop)
            iz = 1 / pole
            z2n = pole ** (data_len - 1)
            c0 = data[:, 0] + z2n * data[:, -1]
            z2n *= z2n * iz
            # for n in range(1, data_len):  # TODO: may have been wrong
            for n in range(1, data_len - 1):
                c0 += (zn + z2n) * data[:, n]
                zn *= pole
                z2n *= iz
            c0 /= 1 - zn**2
    elif boundary.lower() == "zero":
        # TODO(dperdios): not independent of pole number...
        # # zn = 1  # TODO(dperdios): needs attention
        # zn = pole
        # mul = pole * pole / (1 - pole * pole)
        # if horizon < data_len:
        #     # Approximation (accelerated loop)
        #     c0 = data[:, 0]
        #     for n in range(1, horizon):
        #         c0 -= mul * zn * data[:, n]
        #         zn *= pole
        # else:
        #     # Exact expression (full loop)
        #     zN = pole ** (data_len + 1)
        #     c0 = data[:, 0] - data[:, -1] * zN
        #     for n in range(1, data_len - 1):
        #         c0 -= mul * zn * (data[:, n] - zN * data[:, -1 - n])
        #         zn *= pole
        # c0 *= (1 - pole * pole) / (1 - pole ** (2 * data_len + 2))
        raise NotImplementedError("Unknown boundary condition")
    else:
        raise NotImplementedError("Unknown boundary condition")

    return c0


def _init_anticausal_coeff(data: np.ndarray, pole: float, boundary: str):
    """Last coefficient of the anticausal filter"""
    if boundary.lower() == "mirror":
        cn = (pole / (pole * pole - 1)) * (pole * data[:, -2] + data[:, -1])  #  w/ gain
        return cn
    elif boundary.lower() == "zero":
        # TODO(dperdios): not independent of pole number...
        # return -pole * data[:, -1]  # w/ gain
        # # return (1 - pole) ** 2 * data[:, -1]  # w/o gain
        raise NotImplementedError("Unknown boundary condition")
    else:
        raise NotImplementedError("Unknown boundary condition")


@lru_cache(maxsize=1)
def _cupy_array_type():
    """Return CuPy's array type once, or an empty sentinel when unavailable."""
    try:
        import cupy as cp
    except ImportError:
        return ()
    return cp.ndarray


def is_cupy_type(x: object) -> bool:
    """Return whether *x* is a CuPy array without requiring CuPy at runtime."""
    return isinstance(x, _cupy_array_type())


def is_ndarray(x: object) -> bool:
    """Return whether *x* is an explicitly supported dense array type."""
    return isinstance(x, np.ndarray) or is_cupy_type(x)
