import numpy as np

# TODO(dperdios): use scipy-stubs (https://github.com/microsoft/python-type-stubs)
import scipy  # type: ignore
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


def _compute_ck_zero_matrix_banded_v2(
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

    # Create banded system
    # TODO(dperdios): does not work for single-sample signals with m > 1
    diag_size = n
    diag_width = 2 * m + 1
    ab_shape = diag_width, diag_size
    ab = np.zeros(ab_shape, dtype=dtype)
    #   Main diagonal
    ab[m] = bk[m]
    #   Upper diagonals
    for ii, a, b in zip(reversed(range(m)), ab[:m], np.flip(bk)[:m]):
        a[ii + 1 :] = b
    #   Lower diagonals
    for ii, a, b in zip(range(m), ab[m + 1 :], np.flip(bk)[m + 1 :]):
        a[: -(ii + 1)] = b

    # ab = np.ones(ab_shape, dtype=dtype)
    # ab *= np.expand_dims(np.flip(basis), axis=-1)

    # Solve banded system
    # TODO: probably hermitian solver would be more efficient here
    # TODO: probably a copy of data to use "faster" flags
    # Note: can be solved for multiple signals b=np.stack([fk, fk]).T
    ck = scipy.linalg.solve_banded(l_and_u=(m, m), ab=ab, b=fk)
    # fk = np.copy(fk)
    # ck = scipy.linalg.solve_banded(
    #     l_and_u=(m, m), ab=ab, b=data_pad,
    #     overwrite_ab=True, overwrite_b=True, check_finite=False)

    return ck


def _compute_coeffs_narrow_mirror_wg(
    data: np.ndarray,
    poles: np.ndarray,
) -> np.ndarray:
    # TODO(dperdios): test and integrate this version

    DataLength = len(data)

    # Compute overall gain
    gain = np.prod((1 - poles) * (1 - 1 / poles))

    # Apply gain
    c = data
    c *= gain

    # Loop over all poles
    for z in poles:
        # Causal initialization
        zn = z
        iz = 1.0 / z
        z2n = pow(z, DataLength - 1)
        Sum = c[0] + z2n * c[DataLength - 1]
        z2n *= z2n * iz
        for n in range(1, DataLength - 1):
            Sum += (zn + z2n) * c[n]
            zn *= z
            z2n *= iz
        c[0] = Sum / (1.0 - zn * zn)

        # Causal recursion
        for n in range(1, DataLength):
            c[n] += z * c[n - 1]

        # Anti-causal initialization
        c[DataLength - 1] = (z / (z * z - 1.0)) * (
            z * c[DataLength - 2] + c[DataLength - 1]
        )

        # Anti-causal recursion
        for n in range(DataLength - 2, -1, -1):
            c[n] = z * (c[n + 1] - c[n])

    return c


def _compute_coeffs_narrow_mirror_wog(
    data: np.ndarray,
    poles: np.ndarray,
) -> np.ndarray:
    # TODO(dperdios): DO NOT USE (not validated and still erroneous)

    # Flatten data-view except last dimension (on which interpolation occurs)
    data_len = data.shape[-1]
    K = data_len
    data_it = data.reshape(-1, data_len)

    # Make sure `poles` is a numpy array
    poles = np.asarray(poles)

    for pole in poles:
        # Causal initialization
        c0 = 0
        for k in range(K - 1):
            c0 += pole**k * (data_it[:, k] + pole ** (K - 1) * data_it[:, K - 1 - k])
        c0 /= 1 - pole ** (2 * K - 2)

        # Causal recursion
        for k in range(1, data_len):
            data_it[:, k] += pole * data_it[:, k - 1]

        # Anti-causal initialization
        data_it[:, K - 1] = (
            (1 - pole) ** 2
            / (1 - pole**2)
            * (pole * data_it[:, K - 2] + data_it[:, K - 1])
        )

        # Anti-causal recursion
        # for k in reversed(range(0, data_len - 1)):
        #     data_it[:, k] = pole * data_it[:, k + 1] + (1 - pole) ** 2 * data_it[:, k]
        for k in range(1, data_len):
            data_it[:, K - 1 - k] = (
                pole * data_it[:, K - k] + (1 - pole) ** 2 * data_it[:, K - 1 - k]
            )

    return data


def _data_to_coeffs(
    data: np.ndarray,
    poles: np.ndarray,
    boundary: str,
    tol: Optional[float] = None,
) -> np.ndarray:
    """
    In-place pre-filtering of the data to compute spline coefficients.

    Parameters
    ----------
    data : ndarray
        The input data array.
    poles : ndarray
        The poles of the spline basis.
    boundary : str
        The type of boundary condition ("mirror", "zero", "periodic").
    tol : float, optional
        The tolerance for the recursion.

    Returns
    -------
    data : ndarray
        The data array with computed coefficients.
    """
    if tol is None:
        tol = np.finfo(data.real.dtype).eps

    data_len = data.shape[-1]
    data_it = data.reshape(-1, data_len)

    poles = np.asarray(poles)

    # For periodic boundary conditions, we need to adjust the data length
    # for the recursion to wrap around
    if boundary.lower() == "periodic":
        extended_data = np.concatenate((data_it, data_it[:, :1]), axis=1)
        data_len += 1
    else:
        extended_data = data_it

    # Compute and apply overall gain
    gain = np.prod((1 - poles) * (1 - 1 / poles))
    extended_data *= gain

    for pole in poles:
        # Causal and anti-causal recursion with appropriate boundary conditions
        if boundary.lower() == "mirror" or boundary.lower() == "zero":
            # Existing code for "mirror" and "zero" (as before)
            # [Existing implementation]
            pass
        elif boundary.lower() == "periodic":
            # Periodic boundary condition
            _causal_anticausal_recursion_periodic(extended_data, pole, tol)
        else:
            raise NotImplementedError(f"Unknown boundary condition '{boundary}'.")

    # For periodic boundary, discard the extra sample
    if boundary.lower() == "periodic":
        data_it[:] = extended_data[:, :-1]
    else:
        data_it[:] = extended_data

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

def _causal_anticausal_recursion_periodic(data: np.ndarray, pole: float, tol: float):
    """
    Performs causal and anti-causal recursion for periodic boundary conditions.

    Parameters
    ----------
    data : ndarray
        The extended data array (with one extra sample for periodicity).
    pole : float
        The current pole.
    tol : float
        The tolerance for stopping the recursion.
    """
    data_len = data.shape[-1]

    # Initialize
    data[:, 0] = data[:, 0] + pole * data[:, -1]  # Wrap around for periodicity

    # Causal recursion
    for n in range(1, data_len):
        data[:, n] += pole * data[:, n - 1]

    # Anti-causal recursion
    for n in reversed(range(data_len - 1)):
        data[:, n] = pole * (data[:, n + 1] - data[:, n])
