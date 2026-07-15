"""Internal cardinal interpolation-coefficient construction."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .bases.spline_basis import SplineBasis
from .modes.extension_mode import ExtensionMode
from .utils import is_cupy_type, is_ndarray


def prefilter_interpolation_coefficients(
    data: npt.NDArray,
    *,
    bases: Sequence[SplineBasis],
    modes: Sequence[ExtensionMode],
    axes: Sequence[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    backend: str | None = None,
) -> npt.NDArray:
    """Return owned cardinal coefficients along explicitly selected axes.

    This private interface makes basis, boundary, axes, dtype, and backend part
    of one contract.  It intentionally does not cover resize's scale-dependent
    projection filters, whose mathematics and boundaries differ.
    """

    if not is_ndarray(data):
        raise TypeError("'data' must be a NumPy or CuPy array.")
    inferred_backend = "cupy" if is_cupy_type(data) else "numpy"
    if backend is None:
        backend = inferred_backend
    if backend not in ("numpy", "cupy"):
        raise ValueError("'backend' must be 'numpy' or 'cupy'.")
    if backend != inferred_backend:
        raise TypeError(
            f"'backend' is {backend!r}, but 'data' uses {inferred_backend!r}."
        )
    xp = np
    if backend == "cupy":
        import cupy as cp

        xp = cp

    target_dtype = np.dtype(data.dtype if dtype is None else dtype)
    if not (
        np.issubdtype(target_dtype, np.floating)
        or np.issubdtype(target_dtype, np.complexfloating)
    ):
        raise TypeError("'dtype' must be a floating or complex-floating dtype.")
    if axes is None:
        axes = tuple(range(data.ndim))
    try:
        axes = tuple(operator.index(axis) for axis in axes)
    except TypeError as exc:
        raise TypeError("'axes' must be a sequence of integer axes.") from exc
    axes = tuple(axis + data.ndim if axis < 0 else axis for axis in axes)
    if any(axis < 0 or axis >= data.ndim for axis in axes) or len(set(axes)) != len(
        axes
    ):
        raise ValueError("'axes' must contain distinct valid axes.")
    bases = tuple(bases)
    modes = tuple(modes)
    if len(bases) != len(axes) or len(modes) != len(axes):
        raise ValueError("'bases' and 'modes' must contain one entry per axis.")
    if not all(isinstance(basis, SplineBasis) for basis in bases):
        raise TypeError("Every entry in 'bases' must be a SplineBasis.")
    if not all(isinstance(mode, ExtensionMode) for mode in modes):
        raise TypeError("Every entry in 'modes' must be an ExtensionMode.")

    source = data.astype(target_dtype, copy=False)
    if not axes:
        return xp.array(source, copy=True, order="C")
    coefficients = None
    for axis, basis, mode in zip(axes, bases, modes):
        current = source if coefficients is None else coefficients
        axis_last = xp.ascontiguousarray(xp.moveaxis(current, axis, -1))
        filtered = mode.compute_coefficients(data=axis_last, basis=basis)
        if filtered.shape != axis_last.shape or filtered.dtype != target_dtype:
            raise RuntimeError(
                "Coefficient prefilter violated its internal shape/dtype contract."
            )
        coefficients = xp.moveaxis(filtered, -1, axis)

    return xp.ascontiguousarray(coefficients)
