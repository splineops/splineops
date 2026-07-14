# splineops/src/splineops/resize/_pycore/engine.py
from __future__ import annotations
import operator
import numpy as np
from typing import Sequence
from .params import LSParams
from .resize_nd import resize_along_axis
from .utils import calculate_output_size_1d


def _normalize_axes(axes: Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    normalized: list[int] = []
    seen: set[int] = set()
    for value in axes:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("axes entries must be integers")
        try:
            axis = operator.index(value)
        except TypeError as exc:
            raise TypeError("axes entries must be integers") from exc
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {value} is out of range for {ndim}-D input")
        if axis in seen:
            raise ValueError("axes entries must be unique")
        seen.add(axis)
        normalized.append(int(axis))
    return tuple(normalized)

def compute_zoom(
    input_img: np.ndarray,
    output_img: np.ndarray,
    analy_degree: int,
    synthe_degree: int,
    interp_degree: int,
    zoom_factors: Sequence[float],
    shifts: Sequence[float],
    axes: Sequence[int] | None = None,
) -> None:
    """
    Apply per-axis resize using the explicit (interp_degree, analy_degree,
    synthe_degree) triple along each axis.

    This function does not modify the degrees based on zoom: if you request
    a projection (analy_degree >= 0), it is applied for both down-sampling
    and magnification. A same-grid, zero-shift pass is skipped whenever the
    interpolation and synthesis spaces match, including projection methods.
    """
    img = np.asarray(input_img, dtype=np.float64, order="C")
    out = img
    selected_axes = _normalize_axes(axes, out.ndim)
    for ax in selected_axes:
        z = zoom_factors[ax]
        b = shifts[ax]
        p = LSParams(
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            zoom=float(z),
            shift=float(b),
        )

        target_length = calculate_output_size_1d(int(out.shape[ax]), p.zoom)
        if (
            target_length == out.shape[ax]
            and abs(p.shift) <= 1e-15
            and p.synthe_degree == p.interp_degree
        ):
            continue

        out = resize_along_axis(out, ax, p)

    np.copyto(output_img, out)


def python_resize(
    data: np.ndarray,
    zoom_factors: Sequence[float],
    *,
    interp_degree: int,
    analy_degree: int,
    synthe_degree: int,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Pure-Python fallback for :func:`resize_degrees`, with dtype-preserving
    behavior for floats.

    The behavior is fully determined by the three degrees:

      - ``interp_degree`` : interpolation spline degree (0..3)
      - ``analy_degree``  : analysis spline degree (-1..3, -1 = no projection)
      - ``synthe_degree`` : synthesis spline degree (0..3)

    - Input float32 -> internal float64 -> output float32
    - Every other supported real dtype -> internal/output float64
    """
    # Normalize input and remember original dtype
    arr = np.asarray(data, order="C")
    input_dtype = arr.dtype

    # Work with the actual array shape (not necessarily data.shape if it was array-like)
    zoom_factors = [float(z) for z in zoom_factors]
    if len(zoom_factors) != arr.ndim:
        raise ValueError("zoom_factors length must match input rank")
    selected_axes = _normalize_axes(axes, arr.ndim)
    output_shape = list(arr.shape)
    for axis in selected_axes:
        output_shape[axis] = calculate_output_size_1d(
            int(arr.shape[axis]), zoom_factors[axis]
        )

    # Internal buffers are always float64
    img64 = np.asarray(arr, dtype=np.float64, order="C")
    out64 = np.empty(tuple(output_shape), dtype=np.float64)

    # Zero shifts on all axes (centered, no user offset)
    shifts = [0.0] * len(zoom_factors)

    compute_zoom(
        img64,
        out64,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
        interp_degree=interp_degree,
        zoom_factors=zoom_factors,
        shifts=shifts,
        axes=selected_axes,
    )

    # Match the native storage policy: preserve float32 exactly; all other
    # accepted real input dtypes produce float64 unless the caller casts the
    # public result through ``output=``.
    if input_dtype == np.dtype(np.float32):
        return out64.astype(np.float32, copy=False)
    return out64
