# splineops/src/splineops/resize/_pycore/engine.py
from __future__ import annotations
import numpy as np
from typing import Sequence
from .params import LSParams
from .resize_nd import resize_along_axis

# Numerical epsilon for zoom comparisons
_EPS = 1e-12


def compute_zoom(
    input_img: np.ndarray,
    output_img: np.ndarray,
    analy_degree: int,
    synthe_degree: int,
    interp_degree: int,
    zoom_factors: Sequence[float],
    shifts: Sequence[float],
    inversable: bool,
) -> None:
    """
    Apply per-axis resize using the explicit (interp_degree, analy_degree,
    synthe_degree) triple along each axis.

    This function does not modify the degrees based on zoom: if you request
    a projection (analy_degree >= 0), it is applied for both down-sampling
    and magnification. Identity handling is only enabled for pure interpolation
    (analy_degree < 0 and zoom ≈ 1).
    """
    img = np.asarray(input_img, dtype=np.float64, order="C")
    out = img
    for ax, (z, b) in enumerate(zip(zoom_factors, shifts)):
        p = LSParams(
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            zoom=float(z),
            shift=float(b),
            inversable=inversable,
        )

        # Fast identity short-circuit:
        #  - Only for pure interpolation (analy_degree < 0)
        #  - zoom ≈ 1 and zero shift
        if p.analy_degree < 0 and abs(p.zoom - 1.0) <= _EPS and abs(p.shift) <= 1e-15:
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
    inversable: bool = False,
) -> np.ndarray:
    """
    Pure-Python fallback for :func:`resize_degrees`, with dtype-preserving
    behavior for floats.

    The behavior is fully determined by the three degrees:

      - ``interp_degree`` : interpolation spline degree (0..3)
      - ``analy_degree``  : analysis spline degree (-1..3, -1 = no projection)
      - ``synthe_degree`` : synthesis spline degree (0..3)

    - Input float32  -> internal float64 -> output float32
    - Input float64  -> internal float64 -> output float64
    - Other dtypes   -> internal float64 -> output float64
    """
    # Normalize input and remember original dtype
    arr = np.asarray(data, order="C")
    input_dtype = arr.dtype

    # Work with the actual array shape (not necessarily data.shape if it was array-like)
    zoom_factors = [float(z) for z in zoom_factors]
    output_shape = tuple(int(round(n * z)) for n, z in zip(arr.shape, zoom_factors))

    # Internal buffers are always float64
    img64 = np.asarray(arr, dtype=np.float64, order="C")
    out64 = np.empty(output_shape, dtype=np.float64)

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
        inversable=inversable,
    )

    # Preserve float32/float64 at the Python API level.
    if np.issubdtype(input_dtype, np.floating):
        # float32 -> float32, float64 -> float64
        return out64.astype(input_dtype, copy=False)

    # For non-float inputs, keep the previous behavior (return float64).
    return out64
