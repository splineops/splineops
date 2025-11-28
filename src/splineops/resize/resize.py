# splineops/src/splineops/resize/resize.py

# splineops.resize.resize
# =======================
#
# Public resizing APIs:
#
# * `resize`          – preset-based API using a `method` string.
# * `resize_degrees`  – advanced API exposing the three spline degrees
#                       (interp_degree, analy_degree, synthe_degree).
#
# Both will use the native C++ implementation (:mod:`splineops._lsresize`)
# when available, and fall back to the pure-Python reference implementation
# otherwise.

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Dict
import os

import numpy as np
import numpy.typing as npt

from splineops.resize._pycore.engine import python_resize as _python_fallback_resize

# Attempt to import the native acceleration module (optional)
try:
    from splineops._lsresize import resize_nd as _resize_nd_cpp  # type: ignore[attr-defined]
    _HAS_CPP = True
except Exception:  # pragma: no cover - if extension isn't built
    _HAS_CPP = False
    _resize_nd_cpp = None  # type: ignore[assignment]

# Environment switch: "auto" (default), "never", "always"
_ACCEL_ENV = os.environ.get("SPLINEOPS_ACCEL", "auto").lower()

MAX_SUPPORTED_DEGREE = 3


# --------------------------------------------------------------------------- #
# Mapping from `method` strings to (interp_degree, analy_degree, synthe_degree)
# --------------------------------------------------------------------------- #
#
# Each preset is a concrete triple of degrees:
#
#   - interp_degree : degree of the interpolation spline φ  (0..3)
#   - analy_degree  : degree of the analysis spline φ₁      (-1..3, -1 = no projection)
#   - synthe_degree : degree of the synthesis spline φ₂     (0..3)
#
# The behavior is entirely encoded by this triple.

METHOD_MAP: Dict[str, Tuple[int, int, int]] = {
    # Interpolation – no anti-aliasing (analy = -1)
    "fast":      (0, -1, 0),  # nearest
    "linear":    (1, -1, 1),
    "quadratic": (2, -1, 2),
    "cubic":     (3, -1, 3),

    # Antialiasing (projection-based), recommended for down-sampling.
    # These are the classic Muñoz/Unser “oblique” combinations:
    #   (1, 0, 1), (2, 1, 2), (3, 1, 3)
    "linear-antialiasing":    (1, 0, 1),
    "quadratic-antialiasing": (2, 1, 2),
    "cubic-antialiasing":     (3, 1, 3),

    # Legacy aliases (optional) – uncomment if you want to support older names:
    # "linear-fast_antialiasing":    (1, 0, 1),
    # "quadratic-fast_antialiasing": (2, 1, 2),
    # "cubic-fast_antialiasing":     (3, 1, 3),
}

# Helper for naming in messages (if you ever want it)
_DEGREE_TO_NAME = {0: "nearest", 1: "linear", 2: "quadratic", 3: "cubic"}


def _validate_degrees(
    interp_degree: int,
    analy_degree: int,
    synthe_degree: int,
) -> Tuple[int, int, int]:
    """
    Enforce supported degrees and sensible combinations.

    Constraints:
      - 0 <= interp_degree <= 3
      - -1 <= analy_degree <= 3   (-1 = no projection / pure interpolation)
      - 0 <= synthe_degree <= 3
      - if analy_degree >= 0: analy_degree <= interp_degree
      - synthe_degree <= interp_degree
    """
    if not (0 <= interp_degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"interp_degree must be in [0, {MAX_SUPPORTED_DEGREE}]")

    if not (-1 <= analy_degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"analy_degree must be in [-1, {MAX_SUPPORTED_DEGREE}]")

    if not (0 <= synthe_degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"synthe_degree must be in [0, {MAX_SUPPORTED_DEGREE}]")

    if analy_degree >= 0 and analy_degree > interp_degree:
        raise ValueError(
            "analy_degree must be <= interp_degree when analy_degree >= 0 "
            f"(got analy_degree={analy_degree}, interp_degree={interp_degree})"
        )

    if synthe_degree > interp_degree:
        raise ValueError(
            "synthe_degree must be <= interp_degree "
            f"(got synthe_degree={synthe_degree}, interp_degree={interp_degree})"
        )

    return int(interp_degree), int(analy_degree), int(synthe_degree)


def resize_degrees(
    data: npt.NDArray,
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output: Optional[Union[npt.NDArray, np.dtype]] = None,
    output_size: Optional[Tuple[int, ...]] = None,
    interp_degree: int = 3,
    analy_degree: int = -1,
    synthe_degree: Optional[int] = None,
    inversable: bool = False,
) -> npt.NDArray:
    """
    Resize an *N*-dimensional array using explicit spline degrees.

    This is the most general entry point: it exposes the three degrees:

      - interp_degree : degree of the interpolation B-spline φ (0..3)
      - analy_degree  : analysis spline degree (-1..3, -1 = no projection)
      - synthe_degree : synthesis spline degree (0..3)

    Parameters
    ----------
    data : ndarray
        Input array.
    zoom_factors : float or sequence of float, optional
        Per-axis scale factors. Ignored if *output_size* is given.
    output : ndarray or dtype, optional
        If an ``ndarray`` is supplied, the result is written **in-place** into
        that array and returned. If a ``dtype`` is supplied, a new array of that
        dtype is allocated and returned.
    output_size : tuple of int, optional
        Desired shape (overrides *zoom_factors*).
    interp_degree : int, default 3
        Degree of the interpolation B-spline φ (0..3).
    analy_degree : int, default -1
        Degree of the analysis spline φ₁:

          - -1 → no projection (pure interpolation)
          - 0..3 → projection-based resizing (antialiasing, equal-degree projection, etc.)

    synthe_degree : int, optional
        Degree of the synthesis spline φ₂ (output space). Defaults to
        ``interp_degree``. Must be in [0..3] and <= ``interp_degree``.
    inversable : bool, default False
        If True, use a size policy that ensures invertible zoom along each axis.

    Returns
    -------
    ndarray
        Resized data: either a new array or the one supplied via *output*.
    """
    if synthe_degree is None:
        synthe_degree = interp_degree

    interp_degree, analy_degree, synthe_degree = _validate_degrees(
        interp_degree, analy_degree, synthe_degree
    )

    # ----------------------------
    # Resolve target shape / zooms
    # ----------------------------
    if output_size is not None:
        zoom_factors = [
            float(new) / float(old) for new, old in zip(output_size, data.shape)
        ]
    elif zoom_factors is None:
        raise ValueError("Either 'output_size' or 'zoom_factors' must be provided.")
    elif isinstance(zoom_factors, (int, float)):
        zoom_factors = [float(zoom_factors)] * data.ndim
    else:
        zoom_factors = [float(z) for z in zoom_factors]

    # ----------------------------
    # Dispatch to native or fallback
    # ----------------------------
    use_cpp = _HAS_CPP and (_ACCEL_ENV != "never")

    if use_cpp:
        # Keep dtype, only enforce C-order
        arr = np.asarray(data, order="C")
        output_data = _resize_nd_cpp(
            arr,
            list(zoom_factors),
            int(interp_degree),
            int(analy_degree),
            int(synthe_degree),
            bool(inversable),
        )
    else:
        # Pure-Python fallback driven directly by the degree triple
        output_data = _python_fallback_resize(
            data,
            zoom_factors,
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            inversable=inversable,
        )

    # ----------------------------
    # Final casting / in-place write
    # ----------------------------
    if output is not None:
        if isinstance(output, np.ndarray):
            if tuple(output.shape) != tuple(output_data.shape):
                raise ValueError(
                    f"'output' has shape {output.shape}, expected {output_data.shape}"
                )
            np.copyto(output, output_data.astype(output.dtype, copy=False))
            return output
        # output is a dtype
        return np.asarray(output_data, dtype=output)

    return output_data


def resize(
    data: npt.NDArray,
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output: Optional[Union[npt.NDArray, np.dtype]] = None,
    output_size: Optional[Tuple[int, ...]] = None,
    method: str = "cubic",
) -> npt.NDArray:
    """
    Resize an *N*-dimensional array using spline interpolation or an
    antialiasing projection preset.

    This entry point selects both the algorithm and the spline degrees via a
    single ``method`` string, and then delegates to :func:`resize_degrees`.

    Parameters
    ----------
    data : ndarray
        Input array.
    zoom_factors : float or sequence of float, optional
        Per-axis scale factors. Ignored if *output_size* is given.
    output : ndarray or dtype, optional
        If an ``ndarray`` is supplied, the result is written **in-place** into
        that array and returned. If a ``dtype`` is supplied, a new array of that
        dtype is allocated and returned.
    output_size : tuple of int, optional
        Desired shape (overrides *zoom_factors*).
    method : str
        Preset selecting a specific (interp_degree, analy_degree, synthe_degree)
        triple.

        Interpolation (no anti-aliasing, analy = -1):

          - ``"fast"``      – degree 0 (nearest)
          - ``"linear"``    – degree 1
          - ``"quadratic"`` – degree 2
          - ``"cubic"``     – degree 3

        Antialiasing (projection-based, recommended for down-sampling):

          - ``"linear-antialiasing"``    – (interp=1, analy=0, synthe=1)
          - ``"quadratic-antialiasing"`` – (interp=2, analy=1, synthe=2)
          - ``"cubic-antialiasing"``     – (interp=3, analy=1, synthe=3)

    Returns
    -------
    ndarray
        Resized data: either a new array or the one supplied via *output*.
    """
    if method not in METHOD_MAP:  # pragma: no cover
        valid = ", ".join(sorted(METHOD_MAP))
        raise ValueError(f"Unknown method '{method}'. Valid options: {valid}")

    interp_degree, analy_degree, synthe_degree = METHOD_MAP[method]

    return resize_degrees(
        data,
        zoom_factors=zoom_factors,
        output=output,
        output_size=output_size,
        interp_degree=interp_degree,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
        inversable=False,  # preserve previous default
    )
