# splineops/src/splineops/resize/resize.py

# splineops.resize.resize
# =======================
#
# Public resizing APIs:
#
# * `resize`          – backwards-compatible API using a `method` preset string.
# * `resize_degrees`  – advanced API exposing the three spline degrees
#                       (interp_degree, analy_degree, synthe_degree).
#
# Both will use the native C++ implementation (:mod:`splineops._lsresize`)
# when available, and fall back to the pure-Python reference implementation
# otherwise.

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Dict, Literal
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
# Mapping from `method` strings to (algorithm, spline_degree)                 #
# --------------------------------------------------------------------------- #

METHOD_MAP: Dict[
    str,
    Tuple[Literal["interpolation", "oblique", "least-squares"], int],
] = {
    # pure interpolation – no anti-aliasing
    "fast": ("interpolation", 0),
    "linear": ("interpolation", 1),
    "quadratic": ("interpolation", 2),
    "cubic": ("interpolation", 3),
    # oblique projection – fast anti-aliasing
    "linear-fast_antialiasing": ("oblique", 1),
    "quadratic-fast_antialiasing": ("oblique", 2),
    "cubic-fast_antialiasing": ("oblique", 3),
    # least-squares – best anti-aliasing
    "linear-best_antialiasing": ("least-squares", 1),
    "quadratic-best_antialiasing": ("least-squares", 2),
    "cubic-best_antialiasing": ("least-squares", 3),
}

# Helper for ls_oblique_resize ↔︎ degree name (fallback path only)
_DEGREE_TO_NAME = {0: "nearest", 1: "linear", 2: "quadratic", 3: "cubic"}


def _resolve_degrees_for(algo: str, degree: int) -> Tuple[int, int, int]:
    """
    Map (algo, public_degree) -> (interp_degree, analy_degree, synthe_degree)
    to match the original Python implementation's behavior.

    algo ∈ {"interpolation", "oblique", "least-squares"}.
    """
    interp_degree = degree
    synthe_degree = degree
    if algo == "interpolation":
        analy_degree = -1
    elif algo == "least-squares":
        analy_degree = degree
    else:  # "oblique"
        # Oblique uses analy 0 for linear, 1 for quadratic/cubic
        analy_degree = 0 if degree == 1 else 1
    return interp_degree, analy_degree, synthe_degree


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


def _map_degrees_to_python_backend(
    interp_degree: int,
    analy_degree: int,
    synthe_degree: int,
) -> Tuple[str, int]:
    """
    Map (interp_degree, analy_degree, synthe_degree) to the minimal
    (py_method, base_degree) that the pure-Python engine understands.

    The Python fallback supports only the canonical families:

      - Standard interpolation: (n, -1, n)
      - Least-squares:         (n,  n, n)
      - Oblique (fast AA):     (1, 0, 1) or (n, 1, n) for n in {2,3}

    Any other combination requires the C++ backend.
    """
    # Pure interpolation: analy=-1, synthe matches interp
    if analy_degree < 0:
        if synthe_degree != interp_degree:
            raise NotImplementedError(
                "Python fallback only supports synthe_degree == interp_degree "
                "for interpolation mode."
            )
        return "interpolation", interp_degree

    # Least-squares: analy == synthe == interp
    if analy_degree == interp_degree and synthe_degree == interp_degree:
        return "least-squares", interp_degree

    # Oblique presets (same policy as the original implementation)
    if interp_degree == 1 and analy_degree == 0 and synthe_degree == 1:
        return "oblique", interp_degree

    if interp_degree in (2, 3) and analy_degree == 1 and synthe_degree == interp_degree:
        return "oblique", interp_degree

    raise NotImplementedError(
        "This (interp_degree, analy_degree, synthe_degree) combination is only "
        "supported when the C++ backend is available. "
        f"Got interp={interp_degree}, analy={analy_degree}, synthe={synthe_degree}."
    )


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

    This is the most general entry point: it exposes the three degrees used
    in the Muñoz/Unser framework:

      - interp_degree (φ): interpolation spline degree (0..3)
      - analy_degree  (φ₁): analysis spline degree (-1..3, -1 = no projection)
      - synthe_degree (φ₂): synthesis spline degree (0..3)

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
          - 0..3 → LS/oblique-style projection.

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

    # Validate degrees and their relationships
    interp_degree, analy_degree, synthe_degree = _validate_degrees(
        interp_degree, analy_degree, synthe_degree
    )

    # ----------------------------
    # Resolve target shape/zooms
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
        # NOTE: keep dtype, only enforce C-order for the C++ backend.
        arr = np.asarray(data, order="C")

        # _resize_nd_cpp will choose float32 vs float64 based on arr.dtype.
        output_data = _resize_nd_cpp(
            arr,
            list(zoom_factors),
            int(interp_degree),
            int(analy_degree),
            int(synthe_degree),
            bool(inversable),
        )
    else:
        # Python fallback via reference LS/Oblique/Standard solver.
        py_method, base_degree = _map_degrees_to_python_backend(
            interp_degree, analy_degree, synthe_degree
        )
        # Any zoom-dependent policy is handled inside the pure-Python engine;
        # this wrapper simply forwards the request.
        output_data = _python_fallback_resize(
            data,
            zoom_factors,
            py_method,  # "interpolation" | "oblique" | "least-squares"
            base_degree,
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
    Resize an *N*-dimensional array using splines.

    This entry point is **backwards compatible** with the original API and
    selects both the algorithm and the spline degree via a single `method`
    string.

    Under the hood, it maps `method` to explicit degrees and calls
    :func:`resize_degrees`.

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
        Preset selecting **both** the algorithm *and* the spline degree.

        The following values are supported:

        - ``"fast"``: interpolation, degree 0
        - ``"linear"``: interpolation, degree 1
        - ``"quadratic"``: interpolation, degree 2
        - ``"cubic"``: interpolation, degree 3
        - ``"linear-fast_antialiasing"``: oblique, degree 1
        - ``"quadratic-fast_antialiasing"``: oblique, degree 2
        - ``"cubic-fast_antialiasing"``: oblique, degree 3
        - ``"linear-best_antialiasing"``: least-squares, degree 1
        - ``"quadratic-best_antialiasing"``: least-squares, degree 2
        - ``"cubic-best_antialiasing"``: least-squares, degree 3

        Anti-aliasing variants are preferred for down-sampling.

    Returns
    -------
    ndarray
        Resized data: either a new array or the one supplied via *output*.
    """
    # ----------------------------
    # Validate & interpret preset
    # ----------------------------
    if method not in METHOD_MAP:  # pragma: no cover
        valid = ", ".join(sorted(METHOD_MAP))
        raise ValueError(f"Unknown method '{method}'. Valid options: {valid}")

    algo, degree = METHOD_MAP[method]
    interp_degree, analy_degree, synthe_degree = _resolve_degrees_for(algo, degree)

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


# ---------------------------------------------------------------------------
# Legacy reference (no longer used): how `TensorSpline` wired into `resize`
# ---------------------------------------------------------------------------
#
# from splineops.bases.utils import asbasis
# from splineops.spline_interpolation.tensorspline import TensorSpline
#
# def _tensorspline_interpolation(
#     data: npt.NDArray,
#     zoom_factors: Sequence[float],
#     degree: int,
#     modes: Union[str, Sequence[str]] = "auto-or-‘mirror’",
# ) -> npt.NDArray:
#     basis = asbasis(f"bspline{degree}")  # degrees 0..3 supported
#     src_coords = [np.linspace(0, n-1, n, dtype=data.dtype) for n in data.shape]
#     tgt_coords = [np.linspace(0, n-1, int(round(n*z)), dtype=data.dtype)
#                   for n, z in zip(data.shape, zoom_factors)]
#     tensor = TensorSpline(data=data, coordinates=src_coords, bases=basis, modes=modes)
#     return tensor.eval(coordinates=tgt_coords, grid=True)
#
# This approach has been fully replaced by the C++-accelerated path above,
# which now handles both interpolation and projection (Oblique/LS) and
# matches your test-validated coordinate normalization and per-axis sizing.
