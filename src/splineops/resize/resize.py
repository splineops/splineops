# splineops/src/splineops/resize/resize.py

# splineops.resize.resize
# =======================

# One-stop helper that wraps three back-ends

# * **interpolation**   – classic B-spline evaluation (degrees 1-3)
# * **oblique**         – fast anti-aliasing down-sampling using the Muñoz *oblique projection* variant
# * **least-squares**   – highest-quality anti-aliasing down-sampling using Muñoz *LS projection*

# The concrete back-end and spline degree are chosen with a single *method* string
# (see the *method* parameter in :pyfunc:`resize`).

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


# --------------------------------------------------------------------------- #
# Mapping from public `method` strings to (internal_algorithm, spline_degree)  #
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
    to match the Python implementation's behavior.
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

    This function will use the native C++ implementation (:mod:`splineops._lsresize`)
    for all supported presets—**interpolation** (degrees 0-3), **oblique** (deg 1-3),
    and **least-squares** (deg 1-3)—when the extension is available. Otherwise, it
    falls back to the pure-Python reference implementation
    :func:`splineops.resize.ls_oblique_resize.ls_oblique_resize`. You can control
    native vs. Python behavior with the env var ``SPLINEOPS_EXTENSION``:

      - ``SPLINEOPS_ACCEL=auto`` (default): use C++ if available, else Python
      - ``SPLINEOPS_ACCEL=never``: force the Python fallback only

    **Magnification policy (native path):**
        For projection methods (``*-fast_antialiasing`` and ``*-best_antialiasing``),
        the C++ backend applies the same analysis/synthesis model for all zoom
        factors, and only disables projection on axes where ``zoom_factors[i]`` is
        effectively 1 (identity safety).

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
        valid = ", ".join(METHOD_MAP)
        raise ValueError(f"Unknown method '{method}'. Valid options: {valid}")

    algo, degree = METHOD_MAP[method]

    # ----------------------------
    # Resolve target shape/zooms
    # ----------------------------
    if output_size is not None:
        zoom_factors = [float(new) / float(old) for new, old in zip(output_size, data.shape)]
    elif zoom_factors is None:
        raise ValueError("Either 'output_size' or 'zoom_factors' must be provided.")
    elif isinstance(zoom_factors, (int, float)):
        zoom_factors = [float(zoom_factors)] * data.ndim
    else:
        zoom_factors = [float(z) for z in zoom_factors]

    # ----------------------------
    # Dispatch to native or fallback
    # ----------------------------
    interp_degree, analy_degree, synthe_degree = _resolve_degrees_for(algo, degree)

    use_cpp = _HAS_CPP and (_ACCEL_EXP := (_ACCEL_ENV != "never"))
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
            False,  # 'inversable' sizing off to match Python behavior
        )
    else:
        # Python fallback via reference LS/Oblique/Standard solver.
        py_method = algo  # "interpolation" | "oblique" | "least-squares"
        # Magnification policy is enforced per-axis inside python engine,
        # so we can pass 'py_method' as-is.
        output_data = _python_fallback_resize(
            data, zoom_factors, py_method, degree, inversable=False
        )

    # ----------------------------
    # Final casting / in-place write
    # ----------------------------
    if output is not None:
        if isinstance(output, np.ndarray):
            if tuple(output.shape) != tuple(output_data.shape):
                raise ValueError(f"'output' has shape {output.shape}, expected {output_data.shape}")
            np.copyto(output, output_data.astype(output.dtype, copy=False))
            return output
        # output is a dtype
        return np.asarray(output_data, dtype=output)

    return output_data

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
# matches your test-validated coordinate normalization and per-axis
# magnification policy.