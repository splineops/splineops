# splineops/src/splineops/resize/resize.py

# splineops.resize.resize
# =======================

# One-stop helper that wraps three back-ends

# * **interpolation**   – classic B-spline evaluation (degrees 0-9) via :class:`splineops.interpolate.TensorSpline`
# * **oblique**         – fast anti-aliasing down-sampling using the Muñoz *oblique projection* variant
# * **least-squares**   – highest-quality anti-aliasing down-sampling using Muñoz *LS projection*

# The concrete back-end and spline degree are chosen with a single *method* string
# (see the *method* parameter in :pyfunc:`resize`).

# splineops/src/splineops/resize/resize.py

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Dict, Literal
import os

import numpy as np
import numpy.typing as npt

from splineops.bases.utils import asbasis
from splineops.interpolate.tensorspline import TensorSpline
from splineops.resize.ls_oblique_resize import ls_oblique_resize

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
        # Python version uses analy 0 for linear, 1 for quadratic/cubic
        analy_degree = 0 if degree == 1 else 1
    return interp_degree, analy_degree, synthe_degree


def resize(
    data: npt.NDArray,
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output: Optional[Union[npt.NDArray, np.dtype]] = None,
    output_size: Optional[Tuple[int, ...]] = None,
    method: str = "cubic",
    modes: Union[str, Sequence[str]] = "mirror",
) -> npt.NDArray:
    """
    Resize an *N*-dimensional array using splines.

    This function will dynamically use a native C++ implementation for the
    **least-squares** and **oblique** presets (degrees 1–3) when the optional
    module :mod:`splineops._lsresize` is available. Otherwise, it falls back to
    the pure-Python implementation. You can control this with the env var
    ``SPLINEOPS_ACCEL``:

      - ``auto`` (default): use C++ if available, else Python
      - ``never``: always use Python
      - ``always``: require C++; error if not available

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

    method : string
        Preset selecting **both** the algorithm *and* the spline degree:
        
        - **fast**: interpolation, degree 0
        - **linear**: interpolation, degree 1
        - **quadratic**: interpolation, degree 2
        - **cubic**: interpolation, degree 3
        - **linear-fast_antialiasing**: oblique, degree 1
        - **quadratic-fast_antialiasing**: oblique, degree 2
        - **cubic-fast_antialiasing**: oblique, degree 3
        - **linear-best_antialiasing**: least-squares, degree 1
        - **quadratic-best_antialiasing**: least-squares, degree 2
        - **cubic-best_antialiasing**: least-squares, degree 3

        Anti-aliasing variants are preferred for down-sampling.

    modes : str or sequence of str, optional
        Boundary handling passed to
        :class:`splineops.interpolate.TensorSpline`
        (ignored by the anti-aliasing presets).

    Returns
    -------
    ndarray
        Resized data – either a new array or the one supplied via *output*.
    """
    # --------------------------------------------------------------------- #
    # Validate & interpret parameters                                       #
    # --------------------------------------------------------------------- #
    if method not in METHOD_MAP:  # pragma: no cover
        valid = ", ".join(METHOD_MAP)
        raise ValueError(f"Unknown method '{method}'. Valid options: {valid}")

    algo, degree = METHOD_MAP[method]

    if output_size is not None:
        zoom_factors = [new / old for new, old in zip(output_size, data.shape)]
    elif zoom_factors is None:
        raise ValueError("Either 'output_size' or 'zoom_factors' must be provided.")
    elif isinstance(zoom_factors, (int, float)):
        zoom_factors = [float(zoom_factors)] * data.ndim
    else:
        zoom_factors = [float(z) for z in zoom_factors]

    # --------------------------------------------------------------------- #
    # Choose implementation path                                            #
    # --------------------------------------------------------------------- #
    if algo in {"least-squares", "oblique"} and degree in (1, 2, 3):
        interp_degree, analy_degree, synthe_degree = _resolve_degrees_for(algo, degree)

        # C++ availability policy
        use_cpp = _HAS_CPP and (_ACCEL_ENV != "never")
        if _ACCEL_ENV == "always" and not _HAS_CPP:
            raise RuntimeError("SPLINEOPS_ACCEL=always but native extension is not available")

        if use_cpp:
            # Native path: convert to float64 C-order (module expects that)
            arr64 = np.asarray(data, dtype=np.float64, order="C")
            output_data = _resize_nd_cpp(
                arr64,
                zoom_factors,
                int(interp_degree),
                int(analy_degree),
                int(synthe_degree),
                False,   # inversable sizing behavior: keep False to match Python default
            )
        else:
            # Pure-Python fallback (existing implementation)
            output_data = ls_oblique_resize(
                input_img_normalized=data,
                output_size=output_size,
                zoom_factors=zoom_factors,
                method=algo,
                interpolation=_DEGREE_TO_NAME[degree],
            )
    else:
        # Interpolation-only path via TensorSpline (degrees 0–9)
        basis = asbasis(f"bspline{degree}")
        # source grid
        src_coords = [np.linspace(0, n - 1, n, dtype=data.dtype) for n in data.shape]
        # target grid
        tgt_coords = [
            np.linspace(0, n - 1, round(n * z), dtype=data.dtype)
            for n, z in zip(data.shape, zoom_factors)
        ]
        tensor = TensorSpline(data=data, coordinates=src_coords, bases=basis, modes=modes)
        output_data = tensor.eval(coordinates=tgt_coords, grid=True)

    # --------------------------------------------------------------------- #
    # Handle 'output' argument                                              #
    # --------------------------------------------------------------------- #
    if output is not None:
        if isinstance(output, np.ndarray):
            np.copyto(output, output_data)
            return output
        out_arr = np.empty_like(output_data, dtype=output)
        np.copyto(out_arr, output_data)
        return out_arr

    return output_data