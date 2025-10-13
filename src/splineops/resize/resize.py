# splineops/src/splineops/resize/resize.py

# splineops.resize.resize
# =======================

# One-stop helper that wraps three back-ends

# * **interpolation**   – classic B-spline evaluation (degrees 0-9) via :class:`splineops.interpolate.TensorSpline`
# * **oblique**         – fast anti-aliasing down-sampling using the Muñoz *oblique projection* variant
# * **least-squares**   – highest-quality anti-aliasing down-sampling using Muñoz *LS projection*

# The concrete back-end and spline degree are chosen with a single *method* string
# (see the *method* parameter in :pyfunc:`resize`).


from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Dict, Literal

import numpy as np
import numpy.typing as npt

from splineops.bases.utils import asbasis
from splineops.interpolate.tensorspline import TensorSpline
from splineops.resize.ls_oblique_resize import ls_oblique_resize

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

# Helper for ls_oblique_resize ↔︎ degree translation
_DEGREE_TO_NAME = {0: "nearest", 1: "linear", 2: "quadratic", 3: "cubic"}


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

    Parameters
    ----------
    data : ndarray
        Input array.

    zoom_factors : float or sequence of float, optional
        Per-axis scale factors. Ignored if *output_size* is given.

    output : ndarray or dtype, optional
        If an ``ndarray`` is supplied, the result is written **in-place** and
        the same array is returned.

        If a ``dtype`` is supplied, a new array of that dtype is allocated and
        returned.

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

        Note that anti-aliasing variants are preferred when down-sampling.

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
        zoom_factors = [zoom_factors] * data.ndim

    # --------------------------------------------------------------------- #
    # Choose implementation path                                            #
    # --------------------------------------------------------------------- #
    if algo in {"least-squares", "oblique"} and degree in (1, 2, 3):
        # Use Arrate Muñoz' LS/oblique implementation
        output_data = ls_oblique_resize(
            input_img_normalized=data,
            output_size=output_size,
            zoom_factors=zoom_factors,
            method=algo,
            interpolation=_DEGREE_TO_NAME[degree],
        )
    else:
        # Fall back to TensorSpline – handles interpolation for deg 0‒9
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
