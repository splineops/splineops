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
from splineops.resize._pycore.utils import calculate_final_size_1d as _calculate_final_size_1d

# Attempt to import the native acceleration module (optional)
try:
    from splineops._lsresize import resize_nd as _resize_nd_cpp  # type: ignore[attr-defined]
    from splineops._lsresize import ResizePlan as _ResizePlanCpp  # type: ignore[attr-defined]
    _HAS_CPP = True
except Exception:  # pragma: no cover - if extension isn't built
    _HAS_CPP = False
    _resize_nd_cpp = None  # type: ignore[assignment]
    _ResizePlanCpp = None  # type: ignore[assignment]

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


def _normalize_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(n) for n in input_shape)
    if len(shape) == 0:
        raise ValueError("'input_shape' must describe at least one dimension")
    if any(n <= 0 for n in shape):
        raise ValueError("'input_shape' entries must be positive")
    return shape


def _resolve_zoom_for_shape(
    input_shape: Sequence[int],
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output_size: Optional[Tuple[int, ...]] = None,
) -> tuple[float, ...]:
    shape = tuple(int(n) for n in input_shape)
    if output_size is not None:
        if len(output_size) != len(shape):
            raise ValueError("'output_size' length must match 'input_shape' length")
        if any(int(n) <= 0 for n in output_size):
            raise ValueError("'output_size' entries must be positive")
        return tuple(float(new) / float(old) for new, old in zip(output_size, shape))
    if zoom_factors is None:
        raise ValueError("Either 'output_size' or 'zoom_factors' must be provided.")
    if isinstance(zoom_factors, (int, float)):
        zoom = tuple([float(zoom_factors)] * len(shape))
    else:
        zoom = tuple(float(z) for z in zoom_factors)
    if len(zoom) != len(shape):
        raise ValueError("'zoom_factors' length must match 'input_shape' length")
    if any(z <= 0.0 for z in zoom):
        raise ValueError("'zoom_factors' entries must be positive")
    return zoom


def _output_shape_for_plan(
    input_shape: Sequence[int],
    zoom_factors: Sequence[float],
    inversable: bool,
) -> tuple[int, ...]:
    out = []
    for n, z in zip(input_shape, zoom_factors):
        _, out_n = _calculate_final_size_1d(bool(inversable), int(n), float(z))
        out.append(int(out_n))
    return tuple(out)


class ResizePlan:
    """
    Reusable resize plan for repeated same-shape workloads.

    A plan fixes the input shape, target geometry, spline degrees, and size
    policy once, then applies that geometry to many arrays with the same shape.
    When the native extension is available and acceleration is not disabled,
    the plan uses the native backend and reuses its cached per-axis metadata.
    Otherwise it falls back to the pure-Python resize implementation.
    """

    __slots__ = (
        "input_shape",
        "output_shape",
        "zoom_factors",
        "interp_degree",
        "analy_degree",
        "synthe_degree",
        "inversable",
        "method",
        "_native_plan",
    )

    def __init__(
        self,
        input_shape: Sequence[int],
        *,
        zoom_factors: Optional[Union[float, Sequence[float]]] = None,
        output_size: Optional[Tuple[int, ...]] = None,
        method: str = "cubic",
        inversable: bool = False,
    ) -> None:
        if method not in METHOD_MAP:
            valid = ", ".join(sorted(METHOD_MAP))
            raise ValueError(f"Unknown method '{method}'. Valid options: {valid}")
        interp_degree, analy_degree, synthe_degree = METHOD_MAP[method]
        self._init_degrees(
            input_shape,
            zoom_factors=zoom_factors,
            output_size=output_size,
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            inversable=inversable,
            method=method,
        )

    @classmethod
    def from_degrees(
        cls,
        input_shape: Sequence[int],
        *,
        zoom_factors: Optional[Union[float, Sequence[float]]] = None,
        output_size: Optional[Tuple[int, ...]] = None,
        interp_degree: int = 3,
        analy_degree: int = -1,
        synthe_degree: Optional[int] = None,
        inversable: bool = False,
    ) -> "ResizePlan":
        """Create a plan using explicit spline degrees."""
        if synthe_degree is None:
            synthe_degree = interp_degree
        obj = cls.__new__(cls)
        obj._init_degrees(
            input_shape,
            zoom_factors=zoom_factors,
            output_size=output_size,
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            inversable=inversable,
            method=None,
        )
        return obj

    def _init_degrees(
        self,
        input_shape: Sequence[int],
        *,
        zoom_factors: Optional[Union[float, Sequence[float]]],
        output_size: Optional[Tuple[int, ...]],
        interp_degree: int,
        analy_degree: int,
        synthe_degree: int,
        inversable: bool,
        method: Optional[str],
    ) -> None:
        interp_degree, analy_degree, synthe_degree = _validate_degrees(
            interp_degree, analy_degree, synthe_degree
        )
        shape = _normalize_shape(input_shape)
        zoom = _resolve_zoom_for_shape(
            shape,
            zoom_factors=zoom_factors,
            output_size=output_size,
        )

        self.input_shape = shape
        self.output_shape = _output_shape_for_plan(shape, zoom, inversable)
        self.zoom_factors = zoom
        self.interp_degree = interp_degree
        self.analy_degree = analy_degree
        self.synthe_degree = synthe_degree
        self.inversable = bool(inversable)
        self.method = method

        use_cpp = _HAS_CPP and (_ACCEL_ENV != "never")
        if use_cpp:
            self._native_plan = _ResizePlanCpp(  # type: ignore[misc,operator]
                list(self.input_shape),
                list(self.zoom_factors),
                int(self.interp_degree),
                int(self.analy_degree),
                int(self.synthe_degree),
                bool(self.inversable),
            )
            self.output_shape = tuple(int(n) for n in self._native_plan.output_shape)
        else:
            self._native_plan = None

    def apply(
        self,
        data: npt.NDArray,
        output: Optional[Union[npt.NDArray, np.dtype]] = None,
    ) -> npt.NDArray:
        """Apply the planned resize to an array with ``input_shape``."""
        arr = np.asarray(data, order="C")
        if tuple(arr.shape) != self.input_shape:
            raise ValueError(
                f"input has shape {arr.shape}, expected {self.input_shape}"
            )

        if isinstance(output, np.ndarray):
            if tuple(output.shape) != self.output_shape:
                raise ValueError(
                    f"'output' has shape {output.shape}, expected {self.output_shape}"
                )

            if self._native_plan is not None and hasattr(self._native_plan, "apply_into"):
                native_dtype = (
                    np.dtype(np.float32)
                    if arr.dtype == np.dtype(np.float32)
                    else np.dtype(np.float64)
                )
                can_write_native_direct = (
                    output.dtype == native_dtype
                    and output.flags.c_contiguous
                    and output.flags.writeable
                    and not np.may_share_memory(arr, output)
                )
                if can_write_native_direct:
                    return self._native_plan.apply_into(arr, output)

        if self._native_plan is not None:
            output_data = self._native_plan.apply(arr)
        else:
            output_data = _python_fallback_resize(
                arr,
                self.zoom_factors,
                interp_degree=self.interp_degree,
                analy_degree=self.analy_degree,
                synthe_degree=self.synthe_degree,
                inversable=self.inversable,
            )

        if output is not None:
            if isinstance(output, np.ndarray):
                np.copyto(output, output_data.astype(output.dtype, copy=False))
                return output
            return np.asarray(output_data, dtype=output)

        return output_data

    __call__ = apply

    def __repr__(self) -> str:
        label = f"method={self.method!r}" if self.method is not None else (
            "degrees="
            f"({self.interp_degree}, {self.analy_degree}, {self.synthe_degree})"
        )
        return (
            "ResizePlan("
            f"input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}, "
            f"zoom_factors={self.zoom_factors}, "
            f"{label})"
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
