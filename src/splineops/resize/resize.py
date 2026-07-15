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
import numbers
import operator
import os

import numpy as np
import numpy.typing as npt

from splineops.resize._pycore.engine import python_resize as _python_fallback_resize
from splineops.resize._pycore.utils import (
    calculate_output_size_1d as _calculate_output_size_1d,
)

# Attempt to import the native acceleration module (optional)
try:
    from splineops._lsresize import resize_nd as _resize_nd_cpp  # type: ignore[attr-defined]
    from splineops._lsresize import resize_nd_into as _resize_nd_into_cpp  # type: ignore[attr-defined]
    from splineops._lsresize import ResizePlan as _ResizePlanCpp  # type: ignore[attr-defined]

    _HAS_CPP = True
    _NATIVE_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - if extension isn't built
    _HAS_CPP = False
    _NATIVE_IMPORT_ERROR = exc
    _resize_nd_cpp = None  # type: ignore[assignment]
    _resize_nd_into_cpp = None  # type: ignore[assignment]
    _ResizePlanCpp = None  # type: ignore[assignment]

# Environment switch: "auto" (default), "never", "always"
_ACCEL_ENV = os.environ.get("SPLINEOPS_ACCEL", "auto").lower()

MAX_SUPPORTED_DEGREE = 3
_SUPPORTED_REAL_KINDS = frozenset("iuf")


def _require_native_if_requested() -> None:
    if _ACCEL_ENV == "always" and not _HAS_CPP:
        detail = f": {_NATIVE_IMPORT_ERROR}" if _NATIVE_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "SPLINEOPS_ACCEL=always requires the native splineops extension, "
            f"but it could not be imported{detail}"
        ) from _NATIVE_IMPORT_ERROR


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
    "fast": (0, -1, 0),  # nearest
    "linear": (1, -1, 1),
    "quadratic": (2, -1, 2),
    "cubic": (3, -1, 3),
    # Antialiasing (projection-based), recommended for down-sampling.
    # These are the classic Muñoz/Unser “oblique” combinations:
    #   (1, 0, 1), (2, 1, 2), (3, 1, 3)
    "linear-antialiasing": (1, 0, 1),
    "quadratic-antialiasing": (2, 1, 2),
    "cubic-antialiasing": (3, 1, 3),
}


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
    normalized = []
    for name, value in (
        ("interp_degree", interp_degree),
        ("analy_degree", analy_degree),
        ("synthe_degree", synthe_degree),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer")
        try:
            normalized.append(int(operator.index(value)))
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer") from exc

    interp_degree, analy_degree, synthe_degree = normalized

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

    return interp_degree, analy_degree, synthe_degree


def _positive_index(value: object, argument: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"'{argument}' entries must be integers")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"'{argument}' entries must be integers") from exc
    if result <= 0:
        raise ValueError(f"'{argument}' entries must be positive")
    return int(result)


def _normalize_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(_positive_index(n, "input_shape") for n in input_shape)
    if len(shape) == 0:
        raise ValueError("'input_shape' must describe at least one dimension")
    return shape


def _normalize_output_size(
    output_size: Sequence[int], expected_length: int
) -> tuple[int, ...]:
    try:
        values = tuple(output_size)
    except TypeError as exc:
        raise TypeError("'output_size' must be a sequence of integers") from exc
    if len(values) != expected_length:
        raise ValueError("'output_size' length must match the number of resized axes")
    return tuple(_positive_index(n, "output_size") for n in values)


def _normalize_zoom_factors(
    zoom_factors: Union[float, Sequence[float]], expected_length: int
) -> tuple[float, ...]:
    if isinstance(zoom_factors, numbers.Real) and not isinstance(
        zoom_factors, (bool, np.bool_)
    ):
        values = (float(zoom_factors),) * expected_length
    else:
        try:
            raw_values = tuple(zoom_factors)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "'zoom_factors' must be a real number or a sequence of real numbers"
            ) from exc
        if len(raw_values) != expected_length:
            raise ValueError(
                "'zoom_factors' length must match the number of resized axes"
            )
        values_list = []
        for value in raw_values:
            if not isinstance(value, numbers.Real) or isinstance(
                value, (bool, np.bool_)
            ):
                raise TypeError("'zoom_factors' entries must be real numbers")
            values_list.append(float(value))
        values = tuple(values_list)

    if not all(np.isfinite(z) for z in values):
        raise ValueError("'zoom_factors' entries must be finite")
    if not all(z > 0.0 for z in values):
        raise ValueError("'zoom_factors' entries must be positive")
    return values


def _normalize_axes(axes: Optional[Sequence[int]], ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    try:
        values = tuple(axes)
    except TypeError as exc:
        raise TypeError("'axes' must be a sequence of integers") from exc

    normalized = []
    seen = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("'axes' entries must be integers")
        try:
            axis = operator.index(value)
        except TypeError as exc:
            raise TypeError("'axes' entries must be integers") from exc
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {value} is out of range for {ndim}-D input")
        if axis in seen:
            raise ValueError("'axes' entries must be unique")
        seen.add(axis)
        normalized.append(int(axis))
    return tuple(normalized)


def _validate_real_dtype(dtype_like: object, argument: str) -> np.dtype:
    try:
        dtype = np.dtype(dtype_like)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{argument}' must have a real numeric dtype") from exc
    if dtype.kind not in _SUPPORTED_REAL_KINDS:
        raise TypeError(
            f"'{argument}' must have a real integer or floating dtype; got {dtype}"
        )
    return dtype


def _normalize_input(data: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(data, order="C")
    if arr.ndim == 0:
        raise ValueError("'data' must be at least one-dimensional")
    if any(n <= 0 for n in arr.shape):
        raise ValueError("'data' dimensions must be non-empty")
    _validate_real_dtype(arr.dtype, "data")
    return arr


def _validate_output(
    output: Optional[Union[npt.NDArray, np.dtype]],
    expected_shape: Sequence[int],
) -> None:
    if output is None:
        return
    if isinstance(output, np.ndarray):
        _validate_real_dtype(output.dtype, "output")
        if tuple(output.shape) != tuple(expected_shape):
            raise ValueError(
                f"'output' has shape {output.shape}, expected {tuple(expected_shape)}"
            )
        if not output.flags.writeable:
            raise ValueError("'output' must be writeable")
        return
    _validate_real_dtype(output, "output")


def _resolve_zoom_for_shape(
    input_shape: Sequence[int],
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output_size: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
) -> tuple[float, ...]:
    shape = tuple(int(n) for n in input_shape)
    selected_axes = _normalize_axes(axes, len(shape))
    zoom = [1.0] * len(shape)
    if output_size is not None:
        target = _normalize_output_size(output_size, len(selected_axes))
        for axis, new in zip(selected_axes, target):
            zoom[axis] = float(new) / float(shape[axis])
        return tuple(zoom)
    if zoom_factors is None:
        raise ValueError("Either 'output_size' or 'zoom_factors' must be provided.")
    selected_zoom = _normalize_zoom_factors(zoom_factors, len(selected_axes))
    for axis, value in zip(selected_axes, selected_zoom):
        zoom[axis] = value
    return tuple(zoom)


def _output_shape_for_plan(
    input_shape: Sequence[int],
    zoom_factors: Sequence[float],
    axes: Sequence[int],
) -> tuple[int, ...]:
    out = list(int(n) for n in input_shape)
    for axis in axes:
        out[axis] = _calculate_output_size_1d(
            int(input_shape[axis]), float(zoom_factors[axis])
        )
    return tuple(out)


def _resolve_geometry_for_shape(
    input_shape: Sequence[int],
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]],
    output_size: Optional[Sequence[int]],
    axes: Optional[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    """Return normalized axes, full zoom vector, and output shape."""
    shape = tuple(int(n) for n in input_shape)
    selected_axes = _normalize_axes(axes, len(shape))
    normalized_output_size = tuple(output_size) if output_size is not None else None
    full_zoom = _resolve_zoom_for_shape(
        shape,
        zoom_factors=zoom_factors,
        output_size=normalized_output_size,
        axes=selected_axes,
    )

    if normalized_output_size is not None:
        selected_shape = _normalize_output_size(
            normalized_output_size, len(selected_axes)
        )
        output_shape = list(shape)
        for axis, new in zip(selected_axes, selected_shape):
            output_shape[axis] = new
    else:
        output_shape = list(_output_shape_for_plan(shape, full_zoom, selected_axes))

    return (
        selected_axes,
        full_zoom,
        tuple(output_shape),
    )


class ResizePlan:
    """
    Reusable resize plan for repeated same-shape workloads.

    A plan fixes the input shape, target geometry, and spline degrees once,
    then applies that geometry to many arrays with the same shape.
    ``axes`` optionally restricts resizing to unique normalized spatial axes;
    scalar zooms broadcast over those axes and other dimensions remain exact
    identity dimensions. Configuration properties are read-only; construct a
    new plan to use different geometry or spline degrees.
    When the native extension is available and acceleration is not disabled,
    the plan uses the native backend and reuses its cached per-axis metadata.
    Otherwise it falls back to the pure-Python resize implementation.
    """

    __slots__ = (
        "_input_shape",
        "_output_shape",
        "_zoom_factors",
        "_axes",
        "_interp_degree",
        "_analy_degree",
        "_synthe_degree",
        "_method",
        "_native_plan",
    )

    def __init__(
        self,
        input_shape: Sequence[int],
        *,
        zoom_factors: Optional[Union[float, Sequence[float]]] = None,
        output_size: Optional[Sequence[int]] = None,
        axes: Optional[Sequence[int]] = None,
        method: str = "cubic",
    ) -> None:
        if method not in METHOD_MAP:
            valid = ", ".join(sorted(METHOD_MAP))
            raise ValueError(f"Unknown method '{method}'. Valid options: {valid}")
        interp_degree, analy_degree, synthe_degree = METHOD_MAP[method]
        self._init_degrees(
            input_shape,
            zoom_factors=zoom_factors,
            output_size=output_size,
            axes=axes,
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            method=method,
        )

    @classmethod
    def from_degrees(
        cls,
        input_shape: Sequence[int],
        *,
        zoom_factors: Optional[Union[float, Sequence[float]]] = None,
        output_size: Optional[Sequence[int]] = None,
        axes: Optional[Sequence[int]] = None,
        interp_degree: int = 3,
        analy_degree: int = -1,
        synthe_degree: Optional[int] = None,
    ) -> "ResizePlan":
        """Create a plan using explicit spline degrees."""
        if synthe_degree is None:
            synthe_degree = interp_degree
        obj = cls.__new__(cls)
        obj._init_degrees(
            input_shape,
            zoom_factors=zoom_factors,
            output_size=output_size,
            axes=axes,
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            method=None,
        )
        return obj

    def _init_degrees(
        self,
        input_shape: Sequence[int],
        *,
        zoom_factors: Optional[Union[float, Sequence[float]]],
        output_size: Optional[Sequence[int]],
        axes: Optional[Sequence[int]],
        interp_degree: int,
        analy_degree: int,
        synthe_degree: int,
        method: Optional[str],
    ) -> None:
        interp_degree, analy_degree, synthe_degree = _validate_degrees(
            interp_degree, analy_degree, synthe_degree
        )
        shape = _normalize_shape(input_shape)
        selected_axes, zoom, output_shape = _resolve_geometry_for_shape(
            shape,
            zoom_factors=zoom_factors,
            output_size=output_size,
            axes=axes,
        )

        self._input_shape = shape
        self._output_shape = output_shape
        self._zoom_factors = zoom
        self._axes = selected_axes
        self._interp_degree = interp_degree
        self._analy_degree = analy_degree
        self._synthe_degree = synthe_degree
        self._method = method

        _require_native_if_requested()
        use_cpp = _HAS_CPP and (_ACCEL_ENV != "never")
        if use_cpp:
            self._native_plan = _ResizePlanCpp(  # type: ignore[misc,operator]
                list(self.input_shape),
                list(self.zoom_factors),
                int(self.interp_degree),
                int(self.analy_degree),
                int(self.synthe_degree),
                list(self.axes),
            )
            native_shape = tuple(int(n) for n in self._native_plan.output_shape)
            if native_shape != self.output_shape:
                raise RuntimeError(
                    "native resize plan disagrees with the resolved output shape: "
                    f"{native_shape} != {self.output_shape}"
                )
        else:
            self._native_plan = None

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def zoom_factors(self) -> tuple[float, ...]:
        return self._zoom_factors

    @property
    def axes(self) -> tuple[int, ...]:
        return self._axes

    @property
    def interp_degree(self) -> int:
        return self._interp_degree

    @property
    def analy_degree(self) -> int:
        return self._analy_degree

    @property
    def synthe_degree(self) -> int:
        return self._synthe_degree

    @property
    def method(self) -> Optional[str]:
        return self._method

    def apply(
        self,
        data: npt.NDArray,
        output: Optional[Union[npt.NDArray, np.dtype]] = None,
    ) -> npt.NDArray:
        """Apply the planned resize to an array with ``input_shape``."""
        arr = _normalize_input(data)
        if tuple(arr.shape) != self.input_shape:
            raise ValueError(
                f"input has shape {arr.shape}, expected {self.input_shape}"
            )

        _validate_output(output, self.output_shape)

        if isinstance(output, np.ndarray):
            if self._native_plan is not None and hasattr(
                self._native_plan, "apply_into"
            ):
                native_dtype = (
                    np.dtype(np.float32)
                    if arr.dtype == np.dtype(np.float32)
                    else np.dtype(np.float64)
                )
                can_write_native_direct = (
                    output.dtype == native_dtype
                    and output.flags.c_contiguous
                    and output.flags.aligned
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
                axes=self.axes,
            )

        if output is not None:
            if isinstance(output, np.ndarray):
                np.copyto(output, output_data.astype(output.dtype, copy=False))
                return output
            return np.asarray(output_data, dtype=output)

        return output_data

    __call__ = apply

    def __repr__(self) -> str:
        label = (
            f"method={self.method!r}"
            if self.method is not None
            else (
                "degrees="
                f"({self.interp_degree}, {self.analy_degree}, {self.synthe_degree})"
            )
        )
        return (
            "ResizePlan("
            f"input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}, "
            f"zoom_factors={self.zoom_factors}, "
            f"axes={self.axes}, "
            f"{label})"
        )


def resize_degrees(
    data: npt.NDArray,
    *,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output: Optional[Union[npt.NDArray, np.dtype]] = None,
    output_size: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
    interp_degree: int = 3,
    analy_degree: int = -1,
    synthe_degree: Optional[int] = None,
) -> npt.NDArray:
    """
    Resize an *N*-dimensional array using explicit spline degrees.

    This is the most general entry point: it exposes the three degrees:

      - interp_degree : degree of the interpolation B-spline φ (0..3)
      - analy_degree  : analysis spline degree (-1..3, -1 = no projection)
      - synthe_degree : synthesis spline degree (0..3)

    Use this function for custom projection studies, including advanced or
    reference equal-degree least-squares configurations. For routine
    downsampling, prefer :func:`resize` with one of the oblique antialiasing
    presets.

    Parameters
    ----------
    data : ndarray
        Input array.
    zoom_factors : float or sequence of float, optional
        Scale factors for the selected *axes*. A scalar broadcasts to every
        selected axis. Ignored if *output_size* is given.
    output : ndarray or dtype, optional
        If an ``ndarray`` is supplied, the result is written **in-place** into
        that array and returned. If a ``dtype`` is supplied, a new array of that
        dtype is allocated and returned.
    output_size : sequence of int, optional
        Desired lengths for the selected *axes* (overrides *zoom_factors*).
    axes : sequence of int, optional
        Unique axes to resize. Negative axes are normalized. By default all
        axes are resized; unselected axes retain their original lengths and
        are passed to the backend as identity geometry.
    interp_degree : int, default 3
        Degree of the interpolation B-spline φ (0..3).
    analy_degree : int, default -1
        Degree of the analysis spline φ₁:

          - -1 → no projection (pure interpolation)
          - 0..3 → projection-based resizing (antialiasing, equal-degree projection, etc.)

    synthe_degree : int, optional
        Degree of the synthesis spline φ₂ (output space). Defaults to
        ``interp_degree``. Must be in [0..3] and <= ``interp_degree``.
    Returns
    -------
    ndarray
        Resized data: either a new array or the one supplied via *output*.

    Notes
    -----
    Requested zooms determine integer output lengths using half-away-from-zero
    rounding. Resampling then uses the unique endpoint-aligned scale
    ``(M - 1) / (N - 1)``. For a one-point output, interpolation evaluates at
    the symmetric input centre while projection returns the per-line mean.
    A one-point input is treated as a constant and replicated.

    Real integer and floating inputs are accepted. ``float32`` input preserves
    ``float32`` output; all other accepted inputs produce ``float64`` unless an
    explicit real output array or dtype is supplied.
    """
    if synthe_degree is None:
        synthe_degree = interp_degree

    interp_degree, analy_degree, synthe_degree = _validate_degrees(
        interp_degree, analy_degree, synthe_degree
    )

    arr = _normalize_input(data)
    selected_axes, full_zoom, output_shape = _resolve_geometry_for_shape(
        arr.shape,
        zoom_factors=zoom_factors,
        output_size=output_size,
        axes=axes,
    )
    _validate_output(output, output_shape)

    # ----------------------------
    # Dispatch to native or fallback
    # ----------------------------
    _require_native_if_requested()
    use_cpp = _HAS_CPP and (_ACCEL_ENV != "never")

    if use_cpp:
        native_dtype = (
            np.dtype(np.float32)
            if arr.dtype == np.dtype(np.float32)
            else np.dtype(np.float64)
        )
        can_write_native_direct = (
            isinstance(output, np.ndarray)
            and output.dtype == native_dtype
            and output.flags.c_contiguous
            and output.flags.aligned
            and output.flags.writeable
            and not np.may_share_memory(arr, output)
        )
        if can_write_native_direct:
            return _resize_nd_into_cpp(
                arr,
                output,
                list(full_zoom),
                int(interp_degree),
                int(analy_degree),
                int(synthe_degree),
                list(selected_axes),
            )
        output_data = _resize_nd_cpp(
            arr,
            list(full_zoom),
            int(interp_degree),
            int(analy_degree),
            int(synthe_degree),
            list(selected_axes),
        )
    else:
        # Pure-Python fallback driven directly by the degree triple
        output_data = _python_fallback_resize(
            arr,
            full_zoom,
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            axes=selected_axes,
        )

    if tuple(output_data.shape) != output_shape:
        raise RuntimeError(
            "resize backend disagrees with the resolved output shape: "
            f"{output_data.shape} != {output_shape}"
        )

    # ----------------------------
    # Final casting / in-place write
    # ----------------------------
    if output is not None:
        if isinstance(output, np.ndarray):
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
    output_size: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
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
        Scale factors for the selected *axes*. A scalar broadcasts to every
        selected axis. Ignored if *output_size* is given.
    output : ndarray or dtype, optional
        If an ``ndarray`` is supplied, the result is written **in-place** into
        that array and returned. If a ``dtype`` is supplied, a new array of that
        dtype is allocated and returned.
    output_size : sequence of int, optional
        Desired lengths for the selected *axes* (overrides *zoom_factors*).
    axes : sequence of int, optional
        Unique axes to resize. Negative axes are accepted. By default all
        axes are resized; non-selected axes remain unchanged.
    method : str
        Preset selecting a specific (interp_degree, analy_degree, synthe_degree)
        triple.

        Interpolation (no anti-aliasing, analy = -1):

          - ``"fast"``      – degree 0 (nearest)
          - ``"linear"``    – degree 1
          - ``"quadratic"`` – degree 2
          - ``"cubic"``     – degree 3

        Antialiasing (oblique projection, recommended for downsampling):

          - ``"linear-antialiasing"``    – (interp=1, analy=0, synthe=1)
          - ``"quadratic-antialiasing"`` – (interp=2, analy=1, synthe=2)
          - ``"cubic-antialiasing"``     – (interp=3, analy=1, synthe=3)

        Equal-degree least-squares projection is available through
        :func:`resize_degrees` for advanced/reference use, but is not exposed
        as a routine preset.

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
        axes=axes,
        interp_degree=interp_degree,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
    )
