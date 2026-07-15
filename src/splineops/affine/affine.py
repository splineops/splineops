# splineops/src/splineops/affine/affine.py

import json
import numbers
import operator
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence, Tuple
import zipfile
import numpy as np
import numpy.typing as npt
from splineops.spline_interpolation._prefilter import (
    prefilter_interpolation_coefficients,
)
from splineops.spline_interpolation.tensor_spline import TensorSpline
from splineops.spline_interpolation.query_plan import TensorSplineGeometryPlan

_AFFINE_TILE_SIZE = 65_536
_DEFAULT_PLAN_BYTES = 256 * 1024**2
_AFFINE_FIELD_TOKEN = object()


def _normalize_spatial_axes(
    data: np.ndarray,
    spatial_ndim: int,
    spatial_axes: Sequence[int] | None,
) -> tuple[int, ...]:
    if spatial_axes is None:
        if data.ndim != spatial_ndim:
            raise ValueError(
                "'spatial_axes' is required when data contains batch or channel axes."
            )
        return tuple(range(spatial_ndim))
    try:
        axes = tuple(operator.index(axis) for axis in spatial_axes)
    except TypeError as exc:
        raise TypeError("'spatial_axes' must be a sequence of integer axes.") from exc
    if len(axes) != spatial_ndim:
        raise ValueError(f"'spatial_axes' must contain exactly {spatial_ndim} axes.")
    axes = tuple(axis + data.ndim if axis < 0 else axis for axis in axes)
    if any(axis < 0 or axis >= data.ndim for axis in axes) or len(set(axes)) != len(
        axes
    ):
        raise ValueError("'spatial_axes' must contain distinct valid axes.")
    return axes


class AffineCoefficientField:
    """Immutable tagged cardinal coefficients prepared by an affine plan.

    Construct fields with :meth:`AffinePlan.prepare_coefficients`.  The tag
    prevents accidentally applying coefficients with a different input grid,
    spline degree, boundary mode, or precision.  Transform matrix and output
    shape are intentionally not part of the tag, so one prepared field can be
    reused across compatible affine geometries.
    """

    __slots__ = (
        "_values",
        "_spatial_axes",
        "_signature",
        "_configuration",
        "_serialization_metadata",
    )
    _values: np.ndarray
    _spatial_axes: tuple[int, ...]
    _signature: object
    _configuration: dict[str, object]
    _serialization_metadata: dict[str, object]

    def __init__(
        self,
        values,
        spatial_axes,
        signature,
        configuration,
        serialization_metadata,
        *,
        _token=None,
    ):
        if _token is not _AFFINE_FIELD_TOKEN:
            raise TypeError(
                "Construct coefficient fields with "
                "AffinePlan.prepare_coefficients() or load_coefficients()."
            )
        object.__setattr__(self, "_values", values)
        self._values.flags.writeable = False
        object.__setattr__(self, "_spatial_axes", tuple(spatial_axes))
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(self, "_configuration", dict(configuration))
        object.__setattr__(
            self, "_serialization_metadata", dict(serialization_metadata)
        )

    def __setattr__(self, name, value):
        raise AttributeError("AffineCoefficientField is immutable.")

    @property
    def values(self) -> np.ndarray:
        """A copy of the coefficient values."""

        return self._values.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def dtype(self) -> np.dtype:
        return self._values.dtype

    @property
    def nbytes(self) -> int:
        return self._values.nbytes

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return self._spatial_axes

    @property
    def configuration(self) -> dict[str, object]:
        """Copy of the coefficient-space contract."""

        return dict(self._configuration)

    def incompatibility_reason(self, plan) -> str | None:
        """Explain why ``plan`` cannot consume the field, or return ``None``."""

        if not isinstance(plan, AffinePlan):
            return "The supplied object is not an AffinePlan."
        if plan._template._geometry_signature() != self._signature:
            return (
                "Input shape, spline degree, boundary mode, or execution "
                "precision differs from the coefficient field."
            )
        return None

    def is_compatible(self, plan) -> bool:
        """Return whether ``plan`` can apply this coefficient field."""

        return self.incompatibility_reason(plan) is None

    def save(self, path) -> None:
        """Atomically save values and their compatibility contract to NPZ.

        The file contains JSON metadata and a numeric array only; loading never
        enables NumPy object pickles.  Restore it through
        :meth:`AffinePlan.load_coefficients`, which checks the stored contract
        against the receiving plan.  A temporary file in the destination
        directory is flushed and atomically replaces the target, so a failed
        write cannot leave a partially updated archive.
        """

        try:
            target = Path(os.fspath(path))
        except TypeError as exc:
            raise TypeError("'path' must be a filesystem path.") from exc
        if not target.name.endswith(".npz"):
            target = Path(f"{target}.npz")
        metadata = json.dumps(
            self._serialization_metadata, sort_keys=True, separators=(",", ":")
        )
        storage_dtype = self._values.dtype.newbyteorder("<")
        storage_values = self._values.astype(storage_dtype, copy=False)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp.npz",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                values=storage_values,
                metadata=np.asarray(metadata),
            )
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class AffinePlan:
    """Reusable pull-back affine transform on a fixed spatial geometry.

    The plan can cache spline support indexes and weights for repeated frames.
    Non-spatial axes are treated independently, so explicit ``spatial_axes``
    provide batch and channel support without changing :class:`TensorSpline`.

    Parameters
    ----------
    input_shape : tuple of int
        Spatial input shape; two and three dimensions are supported.
    matrix : ndarray
        Pull-back matrix mapping output coordinates to input coordinates.
    offset : ndarray, optional
        Pull-back offset.  Defaults to zero.
    output_shape : tuple of int, optional
        Spatial output shape.  Defaults to ``input_shape``.
    degree : int, optional
        B-spline degree from 0 through 7.
    mode : str, optional
        TensorSpline boundary extension mode.
    dtype : dtype, optional
        Floating execution precision.
    cache_geometry : bool, optional
        Cache support indexes and weights.  Disable for one-shot transforms.
    max_retained_bytes : int, optional
        Hard limit for cached geometry arrays.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        matrix: npt.ArrayLike,
        offset: npt.ArrayLike | None = None,
        *,
        output_shape: tuple[int, ...] | None = None,
        degree: int = 3,
        mode: str = "zero",
        dtype: npt.DTypeLike = np.float64,
        cache_geometry: bool = True,
        max_retained_bytes: int = _DEFAULT_PLAN_BYTES,
    ) -> None:
        try:
            input_shape = tuple(operator.index(length) for length in input_shape)
        except TypeError as exc:
            raise TypeError("'input_shape' must be a sequence of integers.") from exc
        if len(input_shape) not in (2, 3) or any(length <= 0 for length in input_shape):
            raise ValueError(
                "'input_shape' must contain two or three positive lengths."
            )
        if output_shape is None:
            output_shape = input_shape
        try:
            output_shape = tuple(operator.index(length) for length in output_shape)
        except TypeError as exc:
            raise TypeError("'output_shape' must be a sequence of integers.") from exc
        if len(output_shape) != len(input_shape) or any(
            length <= 0 for length in output_shape
        ):
            raise ValueError(
                "'output_shape' must contain positive lengths matching input dimensionality."
            )
        try:
            degree = operator.index(degree)
        except TypeError as exc:
            raise TypeError("'degree' must be an integer from 0 through 7.") from exc
        if isinstance(degree, (bool, np.bool_)) or not 0 <= degree <= 7:
            raise ValueError("'degree' must be an integer from 0 through 7.")
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("'dtype' must be a real floating dtype.")
        if not isinstance(cache_geometry, (bool, np.bool_)):
            raise TypeError("'cache_geometry' must be a boolean.")
        try:
            max_retained_bytes = operator.index(max_retained_bytes)
        except TypeError as exc:
            raise TypeError("'max_retained_bytes' must be a positive integer.") from exc
        if max_retained_bytes <= 0:
            raise ValueError("'max_retained_bytes' must be a positive integer.")

        ndim = len(input_shape)
        matrix = np.asarray(matrix, dtype=dtype)
        if matrix.shape != (ndim, ndim):
            raise ValueError(f"'matrix' must have shape {(ndim, ndim)}.")
        if offset is None:
            offset = np.zeros(ndim, dtype=dtype)
        offset = np.asarray(offset, dtype=dtype)
        if offset.shape != (ndim,):
            raise ValueError(f"'offset' must have shape {(ndim,)}.")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
            raise ValueError("'matrix' and 'offset' must contain only finite values.")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.matrix = np.array(matrix, copy=True)
        self.offset = np.array(offset, copy=True)
        self.matrix.flags.writeable = False
        self.offset.flags.writeable = False
        self.degree = degree
        self.mode = mode
        self.dtype = dtype
        self._basis = f"bspline{degree}"
        self._geometry_plans: tuple[TensorSplineGeometryPlan, ...] = ()
        self._retained_bytes = 0
        coordinates = tuple(
            np.arange(length, dtype=dtype) for length in self.input_shape
        )
        self._template = TensorSpline(
            data=np.zeros(self.input_shape, dtype=dtype),
            coordinates=coordinates,
            bases=self._basis,
            modes=mode,
        )

        if cache_geometry:
            support_entries = sum(basis.support for basis in self._template.bases)
            estimated_bytes = (
                int(np.prod(self.output_shape, dtype=np.int64))
                * support_entries
                * (np.dtype(np.int64).itemsize + dtype.itemsize)
            )
            if estimated_bytes > max_retained_bytes:
                raise MemoryError(
                    "AffinePlan geometry would retain approximately "
                    f"{estimated_bytes} bytes, above max_retained_bytes="
                    f"{max_retained_bytes}. Disable geometry caching for a "
                    "bounded-memory one-shot transform."
                )
            plans: list[TensorSplineGeometryPlan] = []
            output_size = int(np.prod(self.output_shape, dtype=np.int64))
            for start in range(0, output_size, _AFFINE_TILE_SIZE):
                stop = min(start + _AFFINE_TILE_SIZE, output_size)
                transformed = self._transformed_coordinates(start, stop)
                remaining = max_retained_bytes - self._retained_bytes
                geometry_plan: TensorSplineGeometryPlan = self._template.query_plan(
                    tuple(transformed),
                    grid=False,
                    max_retained_bytes=remaining,
                ).detach()
                self._retained_bytes += geometry_plan.retained_bytes
                plans.append(geometry_plan)
            self._geometry_plans = tuple(plans)

    @property
    def geometry_cached(self) -> bool:
        """Whether spline supports and weights are retained by this plan."""

        return bool(self._geometry_plans)

    @property
    def retained_bytes(self) -> int:
        """Bytes retained by geometry and the reusable spline template."""

        return self.geometry_retained_bytes + self.template_retained_bytes

    @property
    def geometry_retained_bytes(self) -> int:
        """Bytes retained by cached spline supports and weights."""

        return self._retained_bytes

    @property
    def template_retained_bytes(self) -> int:
        """Bytes retained by template coefficients and construction coordinates."""

        return self._template._coefficients.nbytes + sum(
            coordinate.nbytes for coordinate in self._template._coordinates
        )

    @property
    def configuration(self) -> dict[str, object]:
        """Copy of the fixed numerical contract represented by this plan."""

        return {
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "degree": self.degree,
            "mode": self.mode,
            "dtype": self.dtype.str,
            "geometry_cached": self.geometry_cached,
        }

    def _transformed_coordinates(self, start: int, stop: int) -> np.ndarray:
        flat_indices = np.arange(start, stop)
        coordinates = np.asarray(
            np.unravel_index(flat_indices, self.output_shape), dtype=self.dtype
        )
        return self.matrix @ coordinates + self.offset[:, np.newaxis]

    def _prefilter_canonical(self, data: np.ndarray) -> np.ndarray:
        spatial_ndim = len(self.input_shape)
        axes = tuple(range(data.ndim - spatial_ndim, data.ndim))
        return prefilter_interpolation_coefficients(
            data,
            bases=self._template.bases,
            modes=self._template.modes,
            axes=axes,
            dtype=self.dtype,
            backend="numpy",
        )

    def _coefficient_serialization_metadata(self, spatial_axes, *, schema_version=2):
        mode_types = [
            f"{type(mode).__module__}.{type(mode).__qualname__}"
            for mode in self._template.modes
        ]
        metadata = {
            "schema": "splineops.affine-coefficients",
            "schema_version": schema_version,
            "input_shape": list(self.input_shape),
            "spatial_axes": list(spatial_axes),
            "degree": self.degree,
            "mode_types": mode_types,
        }
        if schema_version == 1:
            metadata["dtype"] = self.dtype.str
        elif schema_version == 2:
            metadata["dtype"] = self.dtype.name
            metadata["storage_byte_order"] = "little"
        else:
            raise ValueError(
                f"Unsupported affine coefficient schema version {schema_version}."
            )
        return metadata

    def _coefficient_field(self, values, axes):
        return AffineCoefficientField(
            values,
            axes,
            self._template._geometry_signature(),
            {
                "input_shape": self.input_shape,
                "spatial_axes": axes,
                "degree": self.degree,
                "mode": self.mode,
                "dtype": self.dtype.str,
            },
            self._coefficient_serialization_metadata(axes),
            _token=_AFFINE_FIELD_TOKEN,
        )

    def _apply_canonical_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        spatial_ndim = len(self.input_shape)
        batch_shape = coefficients.shape[:-spatial_ndim]
        batch_count = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1
        output_size = int(np.prod(self.output_shape, dtype=np.int64))
        result = np.empty(batch_shape + (output_size,), dtype=self.dtype)
        if self.geometry_cached:
            start = 0
            for geometry_plan in self._geometry_plans:
                stop = start + int(np.prod(geometry_plan.output_shape, dtype=np.int64))
                result[..., start:stop] = geometry_plan._apply_coefficient_array(
                    self._template, coefficients
                )
                start = stop
        else:
            tile_size = max(1, _AFFINE_TILE_SIZE // batch_count)
            for start in range(0, output_size, tile_size):
                stop = min(start + tile_size, output_size)
                transformed = self._transformed_coordinates(start, stop)
                indexes = []
                weights = []
                for axis, coordinate in enumerate(transformed):
                    axis_indexes, axis_weights = self._template._compute_support(
                        axis, coordinate
                    )
                    indexes.append(axis_indexes)
                    weights.append(axis_weights)
                result[..., start:stop] = (
                    self._template._evaluate_precomputed_point_chunk(
                        tuple(indexes),
                        tuple(weights),
                        coefficients=coefficients,
                    )
                )
        return result.reshape(batch_shape + self.output_shape)

    def _normalize_input(self, data, spatial_axes, *, coefficients):
        if not isinstance(data, np.ndarray):
            raise TypeError("'data' must be a NumPy array.")
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise TypeError("'data' must have a real numeric dtype.")
        if coefficients and data.dtype != self.dtype:
            raise TypeError(
                f"Coefficient arrays must have dtype {self.dtype}; received {data.dtype}."
            )
        axes = _normalize_spatial_axes(data, len(self.input_shape), spatial_axes)
        if tuple(data.shape[axis] for axis in axes) != self.input_shape:
            raise ValueError(
                "The dimensions selected by 'spatial_axes' must match input_shape."
            )
        nonspatial_axes = tuple(axis for axis in range(data.ndim) if axis not in axes)
        permutation = nonspatial_axes + axes
        canonical = np.transpose(data, permutation)
        if not coefficients:
            canonical = canonical.astype(self.dtype, copy=False)
        batch_shape = canonical.shape[: len(nonspatial_axes)]
        return axes, permutation, batch_shape, canonical

    @staticmethod
    def _copy_output(result, out):
        if out is None:
            return result
        if not isinstance(out, np.ndarray):
            raise TypeError("'out' must be a NumPy array.")
        if out.shape != result.shape:
            raise ValueError(
                f"'out' must have shape {result.shape}; received {out.shape}."
            )
        if out.dtype != result.dtype:
            raise TypeError(
                f"'out' must have dtype {result.dtype}; received {out.dtype}."
            )
        np.copyto(out, result, casting="no")
        return out

    def prefilter(
        self,
        data: npt.NDArray,
        *,
        spatial_axes: Sequence[int] | None = None,
        out: npt.NDArray | None = None,
    ) -> np.ndarray:
        """Return cardinal coefficients for reuse across affine geometries."""

        _, permutation, _, canonical = self._normalize_input(
            data, spatial_axes, coefficients=False
        )
        canonical = self._prefilter_canonical(canonical)
        result = np.ascontiguousarray(
            np.transpose(canonical, tuple(np.argsort(permutation)))
        )
        return self._copy_output(result, out)

    def prepare_coefficients(
        self,
        data: npt.NDArray,
        *,
        spatial_axes: Sequence[int] | None = None,
    ) -> AffineCoefficientField:
        """Return immutable coefficients with a checked compatibility tag.

        Prefer this method when coefficients will be reused.  ``prefilter``
        remains available for raw-array and explicit-output workflows.
        """

        axes, permutation, _, canonical = self._normalize_input(
            data, spatial_axes, coefficients=False
        )
        canonical = self._prefilter_canonical(canonical)
        values = np.ascontiguousarray(
            np.transpose(canonical, tuple(np.argsort(permutation)))
        )
        return self._coefficient_field(values, axes)

    def load_coefficients(self, path) -> AffineCoefficientField:
        """Load and validate a field saved by ``AffineCoefficientField.save``."""

        try:
            source = open(os.fspath(path), "rb")
        except (OSError, TypeError) as exc:
            raise ValueError("Invalid affine coefficient archive.") from exc
        with source:
            try:
                archive = np.load(source, allow_pickle=False)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError("Invalid affine coefficient archive.") from exc
            with archive:
                if set(archive.files) != {"values", "metadata"}:
                    raise ValueError(
                        "Coefficient archive must contain only 'values' and 'metadata'."
                    )
                try:
                    values = np.array(archive["values"], copy=True, order="C")
                    metadata_value = archive["metadata"]
                    if metadata_value.shape != ():
                        raise ValueError(
                            "Coefficient archive metadata must be scalar JSON."
                        )
                    metadata = json.loads(str(metadata_value.item()))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Invalid affine coefficient archive.") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Coefficient archive metadata must be a JSON object.")
        try:
            axes = tuple(operator.index(axis) for axis in metadata["spatial_axes"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Coefficient archive is missing valid spatial-axis metadata."
            ) from exc
        schema_version = metadata.get("schema_version")
        if schema_version not in (1, 2):
            raise ValueError("Coefficient archive uses an unsupported schema version.")
        expected = self._coefficient_serialization_metadata(
            axes, schema_version=schema_version
        )
        if metadata != expected:
            raise ValueError(
                "Coefficient archive is incompatible with this plan's input "
                "shape, degree, boundary mode, precision, or schema."
            )
        if values.dtype.name != self.dtype.name:
            raise ValueError(
                "Coefficient archive values have an incompatible numerical dtype."
            )
        values = values.astype(self.dtype, copy=False)
        self._normalize_input(values, axes, coefficients=True)
        return self._coefficient_field(values, axes)

    def _apply(
        self,
        data: npt.NDArray,
        *,
        spatial_axes: Sequence[int] | None = None,
        out: npt.NDArray | None = None,
        coefficients: bool,
    ) -> np.ndarray:
        _, permutation, _, canonical = self._normalize_input(
            data, spatial_axes, coefficients=coefficients
        )
        if not coefficients:
            canonical = self._prefilter_canonical(canonical)
        canonical_result = self._apply_canonical_coefficients(canonical)
        result = np.ascontiguousarray(
            np.transpose(canonical_result, tuple(np.argsort(permutation)))
        )
        return self._copy_output(result, out)

    def apply(
        self,
        data: npt.NDArray,
        *,
        spatial_axes: Sequence[int] | None = None,
        out: npt.NDArray | None = None,
    ) -> np.ndarray:
        """Apply the fixed transform to samples or independent batches/channels."""

        return self._apply(data, spatial_axes=spatial_axes, out=out, coefficients=False)

    def apply_coefficients(
        self,
        coefficients: npt.NDArray | AffineCoefficientField,
        *,
        spatial_axes: Sequence[int] | None = None,
        out: npt.NDArray | None = None,
    ) -> np.ndarray:
        """Apply the transform without prefiltering cardinal coefficients again.

        Tagged fields from :meth:`prepare_coefficients` are checked against
        this plan.  Raw arrays from :meth:`prefilter` remain supported for
        backward compatibility; their provenance is necessarily the caller's
        responsibility.
        """

        if isinstance(coefficients, AffineCoefficientField):
            reason = coefficients.incompatibility_reason(self)
            if reason is not None:
                raise ValueError(f"Incompatible coefficient field: {reason}")
            if spatial_axes is None:
                spatial_axes = coefficients.spatial_axes
            else:
                axes = _normalize_spatial_axes(
                    coefficients._values, len(self.input_shape), spatial_axes
                )
                if axes != coefficients.spatial_axes:
                    raise ValueError(
                        "'spatial_axes' differs from the tagged coefficient field."
                    )
            coefficient_array = coefficients._values
        else:
            coefficient_array = coefficients

        return self._apply(
            coefficient_array,
            spatial_axes=spatial_axes,
            out=out,
            coefficients=True,
        )

    __call__ = apply


def affine_transform(
    data: npt.NDArray,
    matrix: npt.ArrayLike,
    offset: npt.ArrayLike | None = None,
    *,
    output_shape: tuple[int, ...] | None = None,
    spatial_axes: Sequence[int] | None = None,
    degree: int = 3,
    mode: str = "zero",
    out: npt.NDArray | None = None,
) -> np.ndarray:
    """Apply a one-shot two- or three-dimensional pull-back affine transform."""

    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be a NumPy array.")
    matrix_array = np.asarray(matrix)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError("'matrix' must be a square two- or three-dimensional matrix.")
    spatial_ndim = matrix_array.shape[0]
    if spatial_ndim not in (2, 3):
        raise ValueError("'matrix' must be a square two- or three-dimensional matrix.")
    axes = _normalize_spatial_axes(data, spatial_ndim, spatial_axes)
    input_shape = tuple(data.shape[axis] for axis in axes)
    work_dtype = (
        data.dtype if np.issubdtype(data.dtype, np.floating) else np.dtype(np.float64)
    )
    plan = AffinePlan(
        input_shape,
        matrix_array,
        offset,
        output_shape=output_shape,
        degree=degree,
        mode=mode,
        dtype=work_dtype,
        cache_geometry=False,
    )
    return plan.apply(data, spatial_axes=axes, out=out)


def rotate(
    data: npt.NDArray,
    angle: float,
    axis: Optional[Tuple[float, float, float]] = None,
    center: Optional[Tuple[float, float, float]] = None,
    degree: int = 3,
    mode: str = "zero",
    *,
    spatial_axes: Sequence[int] | None = None,
    out: npt.NDArray | None = None,
) -> npt.NDArray:
    """
    Rotate 2D or 3D data around a specified center using spline interpolation.

    Parameters
    ----------
    data : ndarray
        2D or 3D input data array to rotate.
    angle : float
        Rotation angle in degrees.
    axis : tuple of float, optional
        The axis of rotation for 3D data. Defaults to (0, 0, 1).
    center : tuple of float, optional
        The center of rotation. Defaults to the array center.
    degree : int, optional
        B-spline degree (0 to 7). Default is 3.
    mode : str, optional
        Boundary handling mode (e.g., "zero", "mirror"). Default is "zero".
    spatial_axes : sequence of int, optional
        Two or three axes to rotate.  Required for arrays that also contain
        batch or channel dimensions; every non-spatial slice is transformed
        independently.
    out : ndarray, optional
        Exact-shape and exact-dtype output buffer.

    Returns
    -------
    rotated_data : ndarray
        The data array after rotation.

    Examples
    --------
    Rotate a 2D array by 45 degrees:

    >>> import numpy as np
    >>> from splineops.affine import rotate
    >>> data = np.array([[1, 2], [3, 4]])
    >>> rotated_data = rotate(data, angle=45)
    >>> rotated_data.shape
    (2, 2)

    Rotate a 3D array around a custom axis:

    >>> data_3d = np.random.rand(4, 4, 4)
    >>> rotated_data_3d = rotate(data_3d, angle=30, axis=(1, 0, 0))
    >>> rotated_data_3d.shape
    (4, 4, 4)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be a NumPy array.")
    if spatial_axes is None:
        ndim = data.ndim
    else:
        try:
            spatial_axes = tuple(spatial_axes)
        except TypeError as exc:
            raise TypeError(
                "'spatial_axes' must be a sequence of integer axes."
            ) from exc
        ndim = len(spatial_axes)
    if ndim not in (2, 3):
        raise ValueError("rotate: exactly two or three spatial axes are supported.")
    axes = _normalize_spatial_axes(data, ndim, spatial_axes)
    input_shape = tuple(data.shape[axis] for axis in axes)
    if any(length == 0 for length in input_shape):
        raise ValueError("'data' dimensions must be non-empty.")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise TypeError("'data' must have a real numeric dtype.")
    if not isinstance(angle, numbers.Real) or isinstance(angle, (bool, np.bool_)):
        raise TypeError("'angle' must be a real number.")
    if not np.isfinite(angle):
        raise ValueError("'angle' must be finite.")

    # Integer samples describe values, not an integer interpolation space.
    # Promote them predictably; preserve supported floating precision.
    work_dtype = (
        data.dtype if np.issubdtype(data.dtype, np.floating) else np.dtype(np.float64)
    )

    # Use specified center or default to array center
    if center is None:
        center_coords = [(dim - 1) / 2.0 for dim in input_shape]
    else:
        if len(center) != ndim:
            raise ValueError("center must have same length as data.ndim")
        center_coords = [float(c) for c in center]
        if not all(np.isfinite(c) for c in center_coords):
            raise ValueError("'center' entries must be finite.")

    # Convert angle to radians
    angle_rad = np.radians(angle)

    if ndim == 2:
        # 2D rotation matrix (pull-back, so angle is negated)
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        R = np.array(
            [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
            dtype=work_dtype,
        )
    else:  # ndim == 3
        # Default axis of rotation (z-axis) if not provided
        if axis is None:
            axis = (0.0, 0.0, 1.0)
        if len(axis) != 3:
            raise ValueError("'axis' must contain three values for 3D rotation.")
        axis_vec = np.array(axis, dtype=work_dtype)
        if not np.all(np.isfinite(axis_vec)):
            raise ValueError("'axis' entries must be finite.")
        norm = np.linalg.norm(axis_vec)
        if norm == 0:
            raise ValueError("axis must be non-zero for 3D rotation")
        axis_vec /= norm  # normalize

        ux, uy, uz = axis_vec
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        one_minus_cos = 1.0 - cos_angle

        R = np.array(
            [
                [
                    cos_angle + ux**2 * one_minus_cos,
                    ux * uy * one_minus_cos - uz * sin_angle,
                    ux * uz * one_minus_cos + uy * sin_angle,
                ],
                [
                    uy * ux * one_minus_cos + uz * sin_angle,
                    cos_angle + uy**2 * one_minus_cos,
                    uy * uz * one_minus_cos - ux * sin_angle,
                ],
                [
                    uz * ux * one_minus_cos - uy * sin_angle,
                    uz * uy * one_minus_cos + ux * sin_angle,
                    cos_angle + uz**2 * one_minus_cos,
                ],
            ],
            dtype=work_dtype,
        )

    center_array = np.asarray(center_coords, dtype=work_dtype)
    offset = center_array - R @ center_array
    plan = AffinePlan(
        input_shape,
        R,
        offset,
        degree=degree,
        mode=mode,
        dtype=work_dtype,
        cache_geometry=False,
    )
    return plan.apply(data, spatial_axes=axes, out=out)
