# splineops/src/splineops/spline_interpolation/tensors_pline.py

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import math
from typing import Sequence, Union, Tuple, cast

from .bases.spline_basis import SplineBasis
from .bases.utils import asbasis
from .modes.extension_mode import ExtensionMode
from .modes.utils import asmode
from .utils import is_cupy_type, is_ndarray
from ._prefilter import prefilter_interpolation_coefficients

TSplineBasis = Union[SplineBasis, str]
TSplineBases = Union[TSplineBasis, Sequence[TSplineBasis]]
TExtensionMode = Union[ExtensionMode, str]
TExtensionModes = Union[TExtensionMode, Sequence[TExtensionMode]]


class TensorSpline:
    """
    A class to handle a tensor spline for multi-dimensional interpolation and approximation.

    This class handles N-dimensional data and allows interpolation with a variety
    of spline bases and extension modes. It supports different bases/modes along
    each axis.

    Parameters
    ----------
    data : ndarray
        The input N-dimensional array to be interpolated.
    coordinates : ndarray
        The coordinates corresponding to the input data.
    bases : str or sequence of str
        The spline bases used for interpolation. It can be a single basis applied across all axes or a sequence of bases for each axis.

        The following spline bases are available:

        - **"bspline0"**, **"bspline0-sym"**: Zero-degree or piecewise constant B-splines and symmetric zero-degree or piecewise constant B-splines.
        - **"bspline1"** to **"bspline9"**: First to ninth-degree B-splines.
        - **"omoms0"**, **"omoms0-sym"**: Zero-degree O-MOMS splines and symmetric zero-degree O-MOMS splines.
        - **"omoms1"** to **"omoms5"**: First to fifth-degree O-MOMS splines.
        - **"omoms2-sym"**, **"omoms4-sym"**: Symmetric second and fourth-degree O-MOMS splines.
        - **"nearest"**, **"nearest-sym"**: Nearest neighbor interpolation.
        - **"linear"**: Linear interpolation.
        - **"keys"**: Keys spline interpolation.

    modes : str or sequence of str
        Signal extension modes used to handle boundaries. It can be a single mode applied across all axes or a sequence of modes for each axis.

        The following extension modes are available for handling boundaries:

        - **"zero" (0 0 0 0 | a b c d | 0 0 0 0)** The input is extended by filling all values beyond the boundary with zeroes.
        - **"mirror" (d c b | a b c d | c b a)** The input is extended by reflecting around the center of the data points adjacent to the border.
        - **"periodic"(d c b | a b c d | a b c)**    The signal is wrapped around cyclically.

    Example
    -------
    1. **1D Interpolation:**

    Here's an example to illustrate 1-dimensional interpolation using the TensorSpline class.

    >>> import numpy as np
    >>> from splineops import TensorSpline
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> coordinates = np.linspace(0, data.size - 1, data.size)
    >>> bases = "linear"  # Linear interpolation
    >>> modes = "mirror"  # Mirror extension mode
    >>> tensor_spline = TensorSpline(data=data, coordinates=coordinates, bases=bases, modes=modes)

    To interpolate the data at a new point:

    >>> eval_coords = np.array([1.5])
    >>> data_eval = tensor_spline(coordinates=eval_coords, grid=False)
    >>> print(data_eval)
    [2.5]

    In this example, the interpolated value at `x = 1.5` is `2.5`, which is the midpoint between `data[1]` (2.0) and `data[2]` (3.0).

    2. **2D Interpolation:**

    Here's a simple example to illustrate 2-dimensional interpolation using the TensorSpline class.

    >>> a = np.arange(12.).reshape((4, 3))
    >>> a
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [ 9., 10., 11.]])
    >>> xx = np.linspace(0, a.shape[0] - 1, a.shape[0])
    >>> yy = np.linspace(0, a.shape[1] - 1, a.shape[1])
    >>> coordinates = xx, yy
    >>> bases = ["bspline1", "bspline1"]  # Linear interpolation along both axes
    >>> modes = ["mirror", "mirror"]      # Mirror extension mode handling along both axes
    >>> tensor_spline = TensorSpline(data=a, coordinates=coordinates, bases=bases, modes=modes)

    To interpolate the array `a` at coordinates `(0.5, 0.5)` and `(2, 1)`:

    >>> eval_coords = np.array([[0.5, 2], [0.5, 1]])
    >>> data_eval_pts = tensor_spline(coordinates=eval_coords, grid=False)
    >>> print(data_eval_pts)
    [2. 7.]

    In this example, the interpolated value at `(0.5, 0.5)` is `2.0`, and the value at `(2, 1)` is `7.0`.

    """

    # Bounds support-index, weight, and gathered-coefficient temporaries. The
    # final output is still allocated normally; callers do not need to tune
    # this implementation detail.
    _EVALUATION_TILE_SIZE = 65_536

    def __init__(
        self,
        data: npt.NDArray,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        bases: TSplineBases,
        modes: TExtensionModes,
    ) -> None:
        """
        Initialize the TensorSpline object with the given data, coordinates, bases, and modes.

        Parameters
        ----------
        data : ndarray
            The input N-dimensional array to be interpolated.
        coordinates : ndarray
            The coordinates corresponding to the input data.
        bases : str or sequence of str
            The spline bases used for interpolation. It can be a single basis applied across all axes or a sequence of bases for each axis.
        modes : str or sequence of str
            Signal extension modes used to handle boundaries. It can be a single mode applied across all axes or a sequence of modes for each axis.

        Example
        -------
        >>> a = np.arange(12.).reshape((4, 3))
        >>> xx = np.linspace(0, a.shape[0] - 1, a.shape[0])
        >>> yy = np.linspace(0, a.shape[1] - 1, a.shape[1])
        >>> coordinates = xx, yy
        >>> tensor_spline = TensorSpline(data=a, coordinates=coordinates, bases="bspline1", modes="mirror")
        """
        # Data
        if not is_ndarray(data):
            raise TypeError("'data' must be a NumPy or CuPy array.")
        if data.ndim == 0:
            raise ValueError("'data' must have at least one dimension.")
        if any(length == 0 for length in data.shape):
            raise ValueError("'data' dimensions must be non-empty.")
        ndim = data.ndim
        self._ndim = ndim

        coordinates = self._normalize_coordinate_sequence(
            coordinates, argument="coordinates", allow_stacked_points=False
        )
        if any(c.ndim != 1 for c in coordinates):
            raise ValueError("Construction coordinates must be one-dimensional.")
        if any(c.size == 0 for c in coordinates):
            raise ValueError("Construction coordinates must be non-empty.")
        if any(is_cupy_type(c) != is_cupy_type(data) for c in coordinates):
            raise TypeError("'data' and construction coordinates must use one backend.")
        if not all(np.isrealobj(c) for c in coordinates):
            raise ValueError("Construction coordinates must be real numbers.")
        if not all(np.issubdtype(c.dtype, np.floating) for c in coordinates):
            raise TypeError("Construction coordinates must have a floating dtype.")
        if not all(bool(np.all(np.isfinite(c))) for c in coordinates):
            raise ValueError("Construction coordinates must be finite.")
        if not all(bool(np.all(np.diff(c) > 0)) for c in coordinates):
            raise ValueError("Construction coordinates must be strictly ascending.")
        for axis, c in enumerate(coordinates):
            if c.size <= 2:
                continue
            diffs = np.diff(c)
            real_dtype = c.real.dtype
            eps = np.finfo(real_dtype).eps
            step = diffs[0]
            atol = 32 * eps * max(1.0, float(abs(step)))
            if not bool(np.allclose(diffs, step, rtol=32 * eps, atol=atol)):
                raise ValueError(
                    "TensorSpline requires a uniform construction grid; "
                    f"coordinates on axis {axis} are nonuniform."
                )

        valid_data_shape = tuple(c.size for c in coordinates)
        if data.shape != valid_data_shape:
            raise ValueError(
                f"Incompatible data shape {data.shape}; expected {valid_data_shape}."
            )
        self._coordinates = tuple(c.copy() for c in coordinates)

        # Pre-computation based on coordinates
        # TODO(dperdios): convert to Python float?
        bounds = tuple([(c[0], c[-1]) for c in coordinates])
        # TODO(dperdios): `bounds` as a public property?
        self._bounds = bounds
        lengths = valid_data_shape
        self._lengths = lengths
        step_seq = []
        for b, l in zip(bounds, lengths):
            if l > 1:
                step = (b[-1] - b[0]) / (l - 1)
            else:
                # Special case for single-sample signal
                step = 1
            step_seq.append(step)
        steps = tuple(step_seq)  # TODO: convert dtype? (can be promoted)
        self._steps = steps
        # TODO(dperdios): cast scalars to real_dtype?

        # DTypes
        dtype = data.dtype
        if not (
            np.issubdtype(dtype, np.floating)
            or np.issubdtype(dtype, np.complexfloating)
        ):
            raise ValueError("Data must be an array of floating point numbers.")
        real_dtype = data.real.dtype
        coords_dtype_seq = tuple(c.dtype for c in coordinates)
        if len(set(coords_dtype_seq)) != 1:
            raise ValueError(
                "Incompatible dtypes in sequence of coordinates. "
                "Expected a consistent dtype. "
                f"Received different dtypes: {tuple(d.name for d in coords_dtype_seq)}"
            )
        coords_dtype = coords_dtype_seq[0]
        if coords_dtype.itemsize != real_dtype.itemsize:
            # TODO(dperdios): maybe automatic cast in the future?
            raise ValueError("Coordinates and data have different floating precisions.")
        self._dtype = dtype
        self._real_dtype = real_dtype

        # Bases
        if isinstance(bases, (SplineBasis, str)):
            # Explicit type cast (special case)
            bases = cast(str, bases)
            bases = ndim * (bases,)
        bases = tuple(asbasis(b) for b in bases)
        if len(bases) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._bases = bases

        # Modes
        if isinstance(modes, (ExtensionMode, str)):
            # Explicit type cast (special case)
            modes = cast(str, modes)
            modes = ndim * (modes,)
        modes = tuple(asmode(m) for m in modes)
        if len(modes) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._modes = modes

        # Compute coefficients
        coefficients = self._compute_coefficients(data=data)
        self._coefficients = coefficients

    # Properties
    @property
    def coefficients(self) -> npt.NDArray:
        return np.copy(self._coefficients)

    @property
    def coordinates(self) -> Tuple[npt.NDArray, ...]:
        """Copies of the uniform construction coordinates for every axis."""
        return tuple(c.copy() for c in self._coordinates)

    @property
    def bases(self) -> Tuple[SplineBasis, ...]:
        return self._bases

    @property
    def modes(self) -> Tuple[ExtensionMode, ...]:
        return self._modes

    @property
    def ndim(self):
        return self._ndim

    def coefficients_from_data(
        self,
        data: npt.NDArray,
        *,
        out: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Prefilter new samples on this spline's construction geometry.

        This is the explicit boundary between samples and cardinal spline
        coefficients.  It is useful when one coefficient field will be
        evaluated by several geometry plans.  Changing samples still require
        prefiltering; this method avoids repeating construction-grid and basis
        validation, not the mathematically required filter itself.

        Parameters
        ----------
        data : ndarray
            Floating or complex-floating samples with the template shape,
            dtype, and array backend.
        out : ndarray, optional
            Exact-shape, exact-dtype destination for the coefficients.
        """

        self._validate_compatible_field(data, argument="data")
        coefficients = self._compute_coefficients(data)
        return self._copy_compatible_field(coefficients, out, argument="out")

    def with_data(self, data: npt.NDArray) -> TensorSpline:
        """Return a compatible spline fitted to new samples.

        The construction coordinates, bases, and modes are shared as immutable
        geometry.  The returned spline owns newly computed coefficients and
        the template remains unchanged.
        """

        coefficients = self.coefficients_from_data(data)
        return self._clone_with_coefficients(coefficients, copy=False)

    def with_coefficients(
        self,
        coefficients: npt.NDArray,
        *,
        copy: bool = True,
    ) -> TensorSpline:
        """Return a compatible spline from precomputed cardinal coefficients.

        ``copy=True`` is the safe default.  With ``copy=False`` the caller must
        not mutate the coefficient array while the returned spline or any
        geometry plan is evaluating it.  This method intentionally performs no
        interpolation prefilter.
        """

        if not isinstance(copy, (bool, np.bool_)):
            raise TypeError("'copy' must be a boolean.")
        self._validate_compatible_field(coefficients, argument="coefficients")
        return self._clone_with_coefficients(coefficients, copy=bool(copy))

    def _validate_compatible_field(self, field, *, argument):
        if not is_ndarray(field):
            raise TypeError(f"'{argument}' must be a NumPy or CuPy array.")
        if is_cupy_type(field) != is_cupy_type(self._coefficients):
            raise TypeError(
                f"'{argument}' and the spline template must use one backend."
            )
        if field.shape != self._lengths:
            raise ValueError(
                f"'{argument}' has shape {field.shape}; expected {self._lengths}."
            )
        if field.dtype != self._dtype:
            raise TypeError(
                f"'{argument}' has dtype {field.dtype}; expected {self._dtype}."
            )

    def _copy_compatible_field(self, field, out, *, argument):
        if out is None:
            return field
        self._validate_compatible_field(out, argument=argument)
        out[...] = field
        return out

    def _clone_with_coefficients(self, coefficients, *, copy):
        xp = self._array_module()
        clone = self.__class__.__new__(self.__class__)
        clone._ndim = self._ndim
        clone._coordinates = self._coordinates
        clone._bounds = self._bounds
        clone._lengths = self._lengths
        clone._steps = self._steps
        clone._dtype = self._dtype
        clone._real_dtype = self._real_dtype
        clone._bases = self._bases
        clone._modes = self._modes
        clone._coefficients = (
            xp.array(coefficients, copy=True, order="C") if copy else coefficients
        )
        return clone

    def _normalize_coordinate_sequence(
        self,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        *,
        argument: str,
        allow_stacked_points: bool,
    ) -> Tuple[npt.NDArray, ...]:
        """Normalize supported coordinate containers without implicit casting."""
        ndim = self._ndim
        if is_ndarray(coordinates):
            array = cast(npt.NDArray, coordinates)
            if ndim == 1:
                return (array,)
            if allow_stacked_points and array.ndim >= 1 and array.shape[0] == ndim:
                return tuple(array[axis] for axis in range(ndim))
            raise ValueError(
                f"'{argument}' must be a {ndim}-item sequence of arrays"
                + (
                    " or an array whose first dimension is ndim."
                    if allow_stacked_points
                    else "."
                )
            )
        try:
            result = tuple(coordinates)
        except TypeError as exc:
            raise TypeError(
                f"'{argument}' must be an array or sequence of arrays."
            ) from exc
        if len(result) != ndim:
            raise ValueError(f"'{argument}' must contain exactly {ndim} arrays.")
        if not all(is_ndarray(c) for c in result):
            raise TypeError(
                f"Every entry in '{argument}' must be a NumPy or CuPy array."
            )
        return cast(Tuple[npt.NDArray, ...], result)

    # Methods
    def __call__(
        self,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        grid: bool = True,
        *,
        out: npt.NDArray | None = None,
        # TODO(dperdios): extrapolate?
    ) -> npt.NDArray:
        return self.eval(coordinates=coordinates, grid=grid, out=out)

    def eval(
        self,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        grid: bool = True,
        *,
        out: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """
        Evaluate the tensor spline at the given coordinates.

        Parameters
        ----------
        coordinates : ndarray
            The coordinates at which to evaluate the tensor spline. If `grid` is True, must be a sequence of 1-D arrays
            representing the grid points along each axis. If `grid` is False, must be a sequence of N-D arrays of the same shape.
        grid : bool, optional
            If True (default), assumes the input coordinates define a grid and evaluates the tensor spline over this grid.
            If False, treats the input coordinates as a list of points at which to evaluate the tensor spline.

        Returns
        -------
        data : ndarray
            The interpolated values at the specified coordinates.

        Example
        -------
        >>> a = np.arange(12.).reshape((4, 3))
        >>> xx = np.linspace(0, a.shape[0] - 1, a.shape[0])
        >>> yy = np.linspace(0, a.shape[1] - 1, a.shape[1])
        >>> coordinates = xx, yy
        >>> tensor_spline = TensorSpline(data=a, coordinates=coordinates, bases="bspline1", modes="mirror")
        >>> eval_coords = np.array([[0.5, 2], [0.5, 1]])
        >>> data_eval_pts = tensor_spline.eval(coordinates=eval_coords, grid=False)
        >>> print(data_eval_pts)
        [2. 7.]
        """
        coordinates = self._prepare_evaluation_coordinates(coordinates, grid=grid)

        result = (
            self._evaluate_grid(coordinates)
            if grid
            else self._evaluate_points(coordinates)
        )
        return self._copy_result_to_output(result, out)

    def _copy_result_to_output(self, result, out):
        """Validate an optional caller buffer and copy *result* into it."""
        if out is None:
            return result
        if not is_ndarray(out):
            raise TypeError("'out' must be a NumPy or CuPy array.")
        if is_cupy_type(out) != is_cupy_type(self._coefficients):
            raise TypeError("'out' and spline coefficients must use one backend.")
        if out.shape != result.shape:
            raise ValueError(
                f"'out' has shape {out.shape}; expected exact shape {result.shape}."
            )
        if out.dtype != result.dtype:
            raise TypeError(
                f"'out' has dtype {out.dtype}; expected exact dtype {result.dtype}."
            )
        out[...] = result
        return out

    def _prepare_evaluation_coordinates(
        self,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        *,
        grid: bool,
    ) -> Tuple[npt.NDArray, ...]:
        """Validate and normalize coordinates for evaluation or query plans."""
        # Grid queries may have shared leading batch
        # dimensions; their last dimension contains the coordinates for one
        # spline axis. Point queries use same-shape arrays, one per spline axis.
        if grid:
            coordinates = self._normalize_coordinate_sequence(
                coordinates, argument="coordinates", allow_stacked_points=False
            )
            if any(c.ndim < 1 for c in coordinates):
                raise ValueError(
                    "Grid coordinate arrays must have at least one dimension."
                )
            batch_shapes = {c.shape[:-1] for c in coordinates}
            if len(batch_shapes) != 1:
                raise ValueError(
                    "Grid coordinate arrays must have identical leading batch dimensions."
                )
        else:
            coordinates = self._normalize_coordinate_sequence(
                coordinates, argument="coordinates", allow_stacked_points=True
            )
            coords_shapes = [c.shape for c in coordinates]
            if len(set(coords_shapes)) != 1:
                raise ValueError(
                    "Point coordinate arrays must have the same shape; "
                    f"received {coords_shapes}."
                )
        if any(
            is_cupy_type(c) != is_cupy_type(self._coefficients) for c in coordinates
        ):
            raise TypeError(
                "Evaluation coordinates and spline coefficients must use one backend."
            )
        if not all(np.isrealobj(c) for c in coordinates):
            raise ValueError("Evaluation coordinates must be real numbers.")
        if not all(bool(np.all(np.isfinite(c))) for c in coordinates):
            raise ValueError("Evaluation coordinates must be finite.")
        return coordinates

    def query_plan(
        self,
        coordinates: Union[npt.NDArray, Sequence[npt.NDArray]],
        grid: bool = True,
        *,
        max_retained_bytes: int = 256 * 2**20,
    ):
        """Precompute support geometry for repeated fixed-coordinate queries.

        Query plans are experimental geometry objects compatible with any
        spline created through :meth:`with_data` or :meth:`with_coefficients`.
        They trade explicit, capped retained memory for faster repeated
        evaluation.  Ordinary calls remain the right choice for coordinates
        that are evaluated only once.
        """
        from .query_plan import TensorSplineGeometryPlan

        return TensorSplineGeometryPlan(
            self,
            coordinates,
            grid=grid,
            max_retained_bytes=max_retained_bytes,
        )

    def _evaluate_points(self, coordinates: Tuple[npt.NDArray, ...]) -> npt.NDArray:
        """Evaluate same-shape point coordinates in bounded-memory tiles."""
        xp = self._array_module()
        query_shape = coordinates[0].shape
        count = math.prod(query_shape)
        output = xp.empty(count, dtype=self._coefficients.dtype)
        flat_coordinates = tuple(c.reshape(-1) for c in coordinates)
        tile_size = self._EVALUATION_TILE_SIZE
        for start in range(0, count, tile_size):
            stop = min(count, start + tile_size)
            chunk = tuple(c[start:stop] for c in flat_coordinates)
            output[start:stop] = self._evaluate_point_chunk(chunk)
        return output.reshape(query_shape)

    def _evaluate_grid(self, coordinates: Tuple[npt.NDArray, ...]) -> npt.NDArray:
        """Evaluate tensor grids through bounded separable contractions."""
        xp = self._array_module()
        batch_shape = coordinates[0].shape[:-1]
        grid_shape = tuple(c.shape[-1] for c in coordinates)
        output = xp.empty(batch_shape + grid_shape, dtype=self._coefficients.dtype)
        batch_indexes = np.ndindex(batch_shape) if batch_shape else ((),)

        for batch_index in batch_indexes:
            vectors = tuple(c[batch_index] if batch_shape else c for c in coordinates)
            indexes = []
            weights = []
            for axis, vector in enumerate(vectors):
                axis_indexes, axis_weights = self._compute_support(axis, vector)
                indexes.append(axis_indexes)
                weights.append(axis_weights)
            output[batch_index] = self._evaluate_separable_grid_from_support(
                tuple(indexes), tuple(weights)
            )
        return output

    def _evaluate_separable_grid_from_support(self, indexes_seq, weights_seq):
        """Contract one tensor-product query axis at a time.

        Every support gather is tiled along the newly evaluated coordinate
        axis.  The required output and intermediate values still scale with
        the grid, while the support-expanded temporary is bounded by the
        evaluation tile target whenever a single remaining slice permits it.
        """
        xp = self._array_module()
        values = self._coefficients
        axis_labels = list(range(self._ndim))

        # Shrinking axes first reduces later intermediates.  This ordering is
        # an internal execution choice; labels restore public axis order.
        order = sorted(
            range(self._ndim),
            key=lambda axis: (
                indexes_seq[axis].shape[1] / self._lengths[axis],
                indexes_seq[axis].shape[1],
            ),
        )
        for logical_axis in order:
            current_axis = axis_labels.index(logical_axis)
            moved = xp.moveaxis(values, current_axis, -1)
            indexes = indexes_seq[logical_axis]
            weights = weights_seq[logical_axis]
            support, query_length = indexes.shape
            other_count = moved.size // moved.shape[-1]
            chunk_length = max(
                1,
                min(
                    query_length,
                    self._EVALUATION_TILE_SIZE // max(1, other_count * support),
                ),
            )
            contracted = xp.empty(
                moved.shape[:-1] + (query_length,), dtype=self._coefficients.dtype
            )
            weight_prefix = (1,) * (moved.ndim - 1)
            for start in range(0, query_length, chunk_length):
                stop = min(start + chunk_length, query_length)
                gathered = xp.take(moved, indexes[:, start:stop], axis=-1)
                chunk_weights = weights[:, start:stop].reshape(
                    weight_prefix + (support, stop - start)
                )
                contracted[..., start:stop] = xp.sum(gathered * chunk_weights, axis=-2)
            values = contracted
            axis_labels.pop(current_axis)
            axis_labels.append(logical_axis)

        permutation = tuple(axis_labels.index(axis) for axis in range(self._ndim))
        return xp.transpose(values, permutation)

    def _evaluate_small_grid(self, coordinates: Tuple[npt.NDArray, ...]) -> npt.NDArray:
        """Use direct separable broadcasting when its temporary is bounded."""
        xp = self._array_module()
        ndim = self._ndim
        indexes_bc = []
        weights_bc = []
        for axis, (coords, basis, mode, data_lim, dx, data_len) in enumerate(
            zip(
                coordinates,
                self._bases,
                self._modes,
                self._bounds,
                self._steps,
                self._lengths,
            )
        ):
            rat_indexes = (coords - data_lim[0]) / dx
            indexes = basis.compute_support_indexes(x=rat_indexes)
            shifted = xp.subtract(
                rat_indexes[np.newaxis], indexes, dtype=self._real_dtype
            )
            indexes, weights = mode.extend_signal(
                indexes=indexes,
                weights=basis(x=shifted),
                length=data_len,
            )
            shape = [1] * (2 * ndim)
            shape[axis] = indexes.shape[0]
            shape[ndim + axis] = coords.size
            indexes_bc.append(indexes.reshape(shape))
            weights_bc.append(weights.reshape(shape))

        weights_product = weights_bc[0]
        for weights in weights_bc[1:]:
            weights_product = weights_product * weights
        return xp.sum(
            self._coefficients[tuple(indexes_bc)] * weights_product,
            axis=tuple(range(ndim)),
        )

    def _array_module(self):
        if is_cupy_type(self._coefficients):
            import cupy as cp

            return cp
        return np

    def _evaluate_point_chunk(
        self, coordinates: Tuple[npt.NDArray, ...]
    ) -> npt.NDArray:
        """Evaluate one same-shape coordinate chunk with vectorized supports."""
        indexes_seq = []
        weights_seq = []
        for axis, coords in enumerate(coordinates):
            indexes_ext, weights_ext = self._compute_support(axis, coords)
            indexes_seq.append(indexes_ext)
            weights_seq.append(weights_ext)

        return self._evaluate_precomputed_point_chunk(indexes_seq, weights_seq)

    def _compute_support(self, axis: int, coordinates: npt.NDArray):
        """Compute extended coefficient indexes and weights for one axis."""
        xp = self._array_module()
        basis = self._bases[axis]
        mode = self._modes[axis]
        x_min, _ = self._bounds[axis]
        rat_indexes = (coordinates - x_min) / self._steps[axis]
        indexes = basis.compute_support_indexes(x=rat_indexes)
        shifted = xp.subtract(rat_indexes[np.newaxis], indexes, dtype=self._real_dtype)
        return mode.extend_signal(
            indexes=indexes,
            weights=basis(x=shifted),
            length=self._lengths[axis],
        )

    def _evaluate_precomputed_point_chunk(
        self, indexes_seq, weights_seq, *, coefficients=None
    ):
        """Gather and combine one chunk of precomputed tensor supports.

        ``coefficients`` may contain leading independent dimensions.  This
        private path lets higher-level plans vectorize batches without changing
        the public ``TensorSpline`` rule that every construction-data axis is a
        spline dimension.
        """
        xp = self._array_module()
        ndim = self._ndim
        if coefficients is None:
            coefficients = self._coefficients
        batch_ndim = coefficients.ndim - ndim
        if batch_ndim < 0 or tuple(coefficients.shape[-ndim:]) != self._lengths:
            raise ValueError("Coefficient array has incompatible trailing dimensions.")

        indexes_bc = []
        weights_bc = []
        query_shape = indexes_seq[0].shape[1:]
        for axis, (indexes, weights) in enumerate(zip(indexes_seq, weights_seq)):
            broadcast_shape = [1] * ndim + list(query_shape)
            broadcast_shape[axis] = indexes.shape[0]
            indexes_bc.append(indexes.reshape(broadcast_shape))
            weights_bc.append(weights.reshape(broadcast_shape))

        axes_sum = tuple(range(batch_ndim, batch_ndim + ndim))
        coefficient_index = (slice(None),) * batch_ndim + tuple(indexes_bc)
        gathered = coefficients[coefficient_index]
        if xp is np:
            labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            required_labels = batch_ndim + ndim + 1
            if required_labels <= len(labels):
                batch_labels = labels[:batch_ndim]
                support_labels = labels[batch_ndim : batch_ndim + ndim]
                query_label = labels[batch_ndim + ndim]
                value_labels = batch_labels + support_labels + query_label
                weight_labels = ",".join(
                    label + query_label for label in support_labels
                )
                equation = (
                    f"{value_labels},{weight_labels}->{batch_labels}{query_label}"
                )
                return np.einsum(equation, gathered, *weights_seq, optimize=False)
        weights_product = weights_bc[0]
        for weights in weights_bc[1:]:
            weights_product = weights_product * weights
        return xp.sum(
            gathered * weights_product,
            axis=axes_sum,
        )

    def _compute_coefficients(self, data: npt.NDArray) -> npt.NDArray:
        """Prefilter every logical axis without relying on reshape views.

        The coefficient filters operate on the last axis.  Repeated cyclic
        transposes happened to remain reshape-compatible in one and two
        dimensions, but intermediate 3-D and higher layouts could make the
        batched reshape allocate a detached copy.  Moving each logical axis to
        the end and accepting the mode's explicit output array keeps all
        filtered coefficients connected to the returned tensor.
        """
        backend = "cupy" if is_cupy_type(data) else "numpy"
        return prefilter_interpolation_coefficients(
            data,
            bases=self._bases,
            modes=self._modes,
            axes=tuple(range(self._ndim)),
            dtype=data.dtype,
            backend=backend,
        )

    def _geometry_signature(self):
        """Hashable numerical-space signature used by geometry plans."""
        backend = "cupy" if is_cupy_type(self._coefficients) else "numpy"
        basis_signature = tuple(
            (type(basis), basis.support, basis.degree, basis.poles)
            for basis in self._bases
        )
        return (
            backend,
            self._real_dtype.str,
            self._lengths,
            tuple((float(low), float(high)) for low, high in self._bounds),
            tuple(float(step) for step in self._steps),
            basis_signature,
            tuple(type(mode) for mode in self._modes),
        )

    @staticmethod
    def _array_module_for(array):
        if is_cupy_type(array):
            import cupy as cp

            return cp
        return np
