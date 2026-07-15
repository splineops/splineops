# splineops/src/splineops/spline_interpolation/tensors_pline.py

import numpy as np
import numpy.typing as npt
import math
from typing import Sequence, Union, Tuple, cast

from .bases.spline_basis import SplineBasis
from .bases.utils import asbasis
from .modes.extension_mode import ExtensionMode
from .modes.utils import asmode
from .utils import is_cupy_type, is_ndarray

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
        # TODO(dperdios): extrapolate?
    ) -> npt.NDArray:
        return self.eval(coordinates=coordinates, grid=grid)

    def eval(
        self, coordinates: Union[npt.NDArray, Sequence[npt.NDArray]], grid: bool = True
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

        if grid:
            return self._evaluate_grid(coordinates)
        return self._evaluate_points(coordinates)

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

        Query plans are experimental and intentionally bound to this spline.
        They trade explicit, capped retained memory for faster repeated
        evaluation.  Ordinary calls remain the right choice for coordinates
        that are evaluated only once.
        """
        from .query_plan import TensorSplineQueryPlan

        return TensorSplineQueryPlan(
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
        """Evaluate tensor grids without materializing full coordinate meshes."""
        xp = self._array_module()
        batch_shape = coordinates[0].shape[:-1]
        grid_shape = tuple(c.shape[-1] for c in coordinates)
        count = math.prod(grid_shape)
        tile_size = self._EVALUATION_TILE_SIZE
        if not batch_shape and count <= tile_size:
            return self._evaluate_small_grid(coordinates)

        output = xp.empty(batch_shape + grid_shape, dtype=self._coefficients.dtype)
        batch_indexes = np.ndindex(batch_shape) if batch_shape else ((),)

        for batch_index in batch_indexes:
            vectors = tuple(c[batch_index] if batch_shape else c for c in coordinates)
            batch_output = xp.empty(count, dtype=self._coefficients.dtype)
            for start in range(0, count, tile_size):
                stop = min(count, start + tile_size)
                flat_indexes = xp.arange(start, stop)
                indexes = xp.unravel_index(flat_indexes, grid_shape)
                points = tuple(
                    vector[indexes[axis]] for axis, vector in enumerate(vectors)
                )
                batch_output[start:stop] = self._evaluate_point_chunk(points)
            output[batch_index] = batch_output.reshape(grid_shape)
        return output

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

    def _evaluate_precomputed_point_chunk(self, indexes_seq, weights_seq):
        """Gather and combine one chunk of precomputed tensor supports."""
        xp = self._array_module()
        ndim = self._ndim

        indexes_bc = []
        weights_bc = []
        query_shape = indexes_seq[0].shape[1:]
        for axis, (indexes, weights) in enumerate(zip(indexes_seq, weights_seq)):
            broadcast_shape = [1] * ndim + list(query_shape)
            broadcast_shape[axis] = indexes.shape[0]
            indexes_bc.append(indexes.reshape(broadcast_shape))
            weights_bc.append(weights.reshape(broadcast_shape))

        weights_product = weights_bc[0]
        for weights in weights_bc[1:]:
            weights_product = weights_product * weights
        axes_sum = tuple(range(ndim))
        return xp.sum(
            self._coefficients[tuple(indexes_bc)] * weights_product,
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
        xp = self._array_module_for(data)
        coefficients = xp.array(data, copy=True, order="C")
        for axis, (basis, mode) in enumerate(zip(self._bases, self._modes)):
            axis_last = xp.ascontiguousarray(xp.moveaxis(coefficients, axis, -1))
            axis_last = mode.compute_coefficients(data=axis_last, basis=basis)
            coefficients = xp.moveaxis(axis_last, -1, axis)

        return coefficients

    @staticmethod
    def _array_module_for(array):
        if is_cupy_type(array):
            import cupy as cp

            return cp
        return np
