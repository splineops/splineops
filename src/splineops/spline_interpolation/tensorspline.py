# splineops/src/splineops/spline_interpolation/tensorspline.py

import numpy as np
import numpy.typing as npt
from typing import Sequence, Union, Tuple, cast

from .bases.splinebasis import SplineBasis
from .bases.utils import asbasis
from .modes.extensionmode import ExtensionMode
from .modes.utils import asmode
from .utils import is_ndarray

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
    >>> from splineops.interpolate.tensorspline import TensorSpline
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
            raise TypeError("Must be an array.")
        ndim = data.ndim
        self._ndim = ndim

        # TODO(dperdios): make `coordinates` optional?
        # TODO(dperdios): `coordinates` need to define a uniform grid.
        #  Note: this is not straightforward to control (numerical errors)
        # Coordinates
        #   1-D special case (either `array` or `(array,)`)
        if is_ndarray(coordinates) and ndim == 1 and len(coordinates) == len(data):
            # Note: we explicitly cast the type to NDArray
            coordinates = cast(npt.NDArray, coordinates)
            # Convert `array` to `(array,)`
            coordinates = (coordinates,)
        if not all(bool(np.all(np.diff(c) > 0)) for c in coordinates):
            raise ValueError("Coordinates must be strictly ascending.")
        valid_data_shape = tuple([c.size for c in coordinates])
        if data.shape != valid_data_shape:
            raise ValueError(
                f"Incompatible data shape. " f"Expected shape: {valid_data_shape}"
            )
        if not all(np.isrealobj(c) for c in coordinates):
            raise ValueError("Must be sequence of real numbers.")
        # TODO(dperdios): useful to keep initial coordinates as property?

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
    def bases(self) -> Tuple[SplineBasis, ...]:
        return self._bases

    @property
    def modes(self) -> Tuple[ExtensionMode, ...]:
        return self._modes

    @property
    def ndim(self):
        return self._ndim

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
        # Check coordinates
        ndim = self._ndim
        if grid:
            # Special 1-D case: "default" grid=True with a 1-D `coords` NDArray
            if is_ndarray(coordinates):
                # Note: we explicitly cast the type to NDArray
                coordinates = cast(npt.NDArray, coordinates)
                if ndim == 1 and coordinates.ndim == 1:
                    coordinates = (coordinates,)
            # N-D cases
            if len(coordinates) != ndim:
                # TODO(dperdios): Sequence of (..., n) arrays (batch dimensions
                #   must be the same!)
                raise ValueError(f"Must be a {ndim}-length sequence of 1-D arrays.")
            if not all([bool(np.all(np.diff(c, axis=-1) > 0)) for c in coordinates]):
                # TODO(dperdios): do they really need to be ascending?
                raise ValueError("Coordinates must be strictly ascending.")
        else:
            # If not `grid`, a sequence of arrays is expected with a length
            #  equal to the number of dimensions. Each array in the sequence
            #  must be of the same shape.
            coords_shapes = [c.shape for c in coordinates]
            if len(coordinates) != ndim or len(set(coords_shapes)) != 1:
                raise ValueError(
                    f"Incompatible sequence of coordinates. "
                    f"Must be a {ndim}-length sequence of same-shape N-D arrays. "
                    f"Current sequence of array shapes: {coords_shapes}."
                )
        if not all(np.isrealobj(c) for c in coordinates):
            raise ValueError("Must be a sequence of real numbers.")

        # Get properties
        real_dtype = self._real_dtype
        bounds_seq = self._bounds
        length_seq = self._lengths
        step_seq = self._steps
        basis_seq = self._bases
        mode_seq = self._modes
        coefficients = self._coefficients
        ndim = self._ndim

        # Rename
        coords_seq = coordinates

        # For-loop over dimensions
        indexes_seq = []
        weights_seq = []
        for coords, basis, mode, data_lim, dx, data_len in zip(
            coords_seq, basis_seq, mode_seq, bounds_seq, step_seq, length_seq
        ):

            # Data limits
            x_min, x_max = data_lim

            # Indexes
            #   Compute rational indexes
            # TODO(dperdios): no difference in using `* fs` or `/ dx`
            # fs = 1 / dx
            # rat_indexes = (coords - x_min) * fs
            rat_indexes = (coords - x_min) / dx
            #   Compute corresponding integer indexes (including support)
            indexes = basis.compute_support_indexes(x=rat_indexes)
            # TODO(dperdios): specify dtype in compute_support_indexes? cast dtype here?
            #  int32 faster than int64? probably not

            # Evaluate basis function (interpolation weights)
            # indexes_shift = np.subtract(indexes, rat_indexes, dtype=real_dtype)
            # shifted_idx = np.subtract(indexes, rat_indexes, dtype=real_dtype)
            shifted_idx = np.subtract(
                rat_indexes[np.newaxis], indexes, dtype=real_dtype
            )
            # TODO(dperdios): casting rules, do we really want it?
            weights = basis(x=shifted_idx)

            # Signal extension
            indexes_ext, weights_ext = mode.extend_signal(
                indexes=indexes, weights=weights, length=data_len
            )

            # TODO(dperdios): Add extrapolate handling?
            # weights[idx_extra] = cval ?? or within extend_signal?

            # Store
            indexes_seq.append(indexes_ext)
            weights_seq.append(weights_ext)

        # Broadcast arrays for tensor product
        if grid:
            # del_axis_base = np.arange(ndim + 1, step=ndim)
            # del_axes = [del_axis_base + ii for ii in range(ndim)]
            # exp_axis_base = np.arange(2 * ndim)
            # exp_axes = [tuple(np.delete(exp_axis_base, a)) for a in del_axes]
            # Batch-compatible axis expansions
            exp_axis_ind_base = np.arange(ndim)  # from start
            exp_axis_coeffs_base = exp_axis_ind_base - ndim  # from end TODO: reverse?
            exp_axes = []
            for ii in range(ndim):
                a = np.concatenate(
                    [
                        np.delete(exp_axis_ind_base, ii),
                        np.delete(exp_axis_coeffs_base, ii),
                    ]
                )
                exp_axes.append(tuple(a))
        else:
            exp_axis_base = np.arange(ndim)
            exp_axes = [tuple(np.delete(exp_axis_base, a)) for a in range(ndim)]

        indexes_bc = []
        weights_bc = []
        for indexes, weights, a in zip(indexes_seq, weights_seq, exp_axes):
            indexes_bc.append(np.expand_dims(indexes, axis=a))
            weights_bc.append(np.expand_dims(weights, axis=a))
        # Note: for interop (CuPy), cannot use prod with a sequence of arrays.
        #  Need explicit stacking before reduction. It is NumPy compatible.
        weights_tp = np.prod(np.stack(np.broadcast_arrays(*weights_bc), axis=0), axis=0)

        # Interpolation (convolution via reduction)
        # TODO(dperdios): might want to change the default reduction axis
        axes_sum = tuple(range(ndim))  # first axes are the indexes
        data = np.sum(coefficients[tuple(indexes_bc)] * weights_tp, axis=axes_sum)

        return data

    def _compute_coefficients(self, data: npt.NDArray) -> npt.NDArray:
        # Prepare data and axes
        # TODO(dperdios): there is probably too many copies along this process
        coefficients = np.copy(data)
        axes = tuple(range(coefficients.ndim))
        axes_roll = tuple(np.roll(axes, shift=-1))

        # TODO(dperdios): could do one less roll by starting with the initiat shape
        for basis, mode in zip(self._bases, self._modes):

            # Roll data w.r.t. dimension
            coefficients = np.transpose(coefficients, axes=axes_roll)

            # Compute coefficients w.r.t. extension `mode` and `basis`
            coefficients = mode.compute_coefficients(data=coefficients, basis=basis)

        return coefficients
