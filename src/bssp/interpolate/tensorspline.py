import numpy as np
import numpy.typing as npt
from typing import Sequence, Union, Tuple

from bssp.bases.splinebasis import SplineBasis
from bssp.bases.utils import asbasis
from bssp.modes.extensionmode import ExtensionMode
from bssp.modes.utils import asmode

TSplineBasis = Union[SplineBasis, str]
TSplineBases = Union[TSplineBasis, Sequence[TSplineBasis]]
TExtensionMode = Union[ExtensionMode, str]
TExtensionModes = Union[TExtensionMode, Sequence[TExtensionMode]]


class TensorSpline:
    def __init__(
        self,
        data: npt.NDArray,
        # TODO(dperdios): samples?
        coords: Union[npt.NDArray, Sequence[npt.NDArray]],
        # TODO(dperdios): coordinates?
        basis: TSplineBases,
        # TODO(dperdios): bases?
        mode: TExtensionModes,
        # TODO(dperdios): modes?
        # TODO(dperdios): extrapolate? only at evaluation time?
        # TODO(dperdios): axis? axes? probably complex
        # TODO(dperdios): optional reduction strategy (e.g., first or last)
    ) -> None:

        # Data
        ndim = data.ndim
        self._ndim = ndim  # TODO(dperdios): public property?

        # TODO(dperdios): optional coordinates?
        # TODO(dperdios): add some tests on the type of `coords`
        # TODO(dperdios): `coords` need to define a uniform grid
        # Coordinates
        #   1-D special case (either `array` or `(array,)`)
        if isinstance(coords, np.ndarray) and ndim == 1 and len(coords) == len(data):
            # Convert `array` to `(array,)`
            coords = (coords,)
        if not all(bool(np.all(np.diff(c) > 0)) for c in coords):
            raise ValueError("Coordinates must be strictly ascending.")
        valid_data_shape = tuple([c.size for c in coords])
        if data.shape != valid_data_shape:
            raise ValueError(
                f"Incompatible data shape. " f"Expected shape: {valid_data_shape}"
            )
        self._coords = coords
        # TODO(dperdios): convert to Python float?
        bounds = tuple([(c[0], c[-1]) for c in coords])
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
        # TODO(dperdios): coords dtype
        # TODO(dperdios): data dtype must be float (for direct filtering)
        dtype = data.dtype
        real_dtype = data.real.dtype
        self._dtype = dtype
        self._real_dtype = real_dtype
        # TODO(dperdios): integer always int64 or int32? any speed difference?
        #  Note: may even be dangerous to use indexes of type int16 for instance
        int_dtype_str = f"i{real_dtype.itemsize}"
        int_dtype = np.dtype(int_dtype_str)
        self._int_dtype = int_dtype

        # Bases
        if isinstance(basis, (SplineBasis, str)):
            basis = ndim * (basis,)
        bases = [asbasis(b) for b in basis]
        if len(bases) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._bases = tuple(bases)

        # Modes
        if isinstance(mode, (ExtensionMode, str)):
            mode = ndim * (mode,)
        modes = [asmode(m) for m in mode]
        if len(modes) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._modes = tuple(modes)

        # Compute coefficients
        coeffs = self._compute_coefficients(data=data)
        self._coeffs = coeffs

    # Properties
    # TODO(dperdios): useful properties?
    @property
    def coeffs(self):
        return np.copy(self._coeffs)

    @property
    def bases(self) -> Tuple[SplineBasis, ...]:
        return self._bases

    # Methods
    def __call__(
        self,
        coords: Union[npt.NDArray, Sequence[npt.NDArray]],
        grid: bool = True,
        # TODO(dperdios): extrapolate?
    ) -> npt.NDArray:
        return self.eval(coords=coords, grid=grid)

    def eval(
        self, coords: Union[npt.NDArray, Sequence[npt.NDArray]], grid: bool = True
    ) -> npt.NDArray:

        # TODO(dperdios): check dtype and/or cast
        ndim = self._ndim
        if grid:
            if len(coords) != ndim:
                # TODO(dperdios): Sequence of (..., n) arrays (batch dimensions
                #   must be the same!)
                raise ValueError(f"Must be a {ndim}-length sequence of 1-D arrays.")
            if not all([bool(np.all(np.diff(c, axis=-1) > 0)) for c in coords]):
                raise ValueError("Coordinates must be strictly ascending.")
        else:
            # If not `grid`, an Array is expected
            if len(coords) != ndim:
                raise ValueError(f"Incompatible shape. Expected shape: ({ndim}, ...). ")

        # Get properties
        real_dtype = self._real_dtype
        bounds_seq = self._bounds
        length_seq = self._lengths
        step_seq = self._steps
        basis_seq = self._bases
        mode_seq = self._modes
        coeffs = self._coeffs
        ndim = len(basis_seq)

        # Rename
        coords_seq = coords

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
        data = np.sum(coeffs[tuple(indexes_bc)] * weights_tp, axis=axes_sum)

        return data

    def _compute_coefficients(self, data: npt.NDArray) -> npt.NDArray:

        # Prepare data and axes
        # TODO(dperdios): there is probably too many copies along this process
        coeffs = np.copy(data)
        axes = tuple(range(coeffs.ndim))
        axes_roll = tuple(np.roll(axes, shift=-1))

        # TODO(dperdios): could do one less roll by starting with the initial
        #  shape
        for basis, mode in zip(self._bases, self._modes):

            # Roll data w.r.t. dimension
            coeffs = np.transpose(coeffs, axes=axes_roll)

            # Compute coefficients w.r.t. extension `mode` and `basis`
            coeffs = mode.compute_coefficients(data=coeffs, basis=basis)

        return coeffs
