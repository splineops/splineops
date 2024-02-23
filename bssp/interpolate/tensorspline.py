import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence, Union, Tuple
from collections import abc

from bssp.bases.splinebasis import SplineBasis
from bssp.interpolate.utils import _compute_ck_zero_matrix_banded_v1
from bssp.interpolate.utils import _compute_ck_zero_matrix_banded_v2
from bssp.interpolate.utils import _compute_coeffs_narrow_mirror_wg
from bssp.interpolate.utils import _data_to_coeffs

from bssp.bases.utils import asbasis
from bssp.utils.interop import is_cupy_type

TSplineBasis = Union[SplineBasis, str]
TSplineBases = Union[TSplineBasis, Sequence[TSplineBasis]]


class TensorSpline:

    def __init__(
            self,
            data: npt.NDArray,
            # TODO(dperdios): samples?
            coords: Union[npt.NDArray, Sequence[npt.NDArray]],
            # TODO(dperdios): coordinates?
            basis: TSplineBases,
            # TODO(dperdios): bases?
            mode: Union[str, Sequence[str]],
            # TODO(dperdios): modes?
            # TODO(dperdios): extrapolate? only at evaluation time?
            # TODO(dperdios): axis? axes? probably complex
            # TODO(dperdios): optional reduction strategy (e.g., first or last)
    ):

        # Data
        ndim = data.ndim
        self._ndim = ndim  # TODO(dperdios): public property?

        # TODO(dperdios): optional coordinates?
        # TODO(dperdios): add some tests on the type of `coords`
        # Coordinates
        #   1-D special case (either `array` or `(array,)`
        if ndim == 1 and len(coords) == len(data):
            # Convert `array` to `(array,)`
            coords = coords,
        if not all(bool(np.all(np.diff(c) > 0)) for c in coords):
            raise ValueError("Coordinates must be strictly ascending.")
        valid_data_shape = tuple([c.size for c in coords])
        if data.shape != valid_data_shape:
            raise ValueError(
                f"Incompatible data shape. "
                f"Expected shape: {valid_data_shape}"
            )
        self._coords = coords
        # TODO(dperdios): convert to Python float?
        bounds = tuple([(c[0], c[-1]) for c in coords])
        # TODO(dperdios): `bounds` as a public property?
        self._bounds = bounds
        lengths = valid_data_shape
        self._lengths = lengths
        steps = []
        for b, l in zip(bounds, lengths):
            if l > 1:
                step = (b[-1] - b[0]) / (l - 1)
            else:
                # Special case for single-sample signal
                step = 1
            steps.append(step)
        steps = tuple(steps)  # TODO: convert dtype? (can be promoted)
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
        int_dtype_str = f'i{real_dtype.itemsize}'
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
        # TODO(dperdios): should we call it extension(s)? mode is quite popular
        if isinstance(mode, str):
            mode = ndim * (mode,)
        if not (isinstance(mode, abc.Sequence)
                and all(isinstance(m, str) for m in mode)):
            raise TypeError(f"Must be a sequence of `{str.__name__}`.")
        if len(mode) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._modes = mode

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
            self,
            coords: Union[npt.NDArray, Sequence[npt.NDArray]],
            grid: bool = True
    ) -> npt.NDArray:

        # TODO(dperdios): check dtype and/or cast
        ndim = self._ndim
        if grid:
            if len(coords) != ndim:
                # TODO(dperdios): Sequence of (..., n) arrays (batch dimensions
                #   must be the same!)
                raise ValueError(
                    f"Must be a {ndim}-length sequence of 1-D arrays.")
            if not all([bool(np.all(np.diff(c, axis=-1) > 0)) for c in coords]):
                raise ValueError("Coordinates must be strictly ascending.")
        else:
            # If not `grid`, an Array is expected
            if len(coords) != ndim:
                raise ValueError(
                    f"Incompatible shape. Expected shape: ({ndim}, ...). ")

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
                coords_seq, basis_seq, mode_seq, bounds_seq, step_seq,
                length_seq
        ):
            x_min, x_max = data_lim

            # Indexes
            #   Compute rational indexes
            # TODO(dperdios): no difference in using `* fs` or `/ dx`
            # fs = 1 / dx
            # rat_indexes = (coords - x_min) * fs
            rat_indexes = (coords - x_min) / dx
            #   Compute corresponding integer indexes (including support)
            indexes = self._compute_support_indexes_1d(basis=basis, ind=rat_indexes)

            # Evaluate basis function (interpolation weights)
            # indexes_shift = np.subtract(indexes, rat_indexes, dtype=real_dtype)
            # shifted_idx = np.subtract(indexes, rat_indexes, dtype=real_dtype)
            shifted_idx = np.subtract(
                rat_indexes[np.newaxis], indexes, dtype=real_dtype)
            # TODO(dperdios): casting rules, do we really want it?
            weights = basis(x=shifted_idx)

            # Signal extension
            # valid = np.logical_and(coords >= x_min, coords <= x_max)
            # bc_l = np.logical_and(valid, indexes < 0)
            # bc_r = np.logical_and(valid, indexes >= data_len)  # TODO: >= or >?
            bc_l = indexes < 0
            bc_r = indexes > data_len - 1
            bc_lr = np.logical_or(bc_l, bc_r)
            if mode == 'zero':
                # Set dumb indexes and weights (zero outside support)
                indexes[bc_lr] = 0  # dumb index
                weights[bc_lr] = 0
            elif mode == 'mirror':
                len_2 = 2 * data_len - 2
                if data_len == 1:
                    ii = np.zeros_like(indexes)
                else:
                    # Sawtooth
                    ii = indexes / len_2 - np.floor(indexes / len_2 + 0.5)
                    ii = np.round(len_2 * np.abs(ii))
                    # ii = np.round(len_2 * np.abs(indexes / len_2 - np.floor(indexes / len_2 + 0.5)))
                indexes[:] = ii
            else:
                # TODO: generic handling of BC errors
                raise NotImplementedError("Unsupported signal extension mode.")

            # Store
            indexes_seq.append(indexes)
            weights_seq.append(weights)

            # TODO(dperdios): remove out of bounds values (e.g., for mirror)?
            #  related to extrapolate
            # indexes *= valid
            # weights *= valid

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
                a = np.concatenate([
                    np.delete(exp_axis_ind_base, ii),
                    np.delete(exp_axis_coeffs_base, ii)
                ])
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
        weights_tp = np.prod(
            np.stack(np.broadcast_arrays(*weights_bc), axis=0), axis=0)

        # Interpolation (convolution via reduction)
        axes_sum = tuple(range(ndim))  # first axes are the indexes
        data = np.sum(coeffs[tuple(indexes_bc)] * weights_tp, axis=axes_sum)

        return data

    # TODO(dperdios): should this be a method of the SplineBasis?
    def _compute_support_indexes_1d(
            self, basis: SplineBasis, ind: npt.NDArray) -> npt.NDArray:

        # Extract property
        int_dtype = self._int_dtype

        # Span and offset
        support = basis.support
        idx_offset = 0.5 if support & 1 else 0.0  # offset for odd support
        idx_span = np.arange(support, dtype=int_dtype, like=ind)
        idx_span -= (support - 1) // 2

        # Floor rational indexes and convert to integers
        # ind_fl = np.array(np.floor(ind + self._idx_offset), dtype=int_dtype)
        ind_fl = (np.floor(ind + idx_offset)).astype(dtype=int_dtype)

        # TODO(dperdios): check fastest axis for computations
        # First axis
        _ns = tuple([support] + ind_fl.ndim * [1])
        idx = ind_fl + np.reshape(idx_span, _ns)
        # # Last axis
        # idx = ind_fl[..., np.newaxis] + idx_span

        return idx

    def _compute_coefficients(self, data: npt.NDArray) -> npt.NDArray:

        coeffs = np.copy(data)
        axes = tuple(range(coeffs.ndim))
        axes_roll = tuple(np.roll(axes, shift=-1))

        # TODO(dperdios): could do one less roll by starting with the initial
        #  shape
        for basis, mode in zip(self._bases, self._modes):

            # Roll data w.r.t. dimension
            coeffs = np.transpose(coeffs, axes=axes_roll)

            # Get poles
            poles = basis.poles

            if poles is not None:
                if mode == 'zero':  # Finite-support coefficients
                    # CuPy compatibility
                    # TODO(dperdios): could use a CuPy-compatible solver
                    # TODO(dperdios): could use dedicated filters
                    #  Note: this would depend on the degree of the bspline
                    need_cupy_compat = is_cupy_type(data)

                    # Reshape for batch-processing
                    coeffs_shape = coeffs.shape
                    coeffs_ns = -1, coeffs.shape[-1]
                    coeffs_rs = np.reshape(coeffs, newshape=coeffs_ns)

                    # CuPy compatibility
                    if need_cupy_compat:
                        # Get as NumPy array
                        coeffs_rs_cp = coeffs_rs
                        coeffs_rs = coeffs_rs_cp.get()

                    # Prepare banded
                    m = (basis.support - 1) // 2
                    bk = basis(np.arange(-m, m + 1))

                    # Compute coefficients (banded solver to be generic)
                    coeffs_rs = _compute_ck_zero_matrix_banded_v1(bk=bk, fk=coeffs_rs.T)
                    # TODO(dperdios): the v2 only has an issue for
                    #  a single-sample signal.
                    #  Note: v2 is probably slightly faster as it does not need
                    #  to create the sub-matrices.
                    # coeffs = _compute_ck_zero_matrix_banded_v2(bk=bk, fk=data)
                    #   Transpose back
                    coeffs_rs = coeffs_rs.T

                    # CuPy compatibility
                    if need_cupy_compat:
                        # Put back as CuPy array (reusing memory)
                        coeffs_rs_cp[:] = np.asarray(
                            coeffs_rs, like=coeffs_rs_cp)
                        coeffs_rs = coeffs_rs_cp

                    # Reshape back to original shape
                    coeffs = np.reshape(coeffs_rs, newshape=coeffs_shape)

                elif mode == 'mirror':  # Narrow mirroring
                    # coeffs = np.copy(data)
                    # TODO(dperdios): wide mirroring too?
                    # TODO(dperdios): the batched version has an issue for
                    #  a single-sample signal (anti-causal init). There is also
                    #  an issue for the boundary condition with single-sample
                    #  signals for the mirror case.
                    # _compute_coeffs_narrow_mirror_wg(data=coeffs, poles=poles)
                    _data_to_coeffs(data=coeffs, poles=poles, boundary=mode)
                else:
                    raise NotImplementedError(
                        "Unsupported signal extension mode.")

        return coeffs
