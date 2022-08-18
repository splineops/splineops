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
TSplineBasis = Union[SplineBasis, str]
TSplineBases = Union[TSplineBasis, Sequence[TSplineBasis]]


# TODO: Mother class?
# TODO: TensorProductSpline, TensorSplineInterpolator, TSpline, NDSpline
class TensorSpline:

    def __init__(
            self,
            data: npt.ArrayLike,  # TODO: input, samples?
            coords: Union[npt.ArrayLike, Sequence[npt.ArrayLike]],  # TODO: coordinates?
            # basis: Union[SplineBasis, Sequence[SplineBasis]],  # TODO: str as well? str only? bases?
            basis: TSplineBases,  # TODO: str as well? str only? bases?
            mode: Union[str, Sequence[str]],  # TODO: modes?
            # TODO: extrapolate?
            # TODO: axis? axes?
            # TODO: optional reduction strategy (e.g., first or last)
    ):

        # Data
        data = np.asarray(data)
        ndim = data.ndim
        self._ndim = ndim

        # TODO(dperdios): optional coordinates?
        # TODO(dperdios): add some tests on the type of `coords`
        # Coordinates
        #   1-D special case (either `array` or `(array,)`
        if ndim == 1 and len(coords) == len(data):
            # Convert `array_like` to `(array_like,)`
            coords = coords,
        coords = tuple(np.asarray(c) for c in coords)
        if not np.all([np.all(np.diff(c) > 0) for c in coords]):
            raise ValueError("Coordinates must be strictly ascending.")
        valid_data_shape = tuple([c.size for c in coords])
        if data.shape != valid_data_shape:
            raise ValueError(
                f"Incompatible data shape. "
                f"Expected shape: {valid_data_shape}"
            )
        self._coords = coords
        bounds = tuple([(c[0], c[-1]) for c in coords])
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
        dtype = data.dtype
        # if dtype.kind not in ('f', 'c'):
        #     ValueError("Invalid dtype. Only supports float and complex-float")
        # if dtype.kind == 'c':
        #     real_dtype_str = f'f{dtype.itemsize // 2:d}'
        #     real_dtype = np.dtype(real_dtype_str)
        # else:
        #     real_dtype = dtype
        real_dtype = data.real.dtype
        self._dtype = dtype
        self._real_dtype = real_dtype
        int_dtype_str = f'i{real_dtype.itemsize}'
        int_dtype = np.dtype(int_dtype_str)
        self._int_dtype = int_dtype

        # Bases
        # TODO(dperdios): check dtype between basis and data (can be complex)?
        # TODO(dperdios): check it is a sequence
        if isinstance(basis, (SplineBasis, str)):
            basis = ndim * (basis,)
        bases = [asbasis(b) for b in basis]
        if len(bases) != ndim:
            raise ValueError(f"Length of the sequence must be {ndim}.")
        self._bases = tuple(bases)

        # if isinstance(basis, SplineBasis):
        #     basis = ndim * (basis,)
        # if not (isinstance(basis, abc.Sequence)
        #         and all(isinstance(b, SplineBasis) for b in basis)):
        #     raise TypeError(f"Must be a sequence of `{SplineBasis.__name__}`.")
        # if len(basis) != ndim:
        #     raise ValueError(f"Length of the sequence must be {ndim}.")
        # self._bases = tuple(basis)

        # Modes
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
    def bases(self) -> Tuple[SplineBasis]:
        return self._bases

    # Methods
    def __call__(self, coords: npt.ArrayLike, grid: bool = True) -> npt.NDArray:
        return self.eval(coords=coords, grid=grid)

    def eval(self, coords: npt.ArrayLike, grid: bool = True) -> npt.NDArray:

        # TODO(dperdios): check dtype and/or cast
        ndim = self._ndim
        if grid:
            coords = tuple(np.asarray(c) for c in coords)
            if not np.all([np.all(np.diff(c, axis=-1) > 0) for c in coords]):
                raise ValueError("Coordinates must be strictly ascending.")
            if len(coords) != ndim:
                raise ValueError(f"Must be a sequence of {ndim}-D arrays.")
        else:
            coords = np.asarray(coords)
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
            fs = 1 / dx
            rat_indexes = (coords - x_min) * fs
            #   Compute corresponding integer indexes (including support)
            indexes = self._compute_support_indexes_1d(basis=basis, ind=rat_indexes)

            # Evaluate basis function (interpolation weights)
            shifted_idx = np.subtract(indexes, rat_indexes, dtype=real_dtype)
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
        # Note: explicit broadcasting is required to avoid the warning
        #  VisibleDeprecationWarning: Creating an ndarray from ragged
        #  nested sequences [...] is deprecated.
        weights_tp = np.prod(np.broadcast_arrays(*weights_bc), axis=0)

        # Interpolation (convolution via reduction)
        axes_sum = tuple(range(ndim))  # first axes are the indexes
        data = np.sum(coeffs[tuple(indexes_bc)] * weights_tp, axis=axes_sum)

        return data

    # TODO(dperdios): should this be a method of the SplineBasis?
    def _compute_support_indexes_1d(
            self, basis: SplineBasis, ind: npt.ArrayLike) -> npt.NDArray:

        # Extract property
        dtype = self._real_dtype
        int_dtype = self._int_dtype

        # Span and offset
        support = basis.support
        idx_offset = 0.5 if support & 1 else 0.0  # offset for even supports
        idx_span = np.arange(support, dtype=int_dtype) - (support - 1) // 2

        # Floor rational indexes and convert to integers
        ind = np.asarray(ind, dtype=dtype)
        # ind_fl = np.array(np.floor(ind + self._idx_offset), dtype=int_dtype)
        ind_fl = np.array(np.floor(ind + idx_offset), dtype=int_dtype)

        # TODO(dperdios): check fastest axis for computations
        # First axis
        # # idx = np.array([s + ind_fl for s in self._idx_span], dtype=int_dtype)
        idx = np.array([s + ind_fl for s in idx_span], dtype=int_dtype)

        # _ns = tuple([self.support] + ind_fl.ndim * [1])
        # idx = ind_fl + self._idx_span.reshape(_ns)
        # _ns = tuple([support] + ind_fl.ndim * [1])
        # idx = ind_fl + np.reshape(idx_span, _ns)
        # # Last axis
        # idx = ind_fl[..., np.newaxis] + self._idx_span

        return idx

    def _compute_coefficients(self, data: npt.NDArray) -> npt.NDArray:

        coeffs = np.copy(data)
        axes = tuple(range(coeffs.ndim))
        axes_roll = tuple(np.roll(axes, shift=-1))

        # TODO(dperdios): could do one less roll by starting with the initial
        #  shape
        for basis, mode in zip(self._bases, self._modes):

            # Transpose data
            coeffs = np.transpose(coeffs, axes=axes_roll)

            # Get poles
            poles = basis.poles

            if poles is not None:
                if mode == 'zero':  # Finite-support coefficients
                    # Reshape for batch-processing
                    coeffs_shape = coeffs.shape
                    coeffs_rs = np.reshape(coeffs, newshape=(-1, coeffs.shape[-1]))

                    # Prepare banded
                    m = (basis.support - 1) // 2
                    bk = basis(np.arange(-m, m + 1))

                    # Reshape back to original shape
                    coeffs_rs = _compute_ck_zero_matrix_banded_v1(bk=bk, fk=coeffs_rs.T)
                    coeffs = np.reshape(coeffs_rs.T, newshape=coeffs_shape)
                    # TODO(dperdios): the v2 only has an issue for
                    #  a single-sample signal.
                    #  Note: v2 is probably slightly faster as it does not need
                    #  to create the sub-matrices.
                    # coeffs = _compute_ck_zero_matrix_banded_v2(bk=bk, fk=data)
                elif mode == 'mirror':  # Narrow mirroring
                    # coeffs = np.copy(data)
                    # TODO(dperdios): the batched version has an issue for
                    #  a single-sample signal (anti-causal init). There is also
                    #  an issue for the boundary condition with single-sample
                    #  signals for the mirror case.
                    # _compute_coeffs_narrow_mirror_wg(data=coeffs, poles=poles)
                    _data_to_coeffs(data=coeffs, poles=poles, boundary=mode)
                else:
                    raise NotImplementedError("Unsupported signal extension mode.")

        return coeffs
