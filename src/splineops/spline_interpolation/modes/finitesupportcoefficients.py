# splineops/src/splineops/spline_interpolation/modes/finitesupportcoefficients.py

import numpy as np
from typing import Tuple
import numpy.typing as npt

from ..bases.splinebasis import SplineBasis
from ..modes.extensionmode import ExtensionMode
from ..utils import _compute_ck_zero_matrix_banded_v1
from ..utils import is_cupy_type

class FiniteSupportCoefficients(ExtensionMode):

    # Methods
    @staticmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        # Alias
        data_len = length

        # Compute out-of-bound-indexes
        bc_l = indexes < 0
        bc_r = indexes > data_len - 1
        bc_lr = np.logical_or(bc_l, bc_r)

        # Update indexes and weights
        indexes_ext = np.copy(indexes)  # TODO(dperdios): in-place update?
        weights_ext = np.copy(weights)  # TODO(dperdios): in-place update?
        indexes_ext[bc_lr] = 0  # dumb index
        # TODO(dperdios): why not use something else as dumb index?
        weights_ext[bc_lr] = 0

        return indexes_ext, weights_ext

    @staticmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:

        # Local copy
        # TODO(dperdios): local copy could be used to accelerate banded solver
        coeffs = np.copy(data)

        # Check whether `basis` has no `poles` (i.e., no computation needed)
        if basis.poles is None:
            return coeffs

        # CuPy compatibility
        # TODO(dperdios): could use a CuPy-compatible solver
        # TODO(dperdios): could use dedicated filters
        #  Note: this would depend on the degree of the bspline
        need_cupy_compat = is_cupy_type(data)

        # Reshape for batch-processing
        coeffs_shape = coeffs.shape
        coeffs_ns = -1, coeffs.shape[-1]
        coeffs_rs = np.reshape(coeffs, coeffs_ns)

        # CuPy compatibility
        if need_cupy_compat:
            # Get as NumPy array
            # TODO(dperdios): type should be handled more properly than
            #  just silencing the CuPy type
            coeffs_rs_cp = coeffs_rs
            coeffs_rs = coeffs_rs_cp.get()  # type: ignore

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
            coeffs_rs_cp[:] = np.asarray(coeffs_rs, like=coeffs_rs_cp)
            coeffs_rs = coeffs_rs_cp

        # Reshape back to original shape
        coeffs = np.reshape(coeffs_rs, coeffs_shape)

        return coeffs
