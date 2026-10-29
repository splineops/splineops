# splineops/src/splineops/spline_interpolation/modes/finite_support_coefficients.py

import numpy as np
from typing import Tuple
import numpy.typing as npt

from ..bases.spline_basis import SplineBasis
from .extension_mode import ExtensionMode
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

        # Preserve the caller's stencil. Zero weights still need valid indexes
        # because evaluation gathers coefficients before multiplying by weights.
        indexes_ext = np.copy(indexes)
        weights_ext = np.copy(weights)
        indexes_ext[bc_lr] = 0
        weights_ext[bc_lr] = 0

        return indexes_ext, weights_ext

    @staticmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:

        # Even the identity case returns independent coefficient storage.
        if basis.poles is None:
            return np.copy(data)

        # The banded solver pads into owned storage, so NumPy inputs need no
        # preliminary copy. CuPy uses a host solve and an owned device buffer
        # for the result; this path is not a device-native solver.
        need_cupy_compat = is_cupy_type(data)
        coeffs = np.copy(data) if need_cupy_compat else data

        # Reshape for batch-processing
        coeffs_shape = coeffs.shape
        coeffs_ns = -1, coeffs.shape[-1]
        coeffs_rs = np.reshape(coeffs, coeffs_ns)

        # CuPy compatibility
        if need_cupy_compat:
            # Get as NumPy array
            coeffs_rs_cp = coeffs_rs
            coeffs_rs = coeffs_rs_cp.get()  # type: ignore

        # Prepare banded
        m = (basis.support - 1) // 2
        bk = basis(np.arange(-m, m + 1))

        # Compute coefficients (banded solver to be generic)
        coeffs_rs = _compute_ck_zero_matrix_banded_v1(bk=bk, fk=coeffs_rs.T)
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
