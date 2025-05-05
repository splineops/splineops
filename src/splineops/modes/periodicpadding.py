"""
PeriodicPadding – periodic extension mode for TensorSpline
---------------------------------------------------------

Implements wrap-around (“cyclic”) boundary conditions:

    … | a  b  c  d | a  b  c  d | a …

Both the index wrapping and the conversion of samples to spline
coefficients are handled here.  For spline orders with poles, the
coefficient calculation is done by diagonalising the circulant
convolution matrix with the DFT (FFT).  A CuPy branch is included so
the mode works transparently on GPU arrays.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import numpy.typing as npt

from splineops.bases.splinebasis import SplineBasis
from splineops.modes.extensionmode import ExtensionMode
from splineops.utils.interop import is_cupy_type


class PeriodicPadding(ExtensionMode):
    """Periodic (wrap-around) boundary condition."""

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _ifft_cyclic_inverse(
        data: np.ndarray,
        basis: SplineBasis,
    ) -> np.ndarray:
        """
        Solve   (basis ⋆ c)[k] = f[k]   on the circle
        by dividing in the Fourier domain.
        Operates on – and returns – NumPy arrays.
        """
        n = data.shape[-1]
        m = (basis.support - 1) // 2

        # One period of the basis (zero-padded to length *n*)
        bk = basis(np.arange(-m, m + 1, dtype=data.real.dtype))
        bk_per = np.zeros(n, dtype=bk.dtype)
        bk_per[: m + 1] = bk[m:]
        bk_per[-m:] = bk[:m]

        # Choose the right FFT flavour depending on data type
        fft  = np.fft.rfftn if np.isrealobj(data) else np.fft.fftn
        ifft = np.fft.irfftn if np.isrealobj(data) else np.fft.ifftn

        Ff = fft(data, axes=(-1,))
        Fb = fft(bk_per, axes=(-1,))

        # Avoid numerical blow-ups
        eps = np.finfo(Fb.real.dtype).eps
        Fb = np.where(np.abs(Fb) < eps, eps, Fb)

        Fc = Ff / Fb
        coeffs = ifft(Fc, axes=(-1,))

        return coeffs.real if np.isrealobj(data) else coeffs

    # ---------------------------------------------------------------- public
    @staticmethod
    def extend_signal(
        indexes: npt.NDArray,
        weights: npt.NDArray,
        length: float,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Wrap indexes modulo *length*.  Weights are unchanged.
        """
        return np.mod(indexes, length, dtype=indexes.dtype), weights

    @staticmethod
    def compute_coefficients(
        data: npt.NDArray,
        basis: SplineBasis,
    ) -> npt.NDArray:
        """
        Convert samples to spline coefficients assuming periodic
        boundaries.  Uses an FFT-based solver for arbitrary order.
        """
        # Orders with no poles need no pre-filter
        if basis.poles is None:
            return np.copy(data)

        # ------------ NumPy path -------------------------------------
        if not is_cupy_type(data):
            return PeriodicPadding._ifft_cyclic_inverse(data, basis)

        # ------------ CuPy path --------------------------------------
        import cupy as cp  # local import to keep CuPy optional

        n = data.shape[-1]
        m = (basis.support - 1) // 2
        bk = basis(cp.arange(-m, m + 1, dtype=data.real.dtype))
        bk_per = cp.zeros(n, dtype=bk.dtype)
        bk_per[: m + 1] = bk[m:]
        bk_per[-m:] = bk[:m]

        Ff = cp.fft.fftn(data, axes=(-1,))
        Fb = cp.fft.fftn(bk_per, axes=(-1,))
        eps = cp.finfo(Fb.real.dtype).eps
        Fb = cp.where(cp.abs(Fb) < eps, eps, Fb)

        Fc = Ff / Fb
        return cp.fft.ifftn(Fc, axes=(-1,))
