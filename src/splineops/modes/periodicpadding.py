"""
PeriodicPadding – periodic extension mode for TensorSpline
----------------------------------------------------------

Implements wrap-around (“cyclic”) boundary conditions

    … | a  b  c  d | a  b  c  d | a …

Both the index wrapping and the conversion of samples to spline
coefficients are handled here.  For spline orders with poles, the
coefficients are obtained by diagonalising the circulant convolution
matrix with the DFT (FFT).  A CuPy branch is included so the mode works
transparently on GPU arrays as well.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import numpy.typing as npt

from splineops.bases.splinebasis import SplineBasis
from splineops.modes.extensionmode import ExtensionMode
from splineops.utils.interop import is_cupy_type


class PeriodicPadding(ExtensionMode):
    """
    Periodic (wrap-around) boundary condition.

    The last signal axis is considered cyclic; indexes are taken modulo
    the signal length and the sample-to-coefficient conversion is solved
    on the circle via FFT.
    """

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _ifft_cyclic_inverse(
        data: np.ndarray,
        basis: SplineBasis,
    ) -> np.ndarray:
        """
        Solve  (basis ⋆ c)[k] = f[k]  on a circle by division in the
        Fourier domain (NumPy implementation, operates on last axis).
        """
        n = data.shape[-1]                   # period length
        m = (basis.support - 1) // 2

        # One period of the basis, zero-padded / wrapped to length *n*
        bk = basis(np.arange(-m, m + 1, dtype=data.real.dtype))
        bk_per = np.zeros(n, dtype=bk.dtype)
        bk_per[: m + 1] = bk[m:]             #  0 … +m
        bk_per[-m:]   = bk[:m]               # -m … -1

        if np.isrealobj(data):
            # --- real FFT branch (need n for odd lengths!) -----------------
            Ff = np.fft.rfft(data, axis=-1)
            Fb = np.fft.rfft(bk_per, n=n, axis=-1)
        else:
            # --- complex branch -------------------------------------------
            Ff = np.fft.fftn(data, axes=(-1,))
            Fb = np.fft.fftn(bk_per, axes=(-1,))

        # Numerical safety
        eps = np.finfo(Fb.real.dtype).eps
        Fb = np.where(np.abs(Fb) < eps, eps, Fb)

        Fc = Ff / Fb

        if np.isrealobj(data):
            coeffs = np.fft.irfft(Fc, n=n, axis=-1)      # exact length n
            return coeffs
        else:
            return np.fft.ifftn(Fc, axes=(-1,))

    # ---------------------------------------------------------------- public
    @staticmethod
    def extend_signal(
        indexes: npt.NDArray,
        weights: npt.NDArray,
        length: float,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Wrap indexes modulo *length*; weights stay unchanged.
        """
        return np.mod(indexes, length), weights

    @staticmethod
    def compute_coefficients(
        data: npt.NDArray,
        basis: SplineBasis,
    ) -> npt.NDArray:
        """
        Convert samples to spline coefficients with periodic boundaries.
        """
        # Orders with no poles (nearest / linear / …) need no pre-filter
        if basis.poles is None:
            return np.copy(data)

        # ----------------------- NumPy path ------------------------------
        if not is_cupy_type(data):
            return PeriodicPadding._ifft_cyclic_inverse(data, basis)

        # ----------------------- CuPy path -------------------------------
        import cupy as cp  # local import keeps CuPy optional

        n = data.shape[-1]
        m = (basis.support - 1) // 2
        bk = basis(cp.arange(-m, m + 1, dtype=data.real.dtype))
        bk_per = cp.zeros(n, dtype=bk.dtype)
        bk_per[: m + 1] = bk[m:]
        bk_per[-m:] = bk[:m]

        if cp.isrealobj(data):
            Fx = cp.fft.rfft(data, axis=-1)
            Fb = cp.fft.rfft(bk_per, n=n, axis=-1)
        else:
            Fx = cp.fft.fftn(data, axes=(-1,))
            Fb = cp.fft.fftn(bk_per, axes=(-1,))

        eps = cp.finfo(Fb.real.dtype).eps
        Fb = cp.where(cp.abs(Fb) < eps, eps, Fb)

        Fc = Fx / Fb

        if cp.isrealobj(data):
            return cp.fft.irfft(Fc, n=n, axis=-1)
        else:
            return cp.fft.ifftn(Fc, axes=(-1,))
