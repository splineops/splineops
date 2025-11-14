# splineops/src/splineops/spline_interpolation/modes/periodicpadding.py

from __future__ import annotations

from typing import Tuple
import numpy as np
import numpy.typing as npt

from ..bases.splinebasis import SplineBasis
from ..modes.extensionmode import ExtensionMode
from ..utils import is_cupy_type

class PeriodicPadding(ExtensionMode):

    # Methods
    @staticmethod
    def _ifft_cyclic_inverse(
        data: np.ndarray, basis: SplineBasis
    ) -> np.ndarray:
        
        n = data.shape[-1]                       # period length
        m = (basis.support - 1) // 2             # half-support

        # Build one period of the basis, wrapped / zero-padded to length n
        bk = basis(np.arange(-m, m + 1, dtype=data.real.dtype))
        bk_per = np.zeros(n, dtype=bk.dtype)
        bk_per[: m + 1] = bk[m:]                 #  0 …  +m
        bk_per[-m:]   = bk[:m]                   # –m … –1

        # Forward DFT of data and basis
        if np.isrealobj(data):
            Ff = np.fft.rfft(data, axis=-1)      # real FFT
            Fb = np.fft.rfft(bk_per, n=n, axis=-1)
        else:
            Ff = np.fft.fftn(data, axes=(-1,))   # complex FFT
            Fb = np.fft.fftn(bk_per, axes=(-1,))

        # Protect against divide-by-0 for very small bins
        eps = np.finfo(Fb.real.dtype).eps
        Fb = np.where(np.abs(Fb) < eps, eps, Fb)

        # Divide in Fourier space, then invert
        Fc = Ff / Fb
        if np.isrealobj(data):
            return np.fft.irfft(Fc, n=n, axis=-1)
        return np.fft.ifftn(Fc, axes=(-1,))

    @staticmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        
        # Wrap indexes modulo *length*; weights unaffected
        return np.mod(indexes, length), weights

    @staticmethod
    def compute_coefficients(
        data: npt.NDArray, basis: SplineBasis
    ) -> npt.NDArray:
        
        # If the basis has no poles (nearest, linear, …) nothing to do
        if basis.poles is None:
            return np.copy(data)

        # ---------- NumPy path ----------
        if not is_cupy_type(data):
            return PeriodicPadding._ifft_cyclic_inverse(data, basis)

        # ---------- CuPy path (same idea, GPU FFT) ----------
        import cupy as cp

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
        return cp.fft.ifftn(Fc, axes=(-1,))
