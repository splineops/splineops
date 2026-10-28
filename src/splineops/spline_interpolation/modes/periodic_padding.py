# splineops/src/splineops/spline_interpolation/modes/periodic_padding.py

from __future__ import annotations

from typing import Tuple
import numpy as np
import numpy.typing as npt

from ..bases.spline_basis import SplineBasis
from .extension_mode import ExtensionMode
from ..utils import is_cupy_type


class PeriodicPadding(ExtensionMode):

    # Methods
    @staticmethod
    def _ifft_cyclic_inverse(data: np.ndarray, basis: SplineBasis) -> np.ndarray:

        n = data.shape[-1]  # period length
        m = (basis.support - 1) // 2  # half-support

        # Build one period of the basis, wrapped / zero-padded to length n
        bk = basis(np.arange(-m, m + 1, dtype=data.real.dtype))
        bk_per = np.zeros(n, dtype=bk.dtype)
        # Fold every integer basis sample into one period. ``np.add.at`` is
        # essential when the support wraps around a short signal more than
        # once; slice assignment silently assumes m < n.
        offsets = np.arange(-m, m + 1)
        np.add.at(bk_per, np.mod(offsets, n), bk)

        # Forward DFT of data and basis
        if np.isrealobj(data):
            Ff = np.fft.rfft(data, axis=-1)  # real FFT
            Fb = np.fft.rfft(bk_per, n=n, axis=-1)
        else:
            Ff = np.fft.fftn(data, axes=(-1,))  # complex FFT
            Fb = np.fft.fftn(bk_per, axes=(-1,))

        # Protect against divide-by-0 for very small bins
        eps = np.finfo(Fb.real.dtype).eps
        Fb = np.where(np.abs(Fb) < eps, eps, Fb)

        # Divide in Fourier space, then invert
        Fc = Ff / Fb
        coefficients: np.ndarray
        if np.isrealobj(data):
            coefficients = np.fft.irfft(Fc, n=n, axis=-1)
        else:
            coefficients = np.fft.ifftn(Fc, axes=(-1,))
        # NumPy 1.x FFTs promote float32/complex64 to double precision.
        # Keep the requested coefficient dtype across supported NumPy versions.
        return coefficients.astype(data.dtype, copy=False)

    @staticmethod
    def extend_signal(
        indexes: npt.NDArray, weights: npt.NDArray, length: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        # Wrap indexes modulo *length*; weights unaffected
        return np.mod(indexes, length), weights

    @staticmethod
    def compute_coefficients(data: npt.NDArray, basis: SplineBasis) -> npt.NDArray:

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
        offsets = cp.arange(-m, m + 1)
        cp.add.at(bk_per, cp.mod(offsets, n), bk)

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
