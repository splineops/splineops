# splineops/src/splineops/multiscale/wavelets/abstract_wavelets.py

"""
abstract_wavelets.py
--------------------
Defines a base class for wavelet analysis & synthesis on 2D (or 3D) signals.
"""

import numpy as np
import operator


class AbstractWavelets:
    """
    Base class for wavelet decomposition with multi-scale analysis & synthesis.

    Subclasses must override:
      - analysis1()  (single-scale wavelet decomposition)
      - synthesis1() (single-scale wavelet reconstruction)

    Parameters
    ----------
    scales : int
        Number of scales for multi-scale decomposition.

    Attributes
    ----------
    scales : int
        Number of scales for repeated analysis/synthesis steps.
    """

    def __init__(self, scales=3):
        self.set_scale(scales)

    def set_scale(self, scale: int):
        """
        Update the number of scales.

        Parameters
        ----------
        scale : int
            New scale value.
        """
        try:
            scale = operator.index(scale)
        except TypeError as exc:
            raise TypeError("'scales' must be a positive integer.") from exc
        if isinstance(scale, (bool, np.bool_)) or scale < 1:
            raise ValueError("'scales' must be a positive integer.")
        self.scales = scale

    def _validate_multiscale_input(self, inp):
        if not isinstance(inp, np.ndarray):
            raise TypeError("Wavelet input must be a NumPy array.")
        if inp.ndim != 2 or any(length == 0 for length in inp.shape):
            raise ValueError("Wavelet input must be a non-empty 2D array.")
        if not np.issubdtype(inp.dtype, np.number) or np.iscomplexobj(inp):
            raise TypeError("Wavelet input must have a real numeric dtype.")
        divisor = 2**self.scales
        if any(length % divisor for length in inp.shape):
            raise ValueError(
                "Both wavelet dimensions must be divisible by "
                f"2**scales ({divisor}); received shape {inp.shape}."
            )

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale wavelet transform of inp -> out. Must be overridden.

        Parameters
        ----------
        inp : np.ndarray
            Input array (2D or 3D typically).

        Returns
        -------
        np.ndarray
            Transformed array (same shape).
        """
        raise NotImplementedError

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale inverse wavelet transform of inp -> out. Must be overridden.

        Parameters
        ----------
        inp : np.ndarray
            Input wavelet coefficients (single scale).

        Returns
        -------
        np.ndarray
            Reconstructed array at that scale.
        """
        raise NotImplementedError

    def analysis(self, inp: np.ndarray) -> np.ndarray:
        """
        Multi-scale wavelet analysis in 2D (or 3D).
        Repeatedly calls analysis1() from fine to coarse.

        Parameters
        ----------
        inp : np.ndarray
            Input array with shape (ny, nx) or (nz, ny, nx).

        Returns
        -------
        np.ndarray
            Full wavelet decomposition (in-place layout).
        """
        self._validate_multiscale_input(inp)
        out = np.copy(inp)
        ny, nx = out.shape[:2]  # for 2D
        for _ in range(self.scales):
            sub = out[:ny, :nx]
            sub_out = self.analysis1(sub)
            out[:ny, :nx] = sub_out
            nx = max(1, nx // 2)
            ny = max(1, ny // 2)
        return out

    def synthesis(self, inp: np.ndarray) -> np.ndarray:
        """
        Multi-scale wavelet synthesis in 2D (or 3D).
        Repeatedly calls synthesis1() from coarsest to finest.

        Parameters
        ----------
        inp : np.ndarray
            Wavelet decomposition array (same shape as original).

        Returns
        -------
        np.ndarray
            Reconstructed array (same shape as input).
        """
        self._validate_multiscale_input(inp)
        out = np.copy(inp)
        ny_full, nx_full = out.shape[:2]
        factor = 2 ** (self.scales - 1)
        nx_coarse = max(1, nx_full // factor)
        ny_coarse = max(1, ny_full // factor)

        nx, ny = nx_coarse, ny_coarse
        for _ in range(self.scales):
            sub = out[:ny, :nx]
            sub_out = self.synthesis1(sub)
            out[:ny, :nx] = sub_out
            nx = min(nx_full, nx * 2)
            ny = min(ny_full, ny * 2)
        return out

    def get_name(self) -> str:
        """
        Returns a descriptive name for the wavelet transform.
        """
        return "AbstractWavelets"

    def get_documentation(self) -> str:
        """
        Returns a short description of the wavelet.
        """
        return "Base class for wavelets."
