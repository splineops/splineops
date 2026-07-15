# splineops/src/splineops/multiscale/wavelets/haar.py

"""
haar.py
-------
Implements a 2D Haar wavelet transform via row->column decomposition
(analysis) and column->row synthesis. Requires ny >= 2 and nx >= 2.
"""

import numpy as np
from .abstract_wavelets import AbstractWavelets


class HaarWavelets(AbstractWavelets):
    """
    A 2D Haar wavelet transform for images with shape (ny, nx).

    Parameters
    ----------
    scales : int
        Number of scales (defaults to 3).

    Attributes
    ----------
    q : float
        Normalization factor, sqrt(2).

    Raises
    ------
    ValueError
        If ny < 2 or nx < 2 at any scale.

    Notes
    -----
    Single-scale steps:
      1) row-wise split
      2) column-wise split
    Synthesis:
      1) column-wise merge
      2) row-wise merge
    """

    def __init__(self, scales=3):
        super().__init__(scales=scales)
        self.q = np.sqrt(2.0)

    def get_name(self):
        """Return 'Haar2D'."""
        return "Haar2D"

    def get_documentation(self):
        """Return short docstring for Haar wavelets."""
        return "Pure 2D Haar wavelets, requiring ny>=2 and nx>=2."

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale 2D Haar analysis (row-split, then col-split).

        Parameters
        ----------
        inp : np.ndarray
            2D array with shape (ny, nx), both >= 2.

        Returns
        -------
        np.ndarray
            Transformed array (same shape).
        """
        out = self._prepare_single_scale_input(inp)
        ny, nx = out.shape[-2:]
        if ny < 2 or nx < 2 or ny % 2 or nx % 2:
            raise ValueError(
                "Haar2D needs even ny>=2 and nx>=2, " f"got shape=({ny},{nx})."
            )

        out = self._split_axis(out, axis=-1)
        return self._split_axis(out, axis=-2)

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale 2D Haar synthesis (inverse of analysis1).

        Parameters
        ----------
        inp : np.ndarray
            2D array (ny, nx), both >= 2.

        Returns
        -------
        np.ndarray
            Reconstructed array.
        """
        out = self._prepare_single_scale_input(inp)
        ny, nx = out.shape[-2:]
        if ny < 2 or nx < 2 or ny % 2 or nx % 2:
            raise ValueError(
                "Haar2D needs even ny>=2 and nx>=2, " f"got shape=({ny},{nx})."
            )

        out = self._merge_axis(out, axis=-2)
        return self._merge_axis(out, axis=-1)

    def _split_axis(self, array, axis):
        moved = np.moveaxis(array, axis, -1)
        return np.moveaxis(self._split(moved), -1, axis)

    def _merge_axis(self, array, axis):
        moved = np.moveaxis(array, axis, -1)
        return np.moveaxis(self._merge(moved), -1, axis)

    def _split(self, v: np.ndarray) -> np.ndarray:
        """
        1D Haar split of vector v (length even).

        A[i] = (v[2i] + v[2i+1]) / sqrt(2)
        D[i] = (v[2i] - v[2i+1]) / sqrt(2)

        Parameters
        ----------
        v : np.ndarray
            1D array, even length >= 2.

        Returns
        -------
        np.ndarray
            Concatenated [A..., D...].
        """
        n = v.shape[-1]
        half = n // 2
        even = v[..., 0::2]
        odd = v[..., 1::2]
        out = np.empty_like(v)
        np.add(even, odd, out=out[..., :half])
        out[..., :half] /= self.q
        np.subtract(even, odd, out=out[..., half:])
        out[..., half:] /= self.q
        return out

    def _merge(self, v: np.ndarray) -> np.ndarray:
        """
        1D Haar merge (inverse of _split).

        First half is A (approx), second half is D (detail).

        Parameters
        ----------
        v : np.ndarray
            1D array with length n, n even, the first half are approx, second half detail.

        Returns
        -------
        np.ndarray
            Reconstructed array of length n.
        """
        n = v.shape[-1]
        half = n // 2
        out = np.empty_like(v)
        approximation = v[..., :half]
        detail = v[..., half:]
        np.add(approximation, detail, out=out[..., 0::2])
        out[..., 0::2] /= self.q
        np.subtract(approximation, detail, out=out[..., 1::2])
        out[..., 1::2] /= self.q
        return out
