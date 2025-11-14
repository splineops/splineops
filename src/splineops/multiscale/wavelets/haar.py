# splineops/src/splineops/multiscale/wavelets/haar.py

"""
haar.py
-------
Implements a 2D Haar wavelet transform via row->column decomposition 
(analysis) and column->row synthesis. Requires ny >= 2 and nx >= 2.
"""

import numpy as np
from .abstractwavelets import AbstractWavelets

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
        ny, nx = inp.shape
        if ny < 2 or nx < 2:
            raise ValueError(f"Haar2D needs ny>=2 and nx>=2, got shape=({ny},{nx}).")

        out = inp.copy()
        # 1) row-wise
        for r in range(ny):
            out[r, :] = self._split(out[r, :])
        # 2) col-wise
        for c in range(nx):
            out[:, c] = self._split(out[:, c])
        return out

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
        ny, nx = inp.shape
        if ny < 2 or nx < 2:
            raise ValueError(f"Haar2D needs ny>=2 and nx>=2, got shape=({ny},{nx}).")

        out = inp.copy()
        # 1) col-merge
        for c in range(nx):
            out[:, c] = self._merge(out[:, c])
        # 2) row-merge
        for r in range(ny):
            out[r, :] = self._merge(out[r, :])
        return out

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
        n = len(v)
        half = n // 2
        out = np.zeros(n, dtype=v.dtype)
        for i in range(half):
            j = 2*i
            a = (v[j] + v[j+1]) / self.q
            d = (v[j] - v[j+1]) / self.q
            out[i] = a
            out[i+half] = d
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
        n = len(v)
        half = n // 2
        out = np.zeros(n, dtype=v.dtype)
        for i in range(half):
            a = v[i]
            d = v[i+half]
            out[2*i]   = (a + d) / self.q
            out[2*i+1] = (a - d) / self.q
        return out
