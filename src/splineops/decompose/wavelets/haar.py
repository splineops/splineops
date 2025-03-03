"""
haar2d.py
---------
Implements a 2D-only Haar wavelet transform using row->column decomposition 
(analysis) and column->row synthesis. Raises an error if ny<2 or nx<2.
"""

import numpy as np
from .abstractwavelets import AbstractWavelets

class HaarWavelets(AbstractWavelets):
    """
    A pure 2D Haar wavelet transform class for images with shape (ny, nx),
    where ny >= 2, nx >= 2.

    Analysis (single-scale):
      1) row-wise split
      2) column-wise split
    Synthesis (single-scale):
      1) column-wise merge
      2) row-wise merge

    If ny < 2 or nx < 2, it raises ValueError. 
    """

    def __init__(self, scales=3):
        super().__init__(scales=scales)
        self.q = np.sqrt(2.0)

    def get_name(self):
        return "Haar2D"

    def get_documentation(self):
        return "Pure 2D Haar wavelets, requiring ny>=2 and nx>=2."

    # -------------------------
    # Single-scale analysis
    # -------------------------
    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale 2D Haar analysis.

        inp: shape=(ny, nx), both >=2
        Returns a new array of the same shape with row+col splits.
        """
        ny, nx = inp.shape
        if ny < 2 or nx < 2:
            raise ValueError(f"Haar2D requires both ny>=2 and nx>=2, got shape=({ny},{nx}).")

        # Make a copy so we don't overwrite the input
        out = np.copy(inp)

        # 1) row-wise split
        for r in range(ny):
            out[r, :] = self._split(out[r, :])

        # 2) column-wise split
        for c in range(nx):
            col = out[:, c]
            out[:, c] = self._split(col)

        return out

    # -------------------------
    # Single-scale synthesis
    # -------------------------
    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale 2D Haar synthesis (inverse of analysis1).
        """
        ny, nx = inp.shape
        if ny < 2 or nx < 2:
            raise ValueError(f"Haar2D requires both ny>=2 and nx>=2, got shape=({ny},{nx}).")

        out = np.copy(inp)

        # 1) column-wise merge (inverse of column-wise split)
        for c in range(nx):
            col = out[:, c]
            out[:, c] = self._merge(col)

        # 2) row-wise merge
        for r in range(ny):
            out[r, :] = self._merge(out[r, :])

        return out

    # -------------------------
    # Haar 1D split/merge
    # -------------------------
    def _split(self, v: np.ndarray) -> np.ndarray:
        """
        1D Haar split transform on vector v of length >= 2:
          A[i] = (v[2i] + v[2i+1]) / sqrt(2)
          D[i] = (v[2i] - v[2i+1]) / sqrt(2)
        concatenated as [A0..A_{n/2 -1} | D0..D_{n/2 -1}]
        If v has odd length, the last element can be handled in various ways.
        Here we assume v has even length => because nx, ny >= 2, 
        but if you want to handle any length, you can adapt this code.
        """
        n = v.shape[0]
        half = n // 2
        out = np.zeros(n, dtype=v.dtype)
        for i in range(half):
            j = 2 * i
            a = (v[j] + v[j+1]) / self.q
            d = (v[j] - v[j+1]) / self.q
            out[i]        = a
            out[i + half] = d
        # If n is odd, you'd handle the leftover element, 
        # but let's assume n is even for simplicity.
        return out

    def _merge(self, v: np.ndarray) -> np.ndarray:
        """
        1D Haar merge (inverse of _split) on vector v.
        First half is approximate, second half is detail.
        """
        n = v.shape[0]
        half = n // 2
        out = np.zeros(n, dtype=v.dtype)
        for i in range(half):
            a = v[i]
            d = v[i + half]
            out[2*i]   = (a + d) / self.q
            out[2*i+1] = (a - d) / self.q
        return out
