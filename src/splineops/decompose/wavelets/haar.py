"""
haar.py
-------
Modified version that handles single-row or single-column inputs gracefully
by skipping the degenerate pass. 
"""

import numpy as np
from .abstractwavelets import AbstractWavelets

class HaarWavelets(AbstractWavelets):

    def __init__(self, scales=3):
        super().__init__(scales=scales)
        self.q = np.sqrt(2.0)

    def get_name(self):
        return "Haar"

    def get_documentation(self):
        return "Haar Wavelets Decomposition (with 2D row->col transform)."

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale analysis pass:
         1) Row transform (split),
         2) Column transform (split),
        skipping any dimension=1 to avoid degenerate transforms.
        """
        out = np.copy(inp)
        ny, nx = out.shape

        # 1) Row pass if more than 1 column
        if nx > 1:
            for r in range(ny):
                out[r, :] = self._split(out[r, :])

        # 2) Column pass if more than 1 row
        if ny > 1:
            for c in range(nx):
                col = out[:, c]
                out[:, c] = self._split(col)

        return out

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale synthesis pass:
         1) Column inverse transform (merge),
         2) Row inverse transform (merge),
        skipping any dimension=1.
        """
        out = np.copy(inp)
        ny, nx = out.shape

        # 1) Column pass if ny > 1
        if ny > 1:
            for c in range(nx):
                col = out[:, c]
                out[:, c] = self._merge(col)

        # 2) Row pass if nx > 1
        if nx > 1:
            for r in range(ny):
                out[r, :] = self._merge(out[r, :])

        return out

    def _split(self, v):
        """
        'Split' step of Haar transform on a 1D vector:
          approximate = (v[2i] + v[2i+1]) / sqrt(2)
          detail      = (v[2i] - v[2i+1]) / sqrt(2)
        """
        n = v.shape[0]
        half = n // 2
        out = np.zeros(n, dtype=v.dtype)
        for i in range(half):
            j = 2*i
            out[i]       = (v[j] + v[j+1]) / self.q
            out[i+half]  = (v[j] - v[j+1]) / self.q

        # If n is odd, the last sample has no pair => you might keep it as is or do something else
        if (n % 2) == 1:
            # For safety, replicate last sample in the last position
            out[-1] = v[-1]
        return out

    def _merge(self, v):
        """
        'Merge' step (inverse of 'split') on a 1D vector.
        """
        n = v.shape[0]
        half = n // 2
        out = np.zeros(n, dtype=v.dtype)
        for i in range(half):
            a = v[i]
            d = v[i + half]
            out[2*i]   = (a + d) / self.q
            out[2*i+1] = (a - d) / self.q

        # handle odd length similarly
        if (n % 2) == 1:
            out[-1] = v[-1]
        return out
