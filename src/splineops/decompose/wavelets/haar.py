"""
haar.py
-------
Implementation of Haar wavelets using the AbstractWavelets interface.
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
        return "Haar Wavelets Decomposition"

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        # Expect 2D array. We'll do a row-wise transform, then col-wise
        # For brevity, let's define a helper:
        def split(v):
            n = v.shape[0]
            half = n//2
            a = np.zeros(n, dtype=v.dtype)
            for i in range(half):
                j = 2*i
                a[i] = (v[j] + v[j+1]) / self.q
                a[i+half] = (v[j] - v[j+1]) / self.q
            return a

        # 1) row transform
        out = np.copy(inp)
        for r in range(out.shape[0]):
            out[r, :] = split(out[r, :])

        # 2) column transform
        for c in range(out.shape[1]):
            col = out[:, c]
            col_split = split(col)
            out[:, c] = col_split

        return out

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        # inverse transform
        def merge(v):
            n = v.shape[0]
            half = n//2
            a = np.zeros(n, dtype=v.dtype)
            for i in range(half):
                a[2*i]   = (v[i] + v[i+half]) / self.q
                a[2*i+1] = (v[i] - v[i+half]) / self.q
            return a

        out = np.copy(inp)
        # inverse column transform
        for c in range(out.shape[1]):
            col = out[:, c]
            col_merged = merge(col)
            out[:, c] = col_merged

        # inverse row transform
        for r in range(out.shape[0]):
            out[r, :] = merge(out[r, :])

        return out
