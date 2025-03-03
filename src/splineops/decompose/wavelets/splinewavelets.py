"""
splinewavelets.py
-----------------
Implements Spline wavelet transform (e.g. orders 1,3,5) via the AbstractWavelets interface.
"""

import numpy as np
from .abstractwavelets import AbstractWavelets
from .filters import SplineFilter

class SplineWavelets(AbstractWavelets):

    def __init__(self, scales=3, order=3):
        super().__init__(scales=scales)
        self.order = order
        self.filter = SplineFilter(order)

    def get_name(self):
        return f"Spline{self.order}"

    def get_documentation(self):
        return f"Spline Wavelets (order={self.order})."

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        # row-wise splitMirror, then col-wise, as your Java code does
        # ...
        # The code basically calls a function:
        out = np.copy(inp)
        # do row pass
        for r in range(out.shape[0]):
            row = out[r,:]
            out[r,:] = self._split_mirror_1d(row, self.filter.h, self.filter.g)
        # do col pass
        for c in range(out.shape[1]):
            col = out[:,c]
            out[:,c] = self._split_mirror_1d(col, self.filter.h, self.filter.g)
        return out

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        # row-wise mergeMirror, then col-wise
        out = np.copy(inp)
        for c in range(out.shape[1]):
            col = out[:, c]
            out[:, c] = self._merge_mirror_1d(col, self.filter.h, self.filter.g)
        for r in range(out.shape[0]):
            row = out[r, :]
            out[r, :] = self._merge_mirror_1d(row, self.filter.h, self.filter.g)
        return out

    def _split_mirror_1d(self, vin, h, g):
        # replicate your logic from SplineWaveletsTool.splitMirror
        return np.copy(vin)   # placeholder

    def _merge_mirror_1d(self, vin, h, g):
        # replicate your logic from SplineWaveletsTool.mergeMirror
        return np.copy(vin)   # placeholder
