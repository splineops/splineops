"""
abstractwavelets.py
-------------------
Defines a base class for wavelet analysis & synthesis on 2D (or 3D) signals.
"""

import numpy as np

class AbstractWavelets:
    """
    Base class for wavelet decomposition with 'analysis' and 'synthesis' methods.
    Subclasses should implement analysis1() and synthesis1() on single-scale.
    """

    def __init__(self, scales=3):
        self.scales = scales

    def set_scale(self, scale: int):
        self.scales = scale

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale wavelet transform of inp -> out. Must be overridden.
        """
        raise NotImplementedError

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale inverse wavelet transform of inp -> out. Must be overridden.
        """
        raise NotImplementedError

    def analysis(self, inp: np.ndarray) -> np.ndarray:
        """
        Multi-scale analysis in 2D or 3D. Here we show a 2D example:
        """
        out = np.copy(inp)
        # successively apply analysis1 at each scale
        nx, ny = out.shape
        for i in range(self.scales):
            # consider sub-image of size (nx, ny), transform in-place
            sub = out[:ny, :nx]   # for 2D, or out[:,...] for 3D
            sub_out = self.analysis1(sub)
            out[:ny, :nx] = sub_out
            nx = max(1, nx//2)
            ny = max(1, ny//2)
        return out

    def synthesis(self, inp: np.ndarray) -> np.ndarray:
        """
        Multi-scale synthesis in 2D or 3D. 
        """
        out = np.copy(inp)
        # starting from coarsest scale
        # e.g. coarsest size is (nx//(2^(scales-1)), ny//(2^(scales-1)))
        factor = 2 ** (self.scales - 1)
        # coarsest dims
        nx_coarse = max(1, out.shape[1] // factor)
        ny_coarse = max(1, out.shape[0] // factor)

        nx = nx_coarse
        ny = ny_coarse
        for i in range(self.scales):
            sub = out[:ny, :nx]
            sub_out = self.synthesis1(sub)
            out[:ny, :nx] = sub_out
            nx = min(out.shape[1], nx * 2)
            ny = min(out.shape[0], ny * 2)
        return out

    def get_name(self) -> str:
        return "AbstractWavelets"

    def get_documentation(self) -> str:
        return "Base class for wavelets."

