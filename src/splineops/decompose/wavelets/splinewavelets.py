"""
splinewavelets.py
-----------------
Defines small wrapper classes for Spline1Wavelets, Spline3Wavelets, Spline5Wavelets,
using the underlying SplineWaveletsTool for the actual analysis/synthesis logic.
"""

import numpy as np
from splineops.decompose.wavelets.abstractwavelets import AbstractWavelets
from .splinewaveletstool import SplineWaveletsTool

class Spline1Wavelets(AbstractWavelets):
    def __init__(self, scales=3):
        super().__init__(scales=scales)
        self.tool = SplineWaveletsTool(scales, order=1)

    def set_scale(self, scale):
        self.scales = scale
        self.tool = SplineWaveletsTool(scale, order=1)

    def get_name(self):
        return "Spline1"

    def get_documentation(self):
        return "Spline Wavelets (order 1)"

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        return self.tool.analysis1(inp)

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        return self.tool.synthesis1(inp)


class Spline3Wavelets(AbstractWavelets):
    def __init__(self, scales=3):
        super().__init__(scales=scales)
        self.tool = SplineWaveletsTool(scales, order=3)

    def set_scale(self, scale):
        self.scales = scale
        self.tool = SplineWaveletsTool(scale, order=3)

    def get_name(self):
        return "Spline3"

    def get_documentation(self):
        return "Spline Wavelets (order 3)"

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        return self.tool.analysis1(inp)

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        return self.tool.synthesis1(inp)


class Spline5Wavelets(AbstractWavelets):
    def __init__(self, scales=3):
        super().__init__(scales=scales)
        self.tool = SplineWaveletsTool(scales, order=5)

    def set_scale(self, scale):
        self.scales = scale
        self.tool = SplineWaveletsTool(scale, order=5)

    def get_name(self):
        return "Spline5"

    def get_documentation(self):
        return "Spline Wavelets (order 5)"

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        return self.tool.analysis1(inp)

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        return self.tool.synthesis1(inp)
