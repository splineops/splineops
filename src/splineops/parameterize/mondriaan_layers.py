"""
CPU-only Mondriaan colour texture (128×128×3 float32).

Very light-weight: no OpenGL – it just returns a NumPy image that
vtk_mondriaan.py uploads into a vtkTexture every frame.
"""
from __future__ import annotations
import numpy as np
from constants import N_LAYERS, TEX_W, TEX_H, LUT_LEN
from spline import bspline3

_rng = np.random.default_rng()

# static lookup table: simple smooth gradient ------------------------------
_g = np.linspace(0, 1, LUT_LEN, dtype='f4')
_LUT = np.stack([_g, _g**2, np.sqrt(_g)], axis=1)

def _initial_mix():
    m = np.array([bspline3(4*k/N_LAYERS - 2) for k in range(N_LAYERS)], 'f4')
    m = np.sqrt(m); return m / m.sum()

class MondriaanLayers:
    def __init__(self):
        self.col  = _rng.random(N_LAYERS).astype('f4')
        self.mix_prev = _initial_mix()
        self.mix_next = np.roll(self.mix_prev.copy(), 1)
        self._t_prev  = -1

        # pre-compute X grid once
        self.x = np.linspace(-1, 1, TEX_W, dtype='f4')[None, :]  # shape (1,W)
        self.hann = 0.5 * (1 + np.cos(np.linspace(0, np.pi*2, TEX_W, dtype='f4')))

    # ---------------------------------------------------------------------
    def _rotate_layers(self):
        """Called once per whole second – cyclic permutation of weights."""
        self.mix_prev[:] = self.mix_next
        self.mix_next = np.roll(self.mix_next, 1)
        # randomise the colour of the new layer 0
        self.col[0] = _rng.random()

    # ---------------------------------------------------------------------
    def update(self, t: float):
        ti, tf = int(t), t - int(t)
        if ti > self._t_prev:
            self._rotate_layers(); self._t_prev = ti

        mix = self.mix_next*tf + self.mix_prev*(1-tf)

        # extremely cheap procedural pattern: sum of weighted Hann stripes
        stripes = np.sin(self.x * np.arange(1, N_LAYERS+1) * np.pi)
        gray = (self.col * mix)[None, :] * stripes
        gray = gray.sum(axis=1)  # shape (1,W)
        gray = np.tile(gray, (TEX_H, 1)) * self.hann  # fade Y edges

        idx = (gray.clip(0,1) * (LUT_LEN-1)).astype('i4')
        return _LUT[idx]
