"""
mondriaan_layers.py  –  CPU-only Mondriaan texture generator
===========================================================

Returns a NumPy RGB image (128×128×3 float32 in [0,1]) every frame.

The pattern here is deliberately simple: 40 vertical sine stripes whose
amplitudes are blended by the rolling B-spline “mixing” weights.  It’s
meant as a stand-in for the far more complex GLSL original—you can
replace the math later without touching the public API.
"""
from __future__ import annotations
import numpy as np
from constants import N_LAYERS, TEX_W, TEX_H, LUT_LEN
from spline import bspline3

_rng = np.random.default_rng()

# ---- static colour look-up table (placeholder) ---------------------------
_g   = np.linspace(0, 1, LUT_LEN, dtype='f4')
_LUT = np.stack([_g, _g**2, np.sqrt(_g)], axis=1)      # (1024,3)

def _initial_mix() -> np.ndarray:
    w = np.array([bspline3(4*k/N_LAYERS - 2) for k in range(N_LAYERS)], 'f4')
    w = np.sqrt(w)
    return w / w.sum()

# --------------------------------------------------------------------------
class MondriaanLayers:
    def __init__(self):
        self.colors    = _rng.random(N_LAYERS).astype('f4')
        self.mix_prev  = _initial_mix()
        self.mix_next  = np.roll(self.mix_prev.copy(), 1)
        self.t_prev    = -1                            # last integer second

        # x-grid (1,128) and stripe wave-numbers (40,1) for vectorised eval
        self.x  = np.linspace(-1, 1, TEX_W, dtype='f4')[None, :]     # (1,W)
        self.k  = np.arange(1, N_LAYERS + 1, dtype='f4')[:, None]    # (L,1)

        # simple vertical fade so top/bottom edges aren’t hard-cut
        self.fade_y = 0.5 * (1 + np.cos(np.linspace(0, np.pi*2, TEX_H, 'f4')))

    # ----------------------------------------------------------------------
    def _roll_once(self):
        """Rotate mixing weights once per whole second and randomise layer 0."""
        self.mix_prev[:] = self.mix_next
        self.mix_next    = np.roll(self.mix_next, 1)
        self.colors[0]   = _rng.random()        # new random colour for layer 0

    # ----------------------------------------------------------------------
    def update(self, t: float) -> np.ndarray:
        """Return a fresh (128×128×3) float32 RGB image for time *t* (sec)."""
        ti, tf = int(t), t - int(t)
        if ti > self.t_prev:
            self._roll_once(); self.t_prev = ti

        # linear interpolation between the two weight sets
        mix = self.mix_next * tf + self.mix_prev * (1 - tf)          # (L,)

        # build sine stripes and mix them
        stripes = np.sin(self.k * np.pi * self.x)                    # (L,W)
        gray_x  = (self.colors * mix)[:, None] * stripes             # (L,W)
        gray_1d = gray_x.sum(axis=0)                                 # (W,)
        gray_2d = gray_1d[None, :] * self.fade_y[:, None]            # (H,W)

        # LUT lookup → RGB
        idx = np.clip((gray_2d * (LUT_LEN - 1)).astype('i4'), 0, LUT_LEN - 1)
        rgb = _LUT[idx]                                              # (H,W,3)
        return rgb.astype('f4')
