"""
mondriaan_layers.py
===================

CPU-only generator of a 128 × 128 × 3 float32 RGB texture that mimics the
“40-layer Mondriaan” idea.  It is intentionally lightweight (pure NumPy)
so it runs everywhere VTK does.

Public API
──────────
    layers = MondriaanLayers()
    rgb    = layers.update(t_seconds)   # ndarray (128,128,3), values ∈ [0,1]
"""
from __future__ import annotations
import numpy as np
from matplotlib import cm                       # <─ colourful lookup table
from constants import N_LAYERS, TEX_W, TEX_H, LUT_LEN
from spline    import bspline3

_rng = np.random.default_rng()

# ── colourful LUT : 1024-sample turbo colormap ----------------------------
_LUT = cm.get_cmap('turbo', LUT_LEN)(np.linspace(0, 1, LUT_LEN)
                                     ).astype('f4')[:, :3]      # shape (1024,3)

def _initial_mix() -> np.ndarray:
    w = np.array([bspline3(4*k/N_LAYERS - 2) for k in range(N_LAYERS)], 'f4')
    w = np.sqrt(w)
    return w / w.sum()

# ───────────────────────────────────────────────────────────────────────────
class MondriaanLayers:
    def __init__(self):
        # per-layer random base colour
        self.colors   = _rng.random(N_LAYERS).astype('f4')

        # B-spline mixing weights that roll every second
        self.mix_prev = _initial_mix()
        self.mix_next = np.roll(self.mix_prev.copy(), 1)
        self.t_prev   = -1                      # last whole second processed

        # x grid (1,128) and stripe wavenumbers (40,1) for vectorised pattern
        self.x = np.linspace(-1, 1, TEX_W, dtype='f4')[None, :]   # (1,W)
        self.k = np.arange(1, N_LAYERS + 1, dtype='f4')[:, None]  # (L,1)

    # ---------------------------------------------------------------------
    def _roll_once(self):
        """Rotate mixing weights & randomise colour of the new front layer."""
        self.mix_prev[:] = self.mix_next
        self.mix_next    = np.roll(self.mix_next, 1)
        self.colors[0]   = _rng.random()        # fresh colour

    # ---------------------------------------------------------------------
    def update(self, t: float) -> np.ndarray:
        """
        Return an RGB image (H,W,3) for the given time *t* in seconds.
        """
        ti, tf = int(t), t - int(t)
        if ti > self.t_prev:
            self._roll_once()
            self.t_prev = ti

        # linear interpolate mixing arrays
        mix = self.mix_next * tf + self.mix_prev * (1 - tf)        # (L,)

        # vertical sine stripes multiplied by per-layer amplitude
        stripes = np.sin(self.k * np.pi * self.x)                  # (L,W)
        gray_1d = (self.colors * mix)[:, None] * stripes           # (L,W)
        gray    = gray_1d.sum(axis=0)[None, :]                     # (1,W)

        # replicate rows to full height (simple planar texture)
        gray = np.repeat(gray, TEX_H, axis=0)                      # (H,W)

        # map through LUT to RGB
        idx = np.clip((gray * (LUT_LEN - 1)).astype('i4'), 0, LUT_LEN - 1)
        rgb = _LUT[idx]                                            # (H,W,3)
        return rgb.astype('f4')
