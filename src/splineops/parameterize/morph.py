"""
morph.py  –  analytic “monopole” surface + rhythmic breathing
=============================================================

• Vertex positions are the original analytic formula from Philippe
  Thévenaz’s screensaver.

• Normals are estimated by centred finite differences.

• A sinusoidal bulge (0.12 units peak-to-peak) along the local normal
  gives the surface visible motion without any heavy B-spline machinery.

Public API
──────────
    m = Morph()
    m.update(t_seconds)          # cache phase; O(1)
    xyz, nrm = m.evaluate(st)    # st ∈ [0,1]²  →  (3,), (3,)   float32
"""
from __future__ import annotations
import numpy as np
from constants import PI2, MESH_W, MESH_H

class Morph:
    def __init__(self,
                 freq_hz: float = 0.3,     # breathing frequency
                 amp: float = 0.06):       # bulge amplitude (radius units)
        self._omega = 2 * np.pi * freq_hz
        self._amp   = amp
        self._phase = 0.0                  # updated every frame

    # ───────────────────────── analytic base surface ──────────────────────
    @staticmethod
    def _monopole(st):
        """Original screensaver surface: st = (u,v) in [0,1]²."""
        u = st[1] * PI2                  # meridional
        v = st[0] * PI2                  # circumferential
        return np.array([
            -np.sin(u)            * np.sin(0.5 * v),
            -0.5 * (1 - np.cos(u)) * np.sin(v),
            -1   + 0.5 * (1 - np.cos(u)) * (1 - np.cos(v))],
            dtype='f4')

    # ─────────────────────────── API methods ──────────────────────────────
    def update(self, t: float):
        """Advance internal phase for time *t* (seconds)."""
        self._phase = self._omega * t

    def evaluate(self, st):
        """
        Return (position, normal) for parametric coords *st* (float pair).
        """
        p0 = self._monopole(st)

        # finite-difference normal
        du = 1.0 / (MESH_H - 1)
        dv = 1.0 / (MESH_W - 1)
        p_du = self._monopole([st[0], st[1] + du]) - p0
        p_dv = self._monopole([st[0] + dv, st[1]]) - p0
        n = np.cross(p_du, p_dv)
        n /= np.linalg.norm(n) + 1e-12     # guard div-by-zero

        # breathing displacement: sinusoid modulated over 4 circumferential waves
        disp = self._amp * np.sin(4 * np.pi * st[0] + self._phase)
        p = p0 + disp * n

        return p.astype('f4'), n.astype('f4')
