"""
Analytic “monopole” surface (same shape as the original screen-saver).

No heavy spline machinery – easy to understand and always valid.
"""
from __future__ import annotations
import numpy as np
from constants import PI2, MESH_W, MESH_H

class Morph:
    # analytic parametric surface -----------------------------------------
    def _monopole(self, st):
        u = st[1] * PI2    # “t” axis
        v = st[0] * PI2    # “s” axis
        return np.array([
            -np.sin(u) * np.sin(0.5 * v),
            -0.5 * (1 - np.cos(u)) * np.sin(v),
            -1   + 0.5 * (1 - np.cos(u)) * (1 - np.cos(v))],
            dtype='f4')

    def update(self, t: float):
        """Nothing to update: surface is purely analytic."""
        pass

    def evaluate(self, st):
        """
        Return (position, normal) for a given (s,t) in [0,1]².
        Normal is estimated by finite differences.
        """
        p = self._monopole(st)
        du = 1.0 / (MESH_H - 1)
        dv = 1.0 / (MESH_W - 1)
        p_du = self._monopole([st[0], st[1] + du]) - p
        p_dv = self._monopole([st[0] + dv, st[1]]) - p
        n = np.cross(p_du, p_dv)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-9 else np.array([0, 0, -1])
        return p, n.astype('f4')
