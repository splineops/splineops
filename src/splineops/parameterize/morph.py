"""
morph.py – analytic “monopole” surface + rhythmic breathing
===========================================================

Implements Philippe Thévenaz’ parameterisation (Monopole.pdf, eq 3.5).
All four edges of the parameter rectangle map to the south pole, matching
the mesh topology used throughout this port.

Public API
──────────
    m = Morph()
    m.update(t_seconds)              # cache breathing phase
    xyz, nrm = m.evaluate((s,t))     # (s,t) either indices or [0,1] floats
"""
from __future__ import annotations
import numpy as np
from constants import MESH_W, MESH_H, PI2


# ───────────────────────── helper ────────────────────────────────────────
def _wrap(x: float, length: int) -> float:
    """Periodic wrap of *x* into [0, length)."""
    return (x + length) % length


# ───────────────────────── Morph class ───────────────────────────────────
class Morph:
    """
    Generate vertex positions and normals for the breathing monopole.

    Parameters
    ----------
    freq_hz : float  – breathing frequency (Hz)
    amp     : float  – amplitude of the bulge along the normal
    """

    def __init__(self, *, freq_hz: float = 0.30, amp: float = 0.06):
        self._omega = 2.0 * np.pi * freq_hz
        self._amp   = amp
        self._phase = 0.0                      # advanced in `update`

    # ─────────────────────── time evolution ──────────────────────────────
    def update(self, t: float):
        """Advance breathing phase to absolute time *t* (seconds)."""
        self._phase = self._omega * t

    # ────────────────── analytical monopole mapping ──────────────────────
    @staticmethod
    def _monopole(st):
        """
        Thévenaz mapping (collapses entire boundary to south pole).

        The formula works for *any* real st, not just integers, which is
        convenient for finite-difference estimates.
        """
        s, t = st
        a_s = np.pi * s / MESH_W         # π·s/W
        a_t = np.pi * t / MESH_H         # π·t/H

        # unified meridional angle
        u = 2.0 * np.arctan2(np.sin(a_t),
                             np.cos(a_t) * np.sin(a_s))
        v = PI2 * s / MESH_W             # ordinary longitude

        x = -np.sin(u) * np.sin(0.5 * v)
        y = -0.5 * (1.0 - np.cos(u)) * np.sin(v)
        z = -1.0 + 0.5 * (1.0 - np.cos(u)) * (1.0 - np.cos(v))
        return np.array([x, y, z], dtype='f4')

    # ───────────────── position & normal at (s,t) ────────────────────────
    def evaluate(self, st):
        """
        Compute (position, unit normal) for parameter coordinates *st*.

        *st* may be either
            • mesh indices    s∈[0, MESH_W), t∈[0, MESH_H)   (ints/floats)
            • normalised      u,v ∈ [0,1]                   (floats)

        Returns
        -------
        xyz, nrm : numpy float32 arrays, shape (3,)
        """
        s_f, t_f = map(float, st)

        # ── accept normalised coordinates too ───────────────────────────
        if 0.0 <= s_f <= 1.0 and 0.0 <= t_f <= 1.0:
            s_f *= (MESH_W - 1)
            t_f *= (MESH_H - 1)

        # central differences (Δ = 1 index) with wrapping
        p0   = self._monopole((s_f,             t_f))
        p_du = self._monopole((s_f, _wrap(t_f + 1.0, MESH_H))) - \
               self._monopole((s_f, _wrap(t_f - 1.0, MESH_H)))
        p_dv = self._monopole((_wrap(s_f + 1.0, MESH_W), t_f)) - \
               self._monopole((_wrap(s_f - 1.0, MESH_W), t_f))

        n = np.cross(p_du, p_dv)
        n_len = np.linalg.norm(n)
        if n_len < 1e-8:                        # exactly at the pole
            n = np.array([0.0, 0.0, -1.0], dtype='f4')
        else:
            n /= n_len

        # breathing displacement: 4 circumferential waves
        theta = PI2 * s_f / MESH_W
        disp  = self._amp * np.sin(4.0 * theta + self._phase)

        p = p0 + disp * n
        return p.astype('f4'), n.astype('f4')
