"""
BrownianVector3 / BrownianRotation4 – B-spline-smoothed random motion.

Used only for camera drift/rotation in the VTK demo, so no external
OpenGL dependencies.
"""
from __future__ import annotations
import numpy as np
from spline import bspline3

rng = np.random.default_rng()

# shared helper -------------------------------------------------------------
def _weights(t_frac: float):
    return np.array([bspline3(t_frac + 1),
                     bspline3(t_frac),
                     bspline3(t_frac - 1),
                     bspline3(t_frac - 2)], dtype='f4')

# --------------------------------------------------------------------------
class BrownianVector3:
    def __init__(self):
        self.ctrl   = rng.standard_normal((4, 3)).astype('f4')
        self.t_prev = -1

    def _roll(self):
        self.ctrl[:-1] = self.ctrl[1:]
        self.ctrl[-1]  = rng.standard_normal(3)

    def update(self, t: float) -> np.ndarray:
        ti, tf = int(t), t - int(t)
        if ti > self.t_prev:
            self._roll(); self.t_prev = ti
        w = _weights(tf)[:, None]                 # (4,1)
        return (w * self.ctrl).sum(axis=0)        # (3,)


# --------------------------------------------------------------------------
class BrownianRotation4:
    def __init__(self):
        self.axis_ctrl = rng.standard_normal((4, 3)).astype('f4')
        self.ang_ctrl  = rng.uniform(0, 2*np.pi, 4).astype('f4')
        self.t_prev    = -1

    def _roll(self):
        self.axis_ctrl[:-1] = self.axis_ctrl[1:]
        self.ang_ctrl[:-1]  = self.ang_ctrl[1:]
        self.axis_ctrl[-1]  = rng.standard_normal(3)
        self.ang_ctrl[-1]  += rng.uniform(-2*np.pi, 2*np.pi)

    def update(self, t: float) -> np.ndarray:
        ti, tf = int(t), t - int(t)
        if ti > self.t_prev:
            self._roll(); self.t_prev = ti
        w = _weights(tf)
        axis  = (w[:, None] * self.axis_ctrl).sum(axis=0)
        angle = (w * self.ang_ctrl).sum()
        axis /= np.linalg.norm(axis) + 1e-12

        c, s = np.cos(angle), np.sin(angle)
        x, y, z = axis
        return np.array([
            [c+(1-c)*x*x,   (1-c)*x*y-s*z, (1-c)*x*z+s*y, 0],
            [(1-c)*y*x+s*z, c+(1-c)*y*y,   (1-c)*y*z-s*x, 0],
            [(1-c)*z*x-s*y, (1-c)*z*y+s*x, c+(1-c)*z*z,   0],
            [0, 0, 0, 1]], dtype='f4')
