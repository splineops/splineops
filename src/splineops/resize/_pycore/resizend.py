# splineops/src/splineops/resize/_pycore/resizend.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Work1D
from .plan1d import make_plan_1d
from .utils import strides_from_shape
from .resize1d import resize_1d_ws

def resize_along_axis(arr: np.ndarray, axis: int, p: LSParams) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64, order="C")
    D = arr.ndim
    N_line = arr.shape[axis]
    plan = make_plan_1d(N_line, p)

    # move axis to front → (N, rest)
    x = np.moveaxis(arr, axis, 0)
    N = x.shape[0]; cols = int(np.prod(x.shape[1:] or (1,)))
    X = x.reshape(N, cols)
    Y = np.empty((plan.outN, cols), dtype=np.float64)

    ws = Work1D()
    for j in range(cols):
        y = resize_1d_ws(X[:, j], p, plan, ws)
        Y[:, j] = y

    out = Y.reshape((plan.outN,) + x.shape[1:])
    out = np.moveaxis(out, 0, axis)
    return out
