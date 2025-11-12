# splineops/src/splineops/resize/_pycore/resizend.py
# splineops/src/splineops/resize/_pycore/resizend.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Work1D
from .plan1d import make_plan_1d
from .resize1d import resize_1d_ws

def resize_along_axis(arr: np.ndarray, axis: int, p: LSParams) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64, order="C")
    N_line = arr.shape[axis]
    plan = make_plan_1d(N_line, p)

    # Move target axis to the end so each line is contiguous
    x_last = np.moveaxis(arr, axis, -1)                        # (..., N)
    cols = int(np.prod(x_last.shape[:-1] or (1,)))
    X = x_last.reshape(cols, N_line)                            # (cols, N) rows are contiguous
    Y = np.empty((cols, plan.outN), dtype=np.float64)          # (cols, outN)

    ws = Work1D()
    for j in range(cols):
        # write directly into Y[j] (no temp allocations)
        resize_1d_ws(X[j], p, plan, ws, out=Y[j])

    out_last = Y.reshape(x_last.shape[:-1] + (plan.outN,))     # (..., outN)
    return np.moveaxis(out_last, -1, axis)
