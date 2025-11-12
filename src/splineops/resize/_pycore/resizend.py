# splineops/src/splineops/resize/_pycore/resizend.py
from __future__ import annotations
import os
import numpy as np
from .params import LSParams
from .plan1d import make_plan_1d
from .filters import (
    get_interpolation_coefficients_batch,
    get_samples_batch,
)
from .diff_integ import do_integ_batch, do_diff_batch

# Tunable batch size (lines processed together)
_BATCH = int(os.environ.get("SPLINEOPS_BLOCK", "64"))
_BATCH = 1 if _BATCH < 1 else _BLOCK if (_BLOCK := _BATCH) else 64  # keep simple; ensure >=1

def resize_along_axis(arr: np.ndarray, axis: int, p: LSParams) -> np.ndarray:
    """
    Batched pure-NumPy path:
      - move target axis to last dim so each line is contiguous
      - process rows in blocks of B lines with vectorized prefilter/integration/diff
      - build extension / padded buffer once per block
      - gather via np.take(..., axis=1) into (B, out_total, win_len_max)
      - accumulate with multiply+sum into (B, out_total)
    """
    a = np.asarray(arr, dtype=np.float64, order="C")
    N_line = a.shape[axis]
    plan = make_plan_1d(N_line, p)

    # Fast identity short-circuit (no projection, zoom==1)
    if (abs(p.zoom - 1.0) <= 1e-12) and (p.analy_degree < 0) and (plan.outN == N_line):
        return a.copy()

    # Move target axis to last so lines are contiguous
    x_last = np.moveaxis(a, axis, -1)                      # (..., N)
    cols = int(np.prod(x_last.shape[:-1] or (1,)))
    X = x_last.reshape(cols, N_line)                       # (cols, N), rows contiguous
    Y = np.empty((cols, plan.outN), dtype=np.float64)      # (cols, outN)

    # Preallocate block work buffers
    B = min(_BATCH, cols) if cols > 0 else 1
    N = N_line
    out_total = plan.out_total
    outN = plan.outN
    length_total = plan.length_total
    full_len = plan.left_pad + plan.length_total + plan.right_pad
    Wmax = plan.win_len_max

    coeffB     = np.empty((B, N), dtype=np.float64)
    extB       = np.empty((B, length_total), dtype=np.float64)
    extFullB   = np.empty((B, full_len), dtype=np.float64)
    yBlock     = np.empty((B, out_total), dtype=np.float64)
    gather3D   = np.empty((B, out_total, Wmax), dtype=np.float64) if (Wmax > 0 and out_total > 0) else None

    # Correlation degree for tail
    corr_degree = p.interp_degree if p.analy_degree < 0 else (p.analy_degree + p.synthe_degree + 1)

    for i in range(0, cols, B):
        b = min(B, cols - i)

        # 1) coefficients (batched IIR)
        np.copyto(coeffB[:b, :], X[i:i+b, :])
        get_interpolation_coefficients_batch(coeffB[:b, :], p.interp_degree)

        # 2) optional integration (in-place); keep per-line averages
        if p.analy_degree >= 0:
            avgB = do_integ_batch(coeffB[:b, :], p.analy_degree + 1)
        else:
            avgB = np.zeros(b, dtype=np.float64)

        # 3) extension (right tail via precomputed mapping)
        extB[:b, :N] = coeffB[:b, :]
        rem = length_total - N
        if rem > 0 and plan.rp_src.size:
            extB[:b, N:] = plan.rp_sign * coeffB[:b, :][:, plan.rp_src]

        # ext_full = [LP | ext | RP]
        if plan.left_pad > 0:
            extFullB[:b, plan.lp_dst] = plan.lp_sign * coeffB[:b, :][:, plan.lp_src]
        extFullB[:b, plan.left_pad : plan.left_pad + length_total] = extB[:b, :]
        if plan.right_pad > 0:
            extFullB[:b, plan.left_pad + length_total :] = extB[:b, -1][:, None]

        # 4) gather + accumulate
        if (Wmax > 0) and (out_total > 0):
            np.take(extFullB[:b, :], plan.idx2d, axis=1, out=gather3D[:b, :, :])   # (b, L, W)
            # multiply+sum (often faster than einsum on many builds)
            np.multiply(gather3D[:b, :, :], plan.weights2d[None, :, :], out=gather3D[:b, :, :])
            np.sum(gather3D[:b, :, :], axis=2, out=yBlock[:b, :])
        else:
            yBlock[:b, :] = 0.0

        # 5) projection tail
        if p.analy_degree >= 0:
            do_diff_batch(yBlock[:b, :], p.analy_degree + 1)
            yBlock[:b, :] += avgB[:, None]
            get_interpolation_coefficients_batch(yBlock[:b, :], corr_degree)
            get_samples_batch(yBlock[:b, :], p.synthe_degree)

        # 6) crop to outN and store
        Y[i:i+b, :] = yBlock[:b, :outN]

    # Reshape back and restore axis
    out_last = Y.reshape(x_last.shape[:-1] + (outN,))
    return np.moveaxis(out_last, -1, axis)
