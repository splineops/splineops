# splineops/src/splineops/resize/_pycore/resize1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D, Work1D
from .filters import get_interpolation_coefficients, get_samples
from .diff_integ import do_integ, do_diff

def _ensure_ws(ws: Work1D, plan: Plan1D, N: int) -> None:
    """Make sure all working buffers have the right shapes for this plan."""
    if ws.coeff.size != N:
        ws.coeff = np.empty(N, dtype=np.float64)
    if ws.ext.size != plan.length_total:
        ws.ext = np.empty(plan.length_total, dtype=np.float64)
    ext_full_len = plan.left_pad + plan.length_total + plan.right_pad
    if ws.ext_full.size != ext_full_len:
        ws.ext_full = np.empty(ext_full_len, dtype=np.float64)
    if ws.y.size != plan.out_total:
        ws.y = np.empty(plan.out_total, dtype=np.float64)
    if ws.gather2d.shape != (plan.out_total, plan.win_len_max):
        ws.gather2d = np.empty((plan.out_total, plan.win_len_max), dtype=np.float64)

def _build_extension_inplace(coeff: np.ndarray, plan: Plan1D, ws: Work1D) -> None:
    """
    Build extension buffers in-place:

      - ws.ext      : mirrored/anti-mirrored extension to 'length_total'
      - ws.ext_full : [left_pad | ext | right_pad]
    """
    N = coeff.size
    ext = ws.ext

    # Base copy
    ext[:N] = coeff

    # Rightwards extension to length_total
    if plan.length_total > N:
        l = np.arange(N, plan.length_total)
        if plan.symmetric_ext:
            period = 2 * N - 2
            if period > 0:
                lk = l % period
                lk = np.where(lk >= N, period - lk, lk)
            else:
                lk = l
            lk = np.clip(lk, 0, N - 1)
            ext[N:] = coeff[lk]
        else:
            period = 2 * N - 3
            if period > 0:
                lk = l % period
                lk = np.where(lk >= N, period - lk, lk)
            else:
                lk = l
            lk = np.clip(lk, 0, N - 1)
            ext[N:] = -coeff[lk]

    # Compose ext_full = [LP | ext | RP]
    ext_full = ws.ext_full
    if plan.left_pad > 0:
        t = np.arange(1, plan.left_pad + 1)
        if plan.symmetric_ext:
            src = np.clip(t, 0, N - 1)
            ext_full[plan.left_pad - t] = coeff[src]
        else:
            src = np.clip(t - 1, 0, N - 1)
            ext_full[plan.left_pad - t] = -coeff[src]

    ext_full[plan.left_pad : plan.left_pad + plan.length_total] = ext

    if plan.right_pad > 0:
        ext_full[plan.left_pad + plan.length_total :] = ext[-1]

def resize_1d_ws(in_line: np.ndarray, p: LSParams, plan: Plan1D, ws: Work1D) -> np.ndarray:
    # 0) Ensure scratch buffers exist (once per plan/shape)
    _ensure_ws(ws, plan, in_line.size)

    # 1) Interpolation coefficients (no extra allocs)
    ws.coeff[...] = in_line  # assignment casts into float64
    get_interpolation_coefficients(ws.coeff, p.interp_degree)

    # 2) Optional integration (in-place)
    average = 0.0
    if p.analy_degree >= 0:
        average = do_integ(ws.coeff, p.analy_degree + 1)

    # 3) Extension buffers (in-place into ws.ext / ws.ext_full)
    _build_extension_inplace(ws.coeff, plan, ws)

    # 4) Accumulate with reuse: gather + row-wise dot without temporaries
    if plan.win_len_max > 0 and plan.out_total > 0:
        np.take(ws.ext_full, plan.idx2d, out=ws.gather2d)

        # Option A (often faster): in-place multiply then sum with out=
        np.multiply(plan.weights2d, ws.gather2d, out=ws.gather2d)
        np.sum(ws.gather2d, axis=1, out=ws.y)

        # Option B (alternative): einsum into out buffer
        # np.einsum("ij,ij->i", plan.weights2d, ws.gather2d, out=ws.y, optimize=True)
    else:
        ws.y[:] = 0.0

    # 5) Projection tail
    if p.analy_degree >= 0:
        do_diff(ws.y, p.analy_degree + 1)
        ws.y += average
        corr_degree = p.interp_degree if p.analy_degree < 0 else (p.analy_degree + p.synthe_degree + 1)
        get_interpolation_coefficients(ws.y, corr_degree)
        get_samples(ws.y, p.synthe_degree)

    # 6) Crop
    return ws.y[:plan.outN].copy()
