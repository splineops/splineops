# splineops/src/splineops/resize/_pycore/resize1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D, Work1D
from .filters import get_interpolation_coefficients, get_samples
from .diff_integ import do_integ, do_diff

def _build_extension(coeff: np.ndarray, plan: Plan1D) -> tuple[np.ndarray, np.ndarray]:
    N = coeff.size
    ext = np.empty(plan.length_total, dtype=np.float64)
    ext[:N] = coeff
    if plan.length_total > N:
        l = np.arange(N, plan.length_total)
        if plan.symmetric_ext:
            period = 2*N - 2
            lk = np.where(period > 0, np.where(l >= period, l % period, l), l)
            lk = np.where(lk >= N, period - lk, lk)
            lk = np.clip(lk, 0, N-1)
            ext[N:] = coeff[lk]
        else:
            period = 2*N - 3
            lk = np.where(period > 0, np.where(l >= period, l % period, l), l)
            lk = np.where(lk >= N, period - lk, lk)
            lk = np.clip(lk, 0, N-1)
            ext[N:] = -coeff[lk]
    # ext_full = [LP | ext | RP]
    ext_full = np.empty(plan.left_pad + plan.length_total + plan.right_pad, dtype=np.float64)
    if plan.left_pad > 0:
        t = np.arange(1, plan.left_pad+1)
        if plan.symmetric_ext:
            src = np.clip(t, 0, N-1)
            ext_full[plan.left_pad - t] = coeff[src]
        else:
            src = np.clip(t-1, 0, N-1)
            ext_full[plan.left_pad - t] = -coeff[src]
    ext_full[plan.left_pad:plan.left_pad + plan.length_total] = ext
    if plan.right_pad > 0:
        ext_full[plan.left_pad + plan.length_total:] = ext[-1]
    return ext, ext_full

def resize_1d_ws(in_line: np.ndarray, p: LSParams, plan: Plan1D, ws: Work1D) -> np.ndarray:
    # 1) interpolation coefficients
    coeff = in_line.astype(np.float64, copy=True)
    get_interpolation_coefficients(coeff, p.interp_degree)

    # 2) optional integration
    average = 0.0
    if p.analy_degree >= 0:
        average = do_integ(coeff, p.analy_degree + 1)

    # 3) extension buffers
    ext, ext_full = _build_extension(coeff, plan)

    # 4) accumulate with gather + dot
    if plan.win_len_max > 0 and plan.out_total > 0:
        gather = ext_full[plan.idx2d]  # (out_total, win_len_max)
        y = (plan.weights2d * gather).sum(axis=1)
    else:
        y = np.zeros(plan.out_total, dtype=np.float64)

    # 5) projection tail
    if p.analy_degree >= 0:
        do_diff(y, p.analy_degree + 1)
        y += average
        # correlation-degree filtering + sampling
        corr_degree = p.interp_degree if p.analy_degree < 0 else (p.analy_degree + p.synthe_degree + 1)
        get_interpolation_coefficients(y, corr_degree)
        get_samples(y, p.synthe_degree)

    # 6) crop
    return y[:plan.outN].copy()
