# splineops/src/splineops/resize/_pycore/resize_1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D, Work1D
from .filters import get_interpolation_coefficients, get_samples
from .diff_integ import do_integ, do_diff


def _ensure_ws(ws: Work1D, plan: Plan1D, N: int) -> None:
    if ws.coeff.size != N:
        ws.coeff = np.empty(N, dtype=np.float64)
    extension_len = 0 if plan.direct_projection else plan.length_total
    if ws.ext.size != extension_len:
        ws.ext = np.empty(extension_len, dtype=np.float64)
    ext_full_len = (
        0
        if plan.direct_projection
        else plan.left_pad + plan.length_total + plan.right_pad
    )
    if ws.ext_full.size != ext_full_len:
        ws.ext_full = np.empty(ext_full_len, dtype=np.float64)
    if ws.y.size != plan.out_total:
        ws.y = np.empty(plan.out_total, dtype=np.float64)
    if ws.gather2d.shape != (plan.out_total, plan.win_len_max):
        ws.gather2d = np.empty((plan.out_total, plan.win_len_max), dtype=np.float64)


def _build_extension_inplace(coeff: np.ndarray, plan: Plan1D, ws: Work1D) -> None:
    N = coeff.size
    ext = ws.ext
    ext[:N] = coeff
    if plan.rp_src.size:
        ext[N:] = plan.rp_sign * coeff[plan.rp_src]

    ext_full = ws.ext_full
    if plan.left_pad > 0:
        ext_full[plan.lp_dst] = plan.lp_sign * coeff[plan.lp_src]
    ext_full[plan.left_pad : plan.left_pad + plan.length_total] = ext
    if plan.right_pad > 0:
        ext_full[plan.left_pad + plan.length_total :] = ext[-1]


def resize_1d_ws(
    in_line: np.ndarray,
    p: LSParams,
    plan: Plan1D,
    ws: Work1D,
    out: np.ndarray | None = None,
) -> np.ndarray:
    # Explicit definitions for endpoint-grid degeneracies.  These mirror the
    # batched N-D path and keep projection away from an undefined zero scale.
    if in_line.size == 1:
        result = np.full(plan.outN, in_line[0], dtype=np.float64)
        if out is not None:
            np.copyto(out, result)
            return out
        return result
    if plan.outN == 1 and p.analy_degree >= 0:
        result = np.asarray([np.mean(in_line, dtype=np.float64)])
        if out is not None:
            np.copyto(out, result)
            return out
        return result
    if plan.outN == 1 and p.interp_degree == 0:
        left = (in_line.size - 1) // 2
        right = in_line.size // 2
        result = np.asarray(
            [0.5 * (float(in_line[left]) + float(in_line[right]))],
            dtype=np.float64,
        )
        if out is not None:
            np.copyto(out, result)
            return out
        return result
    if (
        plan.outN == in_line.size
        and abs(p.shift) <= 1e-15
        and p.synthe_degree == p.interp_degree
    ):
        result = np.asarray(in_line, dtype=np.float64).copy()
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    _ensure_ws(ws, plan, in_line.size)

    # 1) coefficients
    ws.coeff[...] = in_line  # contiguous row from resizend
    get_interpolation_coefficients(ws.coeff, p.interp_degree)

    # 2) optional integration
    average = 0.0
    if p.analy_degree >= 0 and not plan.direct_projection:
        average = do_integ(ws.coeff, p.analy_degree + 1)

    # 3) extension. Direct cross-Gram indices already address the original
    # coefficient line through the full mirror period, so no copy is needed.
    if not plan.direct_projection:
        _build_extension_inplace(ws.coeff, plan, ws)

    # 4) accumulate
    if plan.win_len_max > 0 and plan.out_total > 0:
        source = ws.coeff if plan.direct_projection else ws.ext_full
        np.take(source, plan.idx2d, out=ws.gather2d)
        np.multiply(plan.weights2d, ws.gather2d, out=ws.gather2d)
        np.sum(ws.gather2d, axis=1, out=ws.y)
    else:
        ws.y[:] = 0.0

    # 5) projection tail
    if p.analy_degree >= 0:
        if not plan.direct_projection:
            do_diff(ws.y, p.analy_degree + 1)
            ws.y += average
        corr_degree = (
            p.interp_degree
            if p.analy_degree < 0
            else (p.analy_degree + p.synthe_degree + 1)
        )
        get_interpolation_coefficients(ws.y, corr_degree)
        get_samples(ws.y, p.synthe_degree)

    # 6) crop (optionally write into provided buffer)
    view = ws.y[: plan.outN]
    if out is not None:
        np.copyto(out, view)
        return out
    return view.copy()
