# splineops/src/splineops/resize/_pycore/plan1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D
from .bspline import beta
from .utils import border, calculate_final_size_1d

def make_plan_1d(N: int, p: LSParams) -> Plan1D:
    workN, outN = calculate_final_size_1d(p.inversable, N, p.zoom)

    corr_degree  = (p.interp_degree if p.analy_degree < 0 else p.analy_degree + p.synthe_degree + 1)

    shift = p.shift
    if p.analy_degree >= 0:
        t = (p.analy_degree + 1.0)/2.0
        shift += (t - np.floor(t)) * (1.0/p.zoom - 1.0)

    add_border   = max(border(outN, corr_degree), (p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0) + 1))
    out_total    = outN + add_border
    length_total = N + int(np.ceil(add_border / p.zoom))

    l = np.arange(out_total, dtype=np.float64)
    x = l / p.zoom + shift
    base = (p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0))
    kmin = np.ceil(x - 0.5*(base + 2)).astype(np.int32)
    kmax = np.floor(x + 0.5*(base + 2)).astype(np.int32)
    wlen = (kmax - kmin + 1).astype(np.int32)
    win_len_max = int(wlen.max()) if wlen.size else 0

    tgrid = np.arange(win_len_max, dtype=np.int32)[None, :]
    ks    = (kmin[:, None] + tgrid).astype(np.float64)
    dx    = x[:, None] - ks

    total_deg = p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0) + 1
    fact = (p.zoom ** (p.analy_degree + 1)) if p.analy_degree >= 0 else 1.0
    weights2d = fact * beta(dx, total_deg)

    if win_len_max > 0:
        mask = (tgrid >= wlen[:, None])
        weights2d = weights2d.copy()
        weights2d[mask] = 0.0

    # padding (fix A)
    min_kmin = int(kmin.min()) if kmin.size else 0
    max_kmin = int(kmin.max()) if kmin.size else 0
    LP = max(0, -min_kmin)
    if win_len_max > 0:
        max_idx_needed = max_kmin + (win_len_max - 1)
    else:
        max_idx_needed = int(kmax.max()) if kmax.size else -1
    RP = max(0, max_idx_needed - (length_total - 1))
    full_len = LP + length_total + RP

    row_ptr = np.array([0], dtype=np.int32)
    weights = np.empty(0, dtype=np.float64)

    idx2d = (LP + (kmin[:, None] + tgrid)).astype(np.int64) if win_len_max > 0 else np.empty((out_total, 0), dtype=np.int64)
    if win_len_max > 0 and idx2d.size:
        np.clip(idx2d, 0, full_len - 1, out=idx2d)

    symmetric_ext = ((p.analy_degree + 1) % 2 == 0) if p.analy_degree >= 0 else True

    # ----- precompute extension indices -----
    # left pad destination positions [0..LP-1] (these are exactly the slots before ext)
    lp_dst = np.arange(LP-1, -1, -1, dtype=np.intp) if LP > 0 else np.empty(0, dtype=np.intp)

    # left pad source (indices into coeff)
    if LP > 0:
        t = np.arange(1, LP+1)
        if symmetric_ext:
            lp_src = np.clip(t,   0, N-1).astype(np.intp)
            lp_sign =  1.0
        else:
            lp_src = np.clip(t-1, 0, N-1).astype(np.intp)
            lp_sign = -1.0
    else:
        lp_src  = np.empty(0, dtype=np.intp)
        lp_sign = 1.0 if symmetric_ext else -1.0

    # right extension (ext[N:] = sign * coeff[rp_src])
    rem = length_total - N
    if rem > 0:
        l = np.arange(N, length_total)
        if symmetric_ext:
            period = 2*N - 2
            if period > 0:
                lk = l % period
                lk = np.where(lk >= N, period - lk, lk)
            else:
                lk = l
            rp_sign =  1.0
        else:
            period = 2*N - 3
            if period > 0:
                lk = l % period
                lk = np.where(lk >= N, period - lk, lk)
            else:
                lk = l
            rp_sign = -1.0
        lk = np.clip(lk, 0, N-1)
        rp_src = lk.astype(np.intp)
    else:
        rp_src  = np.empty(0, dtype=np.intp)
        rp_sign = 1.0 if symmetric_ext else -1.0
    # --------------------------------------------

    return Plan1D(
        N=N, outN=outN, out_total=out_total, length_total=length_total,
        symmetric_ext=symmetric_ext,
        left_pad=LP, right_pad=RP,
        kmin=kmin, win_len=wlen, row_ptr=row_ptr,
        weights=weights, win_len_max=win_len_max,
        idx2d=idx2d, weights2d=weights2d,
        lp_dst=lp_dst, lp_src=lp_src, lp_sign=lp_sign,
        rp_src=rp_src, rp_sign=rp_sign
    )
