# splineops/src/splineops/resize/_pycore/plan1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D
from .bspline import beta
from .utils import border, calculate_final_size_1d

def make_plan_1d(N: int, p: LSParams) -> Plan1D:
    workN, outN = calculate_final_size_1d(p.inversable, N, p.zoom)
    total_degree = p.interp_degree + p.analy_degree + 1 if p.analy_degree >= 0 else p.interp_degree
    corr_degree  = (p.interp_degree if p.analy_degree < 0 else p.analy_degree + p.synthe_degree + 1)
    half_support = 0.5 * ( (p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0) + 1) + 1 )
    # match native shift policy for analysis stage
    shift = p.shift
    if p.analy_degree >= 0:
        t = (p.analy_degree + 1.0)/2.0
        shift += (t - np.floor(t)) * (1.0/p.zoom - 1.0)

    add_border   = max(border(outN, corr_degree), (p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0) + 1))
    out_total    = outN + add_border
    length_total = N + int(np.ceil(add_border / p.zoom))

    l = np.arange(out_total, dtype=np.float64)
    x = l / p.zoom + shift
    kmin = np.ceil(x - 0.5*( (p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0)) + 2 )).astype(np.int32)
    kmax = np.floor(x + 0.5*( (p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0)) + 2 )).astype(np.int32)
    wlen = (kmax - kmin + 1).astype(np.int32)
    win_len_max = int(wlen.max()) if wlen.size else 0

    tgrid = np.arange(win_len_max, dtype=np.int32)[None, :]
    ks    = (kmin[:, None] + tgrid).astype(np.float64)
    dx    = x[:, None] - ks

    total_deg = p.interp_degree + (p.analy_degree if p.analy_degree>=0 else 0) + 1
    fact = (p.zoom ** (p.analy_degree + 1)) if p.analy_degree >= 0 else 1.0
    weights2d = fact * beta(dx, total_deg)
    # zero beyond each row's true length
    if win_len_max > 0:
        mask = (tgrid >= wlen[:, None])
        weights2d = weights2d.copy()
        weights2d[mask] = 0.0

    # padding sizes
    # left pad from the most-negative kmin (unchanged),
    # right pad must cover the widest rectangular idx grid.
    min_kmin = int(kmin.min()) if kmin.size else 0
    max_kmin = int(kmin.max()) if kmin.size else 0
    LP = max(0, -min_kmin)

    if win_len_max > 0:
        max_idx_needed = max_kmin + (win_len_max - 1)
    else:
        # No window → fall back to kmax (or -1 when out_total==0)
        max_idx_needed = int(kmax.max()) if kmax.size else -1

    RP = max(0, max_idx_needed - (length_total - 1))
    full_len = LP + length_total + RP

    # CSR style 1-D contiguous weights representation (plus 2-D view kept)
    row_ptr = np.empty(out_total+1, dtype=np.int32)
    row_ptr[0] = 0
    np.cumsum(wlen, out=row_ptr[1:])
    weights = np.zeros(int(row_ptr[-1]), dtype=np.float64)
    if out_total > 0 and win_len_max > 0:
        # pack row-by-row
        idx_write = 0
        for lidx in range(out_total):
            M = int(wlen[lidx])
            if M <= 0: continue
            weights[idx_write:idx_write+M] = weights2d[lidx, :M]
            idx_write += M

    idx2d = (LP + (kmin[:, None] + tgrid)).astype(np.int64) if win_len_max > 0 else np.empty((out_total,0), dtype=np.int64)

    # Safety against any round-off oddities at extreme edges:
    if win_len_max > 0 and idx2d.size:
        np.clip(idx2d, 0, full_len - 1, out=idx2d)

    symmetric_ext = ((p.analy_degree + 1) % 2 == 0) if p.analy_degree >= 0 else True

    return Plan1D(
        N=N, outN=outN, out_total=out_total, length_total=length_total,
        symmetric_ext=symmetric_ext,
        left_pad=LP, right_pad=RP,
        kmin=kmin, win_len=wlen, row_ptr=row_ptr,
        weights=weights, win_len_max=win_len_max,
        idx2d=idx2d, weights2d=weights2d
    )
