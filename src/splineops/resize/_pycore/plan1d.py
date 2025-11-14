# splineops/src/splineops/resize/_pycore/plan1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D
from .bspline import beta
from .utils import border, calculate_final_size_1d

def make_plan_1d(N: int, p: LSParams) -> Plan1D:
    # Final sizes for this axis (working length unused but kept for parity)
    workN, outN = calculate_final_size_1d(p.inversable, N, p.zoom)

    pure_interp = (p.analy_degree < 0)

    # total_degree = n + n1 + 1 (for analy=-1 → total_degree == n)
    total_degree = p.interp_degree + p.analy_degree + 1
    half_support = 0.5 * (total_degree + 1)

    # Correlation degree (projection tail)
    corr_degree = p.interp_degree if pure_interp else (p.analy_degree + p.synthe_degree + 1)

    # Native shift policy for analysis stage (Muñoz correction for analy >= 0)
    shift = float(p.shift)
    if p.analy_degree >= 0:
        t = (p.analy_degree + 1.0) / 2.0
        shift += (t - np.floor(t)) * (1.0 / p.zoom - 1.0)

    # Tail sizing:
    #  - Pure interpolation: no projection tail, only outN samples
    #  - LS / oblique: border-based tail as in C++.
    if pure_interp:
        add_border   = 0
        out_total    = outN
        length_total = N + int(np.ceil(half_support))
    else:
        add_border   = max(border(outN, corr_degree), total_degree)
        out_total    = outN + add_border
        length_total = N + int(np.ceil(add_border / p.zoom))

    # Unified TensorSpline-style geometry for ALL methods:
    #   - Input samples at k = 0 .. N-1
    #   - Visible outputs (0 .. outN-1) span [0, N-1]
    #     => step = (N-1)/(outN-1) when outN > 1
    #   - Tail samples (l >= outN) simply continue with the same step.
    if outN > 1:
        step = (N - 1) / float(outN - 1)
    else:
        step = 0.0

    l = np.arange(out_total, dtype=np.float64)
    x = step * l + shift

    # Window per output position (kmin..kmax)
    kmin = np.ceil(x - half_support).astype(np.int32)
    kmax = np.floor(x + half_support).astype(np.int32)
    wlen = (kmax - kmin + 1).astype(np.int32)
    win_len_max = int(wlen.max()) if wlen.size else 0

    # Distance grid for weights
    tgrid = np.arange(win_len_max, dtype=np.int32)[None, :]
    ks    = (kmin[:, None] + tgrid).astype(np.float64)
    dx    = x[:, None] - ks

    # Analysis scaling factor (Unser–Muñoz step 3 factor)
    fact = (p.zoom ** (p.analy_degree + 1)) if p.analy_degree >= 0 else 1.0

    # Weights for all rows (rectangular), then mask out-of-support columns
    if win_len_max > 0 and out_total > 0:
        weights2d = fact * beta(dx, total_degree)
        mask = (tgrid >= wlen[:, None])
        weights2d[mask] = 0.0
    else:
        weights2d = np.zeros((out_total, 0), dtype=np.float64)

    # --- Padding sizes (match C++: left_pad, right_pad) ---
    min_kmin = int(kmin.min()) if kmin.size else 0
    max_kmax = int(kmax.max()) if kmax.size else -1

    left_pad  = max(0, -min_kmin)
    right_pad = max(0, max_kmax - (length_total - 1))
    full_len  = left_pad + length_total + right_pad

    # CSR packing unused by Python runtime — keep empty shells for compatibility
    row_ptr = np.array([0], dtype=np.int32)
    weights = np.empty(0, dtype=np.float64)

    # Indices for gather into [LP | ext | RP]
    if win_len_max > 0 and out_total > 0:
        idx2d = (left_pad + (kmin[:, None] + tgrid)).astype(np.intp)
        np.clip(idx2d, 0, full_len - 1, out=idx2d)
    else:
        idx2d = np.empty((out_total, 0), dtype=np.intp)

    # Boundary symmetry
    symmetric_ext = ((p.analy_degree + 1) % 2 == 0) if p.analy_degree >= 0 else True

    # ----- Precompute extension indices to avoid per-line mirror math -----
    # Left pad destination slots: [0 .. left_pad-1] in ext_full (just before ext block)
    if left_pad > 0:
        lp_dst = np.arange(left_pad - 1, -1, -1, dtype=np.intp)
        t = np.arange(1, left_pad + 1)
        if symmetric_ext:
            lp_src = np.clip(t, 0, N - 1).astype(np.intp)
            lp_sign = 1.0
        else:
            lp_src = np.clip(t - 1, 0, N - 1).astype(np.intp)
            lp_sign = -1.0
    else:
        lp_dst  = np.empty(0, dtype=np.intp)
        lp_src  = np.empty(0, dtype=np.intp)
        lp_sign = 1.0 if symmetric_ext else -1.0

    # Right extension for ext[N: length_total] = rp_sign * coeff[rp_src]
    rem = length_total - N
    if rem > 0:
        l_idx = np.arange(N, length_total)
        if symmetric_ext:
            period = 2 * N - 2
            if period > 0:
                lk = l_idx % period
                lk = np.where(lk >= N, period - lk, lk)
            else:
                lk = l_idx
            rp_sign = 1.0
        else:
            period = 2 * N - 3
            if period > 0:
                lk = l_idx % period
                lk = np.where(lk >= N, period - lk, lk)
            else:
                lk = l_idx
            rp_sign = -1.0
        lk = np.clip(lk, 0, N - 1)
        rp_src = lk.astype(np.intp)
    else:
        rp_src  = np.empty(0, dtype=np.intp)
        rp_sign = 1.0 if symmetric_ext else -1.0

    return Plan1D(
        N=N, outN=outN, out_total=out_total, length_total=length_total,
        symmetric_ext=symmetric_ext,
        left_pad=left_pad, right_pad=right_pad,
        kmin=kmin, win_len=wlen, row_ptr=row_ptr,
        weights=weights, win_len_max=win_len_max,
        idx2d=idx2d, weights2d=weights2d,
        lp_dst=lp_dst, lp_src=lp_src, lp_sign=lp_sign,
        rp_src=rp_src, rp_sign=rp_sign
    )
