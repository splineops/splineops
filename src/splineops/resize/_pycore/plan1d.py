# splineops/src/splineops/resize/_pycore/plan1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D
from .bspline import beta
from .utils import border, calculate_final_size_1d

def make_plan_1d(N: int, p: LSParams) -> Plan1D:
    # Final sizes for this axis (working length unused but kept for parity)
    workN, outN = calculate_final_size_1d(p.inversable, N, p.zoom)

    # Correlation degree (projection tail)
    corr_degree = p.interp_degree if p.analy_degree < 0 else (p.analy_degree + p.synthe_degree + 1)

    # Native shift policy for analysis stage
    shift = p.shift
    if p.analy_degree >= 0:
        t = (p.analy_degree + 1.0) / 2.0
        shift += (t - np.floor(t)) * (1.0 / p.zoom - 1.0)

    # Total_degree equals n + n1 + 1 (so when analy=-1 → total_degree == n) ---
    total_degree = p.interp_degree + p.analy_degree + 1
    half_support = 0.5 * (total_degree + 1)

    # Tail sizing uses total_degree (matches C++)
    add_border = max(border(outN, corr_degree), total_degree)
    out_total = outN + add_border
    length_total = N + int(np.ceil(add_border / p.zoom))

    # Output sample positions
    l = np.arange(out_total, dtype=np.float64)
    x = l / p.zoom + shift

    # Window per output position (kmin..kmax)
    kmin = np.ceil(x - half_support).astype(np.int32)
    kmax = np.floor(x + half_support).astype(np.int32)
    wlen = (kmax - kmin + 1).astype(np.int32)
    win_len_max = int(wlen.max()) if wlen.size else 0

    # Distance grid for weights
    tgrid = np.arange(win_len_max, dtype=np.int32)[None, :]
    ks = (kmin[:, None] + tgrid).astype(np.float64)
    dx = x[:, None] - ks

    # Analysis scaling factor
    fact = (p.zoom ** (p.analy_degree + 1)) if p.analy_degree >= 0 else 1.0

    # Weights for all rows (rectangular), then zero masked columns
    weights2d = fact * beta(dx, total_degree)
    if win_len_max > 0:
        mask = (tgrid >= wlen[:, None])
        weights2d = weights2d.copy()
        weights2d[mask] = 0.0

    # --- Padding sizes: right pad must cover the widest rectangular index grid ---
    min_kmin = int(kmin.min()) if kmin.size else 0
    max_kmin = int(kmin.max()) if kmin.size else 0
    LP = max(0, -min_kmin)
    if win_len_max > 0:
        max_idx_needed = max_kmin + (win_len_max - 1)
    else:
        max_idx_needed = int(kmax.max()) if kmax.size else -1
    RP = max(0, max_idx_needed - (length_total - 1))
    full_len = LP + length_total + RP

    # Drop CSR packing (unused by Python runtime) — keep empty shells for compatibility
    row_ptr = np.array([0], dtype=np.int32)
    weights = np.empty(0, dtype=np.float64)

    # Indices for gather into [LP | ext | RP]
    idx2d = (
        (LP + (kmin[:, None] + tgrid)).astype(np.intp)
        if win_len_max > 0 else np.empty((out_total, 0), dtype=np.intp)
    )
    if win_len_max > 0 and idx2d.size:
        np.clip(idx2d, 0, full_len - 1, out=idx2d)

    # Boundary symmetry
    symmetric_ext = ((p.analy_degree + 1) % 2 == 0) if p.analy_degree >= 0 else True

    # ----- Precompute extension indices to avoid per-line mirror math -----
    # Left pad destination slots: indices [0 .. LP-1] in ext_full (just before ext block)
    lp_dst = np.arange(LP - 1, -1, -1, dtype=np.intp) if LP > 0 else np.empty(0, dtype=np.intp)

    # Left pad source indices into coeff and its sign
    if LP > 0:
        t = np.arange(1, LP + 1)
        if symmetric_ext:
            lp_src = np.clip(t, 0, N - 1).astype(np.intp)
            lp_sign = 1.0
        else:
            lp_src = np.clip(t - 1, 0, N - 1).astype(np.intp)
            lp_sign = -1.0
    else:
        lp_src = np.empty(0, dtype=np.intp)
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
        rp_src = np.empty(0, dtype=np.intp)
        rp_sign = 1.0 if symmetric_ext else -1.0
    # ---------------------------------------------------------------------

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
