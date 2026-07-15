# splineops/src/splineops/resize/_pycore/plan_1d.py
from __future__ import annotations
import numpy as np
from .params import LSParams, Plan1D
from .bspline import beta
from .utils import border, calculate_output_size_1d

_GAUSS_RULES = {order: np.polynomial.legendre.leggauss(order) for order in range(1, 5)}

_NATIVE_INT_MIN = -(1 << 31)
_NATIVE_INT_MAX = (1 << 31) - 1


def _checked_index_array(
    values: np.ndarray,
    rounding,
    message: str,
) -> np.ndarray:
    rounded = rounding(np.asarray(values, dtype=np.float64))
    if (
        not np.all(np.isfinite(rounded))
        or np.any(rounded < _NATIVE_INT_MIN)
        or np.any(rounded > _NATIVE_INT_MAX)
    ):
        raise OverflowError(message)
    return rounded.astype(np.int64)


def _checked_axis_length(value: int, message: str) -> int:
    value = int(value)
    if value < 0 or value > _NATIVE_INT_MAX:
        raise OverflowError(message)
    return value


def _whole_sample_symmetric_indices(indices: np.ndarray, size: int) -> np.ndarray:
    """Map arbitrary integer indices through the full WSS mirror period."""

    if size <= 1:
        return np.zeros(np.shape(indices), dtype=np.intp)
    period = 2 * int(size) - 2
    folded = np.mod(np.asarray(indices, dtype=np.int64), period)
    folded = np.where(folded >= size, period - folded, folded)
    return folded.astype(np.intp, copy=False)


def _cross_gram_weights_gauss(
    distance: np.ndarray,
    scale: float,
    interp_degree: int,
    analysis_degree: int,
    valid: np.ndarray,
) -> np.ndarray:
    """Stable ``beta_n(./scale) * beta_m`` values for projection rows.

    Splitting at both splines' knots and using enough Gauss points for their
    combined polynomial degree integrates every piece exactly in exact
    arithmetic. Work is chunked so a long axis does not create a large
    temporary knot tensor.
    """

    a = float(scale)
    n = int(interp_degree)
    m = int(analysis_degree)
    input_radius = 0.5 * (n + 1)
    analysis_radius = 0.5 * (m + 1)
    result = np.zeros(distance.shape, dtype=np.float64)
    flat_result = result.ravel()
    flat_distance = np.asarray(distance, dtype=np.float64).ravel()
    flat_valid = np.asarray(valid, dtype=bool).ravel()
    positions = np.flatnonzero(flat_valid)
    if positions.size == 0:
        return result

    input_knots = a * np.arange(-input_radius, input_radius + 0.5, 1.0)
    analysis_knots = np.arange(-analysis_radius, analysis_radius + 0.5, 1.0)
    nodes, gauss_weights = _GAUSS_RULES[(n + m + 2) // 2]
    chunk_size = 4096
    for begin in range(0, positions.size, chunk_size):
        selected = positions[begin : begin + chunk_size]
        x = flat_distance[selected]
        lo = np.maximum(-a * input_radius, x - analysis_radius)
        hi = np.minimum(+a * input_radius, x + analysis_radius)
        overlaps = lo < hi
        if not np.any(overlaps):
            continue

        # Clipping every breakpoint to [lo, hi] is equivalent to filtering
        # it; duplicates simply create zero-width intervals after sorting.
        knots = np.concatenate(
            (
                np.broadcast_to(input_knots, (x.size, input_knots.size)),
                x[:, None] - analysis_knots[None, :],
                lo[:, None],
                hi[:, None],
            ),
            axis=1,
        )
        np.maximum(knots, lo[:, None], out=knots)
        np.minimum(knots, hi[:, None], out=knots)
        knots.sort(axis=1)
        left = knots[:, :-1]
        right = knots[:, 1:]
        half = 0.5 * (right - left)
        midpoint = 0.5 * (right + left)

        subtotal = np.zeros(x.size, dtype=np.float64)
        for node, weight in zip(nodes, gauss_weights):
            t = midpoint + half * node
            subtotal += np.sum(
                half * weight * beta(t / a, n) * beta(x[:, None] - t, m),
                axis=1,
            )
        subtotal[~overlaps] = 0.0
        flat_result[selected] = subtotal
    return result


def make_plan_1d(N: int, p: LSParams) -> Plan1D:
    outN = calculate_output_size_1d(N, p.zoom)

    # The requested zoom is a size request only.  Once the integer output
    # length is known, all resampling geometry is derived from the unique
    # endpoint-aligned scale.  Degenerate axes are executed by explicit fast
    # paths in resize_nd/resize_1d; scale=1 keeps their otherwise-unused plan
    # metadata finite.
    if N > 1 and outN > 1:
        effective_zoom = (outN - 1) / float(N - 1)
        step = 1.0 / effective_zoom
    else:
        effective_zoom = 1.0
        step = 0.0

    pure_interp = p.analy_degree < 0
    visible_projection = (
        not pure_interp and float(p.shift) == 0.0 and N > 1 and outN > 1
    )
    direct_projection = visible_projection and p.analy_degree >= 1

    # total_degree = n + n1 + 1 (for analy=-1 → total_degree == n)
    total_degree = p.interp_degree + p.analy_degree + 1
    half_support = 0.5 * (total_degree + 1)

    # Correlation degree (projection tail)
    corr_degree = (
        p.interp_degree if pure_interp else (p.analy_degree + p.synthe_degree + 1)
    )

    # Native shift policy for analysis stage (Muñoz correction for analy >= 0)
    shift = float(p.shift)
    if p.analy_degree >= 0:
        t = (p.analy_degree + 1.0) / 2.0
        shift += (t - np.floor(t)) * (1.0 / effective_zoom - 1.0)

    # An endpoint-aligned, zero-shift projected signal is mirror-symmetric at
    # output index outN-1.  The boundary-aware finite differences therefore
    # operate directly on the visible sequence, and the output Gram inverse
    # must see that same mirror endpoint.  A tail is retained only for the
    # internal non-zero-shift path, where that symmetry proof does not apply.
    if pure_interp:
        add_border = 0
        out_total = outN
        length_total = _checked_axis_length(
            N + int(np.ceil(max(0.0, shift + half_support))),
            "interpolation extension exceeds native limits",
        )
    elif visible_projection:
        add_border = 0
        out_total = outN
        if direct_projection:
            # Every unwrapped direct-projection coefficient index is mapped through
            # the full whole-sample-symmetric period below.  No materialized
            # extension is needed, even for M=2 strong reductions.
            length_total = N
        else:
            length_total = _checked_axis_length(
                N + int(np.ceil(max(0.0, shift + half_support))),
                "projection extension exceeds native limits",
            )
    else:
        add_border = max(border(outN, corr_degree), total_degree)
        out_total = _checked_axis_length(
            outN + add_border,
            "projection output length exceeds native limits",
        )
        length_total = _checked_axis_length(
            N + int(np.ceil(add_border / effective_zoom)),
            "projection extension exceeds native limits",
        )

    # Unified TensorSpline-style geometry for ALL methods:
    #   - Input samples at k = 0 .. N-1
    #   - Visible outputs (0 .. outN-1) span [0, N-1].
    #   - A single visible output is located at the symmetric centre.  The
    #     projection path handles that case as a line mean before using this
    #     plan; pure interpolation evaluates its spline at this coordinate.
    #   - Tail samples (l >= outN) simply continue with the same step.
    l = np.arange(out_total, dtype=np.float64)
    if direct_projection:
        # Direct compact cross-Gram rows.  In output coordinates the input
        # spline centred at k is centred at effective_zoom*k.
        x = l
        cross_radius = 0.5 * (
            (p.interp_degree + 1) * effective_zoom + p.analy_degree + 1.0
        )
        kmin = _checked_index_array(
            (x - cross_radius) / effective_zoom,
            np.ceil,
            "projection source index exceeds native limits",
        )
        kmax = _checked_index_array(
            (x + cross_radius) / effective_zoom,
            np.floor,
            "projection source index exceeds native limits",
        )
    else:
        if outN == 1 and N > 1:
            x = np.full(out_total, 0.5 * (N - 1) + shift, dtype=np.float64)
        else:
            x = step * l + shift
        kmin = _checked_index_array(
            x - half_support,
            np.ceil,
            "resampling source index exceeds native limits",
        )
        kmax = _checked_index_array(
            x + half_support,
            np.floor,
            "resampling source index exceeds native limits",
        )
    wlen = kmax - kmin + 1
    if np.any(wlen <= 0) or np.any(wlen > _NATIVE_INT_MAX):
        raise OverflowError("resampling support width exceeds native limits")
    win_len_max = int(wlen.max()) if wlen.size else 0
    if int(out_total) * win_len_max > _NATIVE_INT_MAX:
        raise OverflowError("resampling plan has too many weight slots")

    # Distance grid for weights
    tgrid = np.arange(win_len_max, dtype=np.int64)[None, :]
    kgrid = kmin[:, None] + tgrid
    ks = kgrid.astype(np.float64)
    dx = x[:, None] - effective_zoom * ks if direct_projection else x[:, None] - ks

    # Analysis scaling factor (Unser–Muñoz step 3 factor)
    fact = effective_zoom ** (p.analy_degree + 1) if p.analy_degree >= 0 else 1.0

    # Weights for all rows (rectangular), then mask out-of-support columns
    if win_len_max > 0 and out_total > 0:
        mask = tgrid < wlen[:, None]
        if direct_projection:
            weights2d = _cross_gram_weights_gauss(
                dx,
                effective_zoom,
                p.interp_degree,
                p.analy_degree,
                mask,
            )
            # Partition of unity gives an exact DC row sum of one.  Enforce it
            # after floating-point quadrature to avoid accumulating plan error.
            row_sum = np.sum(weights2d, axis=1)
            if np.any(row_sum <= 0.0):
                raise RuntimeError("direct projection cross-Gram row has zero weight")
            weights2d /= row_sum[:, None]
        else:
            weights2d = fact * beta(dx, total_degree)
            weights2d[~mask] = 0.0
    else:
        weights2d = np.zeros((out_total, 0), dtype=np.float64)

    # --- Padding sizes (match C++: left_pad, right_pad) ---
    min_kmin = int(kmin.min()) if kmin.size else 0
    max_kmax = int(kmax.max()) if kmax.size else -1

    if direct_projection:
        left_pad = 0
        right_pad = 0
    else:
        left_pad = max(0, -min_kmin)
        right_pad = max(0, max_kmax - (length_total - 1))
    full_len = _checked_axis_length(
        left_pad + length_total + right_pad,
        "resampling extension exceeds native limits",
    )

    # CSR packing unused by Python runtime — keep empty shells for compatibility
    row_ptr = np.array([0], dtype=np.int32)
    weights = np.empty(0, dtype=np.float64)

    # Indices for gather into [LP | ext | RP]
    if win_len_max > 0 and out_total > 0:
        if direct_projection:
            idx2d = _whole_sample_symmetric_indices(kgrid, N)
        else:
            idx2d = (left_pad + kgrid).astype(np.intp)
            np.clip(idx2d, 0, full_len - 1, out=idx2d)
    else:
        idx2d = np.empty((out_total, 0), dtype=np.intp)

    # Boundary symmetry
    symmetric_ext = (
        True
        if direct_projection
        else (((p.analy_degree + 1) % 2 == 0) if p.analy_degree >= 0 else True)
    )

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
        lp_dst = np.empty(0, dtype=np.intp)
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

    return Plan1D(
        N=N,
        outN=outN,
        out_total=out_total,
        length_total=length_total,
        symmetric_ext=symmetric_ext,
        left_pad=left_pad,
        right_pad=right_pad,
        kmin=kmin,
        win_len=wlen,
        row_ptr=row_ptr,
        weights=weights,
        win_len_max=win_len_max,
        idx2d=idx2d,
        weights2d=weights2d,
        lp_dst=lp_dst,
        lp_src=lp_src,
        lp_sign=lp_sign,
        rp_src=rp_src,
        rp_sign=rp_sign,
        direct_projection=direct_projection,
    )
