# splineops/src/splineops/resize/_pycore/resize_nd.py
from __future__ import annotations
import os
import sys
import threading
from collections import OrderedDict
from time import perf_counter
import numpy as np
from .params import LSParams, Plan1D
from .plan_1d import make_plan_1d
from .utils import calculate_output_size_1d
from .filters import (
    get_interpolation_coefficients_batch,
    get_samples_batch,
)
from .diff_integ import do_integ_batch, do_diff_batch

# ------------------------------ knobs / toggles ------------------------------

# Batch size (lines processed together)
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _bool_env(name: str, default_true: bool = True) -> bool:
    v = os.environ.get(name, "1" if default_true else "0").lower()
    return v not in ("0", "false", "no", "off")

_BATCH  = max(1, _int_env("SPLINEOPS_BLOCK", 256))
_ACCUM  = os.environ.get("SPLINEOPS_ACCUM", "support").lower()
if _ACCUM not in ("mulsum", "einsum", "support"):
    _ACCUM = "support"
_TILE_W = max(0, _int_env("SPLINEOPS_TILE_W", 0))

# Auto-tuner (off by default)
_AUTOTUNE        = _bool_env("SPLINEOPS_AUTOTUNE", False)
_AT_REPEATS      = max(1, _int_env("SPLINEOPS_AT_REPEATS", 1))
# Candidates (comma-separated envs if you want to change them)
_AT_ACCUM_CHOICES = tuple(
    a for a in os.environ.get("SPLINEOPS_AT_ACCUM", "support,einsum,mulsum").lower().split(",")
    if a in ("mulsum", "einsum", "support")
) or ("support", "einsum", "mulsum")
def _parse_list(name: str, default: str) -> list[int]:
    txt = os.environ.get(name, default)
    out = []
    for tok in txt.split(","):
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out

_AT_TILE_CHOICES  = [t for t in _parse_list("SPLINEOPS_AT_TILES", "0,64") if t >= 0]
_AT_BATCH_CHOICES = [b for b in _parse_list("SPLINEOPS_AT_BATCH", "64,128,256,512") if b > 0]

# Plan cache (reuse Plan1D across calls with the same realized signature).
# Both limits are read at use time so a long-lived process can reduce or clear
# retained memory without reloading the module.
_DEFAULT_PLAN_CACHE_SIZE = 32
_DEFAULT_PLAN_CACHE_BYTES = 128 * 1024 * 1024
_PLAN_CACHE_LOCK = threading.RLock()
_PLAN_CACHE: OrderedDict[tuple, tuple[Plan1D, int]] = OrderedDict()
_PLAN_CACHE_BYTES_USED = 0


def _nonnegative_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if text.startswith("+"):
        text = text[1:]
    if not text or not text.isascii() or not text.isdigit():
        return default
    try:
        value = int(text)
    except (ValueError, OverflowError):
        return default
    return value if value <= sys.maxsize else default


def _fallback_env(primary: str, fallback: str, default: int) -> int:
    if primary in os.environ:
        return _nonnegative_env(primary, default)
    return _nonnegative_env(fallback, default)


def _plan_cache_limits() -> tuple[int, int, bool]:
    capacity = _fallback_env(
        "SPLINEOPS_PLAN_CACHE_SIZE",
        "LSRESIZE_PLAN_CACHE_SIZE",
        _DEFAULT_PLAN_CACHE_SIZE,
    )
    byte_capacity = _fallback_env(
        "SPLINEOPS_PLAN_CACHE_BYTES",
        "LSRESIZE_PLAN_CACHE_BYTES",
        _DEFAULT_PLAN_CACHE_BYTES,
    )
    enabled = (
        _bool_env("SPLINEOPS_PLAN_CACHE", True)
        and capacity > 0
        and byte_capacity > 0
    )
    return (
        (capacity, byte_capacity, True)
        if enabled
        else (0, 0, False)
    )


def _plan_memory_bytes(plan: Plan1D) -> int:
    """Conservatively count Python metadata and retained NumPy storage."""

    attributes = vars(plan)
    total = sys.getsizeof(plan) + sys.getsizeof(attributes) + 256
    for value in attributes.values():
        if not isinstance(value, np.ndarray):
            total += sys.getsizeof(value)
            continue

        # nbytes is the logical array payload. A view can retain a larger base,
        # so count the largest ndarray backing store reachable from this field.
        retained = int(value.nbytes)
        base = value.base
        seen = {id(value)}
        while isinstance(base, np.ndarray) and id(base) not in seen:
            seen.add(id(base))
            retained = max(retained, int(base.nbytes))
            base = base.base

        header = sys.getsizeof(value)
        if value.flags.owndata:
            # CPython's ndarray sizeof includes owned payload bytes.
            header = max(0, header - int(value.nbytes))
        total += retained + header
    return total


def _evict_plan_cache_locked(
    capacity: int, byte_capacity: int
) -> list[Plan1D]:
    global _PLAN_CACHE_BYTES_USED

    evicted: list[Plan1D] = []
    while (
        len(_PLAN_CACHE) > capacity
        or _PLAN_CACHE_BYTES_USED > byte_capacity
    ):
        if not _PLAN_CACHE:
            _PLAN_CACHE_BYTES_USED = 0
            break
        _, (plan, memory_bytes) = _PLAN_CACHE.popitem(last=False)
        _PLAN_CACHE_BYTES_USED = max(
            0, _PLAN_CACHE_BYTES_USED - memory_bytes
        )
        evicted.append(plan)
    return evicted


def _build_plan(
    N: int,
    interp: int,
    analy: int,
    synthe: int,
    zoom: float,
    shift: float,
) -> Plan1D:
    p = LSParams(interp_degree=interp, analy_degree=analy, synthe_degree=synthe,
                 zoom=zoom, shift=shift)
    return make_plan_1d(N, p)


def _reset_plan_cache_lock_after_fork() -> None:
    global _PLAN_CACHE_LOCK
    # The cache contains immutable plans and is consistent while the GIL is
    # held at fork. Replace only the possibly inherited-locked synchronization
    # primitive; retaining the plans avoids a cold-cache penalty in the child.
    _PLAN_CACHE_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_plan_cache_lock_after_fork)


def _get_plan(N: int, p: LSParams) -> Plan1D:
    global _PLAN_CACHE_BYTES_USED

    # Nominal zoom is only a request for integer outN. Canonicalizing the
    # cache signature lets all requests resolving to the same grid share one
    # immutable plan.
    outN = calculate_output_size_1d(N, float(p.zoom))
    zoom = float(outN) / float(N)
    key = (
        N,
        p.interp_degree,
        p.analy_degree,
        p.synthe_degree,
        zoom,
        float(p.shift),
    )

    cached = None
    with _PLAN_CACHE_LOCK:
        capacity, byte_capacity, enabled = _plan_cache_limits()
        evicted = _evict_plan_cache_locked(capacity, byte_capacity)
        if enabled:
            entry = _PLAN_CACHE.get(key)
            if entry is not None:
                _PLAN_CACHE.move_to_end(key)
                cached = entry[0]
    del evicted
    if cached is not None:
        return cached

    built = _build_plan(*key)
    memory_bytes = _plan_memory_bytes(built)

    selected = built
    with _PLAN_CACHE_LOCK:
        capacity, byte_capacity, enabled = _plan_cache_limits()
        evicted = _evict_plan_cache_locked(capacity, byte_capacity)
        if enabled and memory_bytes <= byte_capacity:
            entry = _PLAN_CACHE.get(key)
            if entry is not None:
                _PLAN_CACHE.move_to_end(key)
                selected = entry[0]
            else:
                _PLAN_CACHE[key] = (built, memory_bytes)
                _PLAN_CACHE_BYTES_USED += memory_bytes
                evicted.extend(
                    _evict_plan_cache_locked(capacity, byte_capacity)
                )
    del evicted
    return selected

# Cache auto-tuner decisions per "plan signature"
_AT_DECISION_CACHE: dict[tuple, tuple[str, int, int]] = {}
# decision tuple: (accum, tile_w, batch_eff)

# -------------------------------- autotuner ----------------------------------

def _plan_key(plan, p: LSParams) -> tuple:
    # Enough to uniquely identify cost shape
    return (
        int(plan.N), int(plan.out_total), int(plan.win_len_max),
        int(plan.left_pad), int(plan.right_pad), int(plan.outN),
        int(p.interp_degree), int(p.analy_degree), int(p.synthe_degree),
        round(float(p.zoom), 12), round(float(p.shift), 12)
    )

def _bench_block(Xb: np.ndarray, plan, p: LSParams, accum: str, tile_w: int) -> float:
    """
    Time one full pipeline on a small block Xb: (B, N).
    Returns best of _AT_REPEATS in seconds.
    """
    B, N = Xb.shape
    out_total  = plan.out_total
    outN       = plan.outN
    length_total = plan.length_total
    full_len   = plan.left_pad + plan.length_total + plan.right_pad
    Wmax       = plan.win_len_max
    corr_degree = p.interp_degree if p.analy_degree < 0 else (p.analy_degree + p.synthe_degree + 1)

    # allocate once
    coeffB   = np.empty((B, N),            dtype=np.float64)
    extB = (
        None
        if plan.direct_projection
        else np.empty((B, length_total), dtype=np.float64)
    )
    extFullB = (
        None
        if plan.direct_projection
        else np.empty((B, full_len), dtype=np.float64)
    )
    yBlock   = np.empty((B, out_total),    dtype=np.float64)

    use_tiling = (accum != "support") and (tile_w > 0) and (Wmax > tile_w)
    if use_tiling and Wmax > 0 and out_total > 0:
        gather_tile = np.empty((B, out_total, tile_w), dtype=np.float64)
        tmp2D       = np.empty((B, out_total),         dtype=np.float64)
    elif accum == "support" and Wmax > 0 and out_total > 0:
        tmp2D       = np.empty((B, out_total),         dtype=np.float64)
    else:
        gather3D    = np.empty((B, out_total, Wmax),   dtype=np.float64) if (Wmax > 0 and out_total > 0) else None

    best = float("inf")
    for _ in range(_AT_REPEATS):
        t0 = perf_counter()

        # coeffs
        np.copyto(coeffB, Xb)
        get_interpolation_coefficients_batch(coeffB, p.interp_degree)

        # optional integration
        if p.analy_degree >= 0 and not plan.direct_projection:
            avgB = do_integ_batch(coeffB, p.analy_degree + 1)
        else:
            avgB = None

        # Direct cross-Gram indices address coeffB itself; all other methods
        # gather from the materialized extension.
        if plan.direct_projection:
            sampleB = coeffB
        else:
            extB[:, :N] = coeffB
            rem = length_total - N
            if rem > 0 and plan.rp_src.size:
                extB[:, N:] = plan.rp_sign * coeffB[:, plan.rp_src]
            if plan.left_pad > 0:
                extFullB[:, plan.lp_dst] = plan.lp_sign * coeffB[:, plan.lp_src]
            extFullB[:, plan.left_pad : plan.left_pad + length_total] = extB
            if plan.right_pad > 0:
                extFullB[:, plan.left_pad + length_total :] = extB[:, -1][:, None]
            sampleB = extFullB

        # gather + accumulate
        if (Wmax > 0) and (out_total > 0):
            if use_tiling:
                yBlock[:, :] = 0.0
                for t0w in range(0, Wmax, tile_w):
                    t1w = min(Wmax, t0w + tile_w)
                    wtile = plan.weights2d[:, t0w:t1w]  # (L,w)
                    np.take(sampleB, plan.idx2d[:, t0w:t1w], axis=1, out=gather_tile[:, :, :t1w-t0w])
                    if accum == "einsum":
                        yBlock[:, :] += np.einsum('lw,blw->bl', wtile, gather_tile[:, :, :t1w-t0w], optimize=True)
                    else:
                        np.multiply(gather_tile[:, :, :t1w-t0w], wtile[None, :, :], out=gather_tile[:, :, :t1w-t0w])
                        np.sum(gather_tile[:, :, :t1w-t0w], axis=2, out=tmp2D)
                        yBlock[:, :] += tmp2D
            elif accum == "support":
                yBlock[:, :] = 0.0
                for tw in range(Wmax):
                    np.take(sampleB, plan.idx2d[:, tw], axis=1, out=tmp2D)
                    np.multiply(tmp2D, plan.weights2d[None, :, tw], out=tmp2D)
                    yBlock[:, :] += tmp2D
            else:
                np.take(sampleB, plan.idx2d, axis=1, out=gather3D)
                if accum == "einsum":
                    np.einsum('lw,blw->bl', plan.weights2d, gather3D, out=yBlock, optimize=True)
                else:
                    np.multiply(gather3D, plan.weights2d[None, :, :], out=gather3D)
                    np.sum(gather3D, axis=2, out=yBlock)
        else:
            yBlock[:, :] = 0.0

        # projection tail
        if p.analy_degree >= 0:
            if not plan.direct_projection:
                do_diff_batch(yBlock, p.analy_degree + 1)
                yBlock[:, :] += avgB[:, None]
            get_interpolation_coefficients_batch(yBlock, corr_degree)
            get_samples_batch(yBlock, p.synthe_degree)

        _ = yBlock[:, :outN]  # crop (not used)
        dt = perf_counter() - t0
        if dt < best: best = dt

    return best

def _autotune(plan, p: LSParams, X: np.ndarray) -> tuple[str, int, int]:
    """
    Decide (accum, tile_w, batch_eff) once per plan signature and cache it.
    """
    key = _plan_key(plan, p)
    if key in _AT_DECISION_CACHE:
        return _AT_DECISION_CACHE[key]

    # small problems don't benefit; bail
    cols, N = X.shape
    if not _AUTOTUNE or cols < 8 or plan.out_total == 0:
        decision = (_ACCUM, _TILE_W, min(_BATCH, cols))
        _AT_DECISION_CACHE[key] = decision
        return decision

    # Build candidate lists, capped by current problem
    batch_cands = sorted({min(b, cols) for b in _AT_BATCH_CHOICES if b > 0} | {min(_BATCH, cols)})
    tile_cands  = sorted({t for t in _AT_TILE_CHOICES if t >= 0})
    if plan.win_len_max <= 0:
        tile_cands = [0]  # no kernel width to tile
    accum_cands = _AT_ACCUM_CHOICES

    best = (float("inf"), _ACCUM, _TILE_W, min(_BATCH, cols))
    # light sampling: use a small top slice for timing
    for b in batch_cands:
        Xb = X[:b, :]  # (b,N)
        for tile_w in tile_cands:
            for accum in accum_cands:
                t = _bench_block(Xb, plan, p, accum, tile_w)
                if t < best[0]:
                    best = (t, accum, tile_w, b)

    decision = (best[1], best[2], best[3])
    _AT_DECISION_CACHE[key] = decision
    return decision

# ---------------------------------- core -------------------------------------

def resize_along_axis(arr: np.ndarray, axis: int, p: LSParams) -> np.ndarray:
    """
    Batched pure-NumPy path with optional auto-tuning:
      - move target axis to last dim so each line is contiguous
      - (optional) auto-tune accum/tiling/batch on the first block and cache
      - process rows in blocks with vectorized prefilter/integration/diff
      - build extension / padded buffer once per block
      - gather via np.take(..., axis=1) into (B, out_total, win_len_max)
      - accumulate support-wise, with einsum, or with multiply+sum into (B, out_total)
    """
    a = np.asarray(arr, dtype=np.float64, order="C")
    N_line = a.shape[axis]
    outN = calculate_output_size_1d(N_line, p.zoom)

    # Degenerate endpoint grids have no finite endpoint-derived scale.  Give
    # them explicit, symmetric definitions instead of entering the projection
    # integration/difference pipeline:
    #   * one input sample represents a constant line and is replicated;
    #   * one projected output is the line mean (DC-preserving antialiasing);
    #   * degree-0 interpolation at an even-length line centre averages the
    #     two equally-near samples, avoiding an asymmetric tie break.
    if N_line == 1:
        return np.repeat(a, outN, axis=axis)
    if outN == 1 and p.analy_degree >= 0:
        return np.mean(a, axis=axis, dtype=np.float64, keepdims=True)
    if outN == 1 and p.interp_degree == 0:
        left = (N_line - 1) // 2
        right = N_line // 2
        if left == right:
            return np.take(a, [left], axis=axis)
        return 0.5 * (
            np.take(a, [left], axis=axis)
            + np.take(a, [right], axis=axis)
        )

    # When the input and output grids are identical, projection between the
    # same interpolation/synthesis spaces is exactly the identity operator.
    # Base this on integer grid sizes, not on the nominal size-request zoom.
    if (
        outN == N_line
        and abs(p.shift) <= 1e-15
        and p.synthe_degree == p.interp_degree
    ):
        return a.copy()

    plan = _get_plan(N_line, p)

    # Move target axis to last so lines are contiguous
    x_last = np.moveaxis(a, axis, -1)                      # (..., N)
    cols = int(np.prod(x_last.shape[:-1] or (1,)))
    X = x_last.reshape(cols, N_line)                       # (cols, N), rows contiguous

    # Decide runtime strategy (autotune or env)
    if _AUTOTUNE:
        accum_use, tile_use, B_eff = _autotune(plan, p, X)
        B_eff = max(1, min(B_eff, cols))
    else:
        accum_use = _ACCUM
        tile_use  = _TILE_W
        B_eff     = max(1, min(_BATCH, cols))

    Y = np.empty((cols, plan.outN), dtype=np.float64)      # (cols, outN)

    # Preallocate block work buffers according to chosen batch
    N = N_line
    out_total  = plan.out_total
    outN       = plan.outN
    length_total = plan.length_total
    full_len   = plan.left_pad + plan.length_total + plan.right_pad
    Wmax       = plan.win_len_max

    coeffB   = np.empty((B_eff, N),            dtype=np.float64)
    extB = (
        None
        if plan.direct_projection
        else np.empty((B_eff, length_total), dtype=np.float64)
    )
    extFullB = (
        None
        if plan.direct_projection
        else np.empty((B_eff, full_len), dtype=np.float64)
    )
    yBlock   = np.empty((B_eff, out_total),    dtype=np.float64)

    use_tiling = (accum_use != "support") and (tile_use > 0) and (Wmax > tile_use)
    if use_tiling:
        gather_tile = np.empty((B_eff, out_total, tile_use), dtype=np.float64)
        tmp2D       = np.empty((B_eff, out_total),          dtype=np.float64)
    elif accum_use == "support":
        tmp2D       = np.empty((B_eff, out_total),          dtype=np.float64)
    else:
        gather3D    = np.empty((B_eff, out_total, Wmax),    dtype=np.float64) if (Wmax > 0 and out_total > 0) else None

    corr_degree = p.interp_degree if p.analy_degree < 0 else (p.analy_degree + p.synthe_degree + 1)

    for i in range(0, cols, B_eff):
        b = min(B_eff, cols - i)

        # 1) coefficients (batched IIR)
        np.copyto(coeffB[:b, :], X[i:i+b, :])
        get_interpolation_coefficients_batch(coeffB[:b, :], p.interp_degree)

        # 2) optional integration (in-place); keep per-line averages
        if p.analy_degree >= 0 and not plan.direct_projection:
            avgB = do_integ_batch(coeffB[:b, :], p.analy_degree + 1)
        else:
            avgB = None

        # 3) extension (right tail via precomputed mapping). Direct rows gather
        # their already-mirrored indices straight from the coefficient block.
        if plan.direct_projection:
            sampleB = coeffB[:b, :]
        else:
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
            sampleB = extFullB[:b, :]

        # 4) gather + accumulate
        if (Wmax > 0) and (out_total > 0):
            if use_tiling:
                yBlock[:b, :] = 0.0
                for t0 in range(0, Wmax, tile_use):
                    t1 = min(Wmax, t0 + tile_use)
                    wtile = plan.weights2d[:, t0:t1]              # (L, w)
                    # gather tile
                    np.take(sampleB, plan.idx2d[:, t0:t1], axis=1, out=gather_tile[:b, :, :t1-t0])
                    if accum_use == "einsum":
                        yBlock[:b, :] += np.einsum('lw,blw->bl', wtile, gather_tile[:b, :, :t1-t0], optimize=True)
                    else:
                        np.multiply(gather_tile[:b, :, :t1-t0], wtile[None, :, :], out=gather_tile[:b, :, :t1-t0])
                        np.sum(gather_tile[:b, :, :t1-t0], axis=2, out=tmp2D[:b, :])
                        yBlock[:b, :] += tmp2D[:b, :]
            elif accum_use == "support":
                yBlock[:b, :] = 0.0
                for tw in range(Wmax):
                    np.take(sampleB, plan.idx2d[:, tw], axis=1, out=tmp2D[:b, :])
                    np.multiply(tmp2D[:b, :], plan.weights2d[None, :, tw], out=tmp2D[:b, :])
                    yBlock[:b, :] += tmp2D[:b, :]
            else:
                np.take(sampleB, plan.idx2d, axis=1, out=gather3D[:b, :, :])   # (b, L, W)
                if accum_use == "einsum":
                    np.einsum('lw,blw->bl', plan.weights2d, gather3D[:b, :, :], out=yBlock[:b, :], optimize=True)
                else:
                    np.multiply(gather3D[:b, :, :], plan.weights2d[None, :, :], out=gather3D[:b, :, :])
                    np.sum(gather3D[:b, :, :], axis=2, out=yBlock[:b, :])
        else:
            yBlock[:b, :] = 0.0

        # 5) projection tail
        if p.analy_degree >= 0:
            if not plan.direct_projection:
                do_diff_batch(yBlock[:b, :], p.analy_degree + 1)
                yBlock[:b, :] += avgB[:, None]
            get_interpolation_coefficients_batch(yBlock[:b, :], corr_degree)
            get_samples_batch(yBlock[:b, :], p.synthe_degree)

        # 6) crop to outN and store
        Y[i:i+b, :] = yBlock[:b, :outN]

    # Reshape back and restore axis
    out_last = Y.reshape(x_last.shape[:-1] + (outN,))
    return np.moveaxis(out_last, -1, axis)
