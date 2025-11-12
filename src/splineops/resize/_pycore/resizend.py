# splineops/src/splineops/resize/_pycore/resizend.py
from __future__ import annotations
import os
from functools import lru_cache
from time import perf_counter
import numpy as np
from .params import LSParams
from .plan1d import make_plan_1d
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

_BATCH  = max(1, _int_env("SPLINEOPS_BLOCK", 64))
_ACCUM  = os.environ.get("SPLINEOPS_ACCUM", "mulsum").lower()
if _ACCUM not in ("mulsum", "einsum"):
    _ACCUM = "mulsum"
_TILE_W = max(0, _int_env("SPLINEOPS_TILE_W", 0))
_USE_PLAN_CACHE = _bool_env("SPLINEOPS_PLAN_CACHE", True)

# Auto-tuner (off by default)
_AUTOTUNE        = _bool_env("SPLINEOPS_AUTOTUNE", False)
_AT_REPEATS      = max(1, _int_env("SPLINEOPS_AT_REPEATS", 1))
# Candidates (comma-separated envs if you want to change them)
_AT_ACCUM_CHOICES = tuple(
    a for a in os.environ.get("SPLINEOPS_AT_ACCUM", "mulsum,einsum").lower().split(",")
    if a in ("mulsum", "einsum")
) or ("mulsum", "einsum")
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
_AT_BATCH_CHOICES = [b for b in _parse_list("SPLINEOPS_AT_BATCH", "32,64,128") if b > 0]

# Plan cache (reuse Plan1D across calls with same signature)
@lru_cache(maxsize=8)
def _cached_plan(N: int, interp: int, analy: int, synthe: int, zoom: float, shift: float, inversable: bool):
    p = LSParams(interp_degree=interp, analy_degree=analy, synthe_degree=synthe,
                 zoom=zoom, shift=shift, inversable=inversable)
    return make_plan_1d(N, p)

def _get_plan(N: int, p: LSParams):
    if _USE_PLAN_CACHE:
        return _cached_plan(N, p.interp_degree, p.analy_degree, p.synthe_degree, float(p.zoom), float(p.shift), bool(p.inversable))
    return make_plan_1d(N, p)

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
        round(float(p.zoom), 12), round(float(p.shift), 12), bool(p.inversable)
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
    extB     = np.empty((B, length_total), dtype=np.float64)
    extFullB = np.empty((B, full_len),     dtype=np.float64)
    yBlock   = np.empty((B, out_total),    dtype=np.float64)

    use_tiling = (tile_w > 0) and (Wmax > tile_w)
    if use_tiling and Wmax > 0 and out_total > 0:
        gather_tile = np.empty((B, out_total, tile_w), dtype=np.float64)
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
        if p.analy_degree >= 0:
            avgB = do_integ_batch(coeffB, p.analy_degree + 1)
        else:
            avgB = None

        # extension
        extB[:, :N] = coeffB
        rem = length_total - N
        if rem > 0 and plan.rp_src.size:
            extB[:, N:] = plan.rp_sign * coeffB[:, plan.rp_src]
        if plan.left_pad > 0:
            extFullB[:, plan.lp_dst] = plan.lp_sign * coeffB[:, plan.lp_src]
        extFullB[:, plan.left_pad : plan.left_pad + length_total] = extB
        if plan.right_pad > 0:
            extFullB[:, plan.left_pad + length_total :] = extB[:, -1][:, None]

        # gather + accumulate
        if (Wmax > 0) and (out_total > 0):
            if use_tiling:
                yBlock[:, :] = 0.0
                for t0w in range(0, Wmax, tile_w):
                    t1w = min(Wmax, t0w + tile_w)
                    wtile = plan.weights2d[:, t0w:t1w]  # (L,w)
                    np.take(extFullB, plan.idx2d[:, t0w:t1w], axis=1, out=gather_tile[:, :, :t1w-t0w])
                    if accum == "einsum":
                        yBlock[:, :] += np.einsum('lw,blw->bl', wtile, gather_tile[:, :, :t1w-t0w], optimize=True)
                    else:
                        np.multiply(gather_tile[:, :, :t1w-t0w], wtile[None, :, :], out=gather_tile[:, :, :t1w-t0w])
                        np.sum(gather_tile[:, :, :t1w-t0w], axis=2, out=tmp2D)
                        yBlock[:, :] += tmp2D
            else:
                np.take(extFullB, plan.idx2d, axis=1, out=gather3D)
                if accum == "einsum":
                    np.einsum('lw,blw->bl', plan.weights2d, gather3D, out=yBlock, optimize=True)
                else:
                    np.multiply(gather3D, plan.weights2d[None, :, :], out=gather3D)
                    np.sum(gather3D, axis=2, out=yBlock)
        else:
            yBlock[:, :] = 0.0

        # projection tail
        if p.analy_degree >= 0:
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
      - accumulate with multiply+sum or einsum into (B, out_total)
    """
    a = np.asarray(arr, dtype=np.float64, order="C")
    N_line = a.shape[axis]
    plan = _get_plan(N_line, p)

    # Fast identity short-circuit (no projection, zoom==1)
    if (abs(p.zoom - 1.0) <= 1e-12) and (p.analy_degree < 0) and (plan.outN == N_line):
        return a.copy()

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
    extB     = np.empty((B_eff, length_total), dtype=np.float64)
    extFullB = np.empty((B_eff, full_len),     dtype=np.float64)
    yBlock   = np.empty((B_eff, out_total),    dtype=np.float64)

    use_tiling = (tile_use > 0) and (Wmax > tile_use)
    if use_tiling:
        gather_tile = np.empty((B_eff, out_total, tile_use), dtype=np.float64)
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
        if p.analy_degree >= 0:
            avgB = do_integ_batch(coeffB[:b, :], p.analy_degree + 1)
        else:
            avgB = None

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
            if use_tiling:
                yBlock[:b, :] = 0.0
                for t0 in range(0, Wmax, tile_use):
                    t1 = min(Wmax, t0 + tile_use)
                    wtile = plan.weights2d[:, t0:t1]              # (L, w)
                    # gather tile
                    np.take(extFullB[:b, :], plan.idx2d[:, t0:t1], axis=1, out=gather_tile[:b, :, :t1-t0])
                    if accum_use == "einsum":
                        yBlock[:b, :] += np.einsum('lw,blw->bl', wtile, gather_tile[:b, :, :t1-t0], optimize=True)
                    else:
                        np.multiply(gather_tile[:b, :, :t1-t0], wtile[None, :, :], out=gather_tile[:b, :, :t1-t0])
                        np.sum(gather_tile[:b, :, :t1-t0], axis=2, out=tmp2D[:b, :])
                        yBlock[:b, :] += tmp2D[:b, :]
            else:
                np.take(extFullB[:b, :], plan.idx2d, axis=1, out=gather3D[:b, :, :])   # (b, L, W)
                if accum_use == "einsum":
                    np.einsum('lw,blw->bl', plan.weights2d, gather3D[:b, :, :], out=yBlock[:b, :], optimize=True)
                else:
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
