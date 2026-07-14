#!/usr/bin/env python3
"""Experimental stable equal-degree least-squares resize kernels.

This file deliberately does not participate in the public resize path.  It
compares the Muñoz/Blu/Unser running-sum implementation with a direct compact
cross-Gram operator.  The latter evaluates

    q[l] = sum_k c[k] h_a(l - a*k),
    h_a = beta_n(./a) * beta_n,

where ``c`` are the input interpolation coefficients and ``a`` is the unique
endpoint-aligned scale ``(M-1)/(N-1)``.  ``h_a`` is a generalized B-spline
with compact support.  Its weights are generated once per plan by fixed
Gauss-Legendre quadrature; the quadrature is exact in exact arithmetic because
the integrand is piecewise polynomial of degree ``2*n``.

The direct formulation is mathematically the same B-spline inner product as
the finite-difference construction, but it avoids the large antiderivatives
and catastrophic cancellation which make equal-degree cubic LS unsafe on long
lines.

Examples
--------
Quick arbitrary-precision validation and stability sweep::

    python scripts/experiment_resize_stable_ls.py --profile quick

Full N=32..65536 sweep with CSV output::

    python scripts/experiment_resize_stable_ls.py --profile full \
        --output-csv /tmp/resize_stable_ls.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import mpmath as mp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from splineops.resize._pycore.bspline import beta  # noqa: E402
from splineops.resize._pycore.diff_integ import do_diff, do_integ  # noqa: E402
from splineops.resize._pycore.filters import (  # noqa: E402
    get_interpolation_coefficients,
    get_samples,
    sampling_fir,
    spline_poles,
)
from splineops.resize._pycore.params import LSParams, Work1D  # noqa: E402
from splineops.resize._pycore.plan_1d import make_plan_1d  # noqa: E402
from splineops.resize._pycore.resize_1d import resize_1d_ws  # noqa: E402


@dataclass(frozen=True)
class DirectLSPlan:
    n: int
    input_size: int
    output_size: int
    output_total: int
    scale: float
    source_index: np.ndarray
    weight: np.ndarray

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.weight))


@dataclass(frozen=True)
class DirectProjectionPlan:
    interp_degree: int
    analysis_degree: int
    synthesis_degree: int
    input_size: int
    output_size: int
    scale: float
    source_index: np.ndarray
    weight: np.ndarray


@dataclass
class ResultRow:
    degree: int
    input_size: int
    output_size: int
    pattern: str
    method: str
    reference: str
    runtime_ms: float
    plan_ms: float
    rel_l2: float
    max_abs_error: float
    output_min: float
    output_max: float
    finite: bool
    nnz: int


def _params(n: int, zoom: float) -> LSParams:
    return _projection_params(n, n, n, zoom)


def _projection_params(
    interp_degree: int,
    analysis_degree: int,
    synthesis_degree: int,
    zoom: float,
) -> LSParams:
    return LSParams(
        interp_degree=interp_degree,
        analy_degree=analysis_degree,
        synthe_degree=synthesis_degree,
        zoom=float(zoom),
        shift=0.0,
    )


def _mirror_index(k: int, size: int) -> int:
    if size <= 1:
        return 0
    period = 2 * size - 2
    k %= period
    return period - k if k >= size else k


@functools.lru_cache(maxsize=4)
def _gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def generalized_bspline_gauss(x: float, scale: float, degree: int) -> float:
    """Return ``(beta_n(./scale) * beta_n)(x)`` without cancellation.

    The breakpoints of both splines partition the overlap into polynomial
    pieces.  An ``n+1`` point Gauss rule integrates their degree-``2*n``
    product exactly, apart from floating-point evaluation error.
    """

    return generalized_cross_bspline_gauss(x, scale, degree, degree)


def generalized_cross_bspline_gauss(
    x: float,
    scale: float,
    interp_degree: int,
    analysis_degree: int,
) -> float:
    """Convolution ``beta_interp(./a) * beta_analysis`` at ``x``."""

    n = int(interp_degree)
    m = int(analysis_degree)
    a = float(scale)
    input_radius = 0.5 * (n + 1)
    analysis_radius = 0.5 * (m + 1)
    lo = max(-a * input_radius, x - analysis_radius)
    hi = min(+a * input_radius, x + analysis_radius)
    if not lo < hi:
        return 0.0

    knots = np.concatenate(
        (
            a * np.arange(-input_radius, input_radius + 0.5, 1.0),
            x - np.arange(-analysis_radius, analysis_radius + 0.5, 1.0),
            np.asarray([lo, hi]),
        )
    )
    knots = np.unique(np.clip(knots[(knots >= lo) & (knots <= hi)], lo, hi))
    # A q-point rule integrates degree 2q-1.  The product degree is n+m.
    nodes, weights = _gauss_rule((n + m + 2) // 2)

    total = 0.0
    correction = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        width = right - left
        if width <= 16.0 * np.finfo(float).eps * max(1.0, abs(left), abs(right)):
            continue
        half = 0.5 * width
        t = 0.5 * (left + right) + half * nodes
        term = half * float(np.dot(weights, beta(t / a, n) * beta(x - t, m)))
        # Neumaier summation.  There are only O(n) intervals, but this also
        # makes the plan generator insensitive to their ordering.
        summed = total + term
        if abs(total) >= abs(term):
            correction += (total - summed) + term
        else:
            correction += (term - summed) + total
        total = summed
    value = total + correction
    return 0.0 if value < 0.0 and value > -64.0 * np.finfo(float).eps else value


def generalized_bspline_truncated_longdouble(
    x: np.ndarray, scale: float, degree: int
) -> np.ndarray:
    """Vectorized grouped truncated-power formula evaluated in long double.

    This is the fast experiment-plan generator.  The Gauss construction above
    is the cancellation-free production candidate and is used to audit these
    weights; plan execution is identical whichever generator is selected.
    """

    n = int(degree)
    r = n + 1
    power = 2 * n + 1
    a = np.longdouble(scale)
    values = np.asarray(x, dtype=np.longdouble)
    shifted = values + np.longdouble(r) * (a + 1) / 2
    result = np.zeros_like(values)
    for i in range(r + 1):
        for j in range(r + 1):
            positive = np.maximum(shifted - i * a - j, 0)
            result += (
                ((-1) ** (i + j))
                * math.comb(r, i)
                * math.comb(r, j)
                * positive**power
            )
    result /= np.longdouble(math.factorial(power)) * a**n
    radius = np.longdouble(r) * (a + 1) / 2
    result[np.abs(values) >= radius] = 0
    tolerance = 256 * np.finfo(np.longdouble).eps
    result[(result < 0) & (result > -tolerance)] = 0
    return np.asarray(result, dtype=np.float64)


def make_direct_ls_plan(
    input_size: int,
    zoom: float,
    degree: int,
    *,
    kernel: str = "truncated_longdouble",
) -> DirectLSPlan:
    """Build reusable compact cross-Gram weights for one endpoint grid."""

    p = _params(degree, zoom)
    base = make_plan_1d(input_size, p)
    output_size = int(base.outN)
    if input_size <= 1 or output_size <= 1:
        return DirectLSPlan(
            degree,
            input_size,
            output_size,
            output_size,
            1.0,
            np.zeros((output_size, 1), dtype=np.intp),
            np.ones((output_size, 1), dtype=np.float64),
        )

    a = (output_size - 1) / float(input_size - 1)
    radius = 0.5 * (degree + 1) * (a + 1.0)
    # On an endpoint-aligned zero-shift grid the transformed mirror extension
    # is symmetric about output_size-1.  Consequently the cross-inner-product
    # sequence is already a valid length-M mirror sequence and the output Gram
    # inverse must be applied to those M values.  Carrying the historical
    # off-grid tail into the IIR changes its right boundary and is not the same
    # finite-domain orthogonal projection.
    output_total = output_size
    half_width = math.ceil(radius / a) + 2
    offsets = np.arange(-half_width, half_width + 1, dtype=np.int64)
    output_index = np.arange(output_total, dtype=np.float64)
    centers = np.floor(output_index / a).astype(np.int64)
    unwrapped_source = centers[:, None] + offsets[None, :]
    distance = output_index[:, None] - a * unwrapped_source

    if kernel == "truncated_longdouble":
        weight = generalized_bspline_truncated_longdouble(distance, a, degree)
    elif kernel == "gauss":
        weight = np.zeros_like(distance)
        for row in range(output_total):
            for column in range(distance.shape[1]):
                weight[row, column] = generalized_bspline_gauss(
                    float(distance[row, column]), a, degree
                )
    else:
        raise ValueError(f"unknown direct kernel generator: {kernel}")

    period = 2 * input_size - 2
    source = np.mod(unwrapped_source, period)
    source = np.where(source >= input_size, period - source, source).astype(np.intp)

    return DirectLSPlan(
        degree,
        input_size,
        output_size,
        output_total,
        a,
        source,
        weight,
    )


def execute_direct_ls(samples: np.ndarray, plan: DirectLSPlan) -> np.ndarray:
    """Execute a prebuilt direct LS plan using the existing stable IIR tails."""

    x = np.asarray(samples, dtype=np.float64)
    if x.size == 1:
        return np.full(plan.output_size, x[0], dtype=np.float64)
    if plan.output_size == 1:
        return np.asarray([np.mean(x, dtype=np.float64)])
    c = x.copy()
    get_interpolation_coefficients(c, plan.n)
    q = np.sum(plan.weight * c[plan.source_index], axis=1)
    get_interpolation_coefficients(q, 2 * plan.n + 1)
    get_samples(q, plan.n)
    return q[: plan.output_size].copy()


def make_direct_projection_plan(
    input_size: int,
    zoom: float,
    interp_degree: int,
    analysis_degree: int,
    synthesis_degree: int,
) -> DirectProjectionPlan:
    """Small/medium validation plan for general oblique projections."""

    p = _projection_params(
        interp_degree, analysis_degree, synthesis_degree, zoom
    )
    base = make_plan_1d(input_size, p)
    output_size = base.outN
    a = (output_size - 1) / float(input_size - 1)
    radius = 0.5 * ((interp_degree + 1) * a + analysis_degree + 1)
    half_width = math.ceil(radius / a) + 2
    offsets = np.arange(-half_width, half_width + 1, dtype=np.int64)
    output_index = np.arange(output_size, dtype=np.float64)
    centers = np.floor(output_index / a).astype(np.int64)
    unwrapped = centers[:, None] + offsets
    distance = output_index[:, None] - a * unwrapped
    weight = np.zeros_like(distance)
    for row in range(output_size):
        for column in range(distance.shape[1]):
            weight[row, column] = generalized_cross_bspline_gauss(
                float(distance[row, column]),
                a,
                interp_degree,
                analysis_degree,
            )
    period = 2 * input_size - 2
    source = np.mod(unwrapped, period)
    source = np.where(source >= input_size, period - source, source).astype(np.intp)
    return DirectProjectionPlan(
        interp_degree,
        analysis_degree,
        synthesis_degree,
        input_size,
        output_size,
        a,
        source,
        weight,
    )


def execute_direct_projection(
    samples: np.ndarray, plan: DirectProjectionPlan
) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    c = x.copy()
    get_interpolation_coefficients(c, plan.interp_degree)
    q = np.sum(plan.weight * c[plan.source_index], axis=1)
    get_interpolation_coefficients(
        q, plan.analysis_degree + plan.synthesis_degree + 1
    )
    get_samples(q, plan.synthesis_degree)
    return q


def execute_finite_difference(samples: np.ndarray, zoom: float, degree: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    p = _params(degree, zoom)
    return resize_1d_ws(x, p, make_plan_1d(x.size, p), Work1D())


def _initial_causal_typed(c: np.ndarray, pole: np.generic) -> np.generic:
    size = c.size
    if size == 1:
        return c[0]
    zn = pole ** (size - 1)
    value = c[0] + zn * c[-1]
    p1 = pole
    p2 = (zn * zn) / pole
    for i in range(1, size - 1):
        value += (p1 + p2) * c[i]
        p1 *= pole
        p2 /= pole
    return value / (1 - zn * zn)


def _prefilter_typed(c: np.ndarray, degree: int) -> None:
    if degree <= 1 or c.size <= 1:
        return
    typ = c.dtype.type
    poles = [typ(z) for z in spline_poles(degree)]
    normalization = typ(1)
    for pole in poles:
        normalization *= (1 - pole) * (1 - 1 / pole)
    c *= normalization
    for pole in poles:
        c[0] = _initial_causal_typed(c, pole)
        for i in range(1, c.size):
            c[i] += pole * c[i - 1]
        c[-1] = (pole * c[-2] + c[-1]) * pole / (pole * pole - 1)
        for i in range(c.size - 2, -1, -1):
            c[i] = pole * (c[i + 1] - c[i])


def _samples_typed(c: np.ndarray, degree: int) -> None:
    if degree <= 1:
        return
    typ = c.dtype.type
    taps = [typ(v) for v in sampling_fir(degree)]
    result = np.empty_like(c)
    for i in range(c.size):
        value = taps[0] * c[i]
        for j in range(1, len(taps)):
            value += taps[j] * (
                c[_mirror_index(i - j, c.size)] + c[_mirror_index(i + j, c.size)]
            )
        result[i] = value
    c[:] = result


def _integrate_typed(c: np.ndarray, count: int) -> np.generic:
    typ = c.dtype.type

    def average(values: np.ndarray) -> np.generic:
        return (
            typ(2) * np.sum(values, dtype=c.dtype) - values[0] - values[-1]
        ) / typ(2 * values.size - 2)

    saved_average = typ(0)
    if count >= 1:
        saved_average = average(c)
        c -= saved_average
        c[0] *= typ(0.5)
        np.cumsum(c, dtype=c.dtype, out=c)
    if count >= 2:
        original = c.copy()
        c[0] = original[0]
        c[1] = 0
        if c.size > 2:
            c[2:] = -np.cumsum(original[1:-1], dtype=c.dtype)
    if count >= 3:
        mean = average(c)
        c -= mean
        c[0] *= typ(0.5)
        np.cumsum(c, dtype=c.dtype, out=c)
    if count >= 4:
        original = c.copy()
        c[0] = original[0]
        c[1] = 0
        if c.size > 2:
            c[2:] = -np.cumsum(original[1:-1], dtype=c.dtype)
    return saved_average


def _differentiate_typed(c: np.ndarray, count: int) -> None:
    def diff_sa() -> None:
        old = c[-2]
        c[:-1] -= c[1:]
        c[-1] -= old

    def diff_as() -> None:
        c[1:] -= c[:-1]
        c[0] *= 2

    if count == 1:
        diff_as()
    elif count == 2:
        diff_sa(); diff_as()
    elif count == 3:
        diff_as(); diff_sa(); diff_as()
    else:
        diff_sa(); diff_as(); diff_sa(); diff_as()


def _projection_beta_longdouble(x: np.ndarray, degree: int) -> np.ndarray:
    """Stable long-double beta for equal-LS resampling degrees 3, 5 and 7."""

    ax = np.abs(np.asarray(x, dtype=np.longdouble))
    out = np.zeros_like(ax)
    if degree == 3:
        mask = ax < 1
        value = ax[mask]
        out[mask] = value * value * (value - 2) / 2 + np.longdouble(2) / 3
        mask = (~mask) & (ax < 2)
        value = ax[mask] - 2
        out[mask] = -(value * value * value) / 6
        return out
    if degree == 5:
        mask0 = ax < 1
        value = ax[mask0]
        square = value * value
        out[mask0] = square * (square * (np.longdouble(1) / 4 - value / 12) - np.longdouble(1) / 2) + np.longdouble(11) / 20
        mask1 = (~mask0) & (ax < 2)
        value = ax[mask1]
        out[mask1] = value * (value * (value * (value * (value / 24 - np.longdouble(3) / 8) + np.longdouble(5) / 4) - np.longdouble(7) / 4) + np.longdouble(5) / 8) + np.longdouble(17) / 40
        mask2 = (~mask0) & (~mask1) & (ax < 3)
        value = 3 - ax[mask2]
        out[mask2] = value**5 / 120
        return out
    if degree == 7:
        mask0 = ax < 1
        value = ax[mask0]
        square = value * value
        out[mask0] = square * (square * (square * (value / 144 - np.longdouble(1) / 36) + np.longdouble(1) / 9) - np.longdouble(1) / 3) + np.longdouble(151) / 315
        mask1 = (~mask0) & (ax < 2)
        value = ax[mask1]
        out[mask1] = value * (value * (value * (value * (value * (value * (np.longdouble(1) / 20 - value / 240) - np.longdouble(7) / 30) + np.longdouble(1) / 2) - np.longdouble(7) / 18) - np.longdouble(1) / 10) - np.longdouble(7) / 90) + np.longdouble(103) / 210
        mask2 = (~mask0) & (~mask1) & (ax < 3)
        value = ax[mask2]
        out[mask2] = value * (value * (value * (value * (value * (value * (value / 720 - np.longdouble(1) / 36) + np.longdouble(7) / 30) - np.longdouble(19) / 18) + np.longdouble(49) / 18) - np.longdouble(23) / 6) + np.longdouble(217) / 90) - np.longdouble(139) / 630
        mask3 = (~mask0) & (~mask1) & (~mask2) & (ax < 4)
        value = 4 - ax[mask3]
        out[mask3] = value**7 / 5040
        return out
    raise ValueError("equal-LS long-double beta expects degree 3, 5 or 7")


def execute_typed_finite_difference(
    samples: np.ndarray, zoom: float, degree: int, dtype: np.dtype
) -> np.ndarray:
    """Running-sum algorithm in ``dtype`` with endpoint-correct output IIR."""

    p = _params(degree, zoom)
    base = make_plan_1d(len(samples), p)
    typ = np.dtype(dtype).type
    c = np.asarray(samples, dtype=typ).copy()
    _prefilter_typed(c, degree)
    average = _integrate_typed(c, degree + 1)

    # Reuse the exact same extension/index contract as the Python plan.
    ext = np.empty(base.length_total, dtype=typ)
    ext[: c.size] = c
    if base.rp_src.size:
        ext[c.size :] = typ(base.rp_sign) * c[base.rp_src]
    full = np.empty(base.left_pad + base.length_total + base.right_pad, dtype=typ)
    if base.left_pad:
        full[base.lp_dst] = typ(base.lp_sign) * c[base.lp_src]
    full[base.left_pad : base.left_pad + base.length_total] = ext
    if base.right_pad:
        full[base.left_pad + base.length_total :] = ext[-1]
    gathered = full[base.idx2d[: base.outN]]
    if typ is np.longdouble:
        a = typ(base.outN - 1) / typ(len(samples) - 1)
        shift = (typ(degree + 1) / 2 % 1) * (1 / a - 1)
        coordinates = np.arange(base.outN, dtype=typ) / a + shift
        offsets = np.arange(base.win_len_max, dtype=np.int64)
        integer = base.kmin[: base.outN, None].astype(np.longdouble) + offsets
        weight = a ** (degree + 1) * _projection_beta_longdouble(
            coordinates[:, None] - integer, 2 * degree + 1
        )
        weight[offsets[None, :] >= base.win_len[: base.outN, None]] = 0
    else:
        weight = np.asarray(base.weights2d[: base.outN], dtype=typ)
    q = np.sum(
        weight * gathered,
        axis=1,
        dtype=typ,
    )
    _differentiate_typed(q, degree + 1)
    # Differentiation needs a short continuation to compute the last visible
    # samples, but the Gram inverse acts on the visible mirror sequence.  This
    # crop is part of the endpoint-aligned finite-domain contract.
    q = q[: base.outN].copy()
    q += average
    _prefilter_typed(q, 2 * degree + 1)
    _samples_typed(q, degree)
    return q


def execute_longdouble_finite_difference(
    samples: np.ndarray, zoom: float, degree: int
) -> np.ndarray:
    return execute_typed_finite_difference(samples, zoom, degree, np.longdouble)


def execute_deflated_finite_difference(
    samples: np.ndarray,
    zoom: float,
    degree: int,
    direct_plan: DirectLSPlan | None = None,
) -> np.ndarray:
    """Remove low moments, use FD on the residual, restore through direct LS."""

    x = np.asarray(samples, dtype=np.float64)
    coordinate = np.linspace(-1.0, 1.0, x.size)
    vandermonde = np.polynomial.legendre.legvander(coordinate, degree)
    gram = vandermonde.T @ vandermonde
    rhs = vandermonde.T @ x
    trend = vandermonde @ np.linalg.solve(gram, rhs)
    residual = x - trend
    plan = direct_plan or make_direct_ls_plan(x.size, zoom, degree)
    return execute_typed_finite_difference(
        residual, zoom, degree, np.float64
    ) + execute_direct_ls(trend, plan)


# ---------------------------------------------------------------------------
# Arbitrary-precision oracle and finite-difference comparator
# ---------------------------------------------------------------------------


def _mp_poles(degree: int) -> list[mp.mpf]:
    if degree <= 1:
        return []
    if degree == 2:
        return [mp.sqrt(8) - 3]
    if degree == 3:
        return [mp.sqrt(3) - 2]
    if degree == 4:
        return [
            mp.sqrt(664 - mp.sqrt(438976)) + mp.sqrt(304) - 19,
            mp.sqrt(664 + mp.sqrt(438976)) - mp.sqrt(304) - 19,
        ]
    if degree == 5:
        return [
            mp.sqrt(mp.mpf(135) / 2 - mp.sqrt(mp.mpf(17745) / 4))
            + mp.sqrt(mp.mpf(105) / 4)
            - mp.mpf(13) / 2,
            mp.sqrt(mp.mpf(135) / 2 + mp.sqrt(mp.mpf(17745) / 4))
            - mp.sqrt(mp.mpf(105) / 4)
            - mp.mpf(13) / 2,
        ]
    if degree == 7:
        return [
            mp.mpf("-0.5352804307964381655424037816816460718339231523426924148812"),
            mp.mpf("-0.122554615192326690515272264359357343605486549427295558490763"),
            mp.mpf("-0.0091486948096082769285930216516478534156925639545994482648003"),
        ]
    raise ValueError("oracle only needs spline degrees 1, 2, 3, 5 and 7")


def _mp_prefilter(values: list[mp.mpf], degree: int) -> None:
    if degree <= 1 or len(values) <= 1:
        return
    poles = _mp_poles(degree)
    normalization = mp.mpf(1)
    for pole in poles:
        normalization *= (1 - pole) * (1 - 1 / pole)
    for i in range(len(values)):
        values[i] *= normalization
    for pole in poles:
        size = len(values)
        zn = pole ** (size - 1)
        causal = values[0] + zn * values[-1]
        for i in range(1, size - 1):
            causal += (pole**i + pole ** (2 * size - 2 - i)) * values[i]
        values[0] = causal / (1 - zn * zn)
        for i in range(1, size):
            values[i] += pole * values[i - 1]
        values[-1] = (pole * values[-2] + values[-1]) * pole / (pole * pole - 1)
        for i in range(size - 2, -1, -1):
            values[i] = pole * (values[i + 1] - values[i])


def _mp_beta(x: mp.mpf, degree: int) -> mp.mpf:
    """Centered cardinal B-spline through its exact truncated-power form."""

    r = degree + 1
    total = mp.mpf(0)
    shifted = x + mp.mpf(r) / 2
    for k in range(r + 1):
        value = shifted - k
        if value > 0:
            total += (-1) ** k * math.comb(r, k) * value**degree
    return total / math.factorial(degree)


def _mp_cross_kernel(
    x: mp.mpf,
    scale: mp.mpf,
    interp_degree: int,
    analysis_degree: int | None = None,
) -> mp.mpf:
    if analysis_degree is None:
        analysis_degree = interp_degree
    rn = interp_degree + 1
    rm = analysis_degree + 1
    power = interp_degree + analysis_degree + 1
    shifted = x + (mp.mpf(rn) * scale + rm) / 2
    total = mp.mpf(0)
    for i in range(rn + 1):
        for j in range(rm + 1):
            value = shifted - i * scale - j
            if value > 0:
                total += (
                    (-1) ** (i + j)
                    * math.comb(rn, i)
                    * math.comb(rm, j)
                    * value**power
                )
    return total / (math.factorial(power) * scale**interp_degree)


def _mp_samples(values: list[mp.mpf], degree: int) -> None:
    if degree <= 1:
        return
    taps = [[mp.mpf(3) / 4, mp.mpf(1) / 8],
            [mp.mpf(2) / 3, mp.mpf(1) / 6]][degree - 2]
    result: list[mp.mpf] = []
    for i, center in enumerate(values):
        value = taps[0] * center
        for j in range(1, len(taps)):
            value += taps[j] * (
                values[_mirror_index(i - j, len(values))]
                + values[_mirror_index(i + j, len(values))]
            )
        result.append(value)
    values[:] = result


def mp_direct_oracle(
    samples: np.ndarray,
    zoom: float,
    degree: int,
    dps: int = 70,
    *,
    analysis_degree: int | None = None,
    synthesis_degree: int | None = None,
) -> np.ndarray:
    """High-precision direct definition of the endpoint-aligned LS projection."""

    with mp.workdps(dps):
        analysis = degree if analysis_degree is None else analysis_degree
        synthesis = degree if synthesis_degree is None else synthesis_degree
        p = _projection_params(degree, analysis, synthesis, zoom)
        base = make_plan_1d(len(samples), p)
        if base.outN == 1:
            return np.asarray([float(np.mean(samples, dtype=np.float64))])
        scale = mp.mpf(base.outN - 1) / (len(samples) - 1)
        coeff = [mp.mpf(float(v)) for v in samples]
        _mp_prefilter(coeff, degree)
        radius = (mp.mpf(degree + 1) * scale + analysis + 1) / 2
        q: list[mp.mpf] = []
        for l in range(base.outN):
            first = int(mp.ceil((l - radius) / scale))
            last = int(mp.floor((l + radius) / scale))
            value = mp.mpf(0)
            for k in range(first, last + 1):
                value += coeff[_mirror_index(k, len(coeff))] * _mp_cross_kernel(
                    mp.mpf(l) - scale * k, scale, degree, analysis
                )
            q.append(value)
        _mp_prefilter(q, analysis + synthesis + 1)
        _mp_samples(q, synthesis)
        return np.asarray([float(v) for v in q[: base.outN]], dtype=np.float64)


def mp_finite_difference(
    samples: np.ndarray,
    zoom: float,
    degree: int,
    dps: int = 70,
    *,
    analysis_degree: int | None = None,
    synthesis_degree: int | None = None,
    short_difference: bool = False,
) -> np.ndarray:
    """Arbitrary-precision transcription of the running-sum construction."""

    with mp.workdps(dps):
        analysis = degree if analysis_degree is None else analysis_degree
        synthesis = degree if synthesis_degree is None else synthesis_degree
        p = _projection_params(degree, analysis, synthesis, zoom)
        base = make_plan_1d(len(samples), p)
        a = mp.mpf(base.outN - 1) / (len(samples) - 1)
        c = [mp.mpf(float(v)) for v in samples]
        _mp_prefilter(c, degree)

        def mean(values: list[mp.mpf]) -> mp.mpf:
            return (2 * mp.fsum(values) - values[0] - values[-1]) / (2 * len(values) - 2)

        average = mean(c)
        c[0] = (c[0] - average) / 2
        for i in range(1, len(c)):
            c[i] = c[i] - average + c[i - 1]
        integration_count = analysis + 1
        if integration_count >= 2:
            original = c.copy()
            c[0], c[1] = original[0], mp.mpf(0)
            for i in range(2, len(c)):
                c[i] = c[i - 1] - original[i - 1]
        if integration_count >= 3:
            second_mean = mean(c)
            c[0] = (c[0] - second_mean) / 2
            for i in range(1, len(c)):
                c[i] = c[i] - second_mean + c[i - 1]
        if integration_count >= 4:
            original = c.copy()
            c[0], c[1] = original[0], mp.mpf(0)
            for i in range(2, len(c)):
                c[i] = c[i - 1] - original[i - 1]

        extension = c.copy()
        extension.extend(mp.mpf(base.rp_sign) * c[int(i)] for i in base.rp_src)
        shift = (mp.mpf(analysis + 1) / 2 % 1) * (1 / a - 1)
        fact = a ** integration_count
        q: list[mp.mpf] = []
        total_degree = degree + analysis + 1
        half_support = mp.mpf(total_degree + 1) / 2
        resample_size = base.outN if short_difference else base.out_total
        for l in range(resample_size):
            coordinate = l / a + shift
            first = int(mp.ceil(coordinate - half_support))
            last = int(mp.floor(coordinate + half_support))
            value = mp.mpf(0)
            for k in range(first, last + 1):
                if k < 0:
                    if base.symmetric_ext:
                        index, sign = min(-k, len(c) - 1), 1
                    else:
                        index, sign = min(-k - 1, len(c) - 1), -1
                else:
                    index, sign = min(k, len(extension) - 1), 1
                value += sign * extension[index] * fact * _mp_beta(
                    coordinate - k, total_degree
                )
            q.append(value)

        def diff_sa() -> None:
            old = q[-2]
            for i in range(len(q) - 1):
                q[i] -= q[i + 1]
            q[-1] -= old

        def diff_as() -> None:
            for i in range(len(q) - 1, 0, -1):
                q[i] -= q[i - 1]
            q[0] *= 2

        count = integration_count
        if count == 2:
            diff_sa(); diff_as()
        elif count == 3:
            diff_as(); diff_sa(); diff_as()
        elif count == 4:
            diff_sa(); diff_as(); diff_sa(); diff_as()
        else:
            diff_as()
        q = [v + average for v in q[: base.outN]]
        _mp_prefilter(q, analysis + synthesis + 1)
        _mp_samples(q, synthesis)
        return np.asarray([float(v) for v in q[: base.outN]], dtype=np.float64)


def make_pattern(name: str, size: int, seed: int) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, size)
    if name == "ramp":
        return coordinate
    if name == "constant":
        return np.ones(size)
    if name == "low_sine":
        return 0.5 + 0.45 * np.sin(2.0 * np.pi * 3.0 * coordinate + 0.17)
    if name == "random":
        return np.random.default_rng(seed).standard_normal(size)
    if name == "impulse":
        result = np.zeros(size)
        result[size // 2] = 1.0
        return result
    raise ValueError(name)


def _timed(call: Callable[[], np.ndarray], repeats: int) -> tuple[np.ndarray, float]:
    values: list[float] = []
    result = call()
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        values.append((time.perf_counter() - start) * 1000.0)
    return result, statistics.median(values)


def _error(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(candidate, dtype=np.longdouble) - np.asarray(reference, dtype=np.longdouble)
    denominator = max(float(np.linalg.norm(reference)), 1e-300)
    return float(np.linalg.norm(delta) / denominator), float(np.max(np.abs(delta)))


def run_sweep(
    sizes: Iterable[int],
    degrees: Iterable[int],
    patterns: Iterable[str],
    zoom: float,
    repeats: int,
    oracle_max_size: int,
    quad_max_size: int,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for degree in degrees:
        for size in sizes:
            plan_start = time.perf_counter()
            direct_plan = make_direct_ls_plan(size, zoom, degree)
            plan_ms = (time.perf_counter() - plan_start) * 1000.0
            for pattern in patterns:
                samples = make_pattern(pattern, size, seed=1000 * degree + size)
                if size <= oracle_max_size:
                    reference = mp_direct_oracle(samples, zoom, degree)
                    ref_name = "mp_direct"
                    mp_fd = mp_finite_difference(samples, zoom, degree)
                    rel, absolute = _error(mp_fd, reference)
                    rows.append(ResultRow(
                        degree, size, direct_plan.output_size, pattern, "mp_fd", ref_name,
                        float("nan"), 0.0, rel, absolute,
                        float(np.min(mp_fd)), float(np.max(mp_fd)),
                        bool(np.all(np.isfinite(mp_fd))), 0,
                    ))
                else:
                    reference = execute_direct_ls(samples, direct_plan)
                    ref_name = "direct_float64"

                direct, direct_ms = _timed(
                    lambda: execute_direct_ls(samples, direct_plan), repeats
                )
                rel, absolute = _error(direct, reference)
                rows.append(ResultRow(
                    degree, size, direct_plan.output_size, pattern, "direct_float64", ref_name,
                    direct_ms, plan_ms, rel, absolute,
                    float(np.min(direct)), float(np.max(direct)),
                    bool(np.all(np.isfinite(direct))), direct_plan.nnz,
                ))

                fd, fd_ms = _timed(
                    lambda: execute_finite_difference(samples, zoom, degree), repeats
                )
                rel, absolute = _error(fd, reference)
                rows.append(ResultRow(
                    degree, size, direct_plan.output_size, pattern, "fd_float64_current", ref_name,
                    fd_ms, 0.0, rel, absolute,
                    float(np.min(fd)), float(np.max(fd)),
                    bool(np.all(np.isfinite(fd))), 0,
                ))

                fixed_fd, fixed_fd_ms = _timed(
                    lambda: execute_typed_finite_difference(
                        samples, zoom, degree, np.float64
                    ),
                    repeats,
                )
                rel, absolute = _error(fixed_fd, reference)
                rows.append(ResultRow(
                    degree, size, direct_plan.output_size, pattern, "fd_float64_endpoint", ref_name,
                    fixed_fd_ms, 0.0, rel, absolute,
                    float(np.min(fixed_fd)), float(np.max(fixed_fd)),
                    bool(np.all(np.isfinite(fixed_fd))), 0,
                ))

                long_fd, long_ms = _timed(
                    lambda: execute_longdouble_finite_difference(samples, zoom, degree),
                    repeats,
                )
                rel, absolute = _error(long_fd, reference)
                rows.append(ResultRow(
                    degree, size, direct_plan.output_size, pattern, "fd_longdouble", ref_name,
                    long_ms, 0.0, rel, absolute,
                    float(np.min(long_fd)), float(np.max(long_fd)),
                    bool(np.all(np.isfinite(long_fd))), 0,
                ))

                deflated, deflated_ms = _timed(
                    lambda: execute_deflated_finite_difference(
                        samples, zoom, degree, direct_plan
                    ),
                    repeats,
                )
                rel, absolute = _error(deflated, reference)
                rows.append(ResultRow(
                    degree, size, direct_plan.output_size, pattern, "fd_moment_deflated", ref_name,
                    deflated_ms, 0.0, rel, absolute,
                    float(np.min(deflated)), float(np.max(deflated)),
                    bool(np.all(np.isfinite(deflated))), 0,
                ))

                if size <= quad_max_size:
                    start = time.perf_counter()
                    quad = mp_finite_difference(samples, zoom, degree)
                    quad_ms = (time.perf_counter() - start) * 1000.0
                    rel, absolute = _error(quad, reference)
                    rows.append(ResultRow(
                        degree, size, direct_plan.output_size, pattern, "fd_mp70", ref_name,
                        quad_ms, 0.0, rel, absolute,
                        float(np.min(quad)), float(np.max(quad)),
                        bool(np.all(np.isfinite(quad))), 0,
                    ))
                print(
                    f"degree={degree} N={size} pattern={pattern}: "
                    f"direct={direct_ms:.3f} ms fd={fd_ms:.3f} ms "
                    f"fd_rel={_error(fd, reference)[0]:.3e}",
                    flush=True,
                )
    return rows


def print_summary(rows: list[ResultRow]) -> None:
    print("\nWorst error by method and degree")
    print("degree  method                    max rel-L2       max abs")
    for degree in sorted({row.degree for row in rows}):
        methods = sorted({row.method for row in rows if row.degree == degree})
        for method in methods:
            group = [r for r in rows if r.degree == degree and r.method == method]
            print(
                f"{degree:>6}  {method:<24} "
                f"{max(r.rel_l2 for r in group):>12.3e}  "
                f"{max(r.max_abs_error for r in group):>12.3e}"
            )

    print("\nMedian execution time ratio against finite differences")
    for degree in sorted({row.degree for row in rows}):
        ratios = []
        keys = {(r.input_size, r.pattern) for r in rows if r.degree == degree}
        for key in keys:
            direct = next((r for r in rows if r.degree == degree and
                           (r.input_size, r.pattern) == key and
                           r.method == "direct_float64"), None)
            fd = next((r for r in rows if r.degree == degree and
                       (r.input_size, r.pattern) == key and
                       r.method == "fd_float64_endpoint"), None)
            if direct and fd and direct.runtime_ms > 0:
                ratios.append(fd.runtime_ms / direct.runtime_ms)
        if ratios:
            print(
                f"degree {degree}: FD/direct median={statistics.median(ratios):.3f}x "
                f"range=[{min(ratios):.3f}, {max(ratios):.3f}]"
            )


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--sizes", type=parse_ints)
    parser.add_argument("--degrees", type=parse_ints, default=[1, 2, 3])
    parser.add_argument(
        "--patterns",
        type=parse_strings,
        default=["constant", "ramp", "low_sine", "random", "impulse"],
    )
    parser.add_argument("--zoom", type=float, default=0.37)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--oracle-max-size", type=int, default=128)
    parser.add_argument("--quad-max-size", type=int, default=128)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    sizes = args.sizes
    if sizes is None:
        sizes = [32, 128, 4096] if args.profile == "quick" else [32, 128, 4096, 16384, 65536]

    rows = run_sweep(
        sizes=sizes,
        degrees=args.degrees,
        patterns=args.patterns,
        zoom=args.zoom,
        repeats=args.repeats,
        oracle_max_size=args.oracle_max_size,
        quad_max_size=args.quad_max_size,
    )
    print_summary(rows)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
