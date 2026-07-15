# splineops/src/splineops/adaptive_regression_splines/sparsification.py

# Sparsest Piecewise-Linear Interpolation
# =======================================

# This Python implementation computes the sparsest piecewise-linear spline that interpolates
# given data points using total variation regularization on the second derivative. This method
# promotes solutions with the fewest number of knots while maintaining fidelity to the data.

# Author: Thomas Debarre
#         Swiss Federal Institute of Technology Lausanne
#         Biomedical Imaging Group
#         BM-Ecublens
#         CH-1015 Lausanne EPFL, Switzerland

# This script provides functionality for computing optimal sparse splines, evaluating them,
# and performing operations such as sparsification of amplitudes and identification of
# saturation zones.

from typing import Tuple
import numpy as np


def sparsest_interpolant(
    x: np.ndarray, y: np.ndarray, sparsity_tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the sparsest piecewise-linear spline that interpolates the given data points.

    This function implements a method for finding the sparsest linear spline, based on
    total variation regularization on the second derivative. This approach promotes
    solutions with the fewest number of knots while maintaining fidelity to the data.

    The optimal spline can be evaluated using the `linear_spline()` function with the outputs of this function.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates of data points.
    y : ndarray
        Array of y-coordinates of data points.
    sparsity_tol : float, optional
        Threshold for eliminating knots with small amplitude (default is 1e-5).

    Returns
    -------
    knots : ndarray
        Array of knot locations for the optimal sparse spline.
    amplitudes : ndarray
        Corresponding amplitudes of the knots.
    polynomial : ndarray
        Coefficients (b, a) of the linear component p(t) = at + b.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("'x' and 'y' must be one-dimensional arrays.")
    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")
    if x.size < 2:
        raise ValueError("At least two samples are required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("'x' and 'y' must contain only finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("'x' must be strictly increasing.")
    if not np.isfinite(sparsity_tol) or sparsity_tol < 0:
        raise ValueError("'sparsity_tol' must be finite and non-negative.")

    knots = x[1:-1]
    amplitudes_cano, polynomial_cano = _connect_points(x, y)
    amplitudes_cano = _sparsify_amplitudes(
        amplitudes_cano, sparsity_tol
    )  # Set knots below tolerance to exactly zero

    # Identify phantom knots (amplitude = 0) which are outside of saturation zones
    saturations = _saturation_zones(amplitudes_cano, sparsity_tol)
    pruned_bool = np.logical_or(
        saturations != 0, np.abs(amplitudes_cano) > sparsity_tol
    )
    # Remove these phantom knots
    knots_pruned = knots[pruned_bool]
    amplitudes_pruned = amplitudes_cano[pruned_bool]
    saturations_pruned = saturations[pruned_bool]

    # Sparsification of saturation zones
    amplitudes_sparsest = []
    knots_sparsest = []
    i = 0
    last_nz_idx = 0
    num_saturations = (
        0  # Number of consecutive saturation intervals after knots_pruned[i]
    )
    while i < len(knots_pruned):
        if saturations_pruned[i] != 0:
            num_saturations = saturations_pruned[i]
        for j in range(int(np.ceil(num_saturations / 2))):
            new_amp = amplitudes_pruned[i + 2 * j] + amplitudes_pruned[i + 2 * j + 1]
            if new_amp != 0:
                amplitudes_sparsest.append(new_amp)
                barycenter = (
                    amplitudes_pruned[i + 2 * j] * knots_pruned[i + 2 * j]
                    + amplitudes_pruned[i + 2 * j + 1] * knots_pruned[i + 2 * j + 1]
                ) / new_amp
                knots_sparsest.append(barycenter)
        if (num_saturations % 2) == 0:
            # Keep last existing knot if even number of saturations (including 0)
            amplitudes_sparsest.append(amplitudes_pruned[i + num_saturations])
            knots_sparsest.append(knots_pruned[i + num_saturations])

        i += num_saturations + 1
        last_nz_idx += 1
        num_saturations = 0

    knots_array = np.asarray(knots_sparsest)
    amplitudes_array = np.asarray(amplitudes_sparsest)
    indices = np.argsort(knots_array)

    return knots_array[indices], amplitudes_array[indices], polynomial_cano


def linear_spline(
    t: np.ndarray, knots: np.ndarray, amplitudes: np.ndarray, polynomial: np.ndarray
) -> np.ndarray:
    """
    Evaluates a parametrized linear spline at specified location(s) t.

    The spline is represented as:

        s(t) = at + b + sum_{k=0}^{K} a_k (t - τ_k)_+

    where `a` and `b` are the parameters of the linear component, and `a_k` and `τ_k`
    are the amplitudes and locations of the knots, respectively.

    Parameters
    ----------
    t : float or ndarray
        The location(s) where the spline should be evaluated.
    knots : ndarray
        Knot locations of the spline.
    amplitudes : ndarray
        Amplitudes of the knots.
    polynomial : ndarray
        Coefficients (b, a) of the linear component.

    Returns
    -------
    values : float or ndarray
        The evaluated spline values at `t`.
    """

    t = np.asarray(t)
    knots = np.asarray(knots)
    amplitudes = np.asarray(amplitudes)
    polynomial = np.asarray(polynomial)
    if knots.ndim != 1 or amplitudes.ndim != 1:
        raise ValueError("'knots' and 'amplitudes' must be one-dimensional arrays.")
    if knots.size != amplitudes.size:
        raise ValueError("'knots' and 'amplitudes' must have the same size.")
    if polynomial.shape != (2,):
        raise ValueError("'polynomial' must contain exactly two coefficients.")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("'knots' and 'amplitudes' must contain only finite values.")
    if not np.all(np.isfinite(polynomial)) or not np.all(np.isfinite(t)):
        raise ValueError("'t' and 'polynomial' must contain only finite values.")
    if not np.all(np.diff(knots) >= 0):
        raise ValueError("'knots' must be sorted in non-decreasing order.")

    dtype = np.result_type(t, knots, amplitudes, polynomial)
    t = t.astype(dtype, copy=False)
    prefix_amplitudes = np.concatenate(
        (np.zeros(1, dtype=dtype), np.cumsum(amplitudes, dtype=dtype))
    )
    prefix_moments = np.concatenate(
        (np.zeros(1, dtype=dtype), np.cumsum(amplitudes * knots, dtype=dtype))
    )
    active = np.searchsorted(knots, t, side="left")
    return (
        polynomial[0]
        + polynomial[1] * t
        + t * prefix_amplitudes[active]
        - prefix_moments[active]
    )


def _sparsify_amplitudes(
    amplitudes: np.ndarray, sparsity_tol: float = 1e-5
) -> np.ndarray:
    """
    Adjusts amplitudes by setting values below the threshold to zero.

    This operation ensures that the linear spline remains unchanged outside
    regions containing phantom knots.

    Parameters
    ----------
    amplitudes : ndarray
        Array of knot amplitudes.
    sparsity_tol : float, optional
        Threshold below which amplitudes are set to zero (default is 1e-5).

    Returns
    -------
    amplitudes_sparsified : ndarray
        Modified amplitudes after thresholding.
    """

    amplitudes_sparsified = np.array(amplitudes, copy=True)
    zero_indices = np.flatnonzero(np.abs(amplitudes_sparsified) <= sparsity_tol)
    if zero_indices.size == 0:
        return amplitudes_sparsified

    # Collapse each consecutive phantom-knot run into an adjacent retained
    # amplitude before zeroing it. This preserves the far-field slope change
    # while never mutating the caller's array.
    run_starts = np.r_[0, np.flatnonzero(np.diff(zero_indices) != 1) + 1]
    run_stops = np.r_[run_starts[1:], zero_indices.size]
    for start_pos, stop_pos in zip(run_starts, run_stops):
        run = zero_indices[start_pos:stop_pos]
        correction = np.sum(amplitudes_sparsified[run])
        left = int(run[0]) - 1
        right = int(run[-1]) + 1
        if left >= 0:
            amplitudes_sparsified[left] += correction
        elif right < amplitudes_sparsified.size:
            amplitudes_sparsified[right] += correction
        amplitudes_sparsified[run] = 0
    return amplitudes_sparsified


def _saturation_zones(amplitudes: np.ndarray, sparsity_tol: float = 1e-5) -> np.ndarray:
    """
    Identifies saturation zones in the sequence of amplitudes.

    Saturation zones correspond to consecutive segments where the amplitudes
    are small and should be pruned to maintain sparsity.

    Parameters
    ----------
    amplitudes : ndarray
        Array of knot amplitudes.
    sparsity_tol : float, optional
        Threshold for detecting phantom knots (default is 1e-5).

    Returns
    -------
    saturations : ndarray
        Array indicating the number of consecutive saturation zones.
    """

    saturations = np.zeros_like(amplitudes)
    nz_idx = np.nonzero(np.abs(amplitudes) > sparsity_tol)[0]
    if len(nz_idx) > 0:
        sat_idx_start = nz_idx[0]
        for i in range(len(nz_idx) - 1):
            if np.sign(amplitudes[nz_idx[i]]) != np.sign(amplitudes[nz_idx[i + 1]]):
                saturations[sat_idx_start : nz_idx[i] + 1] = nz_idx[i] - sat_idx_start
                sat_idx_start = nz_idx[i + 1]
        saturations[sat_idx_start : nz_idx[-1] + 1] = nz_idx[-1] - sat_idx_start

    return saturations.astype(int)


def _connect_points(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the canonical linear spline solution that connects given data points.

    This function determines the piecewise-linear spline that interpolates the
    given points with minimal complexity.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates.
    y : ndarray
        Array of y-coordinates.

    Returns
    -------
    amplitudes : ndarray
        Amplitudes of the knots.
    polynomial : ndarray
        Coefficients (b, a) of the linear component.
    """

    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    polynomial = np.array([y[0] - slopes[0] * x[0], slopes[0]])
    amplitudes = slopes[1:] - slopes[:-1]
    return amplitudes, polynomial
