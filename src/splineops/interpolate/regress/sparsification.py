from typing import Tuple
import numpy as np

def sparsest_interpolant(
    x: np.ndarray, 
    y: np.ndarray, 
    sparsity_tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Algorithm 1 in [1]_ to compute the linear spline that connects given data points with the fewest knots.

    The optimal spline can be evaluated using the `linear_spline()` function with the outputs of this function.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates of data points
    y : ndarray
        Array of y-coordinates of data points
    sparsity_tol : float
        Knots whose amplitude is smaller in absolute value than this tolerance parameter are discarded.

    Returns
    -------
    knots : ndarray
        Array of knots of the optimal spline
    amplitudes : ndarray
        Array of amplitudes of these knots
    polynomial : ndarray
        Size-2 array (b, a) parametrizing the linear component p(t) = at + b of the optimal spline
    """
    if x.size != y.size:
        raise Exception("x and y must be of the same size")

    knots = x[1:-1]
    amplitudes_cano, polynomial_cano = _connect_points(x, y)

    # Identify phantom knots (amplitude = 0) which are outside of saturation zones
    saturations = _saturation_zones(amplitudes_cano, sparsity_tol)
    pruned_bool = np.logical_or(saturations != 0, np.abs(amplitudes_cano) > sparsity_tol)
    # Remove these phantom knots
    knots_pruned = knots[pruned_bool]
    amplitudes_pruned = amplitudes_cano[pruned_bool]
    saturations_pruned = saturations[pruned_bool]
    #amplitudes_pruned = sparsify_amplitudes(amplitudes_pruned, sparsity_tol)  # Set knots below tolerance to exactly zero

    # Sparsification of saturation zones
    amplitudes_sparsest = np.array([])
    knots_sparsest = np.array([])
    i = 0
    last_nz_idx = 0
    num_saturations = 0  # Number of consecutive saturation intervals after knots_pruned[i]
    while i < len(knots_pruned):
        if saturations_pruned[i] != 0:
            num_saturations = saturations_pruned[i]
        for j in range(int(np.ceil(num_saturations / 2))):
            amplitudes_sparsest = np.append(amplitudes_sparsest, amplitudes_pruned[i+2*j] + amplitudes_pruned[i+2*j+1])
            knots_sparsest = np.append(knots_sparsest, (amplitudes_pruned[i+2*j] * knots_pruned[i+2*j] +
                                   amplitudes_pruned[i+2*j+1] * knots_pruned[i+2*j+1]) / amplitudes_sparsest[-1])
        if (num_saturations % 2) == 0:
            # Keep last existing knot if even number of saturations (including 0)
            amplitudes_sparsest = np.append(amplitudes_sparsest, amplitudes_pruned[i+num_saturations])
            knots_sparsest = np.append(knots_sparsest, knots_pruned[i+num_saturations])

        i += num_saturations + 1
        last_nz_idx += 1
        num_saturations = 0

    return knots_sparsest, amplitudes_sparsest, polynomial_cano

def linear_spline(
    t: np.ndarray, 
    knots: np.ndarray, 
    amplitudes: np.ndarray, 
    polynomial: np.ndarray
) -> np.ndarray:
    """
    Evaluate a parametrized linear spline at location(s) t.

    The mathematical expression of the spline is :math:`s(t) = at + b + \\sum_{k=0}^{K} a_k (t - \\tau_k)_+`, where
    a and b are the parameters of the linear component and a_k and \\tau_k are the amplitudes and the locations of the
    knots, respectively.

    Parameters
    ----------
    t : float or ndarray
        Evaluation point(s) of the spline
    knots : ndarray
        Array of knots of the linear spline
    amplitudes : ndarray
        Array of amplitudes of these knots
    polynomial : ndarray
        Size-2 array (b, a) parametrizing the linear component p(t) = at + b of the spline

    Returns
    -------
    values : float or ndarray
        Value(s) of the spline at the evaluation point(s)
    """
    values = polynomial[0] + polynomial[1] * t
    for i in range(len(knots)):
        values = values + amplitudes[i] * (t - knots[i]) * ((t - knots[i]) > 0)
    return values

def _sparsify_amplitudes(
    amplitudes: np.ndarray, 
    sparsity_tol: float = 1e-5
) -> np.ndarray:
    """ 
    Adjust amplitudes by setting those below the tolerance to zero while conserving the linear spline exactly
    outside areas with phantom knots. 
    """
    zero_indices = np.nonzero(np.abs(amplitudes) <= sparsity_tol)
    amplitudes_sparsified = amplitudes
    amplitudes_sparsified[zero_indices] = 0  # Set knots below tolerance to zero
    i = 0
    while i < len(zero_indices):
        # Compensate close to zero amplitudes on previous knot
        amplitudes_sparsified[zero_indices[i]-1] += amplitudes_sparsified[zero_indices[i]]
        if i == len(zero_indices) - 1:
            break
        j = 0
        # If consecutive phantom knots, compensate all of them on the closest previous true knot
        while i + j + 1 < len(zero_indices) and zero_indices[i+j+1] == zero_indices[i+j] + 1:
            amplitudes_sparsified[zero_indices[i]-1] += amplitudes_sparsified[zero_indices[i+j+1]]
            j += 1
        i += j + 1
    return amplitudes_sparsified

def _saturation_zones(
    amplitudes: np.ndarray, 
    sparsity_tol: float = 1e-5
) -> np.ndarray:
    """ 
    Return array specifying number of consecutive saturation zones (limit points included). 
    """
    saturations = np.zeros_like(amplitudes)
    nz_idx = np.nonzero(np.abs(amplitudes) > sparsity_tol)[0]
    if len(nz_idx) > 0:
        sat_idx_start = nz_idx[0]
        for i in range(len(nz_idx)-1):
            if np.sign(amplitudes[nz_idx[i]]) != np.sign(amplitudes[nz_idx[i+1]]):
                saturations[sat_idx_start:nz_idx[i]+1] = nz_idx[i] - sat_idx_start
                sat_idx_start = nz_idx[i+1]
        saturations[sat_idx_start:nz_idx[-1]+1] = nz_idx[-1] - sat_idx_start

    return saturations.astype(int)

def _connect_points(
    x: np.ndarray, 
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Return parametrization of the canonical linear-spline solution that connects the data points (x[i], y[i]). 
    """
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    polynomial = np.array([y[0] - slopes[0] * x[0], slopes[0]])
    amplitudes = slopes[1:] - slopes[:-1]
    return amplitudes, polynomial
