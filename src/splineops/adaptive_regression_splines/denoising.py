# splineops/src/splineops/adaptive_regression_splines/denoising.py

# Total Variation Denoising with ADMM
# ====================================

# This Python implementation applies total variation (TV) denoising using the Alternating
# Direction Method of Multipliers (ADMM). The method smooths noisy data while preserving
# sharp transitions by solving a convex optimization problem that enforces piecewise smoothness.

# Author: Thomas Debarre
#         Swiss Federal Institute of Technology Lausanne
#         Biomedical Imaging Group
#         BM-Ecublens
#         CH-1015 Lausanne EPFL, Switzerland

# This script provides functions for computing the TV-denoised signal, determining
# the maximum regularization parameter for which linear regression dominates,
# constructing the second-order difference matrix for regularization, and computing
# the L1 proximal operator for sparsity constraints.


from dataclasses import dataclass
from typing import Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class DenoisingDiagnostics:
    """Convergence information for :func:`denoise_y`.

    Closed-form branches (zero regularization and the linear-regression limit)
    report zero iterations and residuals because ADMM is not entered.
    """

    iterations: int
    converged: bool
    primal_residual: float
    dual_residual: float


def denoise_y(
    x: np.ndarray,
    y: np.ndarray,
    lamb: float,
    rho: float = 1.0,
    max_iter: int = int(1e4),
    relative_tol: float = 1e-7,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, DenoisingDiagnostics]:
    """
    Performs total variation denoising on the y-coordinates using ADMM.

    This function solves a convex optimization problem to smooth noisy data while
    preserving sharp transitions. The method is based on the Alternating Direction Method
    of Multipliers (ADMM), a powerful approach for distributed optimization.

    The optimization problem solved is:

        minimize  || y - y_lambda ||_2^2 + λ || Dy ||_1

    where `D` is a discrete difference operator, enforcing piecewise smoothness.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates of data points.
    y : ndarray
        Array of y-coordinates (possibly noisy).
    lamb : float
        Regularization parameter controlling the trade-off between data fidelity and smoothness.
    rho : float, optional
        ADMM penalty parameter (default is 1.0).
    max_iter : int, optional
        Maximum number of ADMM iterations (default is 1e4).
    relative_tol : float, optional
        Tolerance for stopping criterion (default is 1e-7).
    return_diagnostics : bool, optional
        If true, return ``(y_denoised, diagnostics)``.  This opt-in form keeps
        the historical array-only return value as the default.

    Returns
    -------
    y_lambda : ndarray or tuple
        Array of denoised y-coordinates, optionally paired with convergence
        diagnostics.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("'x' and 'y' must be one-dimensional arrays.")
    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")
    if x.size < 3:
        raise ValueError("At least three samples are required for TV denoising.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("'x' and 'y' must contain only finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("'x' must be strictly increasing.")
    if not np.isfinite(lamb) or lamb < 0:
        raise ValueError("'lamb' must be finite and non-negative.")
    if lamb > 0 and (not np.isfinite(rho) or rho <= 0):
        raise ValueError("'rho' must be finite and positive when 'lamb' is positive.")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or int(max_iter) != max_iter
        or max_iter <= 0
    ):
        raise ValueError("'max_iter' must be a positive integer.")
    if not np.isfinite(relative_tol) or relative_tol <= 0:
        raise ValueError("'relative_tol' must be finite and positive.")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise TypeError("'return_diagnostics' must be a boolean.")
    lamb_max, polynomial = _lambda_max(x, y)
    diagnostics = DenoisingDiagnostics(0, True, 0.0, 0.0)
    if lamb == 0:
        # Zero regularization is the identity, including for exactly linear
        # input where ``lamb_max`` is also zero up to floating-point noise.
        y_denoised = y.copy()
    elif lamb >= lamb_max:
        # If lamb is too high, the problem amounts to linear regression
        y_denoised = polynomial[0] * np.ones_like(x) + polynomial[1] * x
    else:
        # Otherwise, solve denoising problem using ADMM
        # Define matrices
        L = _regularization_matrix(x, fmt="csc")  # build as CSC directly
        Lt = L.transpose().tocsr()  # transpose -> CSR
        A = Lt @ L  # CSR @ CSC -> sparse
        M = sp.eye(len(x), format="csc") + rho * A.tocsc()
        # ADMM initialization
        xk, zk, yk = y, L @ y, np.zeros(L.shape[0])
        # Run ADMM
        converged = False
        primal_res = np.inf
        dual_res = np.inf
        for it in range(max_iter):
            z_prev = zk  # Needed for stopping criterion
            # ADMM updates (notations follow Boyd et al. 2011)
            b = y + rho * Lt @ (zk - yk / rho)
            xk = spla.spsolve(M, b)
            Lxk = L @ xk  # Precomputation
            zk = _prox_L1(Lxk + yk / rho, lamb / rho)
            yk += rho * (Lxk - zk)

            # ADMM stopping criterion (as suggested by Boyd et al. 2011)
            primal_res = np.linalg.norm(Lxk - zk)
            dual_res = rho * np.linalg.norm(Lt @ (zk - z_prev))
            primal_eps = relative_tol * max(np.linalg.norm(Lxk), np.linalg.norm(zk))
            dual_eps = relative_tol * np.linalg.norm(Lt @ yk)
            if (primal_res <= primal_eps) and (dual_res <= dual_eps):
                converged = True
                break
        y_denoised = xk
        diagnostics = DenoisingDiagnostics(
            iterations=it + 1,
            converged=converged,
            primal_residual=float(primal_res),
            dual_residual=float(dual_res),
        )
    if return_diagnostics:
        return y_denoised, diagnostics
    return y_denoised


def _lambda_max(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the maximum regularization parameter lambda.

    If lambda exceeds this value, the denoising problem reduces to simple
    linear regression.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates.
    y : ndarray
        Array of y-coordinates.

    Returns
    -------
    lamb_max : float
        Maximum lambda value before regression takes over.
    polynomial : ndarray
        Coefficients (b, a) of the optimal linear regression.
    """

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")
    m = len(x)

    # Compute parameters of polynomial (solution of a 2x2 system)
    det = m * np.sum(x**2) - np.sum(x) ** 2
    polynomial = (1 / det) * np.array(
        [[np.sum(x**2), -np.sum(x)], [-np.sum(x), m]]
    ).dot(np.array([np.sum(y), np.dot(x, y)]))

    h = y - (polynomial[0] * np.ones_like(x) + polynomial[1] * x)
    lamb_max = max(np.abs(x[1:-1] * np.cumsum(h)[:-2] - np.cumsum(h * x)[:-2]))
    return lamb_max, polynomial


def _regularization_matrix(x: np.ndarray, fmt: str = "csc") -> sp.sparray:
    """
    Constructs the second-order difference matrix for total variation regularization.

    This matrix enforces smoothness constraints by penalizing large variations
    in adjacent y-values.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates.

    Returns
    -------
    L : scipy.sparse.diags
        Regularization matrix enforcing smoothness constraints.
    """

    M = len(x)
    v = 1 / (x[1:] - x[:-1])
    return sp.diags(
        [v[:-1], -(v[:-1] + v[1:]), v[1:]], [0, 1, 2], shape=(M - 2, M), format=fmt
    )


def _prox_L1(x: np.ndarray, sigma: float):
    """
    Computes the proximal operator of the L1 norm.

    This function applies soft thresholding, which is the key step in total variation
    denoising.

    Parameters
    ----------
    x : ndarray
        Input vector.
    sigma : float
        Regularization parameter.

    Returns
    -------
    prox : ndarray
        Soft-thresholded output.
    """

    return np.sign(x) * np.maximum(np.abs(x) - sigma, 0)
