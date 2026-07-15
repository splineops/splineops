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
from typing import Iterable, Tuple
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


class DenoisingPlan:
    """Reusable factorization for TV denoising on fixed sample locations.

    The expensive sparse factorization depends on ``x`` and ``rho``, but not
    on the observations or regularization strength.  A plan is therefore most
    useful for denoising many signals sampled at the same locations, or for a
    regularization-parameter sweep.

    Parameters
    ----------
    x : ndarray
        Strictly increasing one-dimensional sample locations.
    rho : float, optional
        Positive ADMM penalty parameter.
    """

    def __init__(self, x: np.ndarray, rho: float = 1.0) -> None:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("'x' must be a one-dimensional array.")
        if x.size < 3:
            raise ValueError("At least three samples are required for TV denoising.")
        if not np.all(np.isfinite(x)):
            raise ValueError("'x' must contain only finite values.")
        if not np.all(np.diff(x) > 0):
            raise ValueError("'x' must be strictly increasing.")
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError("'rho' must be finite and positive.")

        self.x = np.array(x, copy=True)
        self.x.flags.writeable = False
        self.rho = float(rho)
        self._regularizer = _regularization_matrix(self.x, fmt="csc")
        self._regularizer_transpose = self._regularizer.transpose().tocsr()
        normal_matrix = self._regularizer_transpose @ self._regularizer
        system = sp.eye(self.x.size, format="csc") + self.rho * normal_matrix.tocsc()
        self._solve_system = spla.factorized(system)

    @property
    def retained_array_bytes(self) -> int:
        """Known bytes retained by sample and sparse-matrix arrays.

        The sparse solver owns additional implementation-dependent factor
        storage that SciPy does not expose through a stable byte-count API.
        """

        def sparse_bytes(matrix):
            return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

        return (
            self.x.nbytes
            + sparse_bytes(self._regularizer)
            + sparse_bytes(self._regularizer_transpose)
        )

    @property
    def configuration(self) -> dict[str, object]:
        """Copy of the fixed factorization contract."""

        return {"sample_count": self.x.size, "rho": self.rho}

    def solve(
        self,
        y: np.ndarray,
        lamb: float,
        *,
        max_iter: int = int(1e4),
        relative_tol: float = 1e-7,
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, DenoisingDiagnostics]:
        """Denoise one signal while reusing this plan's factorization."""

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("'y' must be a one-dimensional array.")
        if y.size != self.x.size:
            raise ValueError("'x' and 'y' must have the same size.")
        if not np.all(np.isfinite(y)):
            raise ValueError("'y' must contain only finite values.")
        _validate_solver_parameters(lamb, max_iter, relative_tol, return_diagnostics)

        lamb_max, polynomial = _lambda_max(self.x, y)
        diagnostics = DenoisingDiagnostics(0, True, 0.0, 0.0)
        if lamb == 0:
            result = y.copy()
        elif lamb >= lamb_max:
            result = polynomial[0] * np.ones_like(self.x) + polynomial[1] * self.x
        else:
            result, diagnostics = self._solve_admm(
                y,
                float(lamb),
                max_iter=int(max_iter),
                relative_tol=float(relative_tol),
            )
        if return_diagnostics:
            return result, diagnostics
        return result

    def solve_path(
        self,
        y: np.ndarray,
        lambdas: Iterable[float],
        *,
        max_iter: int = int(1e4),
        relative_tol: float = 1e-7,
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, tuple[DenoisingDiagnostics, ...]]:
        """Solve an explicit regularization path with controlled warm starts.

        Solutions are returned in the caller's lambda order.  ADMM state is
        carried only within this method call, so the plan remains free of
        hidden mutable warm-start state and independent calls are repeatable.
        """

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("'y' must be a one-dimensional array.")
        if y.size != self.x.size:
            raise ValueError("'x' and 'y' must have the same size.")
        if not np.all(np.isfinite(y)):
            raise ValueError("'y' must contain only finite values.")
        try:
            lambda_values = np.asarray(tuple(lambdas), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("'lambdas' must be a sequence of real values.") from exc
        if lambda_values.ndim != 1 or lambda_values.size == 0:
            raise ValueError("'lambdas' must be a non-empty one-dimensional sequence.")
        if not np.all(np.isfinite(lambda_values)) or np.any(lambda_values < 0):
            raise ValueError("'lambdas' must contain finite non-negative values.")
        _validate_solver_parameters(
            float(lambda_values[0]), max_iter, relative_tol, return_diagnostics
        )

        lamb_max, polynomial = _lambda_max(self.x, y)
        signal = np.result_type(y.dtype, np.float64).type(1) * y
        state = None
        solutions = []
        diagnostics_sequence = []
        for lamb in lambda_values:
            diagnostics = DenoisingDiagnostics(0, True, 0.0, 0.0)
            if lamb == 0:
                result = signal.copy()
                state = self._state_from_solution(result)
            elif lamb >= lamb_max:
                result = polynomial[0] + polynomial[1] * self.x
                state = self._state_from_solution(result)
            else:
                result, diagnostics, state = self._solve_admm_stateful(
                    signal,
                    float(lamb),
                    max_iter=int(max_iter),
                    relative_tol=float(relative_tol),
                    initial_state=state,
                )
            solutions.append(result)
            diagnostics_sequence.append(diagnostics)
        stacked = np.stack(solutions, axis=0)
        if return_diagnostics:
            return stacked, tuple(diagnostics_sequence)
        return stacked

    def _solve_admm(
        self,
        y: np.ndarray,
        lamb: float,
        *,
        max_iter: int,
        relative_tol: float,
    ) -> tuple[np.ndarray, DenoisingDiagnostics]:
        result, diagnostics, _ = self._solve_admm_stateful(
            y,
            lamb,
            max_iter=max_iter,
            relative_tol=relative_tol,
            initial_state=None,
        )
        return result, diagnostics

    def _state_from_solution(self, solution):
        regularized = self._regularizer @ solution
        return (solution.copy(), regularized, np.zeros_like(regularized))

    def _solve_admm_stateful(
        self,
        y: np.ndarray,
        lamb: float,
        *,
        max_iter: int,
        relative_tol: float,
        initial_state,
    ):
        regularizer = self._regularizer
        regularizer_transpose = self._regularizer_transpose
        rho = self.rho
        signal = np.result_type(y.dtype, np.float64).type(1) * y
        if initial_state is None:
            xk = signal
            zk = regularizer @ signal
            yk = np.zeros(regularizer.shape[0], dtype=signal.dtype)
        else:
            xk, zk, yk = (np.array(value, copy=True) for value in initial_state)
        converged = False
        primal_residual = np.inf
        dual_residual = np.inf

        for iteration in range(max_iter):
            previous_z = zk
            right_hand_side = signal + rho * regularizer_transpose @ (zk - yk / rho)
            xk = self._solve_system(right_hand_side)
            regularized_x = regularizer @ xk
            zk = _prox_L1(regularized_x + yk / rho, lamb / rho)
            yk += rho * (regularized_x - zk)

            primal_residual = np.linalg.norm(regularized_x - zk)
            dual_residual = rho * np.linalg.norm(
                regularizer_transpose @ (zk - previous_z)
            )
            primal_epsilon = relative_tol * max(
                np.linalg.norm(regularized_x), np.linalg.norm(zk)
            )
            dual_epsilon = relative_tol * np.linalg.norm(regularizer_transpose @ yk)
            if primal_residual <= primal_epsilon and dual_residual <= dual_epsilon:
                converged = True
                break

        diagnostics = DenoisingDiagnostics(
            iterations=iteration + 1,
            converged=converged,
            primal_residual=float(primal_residual),
            dual_residual=float(dual_residual),
        )
        return xk, diagnostics, (xk.copy(), zk.copy(), yk.copy())


def _validate_solver_parameters(
    lamb: float,
    max_iter: int,
    relative_tol: float,
    return_diagnostics: bool,
) -> None:
    if not np.isfinite(lamb) or lamb < 0:
        raise ValueError("'lamb' must be finite and non-negative.")
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
    _validate_solver_parameters(lamb, max_iter, relative_tol, return_diagnostics)
    if lamb > 0 and (not np.isfinite(rho) or rho <= 0):
        raise ValueError("'rho' must be finite and positive when 'lamb' is positive.")
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
        plan = DenoisingPlan(x, rho=rho)
        y_denoised, diagnostics = plan._solve_admm(
            y,
            float(lamb),
            max_iter=int(max_iter),
            relative_tol=float(relative_tol),
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
