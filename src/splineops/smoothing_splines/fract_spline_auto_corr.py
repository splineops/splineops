# splineops/src/splineops/smoothing_splines/fract_spline_auto_corr.py

import numpy as np
import numpy.typing as npt

_SUM_CHUNK_SIZE = 32


def fractsplineautocorr(alpha: float, nu: npt.NDArray) -> npt.NDArray:
    """
    Compute the frequency response of the autocorrelation filter
    of a fractional spline of degree `alpha`.

    It uses an acceleration technique to improve the convergence of the infinite
    sum by four orders.

    Parameters
    ----------
    alpha : float
        Fractional degree parameter (must be > -0.5).
    nu : ndarray
        Frequency values (in cycles per sample).

    Returns
    -------
    A : ndarray
        Frequency response of the autocorrelation filter. Its length matches
        that of `nu`.

    Notes
    -----
    This function sums contributions from -N to +N (with N=100) and applies
    an acceleration correction (`U`) to improve the sum convergence.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.smoothing_splines.fract_spline_auto_corr import fractsplineautocorr
    >>> alpha = 0.5
    >>> nu = np.linspace(-0.5, 0.5, 5)
    >>> A = fractsplineautocorr(alpha, nu)
    >>> A
    array([...])

    """
    N = 100  # Number of terms in the summation

    if not np.isfinite(alpha) or alpha <= -0.5:
        raise ValueError("'alpha' must be finite and strictly greater than -0.5.")
    nu = np.asarray(nu)
    if nu.ndim != 1:
        raise ValueError("'nu' must be a one-dimensional array.")
    if not np.all(np.isfinite(nu)):
        raise ValueError("'nu' must contain only finite values.")

    # Sum several shifts at once.  The bounded chunk keeps temporary memory
    # independent of the 201-term truncation while avoiding 201 Python-level
    # full-array operations.
    S = np.zeros(len(nu))
    shifts = np.arange(-N, N + 1)
    for start in range(0, shifts.size, _SUM_CHUNK_SIZE):
        chunk = shifts[start : start + _SUM_CHUNK_SIZE, np.newaxis]
        # np.sinc(x) = sin(pi*x)/(pi*x) in NumPy
        S += np.sum(
            np.abs(np.sinc(nu[np.newaxis, :] + chunk)) ** (2 * alpha + 2), axis=0
        )

    # Acceleration term U
    U = 2 / ((2 * alpha + 1) * N ** (2 * alpha + 1))
    U -= 1 / N ** (2 * alpha + 2)
    U += (alpha + 1) * (1 / 3 + 2 * nu**2) / N ** (2 * alpha + 3)
    U -= (alpha + 1) * (2 * alpha + 3) * nu**2 / N ** (2 * alpha + 4)
    U *= np.abs(np.sin(np.pi * nu) / np.pi) ** (2 * alpha + 2)

    A = S + U
    return A
