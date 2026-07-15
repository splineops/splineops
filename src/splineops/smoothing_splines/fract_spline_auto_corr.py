# splineops/src/splineops/smoothing_splines/fract_spline_auto_corr.py

import numpy as np
import numpy.typing as npt


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

    # Initialize sum
    S = np.zeros(len(nu))
    for n in range(-N, N + 1):
        # np.sinc(x) = sin(pi*x)/(pi*x) in NumPy
        S += np.abs(np.sinc(nu + n)) ** (2 * alpha + 2)

    # Acceleration term U
    U = 2 / ((2 * alpha + 1) * N ** (2 * alpha + 1))
    U -= 1 / N ** (2 * alpha + 2)
    U += (alpha + 1) * (1 / 3 + 2 * nu**2) / N ** (2 * alpha + 3)
    U -= (alpha + 1) * (2 * alpha + 3) * nu**2 / N ** (2 * alpha + 4)
    U *= np.abs(np.sin(np.pi * nu) / np.pi) ** (2 * alpha + 2)

    A = S + U
    return A
