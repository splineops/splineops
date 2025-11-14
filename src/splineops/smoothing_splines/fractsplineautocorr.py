# splineops/src/splineops/smoothing_splines/fractsplineautocorr.py

from typing import Union
import numpy as np
import numpy.typing as npt

def fractsplineautocorr(
    alpha: float,
    nu: npt.NDArray
) -> npt.NDArray:
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
        that of `nu`. If `alpha <= -0.5`, an empty array is returned (and a
        warning is printed).

    Notes
    -----
    This function sums contributions from -N to +N (with N=100) and applies
    an acceleration correction (`U`) to improve the sum convergence.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.interpolate.smooth.fractsplineautocorr import fractsplineautocorr
    >>> alpha = 0.5
    >>> nu = np.linspace(-0.5, 0.5, 5)
    >>> A = fractsplineautocorr(alpha, nu)
    >>> A
    array([...])

    """
    N = 100  # Number of terms in the summation

    if alpha <= -0.5:
        print("The autocorrelation of the fractional splines exists only for "
              "degrees strictly larger than -0.5!")
        # Return an empty array to keep the same ndarray return type
        return np.array([])

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
