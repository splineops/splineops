# splineops/src/splineops/smoothing_splines/fBmper.py

from typing import Tuple
import numpy as np
import numpy.typing as npt
from splineops.smoothing_splines.fractsplineautocorr import fractsplineautocorr

def fBmper(
    epsH: float,
    H: float,
    m: int,
    N: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Generate a fractional (pseudo-)Brownian motion signal (fBM).

    This function synthesizes a random signal of length `m * N` in the frequency
    domain and then transforms it back to the time domain to produce fractional
    Brownian motion with Hurst parameter `H`.

    Parameters
    ----------
    epsH : float
        Scaling (variance) parameter for the fBM.
    H : float
        Hurst parameter (0 < H < 1).
    m : int
        Upsampling factor.
    N : int
        Number of samples at the coarser scale. The output signal will effectively
        have `m * N` samples in the frequency domain before inverse FFT.

    Returns
    -------
    t : ndarray
        Time vector of length ~ `N`.
    y : ndarray
        Generated fractional Brownian motion samples of length ~ `m * N`.

    Notes
    -----
    - The function performs the following steps:
      1. Generate random Gaussian-distributed Fourier coefficients.
      2. Scale those coefficients by frequency-dependent factors involving
         `H` and the fractional spline autocorrelation term.
      3. Inverse FFT the result to obtain the time-domain fBM samples.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.interpolate.smooth.fBmper import fBmper
    >>> epsH = 1.0
    >>> H = 0.7
    >>> m = 4
    >>> N = 128
    >>> t, y = fBmper(epsH, H, m, N)
    >>> t.shape, y.shape
    ((128,), (512,))  # Example: depends on how you interpret the lengths.

    """
    # Generate random Fourier coefficients of length m*N
    Y = np.fft.fft(np.random.randn(m * N))
    Y = Y[1:]  # Drop the zero-frequency bin for now

    omega = np.arange(1, m * N) * 2 * np.pi / (m * N)

    # Scale the Fourier coefficients by the fractional spline autocorrelation
    Y = (
        m ** (-H)
        * epsH
        * Y
        / np.abs(2 * np.sin(omega / 2)) ** (H + 0.5)
        * np.sqrt(fractsplineautocorr(H - 0.5, omega / (2 * np.pi)))
    )

    # Enforce real-valued signal by adjusting the sum of the Fourier coefficients
    Y = np.concatenate(([-np.real(np.sum(Y))], Y))

    # Inverse FFT to get time-domain signal
    y = np.real(np.fft.ifft(Y))

    # Create time vector of length N, spaced by 1/m
    t = np.arange(0, N, 1 / m)

    return t, y
