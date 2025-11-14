# splineops/src/splineops/smoothing_splines/smoothingspline.py

from typing import Tuple
import numpy as np
import numpy.typing as npt
from splineops.smoothing_splines.fractsplineautocorr import fractsplineautocorr
from scipy.fft import fftn, ifftn

def periodize(x: npt.NDArray, m: int) -> npt.NDArray:
    """
    Periodize the input array by concatenating `m` copies of it.

    Parameters
    ----------
    x : ndarray
        Input array to be periodized.
    m : int
        Number of times to concatenate the array.

    Returns
    -------
    xp : ndarray
        The periodized array, which has its size multiplied by `m` along the
        concatenation axis.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.interpolate.smooth.smoothing_spline import periodize
    >>> x = np.array([1, 2, 3])
    >>> periodize(x, 2)
    array([1, 2, 3, 1, 2, 3])
    """
    return np.tile(x, m)


def smoothing_spline(
    y: npt.NDArray,
    lamb: float,
    m: int,
    gamma: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the fractional smoothing spline at m-times upsampling of the input.

    This function returns samples of the smoothing spline for a given input
    sequence, sampled at `m` times the rate of the input. The input is assumed
    to be sampled at integer locations 0..N-1.

    Parameters
    ----------
    y : ndarray
        Input signal of length N.
    lamb : float
        Regularization parameter.
    m : int
        Upsampling factor (integer).
    gamma : float
        Order of the spline operator. Typically gamma = H + 0.5.

    Returns
    -------
    t : ndarray
        The upsampled time vector, of length approximately `N * m`.
    ys : ndarray
        The smoothing spline samples, of length approximately `N * m`.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.interpolate.smooth.smoothing_spline import smoothing_spline
    >>> y = np.array([1., 2., 3.])
    >>> t, ys = smoothing_spline(y, lamb=0.1, m=2, gamma=1.5)
    >>> t.shape, ys.shape
    ((6,), (6,))
    """

    y = np.asarray(y).flatten()
    N = len(y)

    # Compute the FFT of the input signal
    Y = np.fft.fft(y)
    omega = np.arange(1, N * m) * 2 * np.pi / (N * m)

    # Upsample Y
    Ym = periodize(Y, m)

    # Internal calculations
    sinm2g = np.abs(2 * np.sin(m * omega / 2)) ** (2 * gamma)
    sin2g = np.abs(2 * np.sin(omega / 2)) ** (2 * gamma)

    # Calculate A_gamma(omega)
    alpha = gamma - 1
    Ag = fractsplineautocorr(alpha, np.concatenate(([0], omega / (2 * np.pi))))

    # Calculate A_gamma(m * omega)
    Agm = fractsplineautocorr(alpha, np.concatenate(([0], m * omega / (2 * np.pi))))

    # Drop the first element after concatenation (used for shift)
    Ag = Ag[1:]
    Agm = Agm[1:]

    # Compute the smoothing spline filter H_m
    Hm = (m ** (-2 * gamma + 1) * (sinm2g / sin2g) * Ag /
          (Agm + lamb * sinm2g))
    # Insert the DC term at the beginning
    Hm = np.concatenate(([m], Hm))

    # Generate outputs
    ys = np.real(np.fft.ifft(Hm * Ym))
    t = np.arange(0, N, 1 / m)
    return t, ys


def recursive_smoothing_spline(
    signal: npt.NDArray,
    lamb: float = 1.0
) -> npt.NDArray:
    """
    Apply a recursive smoothing spline filter to the input signal.

    Implements a causal and anticausal IIR filter based on the smoothing
    parameter `lamb`.

    Parameters
    ----------
    signal : ndarray
        1D array of data points to smooth.
    lamb : float, optional
        Smoothing parameter controlling the amount of smoothing. Default is 1.0.

    Returns
    -------
    smoothed_signal : ndarray
        1D array of smoothed data, same length as `signal`.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.interpolate.smooth.smoothing_spline import recursive_smoothing_spline
    >>> x = np.array([1., 2., 2., 3., 5.])
    >>> xs = recursive_smoothing_spline(x, lamb=1.0)
    >>> xs
    array([...])
    """
    # Define the filter pole (z1) based on the regularization parameter lamb
    z1 = -lamb / (1 + np.sqrt(1 + 4 * lamb))
    K = len(signal)
    
    # Causal filtering (forward pass)
    y_causal = np.zeros(K, dtype=signal.dtype)
    y_causal[0] = signal[0]
    for k in range(1, K):
        y_causal[k] = signal[k] + z1 * y_causal[k - 1]

    # Anticausal filtering (backward pass)
    smoothed_signal = np.zeros(K, dtype=signal.dtype)
    smoothed_signal[-1] = y_causal[-1]
    for k in range(K - 2, -1, -1):
        smoothed_signal[k] = y_causal[k] + z1 * smoothed_signal[k + 1]
        
    return smoothed_signal


def smoothing_spline_nd(
    data: npt.NDArray,
    lamb: float,
    gamma: float
) -> npt.NDArray:
    """
    Apply multi-dimensional fractional smoothing spline to the input data.

    Parameters
    ----------
    data : ndarray
        Multi-dimensional input data (e.g., image or volume).
    lamb : float
        Regularization parameter.
    gamma : float
        Order of the spline operator (gamma = H + 0.5).

    Returns
    -------
    data_smooth : ndarray
        Smoothed data of the same shape as `data`.

    Examples
    --------
    >>> import numpy as np
    >>> from splineops.interpolate.smooth.smoothing_spline import smoothing_spline_nd
    >>> x = np.random.rand(4, 4)
    >>> x_smooth = smoothing_spline_nd(x, lamb=0.5, gamma=1.0)
    >>> x_smooth.shape
    (4, 4)
    """
    data = np.asarray(data)
    dims = data.shape

    # Compute the frequency grids for each dimension
    freq_grids = np.meshgrid(*[np.fft.fftfreq(n) for n in dims], indexing='ij')
    
    # Vectorized computation of omega_squared
    freq_grids_stacked = np.stack(freq_grids, axis=0)  # Shape: (ndim, dims...)
    omega_squared = np.sum((2 * np.pi * freq_grids_stacked) ** 2, axis=0)

    # Compute the Butterworth-like filter in the Fourier domain
    H = 1 / (1 + lamb * (omega_squared ** gamma))

    # Apply the filter
    data_fft = fftn(data)
    data_smooth_fft = H * data_fft
    data_smooth = np.real(ifftn(data_smooth_fft))

    return data_smooth
