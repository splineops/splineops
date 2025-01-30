import numpy as np
from splineops.interpolate.smoothing_spline.fractsplineautocorr import fractsplineautocorr
from scipy.fft import fftn, ifftn

def periodize(x, m):
    """
    Periodizes the input array by concatenating 'm' copies of it.

    Parameters:
    x (numpy array): Input array.
    m (int): Number of times to concatenate.

    Returns:
    xp (numpy array): Periodized array.
    """
    return np.tile(x, m)

def smoothing_spline(y, lamb, m, gamma):
    """
    Computes the fractional smoothing spline of an input signal.
    Returns samples of the smoothing spline for a given input sequence,
    sampled at "m" times the rate of input. The input is assumed to be
    sampled at integers 0..N-1.

    Parameters:
    y (array-like): Input signal.
    lamb (float): Regularization parameter.
    m (int): Upsampling factor.
    gamma (float): Order of the spline operator (gamma = H + 0.5).

    Returns:
    t (numpy array): Time vector.
    ys (numpy array): Smoothing spline sequence.

    References:
        See above.
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

    Ag = Ag[1:]
    Agm = Agm[1:]

    # Compute the smoothing spline filter H_m
    Hm = (m ** (-2 * gamma + 1) * (sinm2g / sin2g) * Ag /
          (Agm + lamb * sinm2g))
    Hm = np.concatenate(([m], Hm))

    # Generate outputs
    ys = np.real(np.fft.ifft(Hm * Ym))
    t = np.arange(0, N, 1 / m)
    return t, ys

def recursive_smoothing_spline(signal, lamb=1.0):
    """
    Applies recursive smoothing spline filtering to the input signal using
    a causal and anticausal IIR filter.
    
    Parameters:
    - signal: 1D array of data points to smooth
    - lamb: Smoothing parameter controlling the amount of smoothing (lamb)

    Returns:
    - smoothed_signal: 1D array of smoothed data
    """
    # Define the filter pole (z1) based on the regularization parameter lamb
    z1 = -lamb / (1 + np.sqrt(1 + 4 * lamb))
    K = len(signal)
    
    # Causal filtering (forward pass)
    y_causal = np.zeros(K)
    y_causal[0] = signal[0]
    for k in range(1, K):
        y_causal[k] = signal[k] + z1 * y_causal[k - 1]

    # Anticausal filtering (backward pass)
    smoothed_signal = np.zeros(K)
    smoothed_signal[-1] = y_causal[-1]
    for k in range(K - 2, -1, -1):
        smoothed_signal[k] = y_causal[k] + z1 * smoothed_signal[k + 1]
        
    return smoothed_signal

def smoothing_spline_nd(data, lamb, gamma):
    """
    Applies multi-dimensional fractional smoothing spline to the input data.

    Parameters:
    data (ndarray): Multi-dimensional input data (e.g., image, volume).
    lamb (float): Regularization parameter.
    gamma (float): Order of the spline operator (gamma = H + 0.5).

    Returns:
    data_smooth (ndarray): Smoothed data.
    """
    data = np.asarray(data)
    dims = data.shape

    # Compute the frequency grids for each dimension
    freq_grids = np.meshgrid(*[np.fft.fftfreq(n) for n in dims], indexing='ij')
    
    # Vectorized computation of omega_squared
    freq_grids_stacked = np.stack(freq_grids, axis=0)  # Shape: (ndim, dims...)
    omega_squared = np.sum((2 * np.pi * freq_grids_stacked) ** 2, axis=0)

    # Compute the Butterworth-like filter in Fourier domain
    H = 1 / (1 + lamb * omega_squared ** gamma)

    # Apply the filter
    data_fft = fftn(data)
    data_smooth_fft = H * data_fft
    data_smooth = np.real(ifftn(data_smooth_fft))

    return data_smooth
