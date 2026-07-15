# splineops/src/splineops/smoothing_splines/smoothing_spline.py

import operator
from typing import Tuple
import numpy as np
import numpy.typing as npt
from splineops.smoothing_splines.fract_spline_auto_corr import fractsplineautocorr
from scipy.fft import irfftn, rfftn


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
    >>> from splineops.smoothing_splines.smoothing_spline import periodize
    >>> x = np.array([1, 2, 3])
    >>> periodize(x, 2)
    array([1, 2, 3, 1, 2, 3])
    """
    x = np.asarray(x)
    if x.ndim == 0 or x.size == 0:
        raise ValueError("'x' must be a non-empty array.")
    try:
        repetitions = operator.index(m)
    except TypeError as exc:
        raise TypeError("'m' must be a positive integer.") from exc
    if isinstance(m, (bool, np.bool_)) or repetitions <= 0:
        raise ValueError("'m' must be a positive integer.")
    return np.tile(x, repetitions)


def smoothing_spline(
    y: npt.NDArray, lamb: float, m: int, gamma: float
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
    >>> from splineops.smoothing_splines.smoothing_spline import smoothing_spline
    >>> y = np.array([1., 2., 3.])
    >>> t, ys = smoothing_spline(y, lamb=0.1, m=2, gamma=1.5)
    >>> t.shape, ys.shape
    ((6,), (6,))
    """

    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("'y' must be a non-empty one-dimensional array.")
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError("'y' must have a numeric dtype.")
    if np.iscomplexobj(y):
        raise TypeError("'y' must be real-valued.")
    if not np.all(np.isfinite(y)):
        raise ValueError("'y' must contain only finite values.")
    if not np.isfinite(lamb) or lamb < 0:
        raise ValueError("'lamb' must be finite and non-negative.")
    if not np.isfinite(gamma) or gamma <= 0.5:
        raise ValueError("'gamma' must be finite and greater than 0.5.")
    try:
        m = operator.index(m)
    except TypeError as exc:
        raise TypeError("'m' must be a positive integer.") from exc
    if isinstance(m, (bool, np.bool_)) or m <= 0:
        raise ValueError("'m' must be a positive integer.")
    y = y.flatten()
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
    Hm = m ** (-2 * gamma + 1) * (sinm2g / sin2g) * Ag / (Agm + lamb * sinm2g)
    # Insert the DC term at the beginning
    Hm = np.concatenate(([m], Hm))

    # Generate outputs
    ys = np.real(np.fft.ifft(Hm * Ym))
    t = np.arange(0, N, 1 / m)
    return t, ys


def recursive_smoothing_spline(signal: npt.NDArray, lamb: float = 1.0) -> npt.NDArray:
    """
    Apply a recursive first-order smoothing spline filter (piecewise-linear).

    This implements the symmetric all-pole factorization (causal + anticausal)
    for the *first-order* smoothing spline filter.

    Notes
    -----
    - This is NOT the cubic smoother (which requires a higher-order recursion).
    - Includes DC normalization so constant signals remain constant.

    Parameters
    ----------
    signal : ndarray
        1D array of data points to smooth.
    lamb : float, optional
        Smoothing parameter (>= 0). Default is 1.0.

    Returns
    -------
    smoothed_signal : ndarray
        Smoothed data, same length as `signal`.
    """
    x = np.asarray(signal)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("'signal' must be a non-empty one-dimensional array.")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("'signal' must have a numeric dtype.")
    if np.iscomplexobj(x):
        raise TypeError("'signal' must be real-valued.")
    if not np.all(np.isfinite(x)):
        raise ValueError("'signal' must contain only finite values.")
    if not np.isfinite(lamb) or lamb < 0:
        raise ValueError("'lamb' must be finite and non-negative.")
    dtype = np.result_type(x.dtype, np.float64)
    x = x.astype(dtype, copy=False)

    if lamb == 0:
        return x.copy()

    # --- Legacy (kept for reference; not consistent with the paper) ---
    # z1 = -lamb / (1 + np.sqrt(1 + 4 * lamb))

    # Paper-consistent pole for first-order smoothing spline:
    r = np.sqrt(1.0 + 4.0 * lamb)
    z1 = (r - 1.0) / (r + 1.0)  # in (0, 1)

    # DC normalization (preserves constants): (1 - z1)^2 == z1 / lamb
    scale = (1.0 - z1) ** 2

    K = x.size

    # Causal pass (steady-state init)
    c = np.zeros(K, dtype=dtype)
    c[0] = x[0] / (1.0 - z1)
    for k in range(1, K):
        c[k] = x[k] + z1 * c[k - 1]

    # Anticausal pass (steady-state init)
    y = np.zeros(K, dtype=dtype)
    y[-1] = c[-1] / (1.0 - z1)
    for k in range(K - 2, -1, -1):
        y[k] = c[k] + z1 * y[k + 1]

    return scale * y


class SmoothingSplinePlan:
    """Reusable real-FFT smoothing filter for a fixed array shape.

    Constructing the frequency response is independent of the input samples.
    Keeping it in a plan avoids rebuilding frequency grids and the filter when
    smoothing multiple arrays with the same shape and parameters.  The plan
    stores only the non-redundant real-FFT half spectrum.

    Parameters
    ----------
    shape : tuple of int
        Shape of every array that will be passed to :meth:`apply`.
    lamb : float
        Non-negative regularization parameter.
    gamma : float
        Positive order of the spline operator.
    """

    def __init__(self, shape: tuple[int, ...], lamb: float, gamma: float) -> None:
        try:
            shape = tuple(operator.index(length) for length in shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "'shape' must be a non-empty sequence of integers."
            ) from exc
        if not shape or any(
            isinstance(length, (bool, np.bool_)) or length <= 0 for length in shape
        ):
            raise ValueError("'shape' must contain only positive integers.")
        if not np.isfinite(lamb) or lamb < 0:
            raise ValueError("'lamb' must be finite and non-negative.")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("'gamma' must be finite and positive.")

        self.shape = shape
        self.lamb = float(lamb)
        self.gamma = float(gamma)
        spectrum_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        omega_squared = np.zeros(spectrum_shape, dtype=np.float64)
        for axis, length in enumerate(shape):
            frequencies = (
                np.fft.rfftfreq(length)
                if axis == len(shape) - 1
                else np.fft.fftfreq(length)
            )
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = frequencies.size
            angular_frequencies = 2.0 * np.pi * frequencies.reshape(broadcast_shape)
            omega_squared += angular_frequencies**2

        self.frequency_response = 1.0 / (1.0 + self.lamb * omega_squared**self.gamma)
        self.frequency_response.flags.writeable = False

    @property
    def retained_bytes(self) -> int:
        """Number of bytes retained for repeated execution."""

        return self.frequency_response.nbytes

    def apply(
        self, data: npt.NDArray, *, out: npt.NDArray | None = None
    ) -> npt.NDArray:
        """Smooth ``data`` and optionally copy the result into ``out``."""

        data = np.asarray(data)
        if data.shape != self.shape:
            raise ValueError(
                f"'data' must have shape {self.shape}; received {data.shape}."
            )
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("'data' must have a numeric dtype.")
        if np.iscomplexobj(data):
            raise TypeError("'data' must be real-valued.")
        if not np.all(np.isfinite(data)):
            raise ValueError("'data' must contain only finite values.")

        spectrum = rfftn(data)
        result = irfftn(self.frequency_response * spectrum, s=self.shape)
        if out is None:
            return result
        if not isinstance(out, np.ndarray):
            raise TypeError("'out' must be a NumPy array.")
        if out.shape != self.shape:
            raise ValueError(
                f"'out' must have shape {self.shape}; received {out.shape}."
            )
        if out.dtype != result.dtype:
            raise TypeError(
                f"'out' must have dtype {result.dtype}; received {out.dtype}."
            )
        np.copyto(out, result, casting="no")
        return out

    __call__ = apply


def smoothing_spline_nd(data: npt.NDArray, lamb: float, gamma: float) -> npt.NDArray:
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
    >>> from splineops.smoothing_splines.smoothing_spline import smoothing_spline_nd
    >>> x = np.random.rand(4, 4)
    >>> x_smooth = smoothing_spline_nd(x, lamb=0.5, gamma=1.0)
    >>> x_smooth.shape
    (4, 4)
    """
    data = np.asarray(data)
    if data.ndim == 0 or data.size == 0:
        raise ValueError(
            "'data' must be a non-empty array with at least one dimension."
        )
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("'data' must have a numeric dtype.")
    if np.iscomplexobj(data):
        raise TypeError("'data' must be real-valued.")
    if not np.all(np.isfinite(data)):
        raise ValueError("'data' must contain only finite values.")
    if not np.isfinite(lamb) or lamb < 0:
        raise ValueError("'lamb' must be finite and non-negative.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("'gamma' must be finite and positive.")
    return SmoothingSplinePlan(data.shape, lamb=lamb, gamma=gamma).apply(data)
