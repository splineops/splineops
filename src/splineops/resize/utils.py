# splineops/src/splineops/resize/utils.py

import numpy as np
import numpy.typing as npt
from typing import Sequence, Tuple, Union, Optional

def calculate_final_size(
    inversable: bool,
    input_sizes: Sequence[int],
    zoom_factors: Sequence[float]
) -> Tuple[list[int], list[int]]:
    """
    Calculate the working and final sizes for each dimension.

    Parameters
    ----------
    inversable : bool
        Whether to adjust sizes to ensure invertibility.
    input_sizes : Sequence[int]
        The sizes of the input image per dimension.
    zoom_factors : Sequence[float]
        The zoom factors per dimension.

    Returns
    -------
    working_sizes : list of int
        Adjusted working sizes per dimension (may be increased if inversable is True).
    final_sizes : list of int
        Final sizes per dimension after scaling.
    """
    working_sizes = []
    final_sizes = []
    for size, zoom in zip(input_sizes, zoom_factors):
        if inversable:
            # Adjust working size to ensure invertibility
            working_size = size
            s = int(round(round((working_size - 1) * zoom) / zoom))
            while working_size - 1 - s != 0:
                working_size += 1
                s = int(round(round((working_size - 1) * zoom) / zoom))
            final_size = int(round((working_size - 1) * zoom) + 1)
        else:
            working_size = size
            final_size = int(round(working_size * zoom))
        working_sizes.append(working_size)
        final_sizes.append(final_size)
    return working_sizes, final_sizes

def border(size: int, degree: int, tolerance: float = 1e-10) -> int:
    """
    Calculate the border size required based on the spline degree.

    Parameters
    ----------
    size : int
        Size of the dimension.
    degree : int
        Degree of the spline.
    tolerance : float, optional
        Tolerance for numerical calculations (default is 1e-10).

    Returns
    -------
    horizon : int
        The calculated border size.
    """
    if degree in [0, 1]:
        return 0

    # Calculate the pole z based on the degree
    if degree == 2:
        z = np.sqrt(8.0) - 3.0
    elif degree == 3:
        z = np.sqrt(3.0) - 2.0
    elif degree == 4:
        z = np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0
    elif degree == 5:
        z = (np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0)
    elif degree == 6:
        z = -0.488294589303044755130118038883789062112279161239377608394
    elif degree == 7:
        z = -0.5352804307964381655424037816816460718339231523426924148812
    else:
        raise ValueError("Invalid degree (should be [0..7])")

    # Calculate the number of terms needed to reach the desired tolerance
    horizon = 2 + int(np.log(tolerance) / np.log(abs(z)))
    horizon = min(horizon, size)
    return horizon

# Beta function implementation for spline calculations
def beta(x: float, degree: int) -> float:
    """
    Compute the value of the B-spline basis function at a given point x for a specified degree.

    Parameters
    ----------
    x : float
        The point at which to evaluate the B-spline basis function.
    degree : int
        The degree of the B-spline basis function.

    Returns
    -------
    betan : float
        The value of the B-spline basis function at point x.
    """
    betan = 0.0
    if degree == 0:
        if abs(x) < 0.5 or x == -0.5:
            betan = 1.0
    elif degree == 1:
        x = abs(x)
        if x < 1.0:
            betan = 1.0 - x
    elif degree == 2:
        x = abs(x)
        if x < 0.5:
            betan = 3.0 / 4.0 - x * x
        elif x < 1.5:
            x -= 3.0 / 2.0
            betan = x * x * (1.0 / 2.0)
    elif degree == 3:
        x = abs(x)
        if x < 1.0:
            betan = x * x * (x - 2.0) * (1.0 / 2.0) + 2.0 / 3.0
        elif x < 2.0:
            x -= 2.0
            betan = x * x * x * (-1.0 / 6.0)
    elif degree == 4:
        x = abs(x)
        if x < 0.5:
            x *= x
            betan = x * (x * (1.0 / 4.0) - 5.0 / 8.0) + 115.0 / 192.0
        elif x < 1.5:
            betan = x * (x * (x * (5.0 / 6.0 - x * (1.0 / 6.0)) - 5.0 / 4.0) + 5.0 / 24.0) + 55.0 / 96.0
        elif x < 2.5:
            x -= 5.0 / 2.0
            x *= x
            betan = x * x * (1.0 / 24.0)
    elif degree == 5:
        x = abs(x)
        if x < 1.0:
            a = x * x
            betan = a * (a * (1.0 / 4.0 - x * (1.0 / 12.0)) - 1.0 / 2.0) + 11.0 / 20.0
        elif x < 2.0:
            betan = x * (x * (x * (x * (x * (1.0 / 24.0) - 3.0 / 8.0) + 5.0 / 4.0) - 7.0 / 4.0) + 5.0 / 8.0) + 17.0 / 40.0
        elif x < 3.0:
            a = 3.0 - x
            x = a * a
            betan = a * x * x * (1.0 / 120.0)
    elif degree == 6:
        x = abs(x)
        if x < 0.5:
            x *= x
            betan = x * (x * (7.0 / 48.0 - x * (1.0 / 36.0)) - 77.0 / 192.0) + 5887.0 / 11520.0
        elif x < 1.5:
            betan = x * (x * (x * (x * (x * (x * (1.0 / 48.0) - 7.0 / 48.0) + 21.0 / 64.0) - 35.0 / 288.0) - 91.0 / 256.0) - 7.0 / 768.0) + 7861.0 / 15360.0
        elif x < 2.5:
            betan = x * (x * (x * (x * (x * (7.0 / 60.0 - x * (1.0 / 120.0)) - 21.0 / 32.0) + 133.0 / 72.0) - 329.0 / 128.0) + 1267.0 / 960.0) + 1379.0 / 7680.0
        elif x < 3.5:
            x -= 7.0 / 2.0
            x *= x * x
            betan = x * x * (1.0 / 720.0)
    elif degree == 7:
        x = abs(x)
        if x < 1.0:
            a = x * x
            betan = a * (a * (a * (x * (1.0 / 144.0) - 1.0 / 36.0) + 1.0 / 9.0) - 1.0 / 3.0) + 151.0 / 315.0
        elif x < 2.0:
            betan = x * (x * (x * (x * (x * (x * (1.0 / 20.0 - x * (1.0 / 240.0)) - 7.0 / 30.0) + 1.0 / 2.0) - 7.0 / 18.0) - 1.0 / 10.0) - 7.0 / 90.0) + 103.0 / 210.0
        elif x < 3.0:
            betan = x * (x * (x * (x * (x * (x * (x * (1.0 / 720.0) - 1.0 / 36.0) + 7.0 / 30.0) - 19.0 / 18.0) + 49.0 / 18.0) - 23.0 / 6.0) + 217.0 / 90.0) - 139.0 / 630.0
        elif x < 4.0:
            a = 4.0 - x
            x = a * a * a
            betan = x * x * a * (1.0 / 5040.0)
    return betan

# Calculate interpolation coefficients based on degree
def get_interpolation_coefficients(c: npt.NDArray, degree: int) -> None:
    """
    Compute the interpolation coefficients for a signal c given a spline degree.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).
    degree : int
        The degree of the spline.

    Notes
    -----
    This function modifies the input array c in place.
    """
    z = []
    lambda_ = 1.0
    tolerance = 1e-10

    if degree == 0 or degree == 1:
        return
    elif degree == 2:
        z = [np.sqrt(8.0) - 3.0]
    elif degree == 3:
        z = [np.sqrt(3.0) - 2.0]
    elif degree == 4:
        z = [np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
            np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0]
    elif degree == 5:
        z = [np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0,
            np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) - np.sqrt(105.0 / 4.0) - 13.0 / 2.0]
    elif degree == 6:
        z = [-0.488294589303044755130118038883789062112279161239377608394,
            -0.081679271076237512597937765737059080653379610398148178525368,
            -0.00141415180832581775108724397655859252786416905534669851652709]
    elif degree == 7:
        z = [-0.5352804307964381655424037816816460718339231523426924148812,
            -0.122554615192326690515272264359357343605486549427295558490763,
            -0.0091486948096082769285930216516478534156925639545994482648003]
    else:
        raise ValueError("Invalid spline degree (should be [0..7])")

    if len(c) == 1:
        return

    z = np.array(z)
    # Compute normalization factor lambda_
    lambda_ = np.prod((1.0 - z) * (1.0 - 1.0 / z))

    c *= lambda_ # Normalize the signal

    for zk in z:
        # Forward recursion (causal)
        c[0] = get_initial_causal_coefficient(c, zk, tolerance)
        for n in range(1, len(c)):
            c[n] += zk * c[n - 1]

        # Backward recursion (anti-causal)
        c[-1] = get_initial_anti_causal_coefficient(c, zk, tolerance)
        for n in range(len(c) - 2, -1, -1):
            c[n] = zk * (c[n + 1] - c[n])

def get_samples(c: npt.NDArray, degree: int) -> None:
    """
    Extract the samples from the continuous representation of the signal c.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).
    degree : int
        The degree of the spline.

    Notes
    -----
    This function modifies the input array c in place.
    """
    if degree == 0 or degree == 1:
        return
    elif degree == 2:
        h = [3.0 / 4.0, 1.0 / 8.0]
    elif degree == 3:
        h = [2.0 / 3.0, 1.0 / 6.0]
    elif degree == 4:
        h = [115.0 / 192.0, 19.0 / 96.0, 1.0 / 384.0]
    elif degree == 5:
        h = [11.0 / 20.0, 13.0 / 60.0, 1.0 / 120.0]
    elif degree == 6:
        h = [5887.0 / 11520.0, 10543.0 / 46080.0, 361.0 / 23040.0, 1.0 / 46080.0]
    elif degree == 7:
        h = [151.0 / 315.0, 397.0 / 1680.0, 1.0 / 42.0, 1.0 / 5040.0]
    else:
        raise ValueError("Invalid spline degree (should be [0..7])")

    s = np.zeros_like(c)
    symmetric_fir(h, c, s)
    np.copyto(c, s)

def symmetric_fir(h: Sequence[float], c: npt.NDArray, s: npt.NDArray) -> None:
    """
    Perform symmetric FIR filtering on the signal c using filter coefficients h.

    Parameters
    ----------
    h : Sequence[float]
        Filter coefficients.
    c : np.ndarray
        The input signal.
    s : np.ndarray
        The output signal after filtering (must be same size as c).
    """
    if len(c) != len(s):
        raise IndexError("Incompatible size")

    # Implement the FIR filter based on the length of h
    if len(h) == 2:
        if len(c) >= 2:
            s[0] = h[0] * c[0] + 2.0 * h[1] * c[1]
            s[1:-1] = h[0] * c[1:-1] + h[1] * (c[:-2] + c[2:])
            s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2]
        else:
            if len(c) == 1:
                s[0] = (h[0] + 2.0 * h[1]) * c[0]
            else:
                raise ValueError("Invalid length of data")
    
    elif len(h) == 3:
        if len(c) >= 4:
            s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
            s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3])
            s[2:-2] = h[0] * c[2:-2] + h[1] * (c[1:-3] + c[3:-1]) + h[2] * (c[0:-4] + c[4:])
            s[-2] = h[0] * c[-2] + h[1] * (c[-3] + c[-1]) + h[2] * (c[-4] + c[-2])
            s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2] + 2.0 * h[2] * c[-3]
        else:
            if len(c) == 3:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
                s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + 2.0 * h[2] * c[1]
                s[2] = h[0] * c[2] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[0]
            elif len(c) == 2:
                s[0] = (h[0] + 2.0 * h[2]) * c[0] + 2.0 * h[1] * c[1]
                s[1] = (h[0] + 2.0 * h[2]) * c[1] + 2.0 * h[1] * c[0]
            elif len(c) == 1:
                s[0] = (h[0] + 2.0 * (h[1] + h[2])) * c[0]
            else:
                raise ValueError("Invalid length of data")

    elif len(h) == 4:
        if len(c) >= 6:
            s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[3]
            s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]) + h[3] * (c[2] + c[4])
            s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + h[2] * (c[0] + c[4]) + h[3] * (c[1] + c[5])
            s[3:-3] = (h[0] * c[3:-3] + h[1] * (c[2:-4] + c[4:-2]) + h[2] * (c[1:-5] + c[5:-1]) + h[3] * (c[0:-6] + c[6:]))
            s[-3] = h[0] * c[-3] + h[1] * (c[-4] + c[-2]) + h[2] * (c[-5] + c[-1]) + h[3] * (c[-6] + c[-2])
            s[-2] = h[0] * c[-2] + h[1] * (c[-3] + c[-1]) + h[2] * (c[-4] + c[-2]) + h[3] * (c[-5] + c[-3])
            s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2] + 2.0 * h[2] * c[-3] + 2.0 * h[3] * c[-4]
        else:
            if len(c) == 5:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[3]
                s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]) + h[3] * (c[2] + c[4])
                s[2] = h[0] * c[2] + (h[1] + h[3]) * (c[1] + c[3]) + h[2] * (c[0] + c[4])
                s[3] = h[0] * c[3] + h[1] * (c[2] + c[4]) + h[2] * (c[1] + c[3]) + h[3] * (c[0] + c[2])
                s[4] = h[0] * c[4] + 2.0 * h[1] * c[3] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[1]
            elif len(c) == 4:
                s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2] + 2.0 * h[3] * c[3]
                s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]) + 2.0 * h[3] * c[2]
                s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + h[2] * (c[0] + c[2]) + 2.0 * h[3] * c[1]
                s[3] = h[0] * c[3] + 2.0 * h[1] * c[2] + 2.0 * h[2] * c[1] + 2.0 * h[3] * c[0]
            elif len(c) == 3:
                s[0] = h[0] * c[0] + 2.0 * (h[1] + h[3]) * c[1] + 2.0 * h[2] * c[2]
                s[1] = h[0] * c[1] + (h[1] + h[3]) * (c[0] + c[2]) + 2.0 * h[2] * c[1]
                s[2] = h[0] * c[2] + 2.0 * (h[1] + h[3]) * c[1] + 2.0 * h[2] * c[0]
            elif len(c) == 2:
                s[0] = (h[0] + 2.0 * h[2]) * c[0] + 2.0 * (h[1] + h[3]) * c[1]
                s[1] = (h[0] + 2.0 * h[2]) * c[1] + 2.0 * (h[1] + h[3]) * c[0]
            elif len(c) == 1:
                s[0] = (h[0] + 2.0 * (h[1] + h[2] + h[3])) * c[0]
            else:
                raise ValueError("Invalid length of data")
    else:
        raise ValueError("Invalid filter half-length (should be [2..4])")

def get_initial_causal_coefficient(
    c: npt.NDArray, 
    z: float, 
    tolerance: float = 1e-10
) -> float:
    """
    Calculate the initial causal coefficient for the IIR filter.

    Parameters
    ----------
    c : np.ndarray
        The input signal.
    z : float
        The pole of the filter.
    tolerance : float, optional
        The tolerance for numerical calculations (default is 1e-10).

    Returns
    -------
    float
        The initial causal coefficient.
    """
    z1 = z
    zn = z ** (len(c) - 1)
    sum_ = c[0] + zn * c[-1]
    horizon = len(c)

    if tolerance > 0.0:
        horizon = 2 + int(np.log(tolerance) / np.log(np.abs(z)))
        horizon = min(horizon, len(c))

    n = np.arange(1, horizon - 1)
    z1_array = z ** n
    zn_array = zn / (z ** n)
    sum_ += np.sum((z1_array + zn_array) * c[1:horizon-1])

    return sum_ / (1.0 - z ** (2 * len(c) - 2))

def get_initial_anti_causal_coefficient(
    c: npt.NDArray, 
    z: float, 
    tolerance: float = 1e-10
) -> float:
    """
    Calculate the initial anti-causal coefficient for the IIR filter.

    Parameters
    ----------
    c : np.ndarray
        The input signal.
    z : float
        The pole of the filter.
    tolerance : float, optional
        The tolerance for numerical calculations (default is 1e-10).

    Returns
    -------
    float
        The initial anti-causal coefficient.
    """
    return (z * c[-2] + c[-1]) * z / (z * z - 1.0)

def do_integ(c: npt.NDArray, nb: int) -> float:
    """
    Perform integration on the signal c nb times.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).
    nb : int
        Number of integration steps.

    Returns
    -------
    average : float
        The average value after integration.

    Notes
    -----
    This function modifies the input array c in place.
    """
    size = len(c)
    m = 0.0
    average = 0.0

    if nb == 1:
        average = np.sum(c)
        average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
        integ_sa(c, average)

    elif nb == 2:
        average = np.sum(c)
        average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
        integ_sa(c, average)
        integ_as(c, c)

    elif nb == 3:
        average = np.sum(c)
        average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
        integ_sa(c, average)
        integ_as(c, c)
        m = np.sum(c)
        m = (2.0 * m - c[size - 1] - c[0]) / (2.0 * size - 2)
        integ_sa(c, m)

    elif nb == 4:
        average = np.sum(c)
        average = (2.0 * average - c[size - 1] - c[0]) / (2.0 * size - 2)
        integ_sa(c, average)
        integ_as(c, c)
        m = np.sum(c)
        m = (2.0 * m - c[size - 1] - c[0]) / (2.0 * size - 2)
        integ_sa(c, m)
        integ_as(c, c)

    return average

def integ_sa(c: npt.NDArray, m: float) -> None:
    """
    Perform semi-analytical integration on the signal c.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).
    m : float
        The average value to subtract from the signal.

    Notes
    -----
    This function modifies the input array c in place.
    """
    c -= m
    c[0] *= 0.5
    c[1:] += np.cumsum(c[:-1])

def integ_as(c: npt.NDArray, y: npt.NDArray) -> None:
    """
    Perform anti-symmetric integration on the signal c.

    Parameters
    ----------
    c : np.ndarray
        The input signal.
    y : np.ndarray
        The output signal after integration.

    Notes
    -----
    This function modifies the output array y in place.
    """
    z = c.copy()
    y[0] = z[0]
    y[1] = 0
    y[2:] = -np.cumsum(z[1:-1])

def do_diff(c: npt.NDArray, nb: int) -> None:
    """
    Perform differentiation on the signal c nb times.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).
    nb : int
        Number of differentiation steps.

    Notes
    -----
    This function modifies the input array c in place.
    """
    size = len(c)
    if nb == 1:
        diff_as(c)
    elif nb == 2:
        diff_sa(c)
        diff_as(c)
    elif nb == 3:
        diff_as(c)
        diff_sa(c)
        diff_as(c)
    elif nb == 4:
        diff_sa(c)
        diff_as(c)
        diff_sa(c)
        diff_as(c)

def diff_sa(c: npt.NDArray) -> None:
    """
    Perform semi-analytical differentiation on the signal c.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).

    Notes
    -----
    This function modifies the input array c in place.
    """
    old = c[-2]
    c[:-1] -= c[1:]  # Perform the element-wise subtraction
    c[-1] -= old     # Update the last element

def diff_as(c: npt.NDArray) -> None:
    """
    Perform anti-symmetric differentiation on the signal c.

    Parameters
    ----------
    c : np.ndarray
        The input signal (modified in place).

    Notes
    -----
    This function modifies the input array c in place.
    """
    c[1:] -= c[:-1]  # Perform the element-wise subtraction for differentiation
    c[0] *= 2.0      # Update the first element