# splineops/src/splineops/multiscale/pyramid.py

"""
pyramid.py
----------

Implements pyramid decomposition (reduce & expand) in 1D and 2D using
filters derived from spline expansions. Boundary handling is done via
mirror reflection, closely mimicking original C routines:

- "ReduceStandard_1D" / "ExpandStandard_1D"
- "ReduceCentered_1D" / "ExpandCentered_1D"

Usage Example
-------------

.. code-block:: python

    from splineops.multiscale.pyramid import (
        get_pyramid_filter,
        reduce_1d, expand_1d,
        reduce_2d, expand_2d
    )
    import numpy as np

    # Retrieve filter
    g, h, is_centered = get_pyramid_filter("Spline", 3)

    # 1D reduce/expand
    x = np.array([0, 1, 2, 3, 2, 1, 0, -2, -4, -6], dtype=float)
    x_reduced = reduce_1d(x, g, is_centered)
    x_expanded = expand_1d(x_reduced, h, is_centered)

    # 2D reduce/expand
    arr = np.random.rand(8, 8).astype(np.float32)
    arr_reduced = reduce_2d(arr, g, is_centered)
    arr_expanded = expand_2d(arr_reduced, h, is_centered)
"""

import operator

import numpy as np

# -------------------------------------------------------------------------
# 1) Retrieve Filter Coefficients
# -------------------------------------------------------------------------


def get_pyramid_filter(name: str, order: int):
    """
    Retrieve the reduce/expand filters for a particular spline family and order.

    Parameters
    ----------
    name : str
        Filter family name, e.g. "Spline", "Centered Spline".
    order : int
        Spline order (e.g. 3).

    Returns
    -------
    g : np.ndarray
        1D filter for REDUCE operation.
    h : np.ndarray
        1D filter for EXPAND operation.
    is_centered : bool
        True if the filter is a centered variant; False otherwise.

    Raises
    ------
    ValueError
        If the combination of name/order is not implemented.
    """
    # These are sample definitions for demonstration:
    # (Add more for other 'name/order' combos if needed)
    name = name.lower().strip()
    is_centered = False

    if name == "spline" and order == 3:
        # Example: from your PyramidFilterSplinel2(...) with order=3
        g = np.array(
            [
                0.596797,
                0.313287,
                -0.0827691,
                -0.0921993,
                0.0540288,
                0.0436996,
                -0.0302508,
                -0.0225552,
                0.0162251,
                0.0118738,
                -0.00861788,
                -0.00627964,
                0.00456713,
                0.00332464,
                -0.00241916,
                -0.00176059,
                0.00128128,
                0.000932349,
                -0.000678643,
                -0.000493682,
            ]
        )
        h = np.array(
            [
                1.0,
                0.600481,
                0.0,
                -0.127405,
                0.0,
                0.034138,
                0.0,
                -0.00914725,
                0.0,
                0.002451,
                0.0,
                -0.000656743,
            ]
        )
        is_centered = False

    elif name == "centered spline" and order == 3:
        # Example: from your PyramidFilterCentered(...) with order=3
        g = np.array(
            [
                0.708792,
                0.328616,
                -0.165157,
                -0.114448,
                0.0944036,
                0.0543881,
                -0.05193,
                -0.0284868,
                0.0281854,
                0.0152877,
                -0.0152508,
                -0.00825077,
                0.00824629,
                0.00445865,
                -0.0044582,
                -0.00241009,
                0.00241022,
                0.00130278,
                -0.00130313,
                -0.000704109,
                0.000704784,
            ]
        )
        h = np.array(
            [
                1.13726,
                0.625601,
                -0.0870191,
                -0.159256,
                0.0233167,
                0.0426725,
                -0.00624769,
                -0.0114341,
                0.00167406,
                0.00306375,
                -0.000448564,
                -0.000820929,
                0.000120192,
                0.000219967,
                -3.22054e-05,
                -5.894e-05,
            ]
        )
        is_centered = True

    else:
        raise ValueError(f"Filter '{name}' with order={order} not implemented.")

    return g, h, is_centered


# -------------------------------------------------------------------------
# 2) Utility: robust mirror reflection
# -------------------------------------------------------------------------


def wrap_reflect(i: int, n: int) -> int:
    """
    Mirror boundary reflection of index `i` into the range [0..n-1].

    If n >= 2, uses period = 2*(n-1).
    If n < 2, everything maps to 0.

    Parameters
    ----------
    i : int
        Original index (may be out of bounds).
    n : int
        Length of the signal.

    Returns
    -------
    int
        Reflected index within [0..n-1].
    """
    if n < 2:
        return 0
    period = 2 * (n - 1)
    i = i % period  # now in [0..period-1]
    if i >= n:
        i = period - i  # reflect
    return i


def _wrap_reflect_array(index, length):
    if length < 2:
        return np.zeros_like(index, dtype=np.intp)
    period = 2 * (length - 1)
    wrapped = np.mod(index, period)
    return np.where(wrapped >= length, period - wrapped, wrapped).astype(
        np.intp, copy=False
    )


def _validate_centered(centered):
    if not isinstance(centered, (bool, np.bool_)):
        raise TypeError("'centered' must be a boolean.")
    return bool(centered)


# -------------------------------------------------------------------------
# 3) 1D Reduce & Expand
# -------------------------------------------------------------------------


def reduce_1d(signal: np.ndarray, g: np.ndarray, centered: bool) -> np.ndarray:
    """
    Reduce a 1D signal by factor of 2 using filter g.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal of length >= 2.
    g : np.ndarray
        Filter for reduction (REDUCE).
    centered : bool
        Indicates if the filter is a centered variant.

    Returns
    -------
    np.ndarray
        Reduced signal of length roughly n/2.
    """
    signal, g = _validate_1d_inputs(signal, g)
    centered = _validate_centered(centered)
    n = signal.shape[0]
    if n == 1:
        return signal.copy()
    half = n // 2 if n >= 2 else 1
    out = np.zeros(half, dtype=signal.dtype)

    if centered:
        out[:] = _reduce_centered_1d(signal, g)
    else:
        out[:] = _reduce_standard_1d(signal, g)
    return out


def expand_1d(signal: np.ndarray, h: np.ndarray, centered: bool) -> np.ndarray:
    """
    Expand a 1D signal by factor of 2 using filter h.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal (coarse scale).
    h : np.ndarray
        Filter for expansion (EXPAND).
    centered : bool
        Indicates if the filter is a centered variant.

    Returns
    -------
    np.ndarray
        Expanded signal of length ~ 2*n.
    """
    signal, h = _validate_1d_inputs(signal, h)
    centered = _validate_centered(centered)
    n = signal.shape[0]
    outlen = 2 * n if n >= 2 else n
    out = np.zeros(outlen, dtype=signal.dtype)

    if centered:
        out[:] = _expand_centered_1d(signal, h)
    else:
        out[:] = _expand_standard_1d(signal, h)
    return out


def _validate_1d_inputs(signal: np.ndarray, filter_: np.ndarray):
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("'signal' must be a one-dimensional NumPy array.")
    if signal.size == 0:
        raise ValueError("'signal' must be non-empty.")
    if not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
        raise TypeError("'signal' must have a real numeric dtype.")
    filter_ = np.asarray(filter_)
    if filter_.ndim != 1 or filter_.size == 0:
        raise ValueError("The pyramid filter must be a non-empty 1D array.")
    if not np.issubdtype(filter_.dtype, np.number) or np.iscomplexobj(filter_):
        raise TypeError("The pyramid filter must have a real numeric dtype.")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(filter_)):
        raise ValueError("Signal and filter values must be finite.")
    if not np.issubdtype(signal.dtype, np.floating):
        signal = signal.astype(np.float64)
    return signal, filter_


def _validate_nd_input(array, filter_, ndim):
    if not isinstance(array, np.ndarray) or array.ndim != ndim:
        raise ValueError(f"Input must be a {ndim}-dimensional NumPy array.")
    if any(length == 0 for length in array.shape):
        raise ValueError("Input dimensions must be non-empty.")
    flattened = array.reshape(-1)
    validated, filter_ = _validate_1d_inputs(flattened, filter_)
    return validated.reshape(array.shape), filter_


def _validate_spatial_input(array, filter_, spatial_axes):
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if spatial_axes is None:
        if array.ndim != 2:
            raise ValueError(
                "'spatial_axes' is required when input contains batch or channel axes."
            )
        axes = (0, 1)
    else:
        try:
            axes = tuple(operator.index(axis) for axis in spatial_axes)
        except TypeError as exc:
            raise TypeError(
                "'spatial_axes' must be a sequence of integer axes."
            ) from exc
        if len(axes) != 2:
            raise ValueError("'spatial_axes' must contain exactly two axes.")
        axes = tuple(axis + array.ndim if axis < 0 else axis for axis in axes)
        if any(axis < 0 or axis >= array.ndim for axis in axes) or axes[0] == axes[1]:
            raise ValueError("'spatial_axes' must contain two distinct valid axes.")
    if any(array.shape[axis] == 0 for axis in range(array.ndim)):
        raise ValueError("Input dimensions must be non-empty.")
    flattened = array.reshape(-1)
    validated, filter_ = _validate_1d_inputs(flattened, filter_)
    return validated.reshape(array.shape), filter_, axes


def _apply_last_axis(array, filter_, centered, *, expand):
    if array.shape[-1] == 1:
        return array.copy()
    function = (
        _expand_centered_1d
        if expand and centered
        else (
            _expand_standard_1d
            if expand
            else _reduce_centered_1d if centered else _reduce_standard_1d
        )
    )
    return function(array, filter_)


def _apply_axis(array, filter_, centered, axis, *, expand):
    moved = np.moveaxis(array, axis, -1)
    transformed = _apply_last_axis(moved, filter_, centered, expand=expand)
    return np.moveaxis(transformed, -1, axis)


# -------------------------------------------------------------------------
# 4) 2D Reduce & Expand
# -------------------------------------------------------------------------


def reduce_2d(
    image: np.ndarray,
    g: np.ndarray,
    centered: bool,
    *,
    spatial_axes=None,
) -> np.ndarray:
    """
    Reduce two selected image axes by a factor of two.

    Parameters
    ----------
    image : np.ndarray
        Input scalar image or array containing batch/channel dimensions.
    g : np.ndarray
        1D reduce filter.
    centered : bool
        True if using centered reduce logic.
    spatial_axes : sequence of int, optional
        Exactly two dimensions to reduce.  Required for higher-rank arrays;
        every other dimension is preserved.

    Returns
    -------
    np.ndarray
        Array with selected lengths reduced to ``floor(length / 2)`` when the
        length is at least two.  Singleton selected dimensions stay singleton.
    """
    image, g, axes = _validate_spatial_input(image, g, spatial_axes)
    centered = _validate_centered(centered)
    return _apply_axis(
        _apply_axis(image, g, centered, axes[1], expand=False),
        g,
        centered,
        axes[0],
        expand=False,
    )


def expand_2d(
    image: np.ndarray,
    h: np.ndarray,
    centered: bool,
    *,
    spatial_axes=None,
) -> np.ndarray:
    """
    Expand two selected image axes by a factor of two.

    Parameters
    ----------
    image : np.ndarray
        Input scalar image or array containing batch/channel dimensions.
    h : np.ndarray
        1D expand filter.
    centered : bool
        True if using centered expand logic.
    spatial_axes : sequence of int, optional
        Exactly two dimensions to expand.  Required for higher-rank arrays;
        every other dimension is preserved.

    Returns
    -------
    np.ndarray
        Array with selected lengths doubled when they exceed one.  Singleton
        selected dimensions stay singleton.
    """
    image, h, axes = _validate_spatial_input(image, h, spatial_axes)
    centered = _validate_centered(centered)
    return _apply_axis(
        _apply_axis(image, h, centered, axes[1], expand=True),
        h,
        centered,
        axes[0],
        expand=True,
    )


# -------------------------------------------------------------------------
# 5) Internal 1D Routines
# -------------------------------------------------------------------------


def _reduce_standard_1d(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Standard (non-centered) 1D reduction by factor of 2,
    mirror boundary conditions. Matches 'ReduceStandard_1D' from the C code.
    """
    n = x.shape[-1]
    half = n // 2 if n >= 2 else 1
    centers = 2 * np.arange(half)
    values = x[..., centers] * g[0]
    if g.size > 1:
        offsets = np.arange(1, g.size)
        left = _wrap_reflect_array(centers[:, np.newaxis] - offsets, n)
        right = _wrap_reflect_array(centers[:, np.newaxis] + offsets, n)
        values = values + np.sum((x[..., left] + x[..., right]) * g[1:], axis=-1)
    return values.astype(x.dtype, copy=False)


def _expand_standard_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Standard (non-centered) 1D expansion by factor of 2,
    mirror boundary conditions. Matches "ExpandStandard_1D" from the C code.
    """
    n = x.shape[-1]
    outlen = 2 * n if n > 1 else n
    y = np.zeros(x.shape[:-1] + (outlen,), dtype=x.dtype)

    # trivial cases
    if n < 2:
        return x.copy()
    if h.size < 2:
        # replicate each sample
        return np.repeat(x, 2, axis=-1)

    # The C code loops over i in [0..outlen-1],
    # then handles pairs (i-k)/2 and (i+k)/2 for even/odd offsets.
    for parity in (0, 1):
        positions = np.arange(parity, outlen, 2)
        left_offsets = np.arange(parity, h.size, 2)
        right_offsets = np.arange(2 - parity, h.size, 2)
        values = np.zeros(x.shape[:-1] + (positions.size,), dtype=np.result_type(x, h))
        if left_offsets.size:
            left = _wrap_reflect_array(
                (positions[:, np.newaxis] - left_offsets) // 2, n
            )
            values += np.sum(x[..., left] * h[left_offsets], axis=-1)
        if right_offsets.size:
            right = _wrap_reflect_array(
                (positions[:, np.newaxis] + right_offsets) // 2, n
            )
            values += np.sum(x[..., right] * h[right_offsets], axis=-1)
        y[..., positions] = values
    return y


def _reduce_centered_1d(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    1D reduction with 'centered' pyramid logic:
      1) Convolve with g[] in a symmetrical manner,
      2) Then downsample with a Haar step (mean of pairs).
    This matches 'ReduceCentered_1D' from your C code.
    """
    n = x.shape[-1]
    half = n // 2 if n >= 2 else 1
    ytmp = np.zeros_like(x)

    # (a) convolve each sample with mirror boundary
    #     The "centered" code in C used period=2*n for reflection indexing
    #     Then if index >= n => index=2*n-1-index
    positions = np.arange(n)
    values = x * g[0]
    if g.size > 1:
        offsets = np.arange(1, g.size)
        minus = np.mod(positions[:, np.newaxis] - offsets, 2 * n)
        plus = np.mod(positions[:, np.newaxis] + offsets, 2 * n)
        minus = np.where(minus >= n, 2 * n - 1 - minus, minus)
        plus = np.where(plus >= n, 2 * n - 1 - plus, plus)
        values = values + np.sum((x[..., minus] + x[..., plus]) * g[1:], axis=-1)
    ytmp[...] = values

    # (b) downsample 2->1 by averaging pairs
    return 0.5 * (ytmp[..., : 2 * half : 2] + ytmp[..., 1 : 2 * half : 2])


def _expand_centered_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    1D expansion with 'centered' pyramid logic.
    This is the inverse of _reduce_centered_1d:
      1) Upsample with inverse Haar,
      2) Convolve with mirror boundary using h[].

    The logic is taken from 'ExpandCentered_1D' in the original code.
    """

    n = x.shape[-1]
    outlen = 2 * n if n > 1 else n
    y = np.zeros(x.shape[:-1] + (outlen,), dtype=x.dtype)
    if n < 2:
        return x.copy()

    # Step 1) "inverse Haar":
    # Expand x from length n -> length 2n
    # If we consider that the reduce step was (y[i] = (xtmp[2i] + xtmp[2i+1])/2 ),
    # then the inverse is:
    #   xtmp[2i]   = x[i],
    #   xtmp[2i+1] = x[i]  (naive, but the original code did some partial shift).
    # More precisely, from your C code ExpandCentered_1D does partial sum:
    #   for j in reversed range(1..2n-1): y[j] = (y[j] + y[j-1])/2.
    # We'll do a simpler approach: we place x in the even positions, then do
    # half-lifting to fill the odd.
    # For a direct replicate of the C logic, see "ExpandCentered_1D" code.

    # We'll first upsample x into an intermediate "tmp_upsampled" of length 2n
    tmp_upsampled = 0.5 * np.repeat(x, 2, axis=-1)

    # Step 2) convolve with h[] with mirror boundary (like the forward pass but reversed).
    # We'll write the result into y:
    positions = np.arange(outlen)
    values = tmp_upsampled * h[0]
    if h.size > 1:
        offsets = np.arange(1, h.size)
        minus = np.mod(positions[:, np.newaxis] - offsets, 2 * outlen)
        plus = np.mod(positions[:, np.newaxis] + offsets, 2 * outlen)
        minus = np.where(minus >= outlen, 2 * outlen - 1 - minus, minus)
        plus = np.where(plus >= outlen, 2 * outlen - 1 - plus, plus)
        values = values + np.sum(
            (tmp_upsampled[..., minus] + tmp_upsampled[..., plus]) * h[1:],
            axis=-1,
        )
    y[...] = values
    return y
