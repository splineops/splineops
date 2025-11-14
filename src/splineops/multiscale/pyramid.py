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
        g = np.array([
            0.596797, 0.313287, -0.0827691, -0.0921993,
            0.0540288, 0.0436996, -0.0302508, -0.0225552,
            0.0162251, 0.0118738, -0.00861788, -0.00627964,
            0.00456713, 0.00332464, -0.00241916, -0.00176059,
            0.00128128, 0.000932349, -0.000678643, -0.000493682
        ])
        h = np.array([
            1.0, 0.600481, 0.0, -0.127405,
            0.0, 0.034138, 0.0, -0.00914725,
            0.0, 0.002451, 0.0, -0.000656743
        ])
        is_centered = False

    elif name == "centered spline" and order == 3:
        # Example: from your PyramidFilterCentered(...) with order=3
        g = np.array([
            0.708792, 0.328616, -0.165157, -0.114448, 
            0.0944036, 0.0543881, -0.05193, -0.0284868,
            0.0281854, 0.0152877, -0.0152508, -0.00825077,
            0.00824629, 0.00445865, -0.0044582, -0.00241009,
            0.00241022, 0.00130278, -0.00130313, -0.000704109,
            0.000704784
        ])
        h = np.array([
            1.13726, 0.625601, -0.0870191, -0.159256,
            0.0233167, 0.0426725, -0.00624769, -0.0114341,
            0.00167406, 0.00306375, -0.000448564, -0.000820929,
            0.000120192, 0.000219967, -3.22054e-05, -5.894e-05
        ])
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
    period = 2*(n - 1)
    i = i % period  # now in [0..period-1]
    if i >= n:
        i = period - i  # reflect
    return i


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
    n = signal.shape[0]
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
    n = signal.shape[0]
    outlen = 2*n if n >= 2 else n
    out = np.zeros(outlen, dtype=signal.dtype)

    if centered:
        out[:] = _expand_centered_1d(signal, h)
    else:
        out[:] = _expand_standard_1d(signal, h)
    return out


# -------------------------------------------------------------------------
# 4) 2D Reduce & Expand
# -------------------------------------------------------------------------

def reduce_2d(image: np.ndarray, g: np.ndarray, centered: bool) -> np.ndarray:
    """
    Reduce a 2D image by factor of 2 in each dimension.

    Parameters
    ----------
    image : np.ndarray
        Input 2D array (ny, nx).
    g : np.ndarray
        1D reduce filter.
    centered : bool
        True if using centered reduce logic.

    Returns
    -------
    np.ndarray
        Reduced image of shape (ny//2, nx//2) if ny,nx >= 2, else smaller.
    """
    ny, nx = image.shape
    # 1) reduce along X for each row
    row_reduced = []
    for y in range(ny):
        rowdata = image[y, :]
        row_out = reduce_1d(rowdata, g, centered)
        row_reduced.append(row_out)
    tmp = np.vstack(row_reduced)  # shape: (ny, NxOut)

    # 2) reduce along Y for each column
    NxOut = tmp.shape[1]
    NyOut = ny // 2 if ny >= 2 else ny
    out = np.zeros((NyOut, NxOut), dtype=tmp.dtype)
    for x in range(NxOut):
        coldata = tmp[:, x]
        col_out = reduce_1d(coldata, g, centered)
        out[:, x] = col_out

    return out


def expand_2d(image: np.ndarray, h: np.ndarray, centered: bool) -> np.ndarray:
    """
    Expand a 2D image by factor of 2 in each dimension.

    Parameters
    ----------
    image : np.ndarray
        Input 2D array (coarse scale).
    h : np.ndarray
        1D expand filter.
    centered : bool
        True if using centered expand logic.

    Returns
    -------
    np.ndarray
        Expanded image, roughly 2*ny by 2*nx.
    """
    ny, nx = image.shape
    NxOut = nx * 2 if nx >= 2 else nx
    NyOut = ny * 2 if ny >= 2 else ny

    # 1) expand along X for each row
    row_expanded = []
    for y in range(ny):
        rowdata = image[y, :]
        row_out = expand_1d(rowdata, h, centered)
        row_expanded.append(row_out)
    tmp = np.vstack(row_expanded)  # shape: (ny, NxOut)

    # 2) expand along Y for each column
    out = np.zeros((NyOut, NxOut), dtype=tmp.dtype)
    for x in range(NxOut):
        coldata = tmp[:, x]
        col_out = expand_1d(coldata, h, centered)
        out[:, x] = col_out

    return out


# -------------------------------------------------------------------------
# 5) Internal 1D Routines
# -------------------------------------------------------------------------

def _reduce_standard_1d(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Standard (non-centered) 1D reduction by factor of 2, 
    mirror boundary conditions. Matches 'ReduceStandard_1D' from the C code.
    """
    n = x.shape[0]
    half = n // 2 if n >= 2 else 1
    y = np.zeros(half, dtype=x.dtype)

    for kk in range(half):
        k = 2 * kk
        val = x[k] * g[0]
        for i in range(1, g.size):
            i1 = wrap_reflect(k - i, n)
            i2 = wrap_reflect(k + i, n)
            val += g[i] * (x[i1] + x[i2])
        y[kk] = val
    return y


def _expand_standard_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Standard (non-centered) 1D expansion by factor of 2,
    mirror boundary conditions. Matches "ExpandStandard_1D" from the C code.
    """
    n = x.shape[0]
    outlen = 2*n if n > 1 else n
    y = np.zeros(outlen, dtype=x.dtype)

    # trivial cases
    if n < 2:
        return x.copy()
    if h.size < 2:
        # replicate each sample
        for i in range(n):
            j = 2*i
            y[j] = x[i]
            if j+1 < outlen:
                y[j+1] = x[i]
        return y

    # The C code loops over i in [0..outlen-1], 
    # then handles pairs (i-k)/2 and (i+k)/2 for even/odd offsets.
    for i in range(outlen):
        val = 0.0
        # a) loop for k in [ (i % 2), h.size, step=2 ]
        for k in range(i % 2, h.size, 2):
            i1 = (i - k) // 2
            i1 = wrap_reflect(i1, n)
            val += h[k]* x[i1]
        # b) loop for k in [ 2-(i % 2), h.size, step=2 ]
        for k in range(2 - (i % 2), h.size, 2):
            i2 = (i + k) // 2
            i2 = wrap_reflect(i2, n)
            val += h[k]* x[i2]

        y[i] = val
    return y


def _reduce_centered_1d(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    1D reduction with 'centered' pyramid logic:
      1) Convolve with g[] in a symmetrical manner,
      2) Then downsample with a Haar step (mean of pairs).
    This matches 'ReduceCentered_1D' from your C code.
    """
    n = x.shape[0]
    half = n // 2 if n >= 2 else 1
    ytmp = np.zeros(n, dtype=x.dtype)

    # (a) convolve each sample with mirror boundary
    #     The "centered" code in C used period=2*n for reflection indexing
    #     Then if index >= n => index=2*n-1-index
    for k in range(n):
        val = x[k]*g[0]
        for i in range(1, g.size):
            km = (k - i) % (2*n)
            if km >= n:
                km = 2*n - 1 - km
            kp = (k + i) % (2*n)
            if kp >= n:
                kp = 2*n - 1 - kp
            val += g[i]*(x[km] + x[kp])
        ytmp[k] = val

    # (b) downsample 2->1 by averaging pairs
    out = np.zeros(half, dtype=x.dtype)
    for i in range(half):
        k = 2*i
        out[i] = 0.5*(ytmp[k] + ytmp[k+1])
    return out


def _expand_centered_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    1D expansion with 'centered' pyramid logic. 
    This is the inverse of _reduce_centered_1d:
      1) Upsample with inverse Haar,
      2) Convolve with mirror boundary using h[].

    The logic is taken from 'ExpandCentered_1D' in the original code.
    """

    n = x.shape[0]
    outlen = 2*n if n > 1 else n
    y = np.zeros(outlen, dtype=x.dtype)
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
    tmp_upsampled = np.zeros(outlen, dtype=x.dtype)
    for i in range(n):
        j = 2*i
        tmp_upsampled[j] = x[i]
    # Next do the half-ladder: y[j] = (y[j]+ y[j-1])/2 for j=1..end
    for j in range(outlen-1, 0, -1):
        tmp_upsampled[j] = 0.5*(tmp_upsampled[j] + tmp_upsampled[j-1])
    tmp_upsampled[0] *= 0.5

    # Step 2) convolve with h[] with mirror boundary (like the forward pass but reversed).
    # We'll write the result into y:
    for k in range(outlen):
        val = tmp_upsampled[k]*h[0]
        for i in range(1, h.size):
            km = (k - i) % (2*outlen)  # bigger period for reflection
            if km >= outlen:
                km = 2*outlen - 1 - km
            kp = (k + i) % (2*outlen)
            if kp >= outlen:
                kp = 2*outlen - 1 - kp
            val += h[i]*(tmp_upsampled[km] + tmp_upsampled[kp])
        y[k] = val

    return y
