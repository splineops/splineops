"""
splinewaveletstool.py
---------------------
Python translation of SplineWaveletsTool.java logic. 
Uses a SplineFilter to get h[] and g[], then does 'splitMirror' and 
'mergeMirror' passes on rows, then columns.
"""

import numpy as np
from .splinefilter import SplineFilter

def split_mirror_1d(vin: np.ndarray, h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Replicates the 'splitMirror(...)' logic from Java:
      - Low pass part into the first half
      - High pass part into the second half
      - uses period=2*(n-1) mirror boundary for negative indices
      - etc.
    """
    n = vin.shape[0]
    # allocate output
    vout = np.zeros(n, dtype=vin.dtype)
    half = n // 2
    period = 2*(n - 1)  # mirror period

    for i in range(half):
        j = 2*i
        # low pass
        pix = vin[j] * h[0]
        for k in range(1, len(h)):
            jm = j - k
            if jm < 0: 
                jm = jm % period
                if jm >= n:
                    jm = period - jm
            jp = j + k
            if jp >= n:
                jp = jp % period
                if jp >= n:
                    jp = period - jp
            pix += h[k]*(vin[jm] + vin[jp])
        vout[i] = pix

        # high pass
        j2 = j + 1
        pix = vin[j2] * g[0]
        for k in range(1, len(g)):
            jm = j2 - k
            if jm < 0:
                jm = jm % period
                if jm >= n:
                    jm = period - jm
            jp = j2 + k
            if jp >= n:
                jp = jp % period
                if jp >= n:
                    jp = period - jp
            pix += g[k]*(vin[jm] + vin[jp])
        vout[i+half] = pix

    return vout

def merge_mirror_1d(vin: np.ndarray, h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Replicates the 'mergeMirror(...)' logic from Java:
      - inverse of 'splitMirror'
      - uses separate expansions for low pass/h pass 
      - merges them into a 2x-larger array.
    """
    n = vin.shape[0]
    vout = np.zeros(n, dtype=vin.dtype)
    half = n // 2
    # from code: period=2*n2 -1 => n2=half => period=2*half -1
    period = 2*half - 1 if half>0 else 1

    # The Java code has a big loop that reconstructs pairs. 
    # We'll replicate the logic. (See your C or Java mergeMirror code.)
    for i in range(half):
        j = 2*i

        # 1) pix1 for the low pass portion
        pix1 = h[0]*vin[i]
        # "for k in range(2, nh, step=2):" etc...
        # We'll replicate the exact loops from your code 
        # or do a condensed approach. 
        # For brevity, here is a direct approach matching your Java lines:

        # (a) gather from h[] with even steps 
        for k in range(2, len(h), 2):
            i1 = i - (k // 2)
            if i1 < 0:
                i1 = (-i1) % period
                if i1 >= half:
                    i1 = period - i1
            i2 = i + (k // 2)
            if i2 >= half:
                i2 = i2 % period
                if i2 >= half:
                    i2 = period - i2
            pix1 += h[k]*(vin[i1] + vin[i2])

        # 2) pix2 for the high pass portion
        pix2 = 0.0
        # The Java code does a loop for k in [-k02..ng..2]
        # We'll do a direct translation:

        k02 = (len(g)//2)*2 - 1  # e.g. if g.size=47 => k02= 46-1=45
        for k in range(-k02, len(g), 2):
            kk = abs(k)
            i1 = i + (k -1)//2
            if i1 < 0:
                i1 = (-i1 -1) % period
                if i1 >= half:
                    i1 = (period -1) - i1
            if i1 >= half:
                i1 = i1 % period
                if i1 >= half:
                    i1 = (period -1) - i1
            pix2 += g[kk]* vin[i1+half]

        vout[j] = pix1 + pix2

        # next sample j+1
        j += 1
        pix1 = 0.0
        k01 = (len(h)//2)*2 -1
        for kk in range(-k01, len(h), 2):
            kabs = abs(kk)
            i1 = i + (kk+1)//2
            if i1 < 0:
                i1 = (-i1) % period
                if i1 >= half:
                    i1 = period - i1
            if i1 >= half:
                i1 = i1 % period
                if i1 >= half:
                    i1 = period - i1
            pix1 += h[kabs]* vin[i1]

        pix2 = g[0]* vin[i+half]
        for k in range(2, len(g), 2):
            i1 = i - (k//2)
            if i1 < 0:
                i1 = (-i1 -1) % period
                if i1 >= half:
                    i1 = (period -1) - i1
            i2 = i + (k//2)
            if i2 >= half:
                i2 = i2 % period
                if i2 >= half:
                    i2 = (period -1) - i2
            pix2 += g[k]*( vin[i1+half] + vin[i2+half] )

        vout[j] = pix1 + pix2

    return vout


class SplineWaveletsTool:
    """
    Python equivalent of SplineWaveletsTool.java.
    Holds a SplineFilter, and offers analysis1() / synthesis1() 
    that do row -> column passes with splitMirror / mergeMirror.
    """

    def __init__(self, scale: int, order: int):
        self.scale = scale
        # load h/g from SplineFilter
        self.filters = SplineFilter(order)
        # self.filters.h => lowpass
        # self.filters.g => highpass

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        2D version of analysis1:
         1) For each row => splitMirror
         2) For each column => splitMirror
        returns a new array. 
        (For 3D, you'd also do a z-pass.)
        """
        out = np.copy(inp)
        ny, nx = out.shape

        # Row pass
        for y in range(ny):
            row_in = out[y, :]
            row_out = split_mirror_1d(row_in, self.filters.h, self.filters.g)
            out[y, :] = row_out

        # Column pass
        for x in range(nx):
            col_in = out[:, x]
            col_out = split_mirror_1d(col_in, self.filters.h, self.filters.g)
            out[:, x] = col_out

        return out

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        2D version of synthesis1:
         1) For each column => mergeMirror
         2) For each row => mergeMirror
        returns a new array.
        (3D would add a z pass.)
        """
        out = np.copy(inp)
        ny, nx = out.shape

        # Column pass
        for x in range(nx):
            col_in = out[:, x]
            col_out = merge_mirror_1d(col_in, self.filters.h, self.filters.g)
            out[:, x] = col_out

        # Row pass
        for y in range(ny):
            row_in = out[y, :]
            row_out = merge_mirror_1d(row_in, self.filters.h, self.filters.g)
            out[y, :] = row_out

        return out
