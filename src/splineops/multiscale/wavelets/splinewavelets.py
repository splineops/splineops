# splineops/src/splineops/multiscale/wavelets/splinewavelets.py

"""
splinewavelets.py
-----------------
Implements Spline wavelet transforms of orders 1, 3, 5.
"""

import numpy as np
from .abstractwavelets import AbstractWavelets
from .splinefilter import SplineFilter

class SplineWavelets(AbstractWavelets):
    """
    A generic spline wavelet class that references a given 'order' (1,3,5).
    Uses row->column passes with mirror boundary to compute detail.

    Parameters
    ----------
    scales : int
        Number of scales.
    order : int
        Spline order (1,3,5).

    Attributes
    ----------
    filter : SplineFilter
        Holds the arrays h[] (lowpass) and g[] (highpass).
    """

    def __init__(self, scales=3, order=3):
        super().__init__(scales=scales)
        self.order = order
        self.filter = SplineFilter(order)

    def get_name(self):
        """Return 'Spline{order}'."""
        return f"Spline{self.order}"

    def get_documentation(self):
        """Return a short docstring describing the spline wavelet order."""
        return f"Spline Wavelets (order={self.order})."

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale 2D spline wavelet analysis pass:
        - Row pass (splitMirror)
        - Column pass (splitMirror)

        Parameters
        ----------
        inp : np.ndarray
            2D array.

        Returns
        -------
        np.ndarray
            Transformed 2D array (same shape).
        """
        out = np.copy(inp)
        ny, nx = out.shape

        # 1) Row pass if nx>1
        if nx > 1:
            for r in range(ny):
                out[r, :] = self._split_mirror_1d(out[r, :], self.filter.h, self.filter.g)

        # 2) Column pass if ny>1
        if ny > 1:
            for c in range(nx):
                col = out[:, c]
                out[:, c] = self._split_mirror_1d(col, self.filter.h, self.filter.g)

        return out


    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale 2D spline wavelet synthesis pass:
        - Column pass (mergeMirror)
        - Row pass (mergeMirror)

        Parameters
        ----------
        inp : np.ndarray
            2D array of wavelet coefficients (one scale).

        Returns
        -------
        np.ndarray
            Reconstructed array (same shape).
        """
        out = np.copy(inp)
        ny, nx = out.shape

        # 1) Column pass if ny>1
        if ny > 1:
            for c in range(nx):
                out[:, c] = self._merge_mirror_1d(out[:, c], self.filter.h, self.filter.g)

        # 2) Row pass if nx>1
        if nx > 1:
            for r in range(ny):
                out[r, :] = self._merge_mirror_1d(out[r, :], self.filter.h, self.filter.g)

        return out

    # -----------------------------------------------------------------------
    # The key mirror-based 1D "split" (analysis) and "merge" (synthesis)
    # adapted from your SplineWaveletsTool.java => 'splitMirror()' & 'mergeMirror()'.
    # -----------------------------------------------------------------------

    def _split_mirror_1d(self, vin: np.ndarray, h: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        1D mirror-based split for low-pass & high-pass. 
        The first half of the output is the lowpass, the second half is the detail.
        """
        n = vin.shape[0]
        vout = np.zeros(n, dtype=vin.dtype)
        half = n // 2
        period = 2*(n - 1) if n>1 else 1

        for i in range(half):
            j = 2*i
            # Low pass
            pix_low = vin[j]*h[0]
            for k in range(1, len(h)):
                jm = j - k
                if jm<0:
                    jm = jm % period
                    if jm>=n:
                        jm = period - jm
                jp = j + k
                if jp>=n:
                    jp = jp % period
                    if jp>=n:
                        jp = period - jp
                pix_low += h[k]*(vin[jm]+vin[jp])
            vout[i] = pix_low

            # High pass
            j2 = j + 1
            if j2 >= n:
                j2 = j2 % period  # typically for odd n
            pix_high = vin[j2]*g[0]
            for k in range(1, len(g)):
                jm = j2 - k
                if jm<0:
                    jm = jm % period
                    if jm>=n:
                        jm = period - jm
                jp = j2 + k
                if jp>=n:
                    jp = jp % period
                    if jp>=n:
                        jp = period - jp
                pix_high += g[k]*(vin[jm]+vin[jp])
            vout[i+half] = pix_high

        return vout

    def _merge_mirror_1d(self, vin: np.ndarray, h: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Inverse of _split_mirror_1d.
        """
        n = vin.shape[0]
        vout = np.zeros(n, dtype=vin.dtype)
        half = n // 2
        if half < 1:
            return vin.copy()

        period = 2*half - 1 if half>1 else 1

        # We'll replicate the loops from your SplineWaveletsTool.java mergeMirror code:
        # for i in [0..n2-1]:
        #   j = 2*i
        #   ...
        #   j+1 ...
        #   etc.
        k01 = (len(h)//2)*2 - 1
        k02 = (len(g)//2)*2 - 1

        for i in range(half):
            j = 2*i

            # pix1 => from lowpass portion (h filter)
            pix1 = h[0]*vin[i]
            # loop k=2..(step2).. < len(h)
            for k in range(2, len(h), 2):
                i1 = i - (k//2)
                if i1<0:
                    i1 = (-i1) % period
                    if i1>=half:
                        i1 = period- i1
                i2 = i + (k//2)
                if i2>=half:
                    i2 = i2 % period
                    if i2>=half:
                        i2 = period- i2
                pix1 += h[k]*( vin[i1] + vin[i2] )

            # pix2 => from highpass portion (g filter)
            pix2 = 0.0
            for k in range(-k02, len(g), 2):
                kk = abs(k)
                i1 = i + (k-1)//2
                if i1<0:
                    i1 = (-i1-1) % period
                    if i1>=half:
                        i1 = (period-1)-i1
                if i1>=half:
                    i1 = i1 % period
                    if i1>=half:
                        i1 = (period-1)-i1
                pix2 += g[kk]* vin[i1+half]

            vout[j] = pix1 + pix2

            # Next sample j+1
            j = j+1
            pix1 = 0.0
            for k in range(-k01, len(h), 2):
                kk = abs(k)
                i1 = i + (k+1)//2
                if i1<0:
                    i1 = (-i1) % period
                    if i1>=half:
                        i1 = period- i1
                if i1>=half:
                    i1 = i1 % period
                    if i1>=half:
                        i1 = period- i1
                pix1 += h[kk]*vin[i1]

            pix2 = g[0]* vin[i+half]
            for k in range(2, len(g), 2):
                i1 = i - (k//2)
                if i1<0:
                    i1 = (-i1-1) % period
                    if i1>=half:
                        i1 = (period-1)-i1
                i2 = i + (k//2)
                if i2>=half:
                    i2 = i2 % period
                    if i2>=half:
                        i2 = (period-1)-i2
                pix2 += g[k]*( vin[i1+half] + vin[i2+half] )

            vout[j] = pix1 + pix2

        return vout


# ------------------------------------------------------------------------
# If you want to define specialized classes for each order:
#   Spline1Wavelets, Spline3Wavelets, Spline5Wavelets
# you can do so as well:
# ------------------------------------------------------------------------

class Spline1Wavelets(SplineWavelets):
    """Spline Wavelets of order=1."""
    def __init__(self, scales=3):
        super().__init__(scales=scales, order=1)

    def get_name(self):
        return "Spline1"

    def get_documentation(self):
        return "Spline Wavelets (order=1)"


class Spline3Wavelets(SplineWavelets):
    """Spline Wavelets of order=3."""
    def __init__(self, scales=3):
        super().__init__(scales=scales, order=3)

    def get_name(self):
        return "Spline3"

    def get_documentation(self):
        return "Spline Wavelets (order=3)"


class Spline5Wavelets(SplineWavelets):
    """Spline Wavelets of order=5."""
    def __init__(self, scales=3):
        super().__init__(scales=scales, order=5)

    def get_name(self):
        return "Spline5"

    def get_documentation(self):
        return "Spline Wavelets (order=5)"
