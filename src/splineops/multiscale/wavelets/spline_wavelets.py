# splineops/src/splineops/multiscale/wavelets/spline_wavelets.py

"""
spline_wavelets.py
------------------
Implements Spline wavelet transforms of orders 1, 3, 5.
"""

import numpy as np
from .abstract_wavelets import AbstractWavelets
from .spline_filter import SplineFilter


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

    @property
    def exact_reconstruction(self):
        """Whether the bundled tap precision supports a perfect-reconstruction claim."""

        return not self.filter.approximate

    @property
    def reconstruction_contract(self):
        """Published reconstruction classification for the selected filter."""

        return "perfect" if self.exact_reconstruction else "bounded-approximation"

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
        out = self._prepare_single_scale_input(inp)
        ny, nx = out.shape[-2:]
        if ny < 2 or nx < 2 or ny % 2 or nx % 2:
            raise ValueError(
                "Spline wavelets require even dimensions of at least two; "
                f"received {out.shape[-2:]}."
            )

        out = self._apply_axis(out, axis=-1, split=True)
        return self._apply_axis(out, axis=-2, split=True)

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
        out = self._prepare_single_scale_input(inp)
        ny, nx = out.shape[-2:]
        if ny < 2 or nx < 2 or ny % 2 or nx % 2:
            raise ValueError(
                "Spline wavelets require even dimensions of at least two; "
                f"received {out.shape[-2:]}."
            )

        out = self._apply_axis(out, axis=-2, split=False)
        return self._apply_axis(out, axis=-1, split=False)

    def _apply_axis(self, array, axis, *, split):
        moved = np.moveaxis(array, axis, -1)
        function = self._split_mirror_1d if split else self._merge_mirror_1d
        transformed = function(moved, self.filter.h, self.filter.g)
        return np.moveaxis(transformed, -1, axis)

    # -----------------------------------------------------------------------
    # The key mirror-based 1D "split" (analysis) and "merge" (synthesis)
    # adapted from your SplineWaveletsTool.java => 'splitMirror()' & 'mergeMirror()'.
    # -----------------------------------------------------------------------

    def _split_mirror_1d(
        self, vin: np.ndarray, h: np.ndarray, g: np.ndarray
    ) -> np.ndarray:
        """
        1D mirror-based split for low-pass & high-pass.
        The first half of the output is the lowpass, the second half is the detail.
        """
        n = vin.shape[-1]
        half = n // 2
        period = 2 * (n - 1) if n > 1 else 1
        centers = 2 * np.arange(half)

        def reflect(index):
            index = np.mod(index, period)
            return np.where(index >= n, period - index, index)

        low = vin[..., centers] * h[0]
        for k in range(1, len(h)):
            low = low + h[k] * (
                vin[..., reflect(centers - k)] + vin[..., reflect(centers + k)]
            )
        high_centers = centers + 1
        high = vin[..., high_centers] * g[0]
        for k in range(1, len(g)):
            high = high + g[k] * (
                vin[..., reflect(high_centers - k)]
                + vin[..., reflect(high_centers + k)]
            )
        return np.concatenate((low, high), axis=-1).astype(vin.dtype, copy=False)

    def _merge_mirror_1d(
        self, vin: np.ndarray, h: np.ndarray, g: np.ndarray
    ) -> np.ndarray:
        """
        Inverse of _split_mirror_1d.
        """
        n = vin.shape[-1]
        vout = np.zeros_like(vin)
        half = n // 2
        if half < 1:
            return vin.copy()

        period = 2 * half - 1 if half > 1 else 1

        # We'll replicate the loops from your SplineWaveletsTool.java mergeMirror code:
        # for i in [0..n2-1]:
        #   j = 2*i
        #   ...
        #   j+1 ...
        #   etc.
        k01 = (len(h) // 2) * 2 - 1
        k02 = (len(g) // 2) * 2 - 1

        indices = np.arange(half)

        # Even reconstructed samples.
        pix1 = h[0] * vin[..., indices]
        for k in range(2, len(h), 2):
            i1 = indices - (k // 2)
            i1 = np.where(i1 < 0, np.mod(-i1, period), i1)
            i1 = np.where(i1 >= half, period - i1, i1)
            i2 = indices + (k // 2)
            i2 = np.mod(i2, period)
            i2 = np.where(i2 >= half, period - i2, i2)
            pix1 = pix1 + h[k] * (vin[..., i1] + vin[..., i2])

            # pix2 => from highpass portion (g filter)
        pix2 = np.zeros_like(pix1)
        for k in range(-k02, len(g), 2):
            i1 = indices + (k - 1) // 2
            i1 = np.where(i1 < 0, np.mod(-i1 - 1, period), i1)
            i1 = np.mod(i1, period)
            i1 = np.where(i1 >= half, (period - 1) - i1, i1)
            pix2 = pix2 + g[abs(k)] * vin[..., i1 + half]
        vout[..., 0::2] = pix1 + pix2

        # Next sample j+1
        pix1 = np.zeros_like(pix1)
        for k in range(-k01, len(h), 2):
            i1 = indices + (k + 1) // 2
            i1 = np.where(i1 < 0, np.mod(-i1, period), i1)
            i1 = np.mod(i1, period)
            i1 = np.where(i1 >= half, period - i1, i1)
            pix1 = pix1 + h[abs(k)] * vin[..., i1]

        pix2 = g[0] * vin[..., indices + half]
        for k in range(2, len(g), 2):
            i1 = indices - (k // 2)
            i1 = np.where(i1 < 0, np.mod(-i1 - 1, period), i1)
            i1 = np.where(i1 >= half, (period - 1) - i1, i1)
            i2 = np.mod(indices + (k // 2), period)
            i2 = np.where(i2 >= half, (period - 1) - i2, i2)
            pix2 = pix2 + g[k] * (vin[..., i1 + half] + vin[..., i2 + half])
        vout[..., 1::2] = pix1 + pix2

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
    """Approximate order-5 spline wavelets using limited-precision source taps."""

    def __init__(self, scales=3):
        super().__init__(scales=scales, order=5)

    def get_name(self):
        return "Spline5"

    def get_documentation(self):
        return "Spline Wavelets (order=5, bounded approximate reconstruction)"
