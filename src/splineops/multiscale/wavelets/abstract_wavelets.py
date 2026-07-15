# splineops/src/splineops/multiscale/wavelets/abstract_wavelets.py

"""
abstract_wavelets.py
--------------------
Defines a base class for wavelet analysis & synthesis on 2D (or 3D) signals.
"""

import numpy as np
import operator

_BATCH_WORKING_SET_BYTES = 1024**2


class AbstractWavelets:
    """
    Base class for wavelet decomposition with multi-scale analysis & synthesis.

    Subclasses must override:
      - analysis1()  (single-scale wavelet decomposition)
      - synthesis1() (single-scale wavelet reconstruction)

    Parameters
    ----------
    scales : int
        Number of scales for multi-scale decomposition.

    Attributes
    ----------
    scales : int
        Number of scales for repeated analysis/synthesis steps.
    """

    def __init__(self, scales=3):
        self.set_scale(scales)

    def set_scale(self, scale: int):
        """
        Update the number of scales.

        Parameters
        ----------
        scale : int
            New scale value.
        """
        try:
            scale = operator.index(scale)
        except TypeError as exc:
            raise TypeError("'scales' must be a positive integer.") from exc
        if isinstance(scale, (bool, np.bool_)) or scale < 1:
            raise ValueError("'scales' must be a positive integer.")
        self.scales = scale

    def _validate_multiscale_input(self, inp):
        out = self._prepare_single_scale_input(inp)
        divisor = 2**self.scales
        if any(length % divisor for length in out.shape[-2:]):
            raise ValueError(
                "Both wavelet dimensions must be divisible by "
                f"2**scales ({divisor}); received shape {out.shape[-2:]}."
            )
        return out

    def _prepare_single_scale_input(self, inp):
        if not isinstance(inp, np.ndarray):
            raise TypeError("Wavelet input must be a NumPy array.")
        if inp.ndim < 2 or any(length == 0 for length in inp.shape):
            raise ValueError(
                "Wavelet input must have two non-empty trailing wavelet dimensions."
            )
        if not np.issubdtype(inp.dtype, np.number) or np.iscomplexobj(inp):
            raise TypeError("Wavelet input must have a real numeric dtype.")
        if not np.all(np.isfinite(inp)):
            raise ValueError("Wavelet input must contain only finite values.")
        dtype = (
            inp.dtype if np.issubdtype(inp.dtype, np.floating) else np.dtype(np.float64)
        )
        return np.array(inp, dtype=dtype, copy=True, order="C")

    def _prepare_spatial_input(self, inp, spatial_axes):
        if not isinstance(inp, np.ndarray):
            raise TypeError("Wavelet input must be a NumPy array.")
        if spatial_axes is None:
            if inp.ndim != 2:
                raise ValueError(
                    "'spatial_axes' is required when wavelet input contains "
                    "batch or channel axes."
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
            axes = tuple(axis + inp.ndim if axis < 0 else axis for axis in axes)
            if any(axis < 0 or axis >= inp.ndim for axis in axes) or axes[0] == axes[1]:
                raise ValueError("'spatial_axes' must contain two distinct valid axes.")
        if any(length == 0 for length in inp.shape):
            raise ValueError("Wavelet input dimensions must be non-empty.")
        if not np.issubdtype(inp.dtype, np.number) or np.iscomplexobj(inp):
            raise TypeError("Wavelet input must have a real numeric dtype.")
        if not np.all(np.isfinite(inp)):
            raise ValueError("Wavelet input must contain only finite values.")
        spatial_shape = tuple(inp.shape[axis] for axis in axes)
        divisor = 2**self.scales
        if any(length % divisor for length in spatial_shape):
            raise ValueError(
                "Both wavelet dimensions must be divisible by "
                f"2**scales ({divisor}); received shape {spatial_shape}."
            )
        dtype = (
            inp.dtype if np.issubdtype(inp.dtype, np.floating) else np.dtype(np.float64)
        )
        nonspatial_axes = tuple(axis for axis in range(inp.ndim) if axis not in axes)
        permutation = nonspatial_axes + axes
        canonical = np.array(
            np.transpose(inp, permutation), dtype=dtype, copy=True, order="C"
        )
        return canonical, spatial_shape, permutation

    @staticmethod
    def _restore_spatial_output(canonical, permutation):
        return np.ascontiguousarray(
            np.transpose(canonical, tuple(np.argsort(permutation)))
        )

    @staticmethod
    def _spatial_chunks(canonical, spatial_shape):
        """Yield cache-sized groups of independent spatial planes."""

        if canonical.ndim == 2:
            yield canonical
            return
        flattened = canonical.reshape((-1,) + spatial_shape)
        plane_bytes = int(np.prod(spatial_shape, dtype=np.int64)) * canonical.itemsize
        chunk_size = max(1, _BATCH_WORKING_SET_BYTES // plane_bytes)
        for start in range(0, flattened.shape[0], chunk_size):
            yield flattened[start : start + chunk_size]

    def analysis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale wavelet transform of inp -> out. Must be overridden.

        Parameters
        ----------
        inp : np.ndarray
            Input array (2D or 3D typically).

        Returns
        -------
        np.ndarray
            Transformed array (same shape).
        """
        raise NotImplementedError

    def synthesis1(self, inp: np.ndarray) -> np.ndarray:
        """
        Single-scale inverse wavelet transform of inp -> out. Must be overridden.

        Parameters
        ----------
        inp : np.ndarray
            Input wavelet coefficients (single scale).

        Returns
        -------
        np.ndarray
            Reconstructed array at that scale.
        """
        raise NotImplementedError

    def analysis(self, inp: np.ndarray, *, spatial_axes=None) -> np.ndarray:
        """
        Multi-scale wavelet analysis on two selected dimensions.
        Repeatedly calls analysis1() from fine to coarse.

        Parameters
        ----------
        inp : np.ndarray
            Scalar image or array containing batch/channel dimensions.
        spatial_axes : sequence of int, optional
            Exactly two wavelet dimensions.  Required for higher-rank arrays;
            every remaining slice is transformed independently.

        Returns
        -------
        np.ndarray
            Full wavelet decomposition (in-place layout).
        """
        out, spatial_shape, permutation = self._prepare_spatial_input(inp, spatial_axes)
        for chunk in self._spatial_chunks(out, spatial_shape):
            ny, nx = spatial_shape
            for _ in range(self.scales):
                sub = chunk[..., :ny, :nx]
                chunk[..., :ny, :nx] = self.analysis1(sub)
                nx = max(1, nx // 2)
                ny = max(1, ny // 2)
        return self._restore_spatial_output(out, permutation)

    def synthesis(self, inp: np.ndarray, *, spatial_axes=None) -> np.ndarray:
        """
        Multi-scale wavelet synthesis on two selected dimensions.
        Repeatedly calls synthesis1() from coarsest to finest.

        Parameters
        ----------
        inp : np.ndarray
            Wavelet decomposition array (same shape as original).
        spatial_axes : sequence of int, optional
            Exactly two wavelet dimensions.  Required for higher-rank arrays;
            every remaining slice is reconstructed independently.

        Returns
        -------
        np.ndarray
            Reconstructed array (same shape as input).
        """
        out, spatial_shape, permutation = self._prepare_spatial_input(inp, spatial_axes)
        ny_full, nx_full = spatial_shape
        factor = 2 ** (self.scales - 1)
        nx_coarse = max(1, nx_full // factor)
        ny_coarse = max(1, ny_full // factor)
        for chunk in self._spatial_chunks(out, spatial_shape):
            nx, ny = nx_coarse, ny_coarse
            for _ in range(self.scales):
                sub = chunk[..., :ny, :nx]
                chunk[..., :ny, :nx] = self.synthesis1(sub)
                nx = min(nx_full, nx * 2)
                ny = min(ny_full, ny * 2)
        return self._restore_spatial_output(out, permutation)

    def get_name(self) -> str:
        """
        Returns a descriptive name for the wavelet transform.
        """
        return "AbstractWavelets"

    def get_documentation(self) -> str:
        """
        Returns a short description of the wavelet.
        """
        return "Base class for wavelets."
