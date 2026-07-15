"""Reusable fixed-coordinate geometry plans for :class:`TensorSpline`."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tensor_spline import TensorSpline


class TensorSplineGeometryPlan:
    """A memory-capped support/weight plan for compatible tensor splines.

    Construct plans through :meth:`TensorSpline.query_plan`.  A plan is useful
    when the same coordinates are evaluated against changing sample values,
    such as successive image or volume frames.  Plan construction is extra
    work and retained memory, so one-shot calls should use the spline directly.

    The spline used at construction remains the default for backward
    compatibility.  Pass another compatible spline to :meth:`apply` to obtain
    the useful geometry-reuse behavior.
    """

    def __init__(
        self,
        spline: TensorSpline,
        coordinates,
        *,
        grid: bool,
        max_retained_bytes: int,
    ) -> None:
        try:
            max_retained_bytes = operator.index(max_retained_bytes)
        except TypeError as exc:
            raise TypeError("'max_retained_bytes' must be a positive integer.") from exc
        if isinstance(max_retained_bytes, (bool, np.bool_)) or max_retained_bytes <= 0:
            raise ValueError("'max_retained_bytes' must be a positive integer.")

        coordinates = spline._prepare_evaluation_coordinates(coordinates, grid=grid)
        if grid and any(coordinate.ndim != 1 for coordinate in coordinates):
            raise ValueError(
                "Query plans currently require unbatched grid coordinates."
            )

        self._default_spline: TensorSpline | None = spline
        self._geometry_signature = spline._geometry_signature()
        self._grid = bool(grid)
        self._query_shape = (
            tuple(coordinate.size for coordinate in coordinates)
            if grid
            else coordinates[0].shape
        )
        indexes_sequence = []
        weights_sequence = []
        retained_bytes = 0
        for axis, coordinate in enumerate(coordinates):
            indexes, weights = spline._compute_support(axis, coordinate.reshape(-1))
            indexes = indexes.copy()
            weights = weights.copy()
            retained_bytes += indexes.nbytes + weights.nbytes
            if retained_bytes > max_retained_bytes:
                raise MemoryError(
                    "TensorSpline query plan would retain "
                    f"{retained_bytes} bytes, above max_retained_bytes="
                    f"{max_retained_bytes}."
                )
            if isinstance(indexes, np.ndarray):
                indexes.flags.writeable = False
                weights.flags.writeable = False
            indexes_sequence.append(indexes)
            weights_sequence.append(weights)
        self._indexes = tuple(indexes_sequence)
        self._weights = tuple(weights_sequence)
        self._retained_bytes = retained_bytes

    @property
    def retained_bytes(self) -> int:
        """Bytes retained by support indexes and weights."""
        return self._retained_bytes

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape returned by :meth:`apply`."""
        return self._query_shape

    @property
    def grid(self) -> bool:
        """Whether the retained coordinates describe a tensor grid."""

        return self._grid

    @property
    def attached(self) -> bool:
        """Whether a construction spline remains available as the default."""

        return self._default_spline is not None

    def incompatibility_reason(self, spline: TensorSpline) -> str | None:
        """Explain why ``spline`` cannot use this geometry, or return ``None``."""

        from .tensor_spline import TensorSpline

        if not isinstance(spline, TensorSpline):
            return "The supplied object is not a TensorSpline."
        if spline._geometry_signature() != self._geometry_signature:
            return (
                "Construction coordinates, bases, modes, backend, shape, or "
                "real precision differ from this geometry plan."
            )
        return None

    def is_compatible(self, spline: TensorSpline) -> bool:
        """Return whether ``spline`` can be evaluated by this geometry plan."""

        return self.incompatibility_reason(spline) is None

    def __call__(self, spline: TensorSpline | None = None, *, out=None):
        return self.apply(spline, out=out)

    def detach(self) -> TensorSplineGeometryPlan:
        """Drop the construction spline while retaining reusable geometry.

        Detached plans must receive an explicit compatible spline in
        :meth:`apply`.  This is useful for higher-level reusable operators that
        should not retain the samples used to define their geometry.
        """

        self._default_spline = None
        return self

    def apply(self, spline: TensorSpline | None = None, *, out=None):
        """Evaluate a compatible spline using the precomputed geometry.

        Parameters
        ----------
        spline : TensorSpline, optional
            Spline with the same construction grid, bases, modes, backend and
            real precision as the spline used to construct this plan.  If
            omitted, the construction spline is evaluated for compatibility
            with the original experimental API.
        out : ndarray, optional
            Exact-shape, exact-dtype output buffer on the same array backend.
        """
        if spline is None:
            spline = self._default_spline
        if spline is None:
            raise TypeError(
                "A detached geometry plan requires an explicit compatible spline."
            )
        incompatibility = self.incompatibility_reason(spline)
        if incompatibility is not None:
            raise ValueError(
                "The supplied TensorSpline is incompatible with this geometry "
                f"plan: {incompatibility}"
            )
        if self._grid:
            result = spline._evaluate_separable_grid_from_support(
                self._indexes, self._weights
            )
        else:
            result = self._apply_points(spline)
        return spline._copy_result_to_output(result, out)

    def _apply_points(self, spline):
        xp = spline._array_module()
        count = int(np.prod(self._query_shape, dtype=np.int64))
        output = xp.empty(count, dtype=spline._coefficients.dtype)
        for start in range(0, count, spline._EVALUATION_TILE_SIZE):
            stop = min(count, start + spline._EVALUATION_TILE_SIZE)
            indexes = tuple(index[:, start:stop] for index in self._indexes)
            weights = tuple(weight[:, start:stop] for weight in self._weights)
            output[start:stop] = spline._evaluate_precomputed_point_chunk(
                indexes, weights
            )
        return output.reshape(self._query_shape)

    def _apply_coefficient_array(self, spline, coefficients):
        """Evaluate leading batches of compatible coefficients.

        This private bridge is used by higher-level plans such as
        ``AffinePlan``.  It deliberately does not expand ``TensorSpline``'s
        public construction-data contract with inferred channel dimensions.
        """

        if self._grid:
            raise ValueError("Batched coefficient evaluation requires point geometry.")
        if not isinstance(coefficients, np.ndarray):
            raise TypeError("'coefficients' must be a NumPy array.")
        spatial_ndim = spline.ndim
        if coefficients.ndim < spatial_ndim or tuple(
            coefficients.shape[-spatial_ndim:]
        ) != tuple(spline._lengths):
            raise ValueError("Coefficient array has incompatible trailing dimensions.")
        if coefficients.dtype != spline._dtype:
            raise TypeError(
                f"Coefficient array must have dtype {spline._dtype}; "
                f"received {coefficients.dtype}."
            )

        batch_shape = coefficients.shape[:-spatial_ndim]
        batch_count = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1
        count = int(np.prod(self._query_shape, dtype=np.int64))
        output = np.empty(batch_shape + (count,), dtype=coefficients.dtype)
        tile_size = max(1, spline._EVALUATION_TILE_SIZE // batch_count)
        for start in range(0, count, tile_size):
            stop = min(count, start + tile_size)
            indexes = tuple(index[:, start:stop] for index in self._indexes)
            weights = tuple(weight[:, start:stop] for weight in self._weights)
            output[..., start:stop] = spline._evaluate_precomputed_point_chunk(
                indexes, weights, coefficients=coefficients
            )
        return output.reshape(batch_shape + self._query_shape)


# Backward-compatible name for the first experimental release.  The object is
# now geometry-reusable; only the historical spelling remains an alias.
TensorSplineQueryPlan = TensorSplineGeometryPlan
