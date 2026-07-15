"""Experimental fixed-coordinate evaluation plans for :class:`TensorSpline`."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tensor_spline import TensorSpline


class TensorSplineQueryPlan:
    """A memory-capped support/weight plan bound to one ``TensorSpline``.

    Construct plans through :meth:`TensorSpline.query_plan`.  A plan is useful
    when the same coordinates are evaluated repeatedly.  Plan construction is
    extra work and retained memory, so one-shot calls should use the spline
    directly.
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

        self._spline = spline
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

    def __call__(self):
        return self.apply()

    def apply(self):
        """Evaluate the bound spline using the precomputed query geometry."""
        if self._grid:
            return self._apply_grid()
        return self._apply_points()

    def _apply_points(self):
        spline = self._spline
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

    def _apply_grid(self):
        spline = self._spline
        xp = spline._array_module()
        count = int(np.prod(self._query_shape, dtype=np.int64))
        output = xp.empty(count, dtype=spline._coefficients.dtype)
        for start in range(0, count, spline._EVALUATION_TILE_SIZE):
            stop = min(count, start + spline._EVALUATION_TILE_SIZE)
            flat_indexes = xp.arange(start, stop)
            grid_indexes = xp.unravel_index(flat_indexes, self._query_shape)
            indexes = tuple(
                index[:, grid_indexes[axis]] for axis, index in enumerate(self._indexes)
            )
            weights = tuple(
                weight[:, grid_indexes[axis]]
                for axis, weight in enumerate(self._weights)
            )
            output[start:stop] = spline._evaluate_precomputed_point_chunk(
                indexes, weights
            )
        return output.reshape(self._query_shape)
