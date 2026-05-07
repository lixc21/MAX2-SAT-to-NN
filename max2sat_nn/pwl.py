"""Continuous piecewise-linear function (used for exact Lipschitz of nets)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class PWLFunc:
    """Continuous PWL function on a closed interval.

    Stores sorted breakpoints ``xs`` and matching ordinate values ``ys``;
    the function is the linear interpolant between consecutive points.
    """

    __slots__ = ("xs", "ys")

    def __init__(self, xs: Sequence[float], ys: Sequence[float]):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape:
            raise ValueError("xs and ys must be 1-D arrays of identical shape")
        if xs.size < 2:
            raise ValueError("PWLFunc requires at least two breakpoints")
        if not np.all(np.diff(xs) >= -1e-12):
            raise ValueError("xs must be non-decreasing")
        self.xs = xs
        self.ys = ys

    @property
    def domain(self) -> Tuple[float, float]:
        return float(self.xs[0]), float(self.xs[-1])

    @property
    def n_pieces(self) -> int:
        return self.xs.size - 1

    def evaluate(self, x):
        return np.interp(np.asarray(x, dtype=float), self.xs, self.ys)

    __call__ = evaluate

    def lipschitz(self) -> float:
        """Exact Lipschitz constant on the domain (max |slope| over pieces)."""
        dx = np.diff(self.xs)
        dy = np.diff(self.ys)
        # Guard against zero-width segments.
        mask = dx > 0
        if not mask.any():
            return 0.0
        return float(np.max(np.abs(dy[mask] / dx[mask])))
