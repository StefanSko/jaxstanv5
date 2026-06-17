"""Finite interval constraint metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Constraint metadata for values in a finite open interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate finite ordered interval bounds."""
        if isinstance(self.lower, bool) or isinstance(self.upper, bool):
            raise TypeError("Interval bounds must be finite real numbers, not bool")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("Interval bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("Interval lower bound must be less than upper bound")

    @property
    def width(self) -> float:
        """Return finite interval width."""
        return self.upper - self.lower


@dataclass(frozen=True)
class UnitInterval:
    """Constraint metadata for values in the open unit interval."""
