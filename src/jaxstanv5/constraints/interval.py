"""Finite interval constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jaxstanv5.constraints.core import (
    ConstrainedValue,
    LogAbsDetJacobian,
    UnconstrainedValue,
)


@dataclass(frozen=True)
class Interval:
    """Constraint for values in a finite open interval."""

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

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map constrained interval values to unconstrained real values."""
        unit_value = (jnp.asarray(x) - self.lower) / self.width
        return jnp.log(unit_value) - jnp.log1p(-unit_value)

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained real values to interval-constrained values."""
        return self.lower + self.width * jax.nn.sigmoid(jnp.asarray(y))

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        """Return log absolute determinant of inverse-transform Jacobian."""
        unconstrained = jnp.asarray(y)
        return (
            jnp.log(self.width) - jax.nn.softplus(-unconstrained) - jax.nn.softplus(unconstrained)
        )

    @property
    def width(self) -> float:
        """Return finite interval width."""
        return self.upper - self.lower


@dataclass(frozen=True)
class UnitInterval:
    """Constraint for values in the open unit interval."""

    _interval: Interval = field(default_factory=lambda: Interval(0.0, 1.0), init=False, repr=False)

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map unit-interval values to unconstrained real values."""
        return self._interval.transform(x)

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained real values to unit-interval values."""
        return self._interval.inverse_transform(y)

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        """Return log absolute determinant of inverse-transform Jacobian."""
        return self._interval.log_abs_det_jacobian(y)
