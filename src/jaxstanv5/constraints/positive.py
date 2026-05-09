"""Positive constraint."""

from dataclasses import dataclass

import jax.numpy as jnp

from jaxstanv5.constraints.core import (
    ConstrainedValue,
    LogAbsDetJacobian,
    UnconstrainedValue,
)


@dataclass(frozen=True)
class Positive:
    """Constraint for strictly positive values using log/exp transforms."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map positive constrained values to unconstrained real values."""
        return jnp.log(x)

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained real values to positive constrained values."""
        return jnp.exp(y)

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        """Return log absolute determinant of ``exp`` Jacobian at ``y``."""
        return jnp.asarray(y)
