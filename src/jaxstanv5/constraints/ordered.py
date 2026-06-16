"""Ordered vector constraint."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxstanv5.constraints.core import (
    ConstrainedValue,
    LogAbsDetJacobian,
    UnconstrainedValue,
)

if TYPE_CHECKING:
    import jax.numpy as jnp
else:
    from jaxstanv5._jax_lazy import jnp


@dataclass(frozen=True)
class Ordered:
    """Constraint for strictly increasing vectors along the last axis."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map ordered constrained vectors to unconstrained real vectors."""
        constrained = jnp.asarray(x)
        if constrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        first = constrained[..., :1]
        log_differences = jnp.log(constrained[..., 1:] - constrained[..., :-1])
        return jnp.concatenate((first, log_differences), axis=-1)

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained real vectors to strictly ordered constrained vectors."""
        unconstrained = jnp.asarray(y)
        if unconstrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        first = unconstrained[..., :1]
        increments = jnp.exp(unconstrained[..., 1:])
        tail = first + jnp.cumsum(increments, axis=-1)
        return jnp.concatenate((first, tail), axis=-1)

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        """Return log absolute determinant of inverse-transform Jacobian."""
        unconstrained = jnp.asarray(y)
        if unconstrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        return jnp.sum(unconstrained[..., 1:], axis=-1)
