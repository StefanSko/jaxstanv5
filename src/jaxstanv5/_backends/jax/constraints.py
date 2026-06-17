"""JAX implementations of constraint operations."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp

from jaxstanv5.constraints.core import ConstrainedValue, Constraint, UnconstrainedValue
from jaxstanv5.constraints.interval import Interval, UnitInterval
from jaxstanv5.constraints.ordered import Ordered
from jaxstanv5.constraints.positive import Positive


@runtime_checkable
class _PythonConstraint(Protocol):
    """Compatibility protocol for Python-defined JAX constraints."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map constrained values to unconstrained values."""
        ...

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained values to constrained values."""
        ...

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> object:
        """Return inverse-transform log absolute determinant."""
        ...


def transform(constraint: Constraint, x: ConstrainedValue) -> jax.Array:
    """Map constrained values to unconstrained values with the JAX backend."""
    if isinstance(constraint, Positive):
        return jnp.log(x)
    if isinstance(constraint, Interval):
        unit_value = (jnp.asarray(x) - constraint.lower) / constraint.width
        return jnp.log(unit_value) - jnp.log1p(-unit_value)
    if isinstance(constraint, UnitInterval):
        return transform(Interval(0.0, 1.0), x)
    if isinstance(constraint, Ordered):
        constrained = jnp.asarray(x)
        if constrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        first = constrained[..., :1]
        log_differences = jnp.log(constrained[..., 1:] - constrained[..., :-1])
        return jnp.concatenate((first, log_differences), axis=-1)
    if isinstance(constraint, _PythonConstraint):
        return cast(jax.Array, constraint.transform(x))
    raise TypeError(f"Unsupported constraint: {type(constraint).__name__}")


def inverse_transform(constraint: Constraint, y: UnconstrainedValue) -> jax.Array:
    """Map unconstrained values to constrained values with the JAX backend."""
    if isinstance(constraint, Positive):
        return jnp.exp(y)
    if isinstance(constraint, Interval):
        return constraint.lower + constraint.width * jax.nn.sigmoid(jnp.asarray(y))
    if isinstance(constraint, UnitInterval):
        return inverse_transform(Interval(0.0, 1.0), y)
    if isinstance(constraint, Ordered):
        unconstrained = jnp.asarray(y)
        if unconstrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        first = unconstrained[..., :1]
        increments = jnp.exp(unconstrained[..., 1:])
        tail = first + jnp.cumsum(increments, axis=-1)
        return jnp.concatenate((first, tail), axis=-1)
    if isinstance(constraint, _PythonConstraint):
        return cast(jax.Array, constraint.inverse_transform(y))
    raise TypeError(f"Unsupported constraint: {type(constraint).__name__}")


def log_abs_det_jacobian(constraint: Constraint, y: UnconstrainedValue) -> jax.Array:
    """Return inverse-transform log absolute determinant with the JAX backend."""
    if isinstance(constraint, Positive):
        return jnp.asarray(y)
    if isinstance(constraint, Interval):
        unconstrained = jnp.asarray(y)
        return (
            jnp.log(constraint.width)
            - jax.nn.softplus(-unconstrained)
            - jax.nn.softplus(unconstrained)
        )
    if isinstance(constraint, UnitInterval):
        return log_abs_det_jacobian(Interval(0.0, 1.0), y)
    if isinstance(constraint, Ordered):
        unconstrained = jnp.asarray(y)
        if unconstrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        return jnp.sum(unconstrained[..., 1:], axis=-1)
    if isinstance(constraint, _PythonConstraint):
        return constraint.log_abs_det_jacobian(y)
    raise TypeError(f"Unsupported constraint: {type(constraint).__name__}")
