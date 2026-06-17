"""JAX backend constraint operation tests."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5._backends.jax.constraints import (
    inverse_transform,
    log_abs_det_jacobian,
    transform,
)
from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval


def test_positive_constraint_operations_are_backend_operations() -> None:
    constrained = jnp.asarray([0.5, 2.0])
    unconstrained = transform(Positive(), constrained)

    assert jnp.allclose(inverse_transform(Positive(), unconstrained), constrained)
    assert jnp.allclose(log_abs_det_jacobian(Positive(), unconstrained), unconstrained)


def test_interval_constraint_operations_are_backend_operations() -> None:
    constraint = Interval(-1.0, 3.0)
    constrained = jnp.asarray([0.0, 2.0])
    unconstrained = transform(constraint, constrained)

    assert jnp.allclose(inverse_transform(constraint, unconstrained), constrained)
    assert log_abs_det_jacobian(constraint, unconstrained).shape == (2,)


def test_unit_interval_constraint_operations_are_backend_operations() -> None:
    constrained = jnp.asarray([0.2, 0.8])
    unconstrained = transform(UnitInterval(), constrained)

    assert jnp.allclose(inverse_transform(UnitInterval(), unconstrained), constrained)


def test_ordered_constraint_operations_are_backend_operations() -> None:
    constrained = jnp.asarray([-1.0, 0.5, 2.0])
    unconstrained = transform(Ordered(), constrained)

    assert jnp.allclose(inverse_transform(Ordered(), unconstrained), constrained)
    assert jnp.allclose(log_abs_det_jacobian(Ordered(), unconstrained), unconstrained[1:].sum())
