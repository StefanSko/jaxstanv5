"""Tests for positive constraint."""

import jax.numpy as jnp
from bayeswire.constraints import Positive

from jaxstanv5._backends.jax.constraints import (
    inverse_transform,
    log_abs_det_jacobian,
    transform,
)


def test_positive_constraint_round_trips_constrained_values() -> None:
    constraint = Positive()
    constrained = jnp.asarray([0.25, 1.0, 4.0])

    unconstrained = transform(constraint, constrained)
    actual = inverse_transform(constraint, unconstrained)

    assert jnp.allclose(actual, constrained)


def test_positive_constraint_log_abs_det_jacobian_matches_inverse_transform() -> None:
    constraint = Positive()
    unconstrained = jnp.asarray([-1.0, 0.0, 2.0])

    actual = log_abs_det_jacobian(constraint, unconstrained)

    expected = unconstrained
    assert jnp.allclose(actual, expected)
