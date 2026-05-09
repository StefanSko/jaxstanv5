"""Tests for positive constraint."""

import jax.numpy as jnp

from jaxstanv5.constraints import Positive


def test_positive_constraint_round_trips_constrained_values() -> None:
    constraint = Positive()
    constrained = jnp.asarray([0.25, 1.0, 4.0])

    unconstrained = constraint.transform(constrained)
    actual = constraint.inverse_transform(unconstrained)

    assert jnp.allclose(actual, constrained)


def test_positive_constraint_log_abs_det_jacobian_matches_inverse_transform() -> None:
    constraint = Positive()
    unconstrained = jnp.asarray([-1.0, 0.0, 2.0])

    actual = constraint.log_abs_det_jacobian(unconstrained)

    expected = unconstrained
    assert jnp.allclose(actual, expected)
