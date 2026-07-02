"""Tests for ordered vector constraint."""

import jax.numpy as jnp
import pytest
from bayeswire.constraints import Ordered

from jaxstanv5._backends.jax.constraints import (
    inverse_transform,
    log_abs_det_jacobian,
    transform,
)


def test_ordered_constraint_round_trips_constrained_vectors() -> None:
    constraint = Ordered()
    constrained = jnp.asarray([-1.5, 0.25, 2.0])

    unconstrained = transform(constraint, constrained)
    actual = inverse_transform(constraint, unconstrained)

    assert jnp.allclose(actual, constrained)


def test_ordered_constraint_inverse_transform_produces_strictly_increasing_vectors() -> None:
    constraint = Ordered()
    unconstrained = jnp.asarray([-1.0, -2.0, 0.5, 1.0])

    actual = jnp.asarray(inverse_transform(constraint, unconstrained))

    assert actual.shape == (4,)
    assert jnp.all(actual[1:] > actual[:-1])


def test_ordered_constraint_log_abs_det_jacobian_sums_tail_unconstrained_values() -> None:
    constraint = Ordered()
    unconstrained = jnp.asarray([-1.0, -2.0, 0.5, 1.0])

    actual = log_abs_det_jacobian(constraint, unconstrained)

    assert jnp.allclose(actual, -0.5)


def test_ordered_constraint_supports_leading_dimensions() -> None:
    constraint = Ordered()
    unconstrained = jnp.asarray([[-1.0, 0.0, 0.5], [2.0, -0.25, 0.75]])

    constrained = jnp.asarray(inverse_transform(constraint, unconstrained))
    round_trip = transform(constraint, constrained)
    jacobian = log_abs_det_jacobian(constraint, unconstrained)

    assert constrained.shape == (2, 3)
    assert jnp.all(constrained[..., 1:] > constrained[..., :-1])
    assert jnp.allclose(round_trip, unconstrained)
    assert jnp.allclose(jacobian, jnp.asarray([0.5, 0.5]))


def test_ordered_constraint_rejects_scalar_values() -> None:
    constraint = Ordered()

    with pytest.raises(ValueError, match="requires vector"):
        transform(constraint, jnp.asarray(1.0))

    with pytest.raises(ValueError, match="requires vector"):
        inverse_transform(constraint, jnp.asarray(1.0))

    with pytest.raises(ValueError, match="requires vector"):
        log_abs_det_jacobian(constraint, jnp.asarray(1.0))
