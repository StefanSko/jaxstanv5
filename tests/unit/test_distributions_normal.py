"""Tests for Normal distribution."""

import math

import jax.numpy as jnp

from jaxstanv5.distributions import Normal


def test_standard_normal_log_prob_matches_expected_scalar_value() -> None:
    dist = Normal(0.0, 1.0)

    actual = dist.log_prob(jnp.asarray(0.0))

    expected = -0.5 * math.log(2.0 * math.pi)
    assert jnp.allclose(actual, expected)


def test_normal_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = Normal(0.0, 2.0)
    values = jnp.asarray([0.0, 2.0])

    actual = dist.log_prob(values)

    expected = jnp.asarray(
        [
            -math.log(2.0) - 0.5 * math.log(2.0 * math.pi),
            -0.5 - math.log(2.0) - 0.5 * math.log(2.0 * math.pi),
        ]
    )
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)
