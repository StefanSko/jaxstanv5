"""Tests for Normal distribution."""

import math

import jax
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


def test_normal_sample_is_deterministic_for_seed() -> None:
    dist = Normal(1.0, 2.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.allclose(first, second)


def test_normal_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Normal(jnp.asarray([0.0, 1.0, 2.0]), 0.5)

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(jnp.isfinite(samples))
