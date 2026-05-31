"""Tests for Poisson distribution."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import Poisson


def test_poisson_log_prob_matches_expected_scalar_value() -> None:
    dist = Poisson(2.0)

    actual = dist.log_prob(jnp.asarray(3.0))

    expected = 3.0 * math.log(2.0) - 2.0 - math.log(6.0)
    assert jnp.allclose(actual, expected)


def test_poisson_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Poisson(2.0)

    negative = dist.log_prob(jnp.asarray(-1.0))
    fractional = dist.log_prob(jnp.asarray(1.5))

    assert negative == -jnp.inf
    assert fractional == -jnp.inf


def test_poisson_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = Poisson(jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([0.0, 3.0])

    actual = dist.log_prob(values)

    expected = values * jnp.log(jnp.asarray([1.0, 2.0])) - jnp.asarray([1.0, 2.0])
    expected -= gammaln(values + 1.0)
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_poisson_batch_shape_matches_rate_shape() -> None:
    dist = Poisson(jnp.ones((2, 3)))

    assert dist.batch_shape() == (2, 3)


def test_poisson_sample_is_deterministic_for_seed() -> None:
    dist = Poisson(2.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0)
    assert jnp.allclose(first, second)


def test_poisson_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Poisson(jnp.asarray([1.0, 2.0, 3.0]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0)
