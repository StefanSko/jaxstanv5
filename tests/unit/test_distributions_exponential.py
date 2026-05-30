"""Tests for Exponential distribution."""

import math

import jax
import jax.numpy as jnp

from jaxstanv5.distributions import Exponential


def test_exponential_log_prob_matches_expected_scalar_value() -> None:
    dist = Exponential(2.0)

    actual = dist.log_prob(jnp.asarray(0.5))

    expected = math.log(2.0) - 1.0
    assert jnp.allclose(actual, expected)


def test_exponential_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Exponential(2.0)

    actual = dist.log_prob(jnp.asarray(-0.1))

    assert actual == -jnp.inf


def test_exponential_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = Exponential(jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([0.5, 0.5])

    actual = dist.log_prob(values)

    expected = jnp.asarray([math.log(1.0) - 0.5, math.log(2.0) - 1.0])
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_exponential_batch_shape_matches_rate_shape() -> None:
    dist = Exponential(jnp.ones((2, 3)))

    assert dist.batch_shape() == (2, 3)


def test_exponential_cdf_and_icdf_are_inverse_for_central_probabilities() -> None:
    dist = Exponential(2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = dist.icdf(probabilities)

    assert jnp.allclose(dist.cdf(values), probabilities, atol=1e-6)


def test_exponential_sample_is_deterministic_for_seed() -> None:
    dist = Exponential(2.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.allclose(first, second)


def test_exponential_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Exponential(jnp.asarray([1.0, 2.0, 3.0]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)
