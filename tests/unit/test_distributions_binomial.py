"""Tests for Binomial distribution."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import Binomial


def test_binomial_log_prob_matches_expected_scalar_value() -> None:
    dist = Binomial(10.0, 0.25)

    actual = dist.log_prob(jnp.asarray(3.0))

    expected = math.log(120.0) + 3.0 * math.log(0.25) + 7.0 * math.log(0.75)
    assert jnp.allclose(actual, expected)


def test_binomial_log_prob_handles_boundary_probabilities() -> None:
    certain_failure = Binomial(4.0, 0.0)
    certain_success = Binomial(4.0, 1.0)

    assert jnp.allclose(certain_failure.log_prob(jnp.asarray(0.0)), 0.0, atol=1e-6)
    assert certain_failure.log_prob(jnp.asarray(1.0)) == -jnp.inf
    assert jnp.allclose(certain_success.log_prob(jnp.asarray(4.0)), 0.0, atol=1e-6)
    assert certain_success.log_prob(jnp.asarray(3.0)) == -jnp.inf


def test_binomial_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Binomial(5.0, 0.4)

    negative = dist.log_prob(jnp.asarray(-1.0))
    too_large = dist.log_prob(jnp.asarray(6.0))
    fractional = dist.log_prob(jnp.asarray(1.5))
    non_integer_count = Binomial(5.5, 0.4).log_prob(jnp.asarray(2.0))
    invalid_prob = Binomial(5.0, -0.1).log_prob(jnp.asarray(2.0))

    assert negative == -jnp.inf
    assert too_large == -jnp.inf
    assert fractional == -jnp.inf
    assert non_integer_count == -jnp.inf
    assert invalid_prob == -jnp.inf


def test_binomial_log_prob_broadcasts_over_vector_inputs() -> None:
    total_count = jnp.asarray([2.0, 4.0])
    probs = jnp.asarray([0.25, 0.5])
    values = jnp.asarray([1.0, 3.0])
    dist = Binomial(total_count, probs)

    actual = dist.log_prob(values)

    expected = gammaln(total_count + 1.0) - gammaln(values + 1.0)
    expected -= gammaln(total_count - values + 1.0)
    expected += values * jnp.log(probs) + (total_count - values) * jnp.log1p(-probs)
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_binomial_batch_shape_broadcasts_count_and_probability_shapes() -> None:
    dist = Binomial(jnp.ones((2, 1)), jnp.ones((3,)) * 0.25)

    assert dist.batch_shape() == (2, 3)


def test_binomial_sample_is_deterministic_for_seed() -> None:
    dist = Binomial(5.0, 0.5)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.all(first <= 5.0)
    assert jnp.allclose(first, second)


def test_binomial_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Binomial(jnp.asarray([2.0, 3.0, 4.0]), jnp.asarray([0.2, 0.5, 0.8]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= jnp.asarray([2.0, 3.0, 4.0]))
