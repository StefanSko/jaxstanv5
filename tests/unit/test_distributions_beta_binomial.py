"""Tests for Beta-binomial distribution."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import BetaBinomial


def beta_log_function(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Return log B(a, b)."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def test_beta_binomial_log_prob_matches_expected_scalar_value() -> None:
    dist = BetaBinomial(10.0, 2.0, 5.0)

    actual = dist.log_prob(jnp.asarray(3.0))

    expected = math.log(120.0)
    expected += float(beta_log_function(jnp.asarray(5.0), jnp.asarray(12.0)))
    expected -= float(beta_log_function(jnp.asarray(2.0), jnp.asarray(5.0)))
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_beta_binomial_log_prob_is_negative_infinity_outside_support() -> None:
    dist = BetaBinomial(5.0, 2.0, 3.0)

    negative = dist.log_prob(jnp.asarray(-1.0))
    too_large = dist.log_prob(jnp.asarray(6.0))
    fractional = dist.log_prob(jnp.asarray(1.5))
    non_integer_count = BetaBinomial(5.5, 2.0, 3.0).log_prob(jnp.asarray(2.0))
    invalid_alpha = BetaBinomial(5.0, 0.0, 3.0).log_prob(jnp.asarray(2.0))
    invalid_beta = BetaBinomial(5.0, 2.0, -1.0).log_prob(jnp.asarray(2.0))

    assert negative == -jnp.inf
    assert too_large == -jnp.inf
    assert fractional == -jnp.inf
    assert non_integer_count == -jnp.inf
    assert invalid_alpha == -jnp.inf
    assert invalid_beta == -jnp.inf


def test_beta_binomial_log_prob_broadcasts_over_vector_inputs() -> None:
    total_count = jnp.asarray([2.0, 4.0])
    alpha = jnp.asarray([1.5, 2.0])
    beta = jnp.asarray([3.0, 4.0])
    values = jnp.asarray([1.0, 3.0])
    dist = BetaBinomial(total_count, alpha, beta)

    actual = dist.log_prob(values)

    expected = gammaln(total_count + 1.0) - gammaln(values + 1.0)
    expected -= gammaln(total_count - values + 1.0)
    expected += beta_log_function(values + alpha, total_count - values + beta)
    expected -= beta_log_function(alpha, beta)
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_beta_binomial_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = BetaBinomial(jnp.ones((2, 1)), jnp.ones((3,)), 2.0)

    assert dist.batch_shape() == (2, 3)


def test_beta_binomial_sample_is_deterministic_for_seed() -> None:
    dist = BetaBinomial(5.0, 2.0, 3.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.all(first <= 5.0)
    assert jnp.allclose(first, second)


def test_beta_binomial_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = BetaBinomial(jnp.asarray([2.0, 3.0, 4.0]), jnp.asarray([1.0, 2.0, 3.0]), 4.0)

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= jnp.asarray([2.0, 3.0, 4.0]))
