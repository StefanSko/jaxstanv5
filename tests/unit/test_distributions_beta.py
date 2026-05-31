"""Tests for Beta distribution."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import Beta


def beta_log_function(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Return log B(a, b)."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def test_beta_log_prob_matches_expected_scalar_value() -> None:
    dist = Beta(2.0, 5.0)

    actual = dist.log_prob(jnp.asarray(0.3))

    expected = (2.0 - 1.0) * math.log(0.3)
    expected += (5.0 - 1.0) * math.log1p(-0.3)
    expected -= float(beta_log_function(jnp.asarray(2.0), jnp.asarray(5.0)))
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_beta_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Beta(2.0, 3.0)

    zero = dist.log_prob(jnp.asarray(0.0))
    one = dist.log_prob(jnp.asarray(1.0))
    negative = dist.log_prob(jnp.asarray(-0.1))
    too_large = dist.log_prob(jnp.asarray(1.1))
    invalid_alpha = Beta(0.0, 3.0).log_prob(jnp.asarray(0.5))
    invalid_beta = Beta(2.0, -1.0).log_prob(jnp.asarray(0.5))

    assert zero == -jnp.inf
    assert one == -jnp.inf
    assert negative == -jnp.inf
    assert too_large == -jnp.inf
    assert invalid_alpha == -jnp.inf
    assert invalid_beta == -jnp.inf


def test_beta_log_prob_broadcasts_over_vector_inputs() -> None:
    alpha = jnp.asarray([1.5, 2.0])
    beta = jnp.asarray([3.0, 4.0])
    values = jnp.asarray([0.25, 0.75])
    dist = Beta(alpha, beta)

    actual = dist.log_prob(values)

    expected = (alpha - 1.0) * jnp.log(values)
    expected += (beta - 1.0) * jnp.log1p(-values)
    expected -= beta_log_function(alpha, beta)
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_beta_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = Beta(jnp.ones((2, 1)), jnp.ones((3,)))

    assert dist.batch_shape() == (2, 3)


def test_beta_sample_is_deterministic_for_seed() -> None:
    dist = Beta(2.0, 3.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first > 0.0)
    assert jnp.all(first < 1.0)
    assert jnp.allclose(first, second)


def test_beta_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Beta(jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([4.0, 5.0, 6.0]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples > 0.0)
    assert jnp.all(samples < 1.0)
