"""Tests for Negative-binomial distribution."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import NegativeBinomial


def test_negative_binomial_log_prob_matches_expected_scalar_value() -> None:
    dist = NegativeBinomial(3.0, 2.0)

    actual = dist.log_prob(jnp.asarray(4.0))

    expected = gammaln(6.0) - gammaln(2.0) - gammaln(5.0)
    expected += 2.0 * jnp.log(2.0 / 5.0) + 4.0 * jnp.log(3.0 / 5.0)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_negative_binomial_log_prob_is_negative_infinity_outside_support() -> None:
    dist = NegativeBinomial(3.0, 2.0)

    negative = dist.log_prob(jnp.asarray(-1.0))
    fractional = dist.log_prob(jnp.asarray(1.5))
    invalid_mean = NegativeBinomial(0.0, 2.0).log_prob(jnp.asarray(2.0))
    invalid_overdispersion = NegativeBinomial(3.0, -1.0).log_prob(jnp.asarray(2.0))

    assert negative == -jnp.inf
    assert fractional == -jnp.inf
    assert invalid_mean == -jnp.inf
    assert invalid_overdispersion == -jnp.inf


def test_negative_binomial_log_prob_broadcasts_over_vector_inputs() -> None:
    mean = jnp.asarray([2.0, 4.0])
    overdispersion = jnp.asarray([3.0, 5.0])
    values = jnp.asarray([1.0, 3.0])
    dist = NegativeBinomial(mean, overdispersion)

    actual = dist.log_prob(values)

    total = mean + overdispersion
    expected = gammaln(values + overdispersion) - gammaln(overdispersion)
    expected -= gammaln(values + 1.0)
    expected += overdispersion * jnp.log(overdispersion / total)
    expected += values * jnp.log(mean / total)
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_negative_binomial_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = NegativeBinomial(jnp.ones((2, 1)), jnp.ones((3,)))

    assert dist.batch_shape() == (2, 3)


def test_negative_binomial_sample_is_deterministic_for_seed() -> None:
    dist = NegativeBinomial(3.0, 2.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.allclose(first, second)


def test_negative_binomial_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = NegativeBinomial(jnp.asarray([2.0, 3.0, 4.0]), jnp.asarray([1.0, 2.0, 3.0]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)
