"""Tests for discrete count distributions."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import Bernoulli, BetaBinomial, Binomial, NegativeBinomial, Poisson


def beta_log_function(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return log B(a, b)."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def test_poisson_log_prob_matches_expected_scalar_value() -> None:
    dist = Poisson(2.0)

    actual = dist.log_prob(jnp.asarray(3.0))

    expected = 3.0 * math.log(2.0) - 2.0 - math.log(6.0)
    assert jnp.allclose(actual, expected)


def test_poisson_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Poisson(2.0)

    negative = dist.log_prob(jnp.asarray(-1.0))
    fractional = dist.log_prob(jnp.asarray(1.5))
    zero_rate = Poisson(0.0).log_prob(jnp.asarray(0.0))
    zero_rate_gradient = jax.grad(lambda rate: Poisson(rate).log_prob(jnp.asarray(0.0)))(0.0)

    assert negative == -jnp.inf
    assert fractional == -jnp.inf
    assert zero_rate == -jnp.inf
    assert jnp.isfinite(zero_rate_gradient)


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


def test_bernoulli_log_prob_matches_expected_values() -> None:
    dist = Bernoulli(0.25)
    values = jnp.asarray([0.0, 1.0])

    actual = dist.log_prob(values)

    expected = jnp.asarray([math.log(0.75), math.log(0.25)])
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_bernoulli_log_prob_handles_boundary_probabilities() -> None:
    certain_failure = Bernoulli(0.0)
    certain_success = Bernoulli(1.0)

    assert jnp.allclose(certain_failure.log_prob(jnp.asarray(0.0)), 0.0, atol=1e-6)
    assert certain_failure.log_prob(jnp.asarray(1.0)) == -jnp.inf
    assert jnp.allclose(certain_success.log_prob(jnp.asarray(1.0)), 0.0, atol=1e-6)
    assert certain_success.log_prob(jnp.asarray(0.0)) == -jnp.inf


def test_bernoulli_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Bernoulli(0.4)

    negative = dist.log_prob(jnp.asarray(-1.0))
    too_large = dist.log_prob(jnp.asarray(2.0))
    fractional = dist.log_prob(jnp.asarray(0.5))
    invalid_low_prob = Bernoulli(-0.1).log_prob(jnp.asarray(0.0))
    invalid_high_prob = Bernoulli(1.1).log_prob(jnp.asarray(1.0))

    assert negative == -jnp.inf
    assert too_large == -jnp.inf
    assert fractional == -jnp.inf
    assert invalid_low_prob == -jnp.inf
    assert invalid_high_prob == -jnp.inf


def test_bernoulli_log_prob_broadcasts_over_vector_inputs() -> None:
    probs = jnp.asarray([0.25, 0.5])
    values = jnp.asarray([0.0, 1.0])
    dist = Bernoulli(probs)

    actual = dist.log_prob(values)

    expected = jnp.asarray([math.log(0.75), math.log(0.5)])
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_bernoulli_log_prob_matches_binomial_with_one_trial() -> None:
    probs = jnp.asarray([0.0, 0.25, 0.75, 1.0])
    values = jnp.asarray([0.0, 1.0, 0.5, 2.0])

    actual = Bernoulli(probs).log_prob(values)
    expected = Binomial(1.0, probs).log_prob(values)

    assert jnp.allclose(actual, expected, atol=1e-6)


def test_bernoulli_batch_shape_matches_probability_shape() -> None:
    dist = Bernoulli(jnp.ones((2, 3)) * 0.25)

    assert dist.batch_shape() == (2, 3)


def test_bernoulli_sample_is_deterministic_for_seed() -> None:
    dist = Bernoulli(0.5)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0)
    assert jnp.all(first <= 1)
    assert jnp.allclose(first, second)


def test_bernoulli_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Bernoulli(jnp.asarray([0.2, 0.5, 0.8]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0)
    assert jnp.all(samples <= 1)


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


def test_discrete_distribution_samples_have_integer_dtype() -> None:
    key = jax.random.PRNGKey(789)
    distributions = (
        Bernoulli(0.5),
        Poisson(2.0),
        Binomial(10.0, 0.5),
        BetaBinomial(10.0, 2.0, 3.0),
        NegativeBinomial(3.0, 2.0),
    )

    for dist in distributions:
        sample = dist.sample(key, sample_shape=(2,))
        assert jnp.issubdtype(sample.dtype, jnp.integer)


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
