"""Tests for HalfNormal distribution."""

import math

import jax
import jax.numpy as jnp

from jaxstanv5.distributions import HalfNormal


def test_half_normal_log_prob_matches_expected_scalar_value() -> None:
    dist = HalfNormal(2.0)

    actual = dist.log_prob(jnp.asarray(0.0))

    expected = 0.5 * math.log(2.0 / math.pi) - math.log(2.0)
    assert jnp.allclose(actual, expected)


def test_half_normal_log_prob_is_negative_infinity_outside_support() -> None:
    dist = HalfNormal(2.0)

    actual = dist.log_prob(jnp.asarray(-0.1))

    assert actual == -jnp.inf


def test_half_normal_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = HalfNormal(jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([0.0, 2.0])

    actual = dist.log_prob(values)

    expected = jnp.asarray(
        [
            0.5 * math.log(2.0 / math.pi),
            0.5 * math.log(2.0 / math.pi) - math.log(2.0) - 0.5,
        ]
    )
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_half_normal_batch_shape_matches_scale_shape() -> None:
    dist = HalfNormal(jnp.ones((2, 3)))

    assert dist.batch_shape() == (2, 3)


def test_half_normal_cdf_and_icdf_are_inverse_for_central_probabilities() -> None:
    dist = HalfNormal(2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = dist.icdf(probabilities)

    assert jnp.allclose(dist.cdf(values), probabilities, atol=1e-6)


def test_half_normal_sample_is_deterministic_for_seed() -> None:
    dist = HalfNormal(2.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.allclose(first, second)


def test_half_normal_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = HalfNormal(jnp.asarray([1.0, 2.0, 3.0]))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)
