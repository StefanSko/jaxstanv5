"""Tests for explicitly truncated distributions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr

from jaxstanv5._backends.jax.distributions import cdf, icdf, log_prob
from jaxstanv5._backends.jax.distributions import sample as distribution_sample
from jaxstanv5.distributions import Normal, Truncated


def test_truncated_normal_log_prob_includes_normalizer() -> None:
    dist = Truncated(Normal(1.0, 2.0), lower=0.0)
    value = jnp.asarray(0.5)

    actual = log_prob(dist, value)

    base = -0.5 * ((value - 1.0) / 2.0) ** 2 - math.log(2.0) - 0.5 * math.log(2.0 * math.pi)
    expected = base - jnp.log(ndtr(0.5))
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_truncated_normal_log_prob_is_stable_in_upper_tail() -> None:
    dist = Truncated(Normal(0.0, 1.0), lower=10.0)
    value = jnp.asarray(10.5)

    actual = log_prob(dist, value)

    base = -0.5 * value**2 - 0.5 * math.log(2.0 * math.pi)
    expected = base - log_ndtr(jnp.asarray(-10.0))
    assert jnp.isfinite(actual)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_truncated_normal_log_prob_rejects_values_outside_bounds() -> None:
    dist = Truncated(Normal(0.0, 1.0), lower=-1.0, upper=1.0)

    assert log_prob(dist, jnp.asarray(-2.0)) == -jnp.inf
    assert log_prob(dist, jnp.asarray(2.0)) == -jnp.inf


def test_truncated_normal_cdf_and_icdf_are_inverse_inside_bounds() -> None:
    dist = Truncated(Normal(0.0, 1.0), lower=-1.0, upper=2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = icdf(dist, probabilities)

    assert jnp.all(values >= -1.0)
    assert jnp.all(values <= 2.0)
    assert jnp.allclose(cdf(dist, values), probabilities, atol=1e-6)


def test_truncated_normal_cdf_icdf_and_sample_are_stable_in_upper_tail() -> None:
    dist = Truncated(Normal(0.0, 1.0), lower=10.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = icdf(dist, probabilities)
    draws = distribution_sample(dist, jax.random.PRNGKey(2), sample_shape=(5,))

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(values >= 10.0)
    assert jnp.allclose(cdf(dist, values), probabilities, atol=1e-5)
    assert jnp.all(jnp.isfinite(draws))
    assert jnp.all(draws >= 10.0)


def test_truncated_normal_sample_stays_inside_bounds() -> None:
    dist = Truncated(Normal(jnp.asarray([0.0, 1.0]), 1.0), lower=0.0, upper=2.0)

    draws = distribution_sample(dist, jax.random.PRNGKey(1), sample_shape=(20,))

    assert draws.shape == (20, 2)
    assert jnp.all(draws >= 0.0)
    assert jnp.all(draws <= 2.0)
