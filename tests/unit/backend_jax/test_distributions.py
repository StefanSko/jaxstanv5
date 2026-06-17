"""JAX backend distribution operation tests."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxstanv5._backends.jax.distributions import (
    batch_shape,
    cdf,
    event_shape,
    icdf,
    log_prob,
    sample,
)
from jaxstanv5.distributions import Bernoulli, Binomial, MultivariateNormal, Normal


def test_normal_log_prob_is_backend_operation() -> None:
    actual = log_prob(Normal(0.0, 2.0), jnp.asarray(1.0))
    expected = -0.5 * (1.0 / 2.0) ** 2 - math.log(2.0) - 0.5 * math.log(2.0 * math.pi)

    assert jnp.allclose(actual, expected)


def test_count_log_prob_is_backend_operation() -> None:
    actual = log_prob(Binomial(4.0, 0.25), jnp.asarray(1.0))
    expected = math.log(4.0) + math.log(0.25) + 3.0 * math.log(0.75)

    assert jnp.allclose(actual, expected)


def test_distribution_shapes_and_sampling_are_backend_operations() -> None:
    distribution = Normal(jnp.asarray([0.0, 1.0]), 2.0)

    assert batch_shape(distribution) == (2,)
    assert event_shape(distribution) == ()
    assert sample(distribution, jax.random.PRNGKey(1), sample_shape=(3,)).shape == (3, 2)


def test_inverse_cdf_operations_are_backend_operations() -> None:
    distribution = Normal(0.0, 1.0)
    probability = cdf(distribution, jnp.asarray(0.25))

    assert jnp.allclose(icdf(distribution, probability), 0.25, atol=1e-6)


def test_multivariate_event_shape_is_backend_operation() -> None:
    distribution = MultivariateNormal(jnp.zeros((2,)), jnp.eye(2))

    assert batch_shape(distribution) == ()
    assert event_shape(distribution) == (2,)
    assert sample(distribution, jax.random.PRNGKey(2), sample_shape=(4,)).shape == (4, 2)


def test_discrete_sampling_uses_backend_operation() -> None:
    draws = sample(Bernoulli(0.8), jax.random.PRNGKey(3), sample_shape=(10,))

    assert draws.shape == (10,)
    assert jnp.all((draws == 0) | (draws == 1))
