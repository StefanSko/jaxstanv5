"""Tests for multivariate distributions."""

import math

import jax
import jax.numpy as jnp
import pytest
from bayeswire.distributions import MultivariateNormal
from jax.scipy.linalg import solve_triangular

from jaxstanv5._backends.jax.distributions import (
    event_shape,
    log_prob,
)
from jaxstanv5._backends.jax.distributions import (
    sample as distribution_sample,
)


def test_multivariate_normal_log_prob_matches_standard_identity_value() -> None:
    dist = MultivariateNormal(jnp.zeros((2,)), jnp.eye(2))

    actual = log_prob(dist, jnp.asarray([0.0, 0.0]))

    expected = -math.log(2.0 * math.pi)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_sample_rejects_non_lower_triangular_scale() -> None:
    dist = MultivariateNormal(jnp.zeros((2,)), jnp.asarray([[1.0, 0.9], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="lower-triangular"):
        distribution_sample(dist, jax.random.PRNGKey(0))


def test_multivariate_normal_log_prob_masks_non_positive_diagonal() -> None:
    dist = MultivariateNormal(jnp.zeros((2,)), jnp.asarray([[1.0, 0.0], [0.0, -1.0]]))

    actual = log_prob(dist, jnp.asarray([0.0, 0.0]))

    assert actual == -jnp.inf


def test_multivariate_normal_log_prob_matches_cholesky_formula() -> None:
    mean = jnp.asarray([1.0, -0.5])
    chol = jnp.asarray([[2.0, 0.0], [0.5, 1.5]])
    value = jnp.asarray([2.0, 1.0])
    dist = MultivariateNormal(mean, chol)

    actual = log_prob(dist, value)

    solved = solve_triangular(chol, value - mean, lower=True)
    expected = (
        -0.5 * jnp.sum(solved**2) - jnp.sum(jnp.log(jnp.diagonal(chol))) - math.log(2.0 * math.pi)
    )
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_allows_scalar_broadcast_mean() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(3))

    actual = log_prob(dist, jnp.asarray([0.0, 0.0, 0.0]))

    expected = -1.5 * math.log(2.0 * math.pi)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_batches_over_leading_vector_dimensions() -> None:
    dist = MultivariateNormal(jnp.zeros((2,)), jnp.eye(2))
    values = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])

    actual = log_prob(dist, values)

    expected = jnp.asarray([-math.log(2.0 * math.pi), -0.5 - math.log(2.0 * math.pi)])
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_batches_over_scale_tril() -> None:
    dist = MultivariateNormal(
        mean=jnp.zeros((2, 2)),
        scale_tril=jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)]),
    )

    actual = log_prob(dist, jnp.zeros((2, 2)))

    expected = jnp.asarray(
        [
            -math.log(2.0 * math.pi),
            -2.0 * math.log(2.0) - math.log(2.0 * math.pi),
        ]
    )
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_broadcasts_value_over_batched_scale_tril() -> None:
    dist = MultivariateNormal(
        mean=jnp.zeros((2,)),
        scale_tril=jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)]),
    )

    actual = log_prob(dist, jnp.asarray([0.0, 0.0]))

    expected = jnp.asarray(
        [
            -math.log(2.0 * math.pi),
            -2.0 * math.log(2.0) - math.log(2.0 * math.pi),
        ]
    )
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_event_shape_matches_cholesky_dimension() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(4))

    assert event_shape(dist) == (4,)


def test_multivariate_normal_sample_remains_jittable() -> None:
    @jax.jit
    def draw(scale_tril: jax.Array) -> jax.Array:
        return distribution_sample(MultivariateNormal(0.0, scale_tril), jax.random.PRNGKey(0))

    sample = draw(jnp.eye(3))

    assert sample.shape == (3,)


def test_multivariate_normal_sample_draws_one_event_vector() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(3))

    sample = distribution_sample(dist, jax.random.PRNGKey(123))

    assert sample.shape == (3,)
    assert jnp.all(jnp.isfinite(sample))


def test_multivariate_normal_sample_draws_leading_iid_vectors() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(3))

    samples = distribution_sample(dist, jax.random.PRNGKey(456), sample_shape=(5,))

    assert samples.shape == (5, 3)
    assert jnp.all(jnp.isfinite(samples))
