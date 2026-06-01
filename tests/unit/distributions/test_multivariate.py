"""Tests for multivariate distributions."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from jaxstanv5.distributions import MultivariateNormal


def test_multivariate_normal_log_prob_matches_standard_identity_value() -> None:
    dist = MultivariateNormal(jnp.zeros((2,)), jnp.eye(2))

    actual = dist.log_prob(jnp.asarray([0.0, 0.0]))

    expected = -math.log(2.0 * math.pi)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_matches_cholesky_formula() -> None:
    mean = jnp.asarray([1.0, -0.5])
    chol = jnp.asarray([[2.0, 0.0], [0.5, 1.5]])
    value = jnp.asarray([2.0, 1.0])
    dist = MultivariateNormal(mean, chol)

    actual = dist.log_prob(value)

    solved = solve_triangular(chol, value - mean, lower=True)
    expected = (
        -0.5 * jnp.sum(solved**2) - jnp.sum(jnp.log(jnp.diagonal(chol))) - math.log(2.0 * math.pi)
    )
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_allows_scalar_broadcast_mean() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(3))

    actual = dist.log_prob(jnp.asarray([0.0, 0.0, 0.0]))

    expected = -1.5 * math.log(2.0 * math.pi)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_batches_over_leading_vector_dimensions() -> None:
    dist = MultivariateNormal(jnp.zeros((2,)), jnp.eye(2))
    values = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])

    actual = dist.log_prob(values)

    expected = jnp.asarray([-math.log(2.0 * math.pi), -0.5 - math.log(2.0 * math.pi)])
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_multivariate_normal_log_prob_batches_over_scale_tril() -> None:
    dist = MultivariateNormal(
        mean=jnp.zeros((2, 2)),
        scale_tril=jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)]),
    )

    actual = dist.log_prob(jnp.zeros((2, 2)))

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

    actual = dist.log_prob(jnp.asarray([0.0, 0.0]))

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

    assert dist.event_shape() == (4,)


def test_multivariate_normal_sample_draws_one_event_vector() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(3))

    sample = dist.sample(jax.random.PRNGKey(123))

    assert sample.shape == (3,)
    assert jnp.all(jnp.isfinite(sample))


def test_multivariate_normal_sample_draws_leading_iid_vectors() -> None:
    dist = MultivariateNormal(0.0, jnp.eye(3))

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(5,))

    assert samples.shape == (5, 3)
    assert jnp.all(jnp.isfinite(samples))
