"""Tests for Uniform distribution."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxstanv5.distributions import Uniform


def test_uniform_log_prob_matches_expected_values() -> None:
    dist = Uniform(-1.0, 3.0)
    values = jnp.asarray([-2.0, -1.0, 1.0, 3.0, 4.0])

    actual = dist.log_prob(values)

    expected_inside = -math.log(4.0)
    expected = jnp.asarray([-jnp.inf, expected_inside, expected_inside, expected_inside, -jnp.inf])
    assert jnp.allclose(actual, expected)


def test_uniform_sample_is_deterministic_and_broadcasts_shape() -> None:
    dist = Uniform(jnp.asarray([0.0, 10.0]), jnp.asarray([1.0, 12.0]))
    key = jax.random.PRNGKey(1)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4, 2)
    assert jnp.allclose(first, second)
    assert jnp.all(first >= jnp.asarray([0.0, 10.0]))
    assert jnp.all(first <= jnp.asarray([1.0, 12.0]))


def test_uniform_cdf_and_icdf_are_inverse_inside_support() -> None:
    dist = Uniform(-2.0, 2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = dist.icdf(probabilities)

    assert jnp.allclose(dist.cdf(values), probabilities, atol=1e-6)


def test_uniform_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = Uniform(jnp.zeros((2, 1)), jnp.ones((3,)))

    assert dist.batch_shape() == (2, 3)
