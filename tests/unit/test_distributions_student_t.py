"""Tests for StudentT distribution."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions import StudentT


def _student_t_log_prob(df: float, loc: float, scale: float, value: float) -> jax.Array:
    standardized = (value - loc) / scale
    return (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * jnp.log(df * math.pi)
        - jnp.log(scale)
        - 0.5 * (df + 1.0) * jnp.log1p(standardized**2 / df)
    )


def test_student_t_log_prob_matches_expected_scalar_value() -> None:
    dist = StudentT(4.0, 1.0, 2.0)

    actual = dist.log_prob(jnp.asarray(1.5))

    expected = _student_t_log_prob(4.0, 1.0, 2.0, 1.5)
    assert jnp.allclose(actual, expected)


def test_student_t_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = StudentT(jnp.asarray([4.0, 8.0]), 1.0, jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([1.5, -0.5])

    actual = dist.log_prob(values)

    expected = jnp.asarray(
        [
            _student_t_log_prob(4.0, 1.0, 1.0, 1.5),
            _student_t_log_prob(8.0, 1.0, 2.0, -0.5),
        ]
    )
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_student_t_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = StudentT(jnp.ones((2, 1)) * 4.0, jnp.zeros((3,)), 1.0)

    assert dist.batch_shape() == (2, 3)


def test_student_t_sample_is_deterministic_for_seed() -> None:
    dist = StudentT(4.0, 1.0, 2.0)
    key = jax.random.PRNGKey(123)

    first = dist.sample(key, sample_shape=(4,))
    second = dist.sample(key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(jnp.isfinite(first))
    assert jnp.allclose(first, second)


def test_student_t_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = StudentT(4.0, jnp.asarray([0.0, 1.0, 2.0]), 0.5)

    samples = dist.sample(jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(jnp.isfinite(samples))
