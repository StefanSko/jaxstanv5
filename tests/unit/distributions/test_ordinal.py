"""Tests for ordinal distributions."""

import jax
import jax.numpy as jnp
import pytest

from jaxstanv5._backends.jax.distributions import (
    batch_shape,
    log_prob,
)
from jaxstanv5._backends.jax.distributions import (
    sample as distribution_sample,
)
from jaxstanv5.distributions import OrderedLogistic


def _manual_ordered_logistic_probs(eta: jax.Array, cutpoints: jax.Array) -> jax.Array:
    cumulative = jax.nn.sigmoid(cutpoints - eta[..., None])
    return jnp.concatenate(
        (
            cumulative[..., :1],
            cumulative[..., 1:] - cumulative[..., :-1],
            1.0 - cumulative[..., -1:],
        ),
        axis=-1,
    )


def test_ordered_logistic_log_prob_matches_expected_scalar_categories() -> None:
    eta = jnp.asarray(0.0)
    cutpoints = jnp.asarray([-1.0, 1.0])
    dist = OrderedLogistic(eta, cutpoints)

    actual = log_prob(dist, jnp.asarray([0.0, 1.0, 2.0]))

    expected = jnp.log(_manual_ordered_logistic_probs(eta, cutpoints))
    assert actual.shape == (3,)
    assert jnp.allclose(actual, expected)


def test_ordered_logistic_log_prob_is_negative_infinity_outside_support() -> None:
    dist = OrderedLogistic(0.0, jnp.asarray([-1.0, 1.0]))

    negative = log_prob(dist, jnp.asarray(-1.0))
    too_large = log_prob(dist, jnp.asarray(3.0))
    fractional = log_prob(dist, jnp.asarray(0.5))

    assert negative == -jnp.inf
    assert too_large == -jnp.inf
    assert fractional == -jnp.inf


def test_ordered_logistic_log_prob_requires_ordered_cutpoints() -> None:
    dist = OrderedLogistic(0.0, jnp.asarray([1.0, -1.0]))

    actual = log_prob(dist, jnp.asarray([0.0, 1.0, 2.0]))

    assert jnp.all(actual == -jnp.inf)


def test_ordered_logistic_log_prob_broadcasts_over_observations() -> None:
    eta = jnp.asarray([-1.0, 0.0, 1.0])
    cutpoints = jnp.asarray([-0.5, 0.75])
    values = jnp.asarray([0.0, 1.0, 2.0])
    dist = OrderedLogistic(eta, cutpoints)

    actual = log_prob(dist, values)

    probabilities = _manual_ordered_logistic_probs(eta, cutpoints)
    expected = jnp.log(jnp.asarray([probabilities[0, 0], probabilities[1, 1], probabilities[2, 2]]))
    assert actual.shape == (3,)
    assert jnp.allclose(actual, expected)


def test_ordered_logistic_log_prob_matches_bernoulli_when_one_cutpoint() -> None:
    eta = jnp.asarray([-1.0, 0.0, 1.0])
    cutpoints = jnp.asarray([0.25])
    values = jnp.asarray([0.0, 1.0, 1.0])
    dist = OrderedLogistic(eta, cutpoints)

    actual = log_prob(dist, values)

    probability_category_one = jax.nn.sigmoid(eta - cutpoints[0])
    probability = jnp.where(values == 1.0, probability_category_one, 1.0 - probability_category_one)
    assert jnp.allclose(actual, jnp.log(probability))


def test_ordered_logistic_rejects_scalar_cutpoints() -> None:
    dist = OrderedLogistic(0.0, 1.0)

    with pytest.raises(ValueError, match="cutpoints must be a vector"):
        log_prob(dist, jnp.asarray(0.0))


def test_ordered_logistic_batch_shape_broadcasts_eta_and_cutpoint_batch_shape() -> None:
    dist = OrderedLogistic(jnp.zeros((2, 1)), jnp.zeros((3, 2)))

    assert batch_shape(dist) == (2, 3)


def test_ordered_logistic_sample_rejects_unordered_cutpoints() -> None:
    dist = OrderedLogistic(0.0, jnp.asarray([2.0, -2.0]))

    with pytest.raises(ValueError, match="strictly increasing"):
        distribution_sample(dist, jax.random.PRNGKey(1), sample_shape=(10,))


def test_ordered_logistic_sample_remains_jittable() -> None:
    @jax.jit
    def draw(cutpoints: jax.Array) -> jax.Array:
        return distribution_sample(
            OrderedLogistic(0.0, cutpoints), jax.random.PRNGKey(3), sample_shape=(4,)
        )

    sample = draw(jnp.asarray([-1.0, 1.0]))

    assert sample.shape == (4,)


def test_ordered_logistic_sample_clamps_tiny_negative_probabilities_before_log() -> None:
    cutpoints = jnp.asarray([0.0, jnp.finfo(jnp.float32).tiny])
    dist = OrderedLogistic(jnp.asarray(0.0, dtype=jnp.float32), cutpoints)

    sample = distribution_sample(dist, jax.random.PRNGKey(2), sample_shape=(10,))

    assert sample.shape == (10,)
    assert jnp.all(sample >= 0)
    assert jnp.all(sample <= 2)


def test_ordered_logistic_sample_is_deterministic_for_seed() -> None:
    dist = OrderedLogistic(jnp.asarray([-1.0, 0.0, 1.0]), jnp.asarray([-0.5, 0.75]))
    key = jax.random.PRNGKey(123)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4, 3)
    assert jnp.all(first >= 0)
    assert jnp.all(first <= 2)
    assert jnp.allclose(first, second)
