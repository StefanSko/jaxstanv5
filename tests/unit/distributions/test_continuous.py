"""Tests for scalar continuous distributions."""

import math

import jax
import jax.numpy as jnp
from bayeswire.distributions import Beta, Exponential, HalfNormal, Normal, StudentT, Uniform
from jax.scipy.special import gammaln

from jaxstanv5._backends.jax.distributions import (
    batch_shape,
    cdf,
    icdf,
    log_prob,
)
from jaxstanv5._backends.jax.distributions import (
    sample as distribution_sample,
)


def beta_log_function(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return log B(a, b)."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def _student_t_log_prob(df: float, loc: float, scale: float, value: float) -> jax.Array:
    standardized = (value - loc) / scale
    return (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * jnp.log(df * math.pi)
        - jnp.log(scale)
        - 0.5 * (df + 1.0) * jnp.log1p(standardized**2 / df)
    )


def test_standard_normal_log_prob_matches_expected_scalar_value() -> None:
    dist = Normal(0.0, 1.0)

    actual = log_prob(dist, jnp.asarray(0.0))

    expected = -0.5 * math.log(2.0 * math.pi)
    assert jnp.allclose(actual, expected)


def test_normal_invalid_scale_returns_negative_infinity_with_finite_gradient() -> None:
    value = log_prob(Normal(0.0, -1.0), jnp.asarray(0.0))
    gradient = jax.grad(lambda scale: log_prob(Normal(0.0, scale), jnp.asarray(0.0)))(0.0)

    assert value == -jnp.inf
    assert jnp.isfinite(gradient)


def test_normal_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = Normal(0.0, 2.0)
    values = jnp.asarray([0.0, 2.0])

    actual = log_prob(dist, values)

    expected = jnp.asarray(
        [
            -math.log(2.0) - 0.5 * math.log(2.0 * math.pi),
            -0.5 - math.log(2.0) - 0.5 * math.log(2.0 * math.pi),
        ]
    )
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_normal_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = Normal(jnp.zeros((2, 1)), jnp.ones((3,)))

    assert batch_shape(dist) == (2, 3)


def test_normal_cdf_and_icdf_are_inverse_for_central_probabilities() -> None:
    dist = Normal(1.0, 2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = icdf(dist, probabilities)

    assert jnp.allclose(cdf(dist, values), probabilities, atol=1e-6)


def test_normal_icdf_and_cdf_are_inverse_for_values() -> None:
    dist = Normal(jnp.asarray([0.0, 1.0]), jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([-0.5, 3.0])

    probabilities = cdf(dist, values)

    assert jnp.allclose(icdf(dist, probabilities), values, atol=1e-6)


def test_normal_sample_is_deterministic_for_seed() -> None:
    dist = Normal(1.0, 2.0)
    key = jax.random.PRNGKey(123)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.allclose(first, second)


def test_normal_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Normal(jnp.asarray([0.0, 1.0, 2.0]), 0.5)

    samples = distribution_sample(dist, jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(jnp.isfinite(samples))


def test_half_normal_log_prob_matches_expected_scalar_value() -> None:
    dist = HalfNormal(2.0)

    actual = log_prob(dist, jnp.asarray(0.0))

    expected = 0.5 * math.log(2.0 / math.pi) - math.log(2.0)
    assert jnp.allclose(actual, expected)


def test_half_normal_log_prob_is_negative_infinity_outside_support() -> None:
    dist = HalfNormal(2.0)

    actual = log_prob(dist, jnp.asarray(-0.1))
    invalid_scale = log_prob(HalfNormal(-1.0), jnp.asarray(0.1))
    gradient = jax.grad(lambda scale: log_prob(HalfNormal(scale), jnp.asarray(0.1)))(0.0)

    assert actual == -jnp.inf
    assert invalid_scale == -jnp.inf
    assert jnp.isfinite(gradient)


def test_half_normal_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = HalfNormal(jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([0.0, 2.0])

    actual = log_prob(dist, values)

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

    assert batch_shape(dist) == (2, 3)


def test_half_normal_cdf_and_icdf_are_inverse_for_central_probabilities() -> None:
    dist = HalfNormal(2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = icdf(dist, probabilities)

    assert jnp.allclose(cdf(dist, values), probabilities, atol=1e-6)


def test_half_normal_sample_is_deterministic_for_seed() -> None:
    dist = HalfNormal(2.0)
    key = jax.random.PRNGKey(123)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.allclose(first, second)


def test_half_normal_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = HalfNormal(jnp.asarray([1.0, 2.0, 3.0]))

    samples = distribution_sample(dist, jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)


def test_student_t_log_prob_matches_expected_scalar_value() -> None:
    dist = StudentT(4.0, 1.0, 2.0)

    actual = log_prob(dist, jnp.asarray(1.5))

    expected = _student_t_log_prob(4.0, 1.0, 2.0, 1.5)
    assert jnp.allclose(actual, expected)


def test_student_t_invalid_parameters_return_negative_infinity_with_finite_gradients() -> None:
    invalid_df = log_prob(StudentT(-1.0, 0.0, 1.0), jnp.asarray(0.0))
    invalid_scale = log_prob(StudentT(4.0, 0.0, -1.0), jnp.asarray(0.0))
    df_gradient = jax.grad(lambda df: log_prob(StudentT(df, 0.0, 1.0), jnp.asarray(0.0)))(0.0)
    scale_gradient = jax.grad(lambda scale: log_prob(StudentT(4.0, 0.0, scale), jnp.asarray(0.0)))(
        0.0
    )

    assert invalid_df == -jnp.inf
    assert invalid_scale == -jnp.inf
    assert jnp.isfinite(df_gradient)
    assert jnp.isfinite(scale_gradient)


def test_student_t_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = StudentT(jnp.asarray([4.0, 8.0]), 1.0, jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([1.5, -0.5])

    actual = log_prob(dist, values)

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

    assert batch_shape(dist) == (2, 3)


def test_student_t_sample_is_deterministic_for_seed() -> None:
    dist = StudentT(4.0, 1.0, 2.0)
    key = jax.random.PRNGKey(123)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(jnp.isfinite(first))
    assert jnp.allclose(first, second)


def test_student_t_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = StudentT(4.0, jnp.asarray([0.0, 1.0, 2.0]), 0.5)

    samples = distribution_sample(dist, jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(jnp.isfinite(samples))


def test_exponential_log_prob_matches_expected_scalar_value() -> None:
    dist = Exponential(2.0)

    actual = log_prob(dist, jnp.asarray(0.5))

    expected = math.log(2.0) - 1.0
    assert jnp.allclose(actual, expected)


def test_exponential_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Exponential(2.0)

    actual = log_prob(dist, jnp.asarray(-0.1))
    invalid_rate = log_prob(Exponential(-1.0), jnp.asarray(0.1))
    gradient = jax.grad(lambda rate: log_prob(Exponential(rate), jnp.asarray(0.1)))(0.0)

    assert actual == -jnp.inf
    assert invalid_rate == -jnp.inf
    assert jnp.isfinite(gradient)


def test_exponential_log_prob_broadcasts_over_vector_inputs() -> None:
    dist = Exponential(jnp.asarray([1.0, 2.0]))
    values = jnp.asarray([0.5, 0.5])

    actual = log_prob(dist, values)

    expected = jnp.asarray([math.log(1.0) - 0.5, math.log(2.0) - 1.0])
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_exponential_batch_shape_matches_rate_shape() -> None:
    dist = Exponential(jnp.ones((2, 3)))

    assert batch_shape(dist) == (2, 3)


def test_exponential_cdf_and_icdf_are_inverse_for_central_probabilities() -> None:
    dist = Exponential(2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = icdf(dist, probabilities)

    assert jnp.allclose(cdf(dist, values), probabilities, atol=1e-6)


def test_exponential_sample_is_deterministic_for_seed() -> None:
    dist = Exponential(2.0)
    key = jax.random.PRNGKey(123)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first >= 0.0)
    assert jnp.allclose(first, second)


def test_exponential_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Exponential(jnp.asarray([1.0, 2.0, 3.0]))

    samples = distribution_sample(dist, jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples >= 0.0)


def test_uniform_log_prob_matches_expected_values() -> None:
    dist = Uniform(-1.0, 3.0)
    values = jnp.asarray([-2.0, -1.0, 1.0, 3.0, 4.0])

    actual = log_prob(dist, values)

    expected_inside = -math.log(4.0)
    expected = jnp.asarray([-jnp.inf, expected_inside, expected_inside, expected_inside, -jnp.inf])
    assert jnp.allclose(actual, expected)


def test_uniform_invalid_bounds_return_negative_infinity_with_finite_gradients() -> None:
    value = log_prob(Uniform(1.0, 0.0), jnp.asarray(0.5))
    gradient = jax.grad(lambda high: log_prob(Uniform(0.0, high), jnp.asarray(0.5)))(0.0)

    assert value == -jnp.inf
    assert jnp.isfinite(gradient)


def test_uniform_sample_is_deterministic_and_broadcasts_shape() -> None:
    dist = Uniform(jnp.asarray([0.0, 10.0]), jnp.asarray([1.0, 12.0]))
    key = jax.random.PRNGKey(1)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4, 2)
    assert jnp.allclose(first, second)
    assert jnp.all(first >= jnp.asarray([0.0, 10.0]))
    assert jnp.all(first <= jnp.asarray([1.0, 12.0]))


def test_uniform_cdf_and_icdf_are_inverse_inside_support() -> None:
    dist = Uniform(-2.0, 2.0)
    probabilities = jnp.asarray([0.1, 0.5, 0.9])

    values = icdf(dist, probabilities)

    assert jnp.allclose(cdf(dist, values), probabilities, atol=1e-6)


def test_uniform_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = Uniform(jnp.zeros((2, 1)), jnp.ones((3,)))

    assert batch_shape(dist) == (2, 3)


def test_beta_log_prob_matches_expected_scalar_value() -> None:
    dist = Beta(2.0, 5.0)

    actual = log_prob(dist, jnp.asarray(0.3))

    expected = (2.0 - 1.0) * math.log(0.3)
    expected += (5.0 - 1.0) * math.log1p(-0.3)
    expected -= float(beta_log_function(jnp.asarray(2.0), jnp.asarray(5.0)))
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_beta_log_prob_is_negative_infinity_outside_support() -> None:
    dist = Beta(2.0, 3.0)

    zero = log_prob(dist, jnp.asarray(0.0))
    one = log_prob(dist, jnp.asarray(1.0))
    negative = log_prob(dist, jnp.asarray(-0.1))
    too_large = log_prob(dist, jnp.asarray(1.1))
    invalid_alpha = log_prob(Beta(0.0, 3.0), jnp.asarray(0.5))
    invalid_beta = log_prob(Beta(2.0, -1.0), jnp.asarray(0.5))

    assert zero == -jnp.inf
    assert one == -jnp.inf
    assert negative == -jnp.inf
    assert too_large == -jnp.inf
    assert invalid_alpha == -jnp.inf
    assert invalid_beta == -jnp.inf


def test_beta_log_prob_broadcasts_over_vector_inputs() -> None:
    alpha = jnp.asarray([1.5, 2.0])
    beta = jnp.asarray([3.0, 4.0])
    values = jnp.asarray([0.25, 0.75])
    dist = Beta(alpha, beta)

    actual = log_prob(dist, values)

    expected = (alpha - 1.0) * jnp.log(values)
    expected += (beta - 1.0) * jnp.log1p(-values)
    expected -= beta_log_function(alpha, beta)
    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)


def test_beta_batch_shape_broadcasts_parameter_shapes() -> None:
    dist = Beta(jnp.ones((2, 1)), jnp.ones((3,)))

    assert batch_shape(dist) == (2, 3)


def test_beta_sample_is_deterministic_for_seed() -> None:
    dist = Beta(2.0, 3.0)
    key = jax.random.PRNGKey(123)

    first = distribution_sample(dist, key, sample_shape=(4,))
    second = distribution_sample(dist, key, sample_shape=(4,))

    assert first.shape == (4,)
    assert jnp.all(first > 0.0)
    assert jnp.all(first < 1.0)
    assert jnp.allclose(first, second)


def test_beta_sample_broadcasts_distribution_shape_after_sample_shape() -> None:
    dist = Beta(jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([4.0, 5.0, 6.0]))

    samples = distribution_sample(dist, jax.random.PRNGKey(456), sample_shape=(2,))

    assert samples.shape == (2, 3)
    assert jnp.all(samples > 0.0)
    assert jnp.all(samples < 1.0)
