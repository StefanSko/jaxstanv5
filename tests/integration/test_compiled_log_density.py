"""Integration tests for compiling declared models to log-density functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from _helpers import bind_model
from jax.scipy.special import gammaln

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval
from jaxstanv5.distributions import (
    Beta,
    BetaBinomial,
    Binomial,
    NegativeBinomial,
    Normal,
    OrderedLogistic,
    Poisson,
)
from jaxstanv5.math import exp, sigmoid


@model
class SimpleNormal:
    """Scalar normal with known scale and no constraints."""

    mu = Param(Normal(0, 1))
    y = Observed(Normal(mu, 1))


@model
class ConstrainedNormal:
    """Scalar normal with constrained scale."""

    sigma = Param(Normal(0, 1), constraint=Positive())
    y = Observed(Normal(0, sigma))


@model
class IntervalConstrainedNormal:
    """Scalar normal prior constrained to a finite interval."""

    x = Param(Normal(0, 1), constraint=Interval(-2.0, 3.0))


@model
class UnitIntervalConstrainedBeta:
    """Scalar Beta prior constrained to the unit interval."""

    theta = Param(Beta(2.0, 3.0), constraint=UnitInterval())


@model
class LinearPredictor:
    """One-predictor linear model with known scale."""

    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, 1))


@model
class PoissonLogDensity:
    """Poisson count model using symbolic exponential rate."""

    eta = Param(Normal(0, 1))
    rate = exp(eta)
    y = Observed(Poisson(rate))


@model
class MatrixColumnPredictor:
    """Linear model using ordinary matrix-column declaration indexing."""

    x = Data.matrix()
    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    mu = alpha + beta * x[:, 0]
    y = Observed(Normal(mu, 1))


@model
class MatrixRowAndDataIndexPredictor:
    """Model exercising row slicing and data/static tuple indexing."""

    n_obs = Data.scalar()
    x = Data.matrix()
    row_idx = Data.vector()
    beta = Param(Normal(0, 1), size=n_obs)
    y = Observed(Normal(x[0, :] + x[row_idx, 0] + beta, 1))


@model
class SeparatedAdvancedIndexPredictor:
    """Model with a scalar index before a slice and a data index after it."""

    x = Data.array(rank=3)
    group_idx = Data.vector()
    theta = Param(Normal(0, 1))
    y = Observed(Normal(theta + x[0, :, group_idx][3, :], 1))


@model
class IndexedNormal:
    """Hierarchical normal with data-indexed group effects."""

    n_groups = Data.scalar()
    group_idx = Data.vector()
    theta = Param(Normal(0, 1), size=n_groups)
    y = Observed(Normal(theta[group_idx], 1))


@model
class BinomialLogisticDensity:
    """Binomial count model using symbolic logistic success probabilities."""

    eta = Param(Normal(0, 1))
    trials = Data.vector()
    y = Observed(Binomial(trials, sigmoid(eta)))


@model
class BetaBinomialLogisticDensity:
    """Beta-binomial count model using symbolic logistic mean and concentration."""

    eta = Param(Normal(0, 1))
    log_concentration = Param(Normal(math.log(20.0), 0.5))
    trials = Data.vector()
    p = sigmoid(eta)
    concentration = exp(log_concentration)
    a = p * concentration
    b = (1.0 - p) * concentration
    y = Observed(BetaBinomial(trials, a, b))


@model
class BetaLogisticDensity:
    """Beta model using symbolic logistic mean and concentration."""

    eta = Param(Normal(0, 1))
    log_concentration = Param(Normal(math.log(20.0), 0.5))
    p = sigmoid(eta)
    concentration = exp(log_concentration)
    a = p * concentration
    b = (1.0 - p) * concentration
    y = Observed(Beta(a, b))


@model
class NegativeBinomialLogRateDensity:
    """Negative-binomial count model using symbolic mean and overdispersion."""

    eta = Param(Normal(0, 1))
    log_overdispersion = Param(Normal(math.log(5.0), 0.5))
    mean = exp(eta)
    overdispersion = exp(log_overdispersion)
    y = Observed(NegativeBinomial(mean, overdispersion))


@model
class OrderedLogisticDensity:
    """Ordinal model using ordered cutpoints and zero-based labels."""

    n_cutpoints = Data.scalar()
    x = Data.vector()

    beta = Param(Normal(0.0, 1.0))
    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())

    eta = beta * x
    y = Observed(OrderedLogistic(eta, cutpoints))


@model
class MeasurementErrorLogDensity:
    """Measurement-error model with latent vectors and two observed sites."""

    n_states = Data.scalar()
    age = Data.vector()
    marriage_sd = Data.vector()
    divorce_sd = Data.vector()

    alpha = Param(Normal(0, 1))
    b_age = Param(Normal(0, 1))
    b_marriage = Param(Normal(0, 1))
    sigma = Param(Normal(0, 1), constraint=Positive())

    marriage_true = Param(Normal(0, 1), size=n_states)
    divorce_mu = alpha + b_age * age + b_marriage * marriage_true
    divorce_true = Param(Normal(divorce_mu, sigma), size=n_states)

    marriage_obs = Observed(Normal(marriage_true, marriage_sd))
    divorce_obs = Observed(Normal(divorce_true, divorce_sd))


def normal_log_prob(x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Element-wise normal log-density matching ``Normal.log_prob``."""
    standardized = (x - loc) / scale
    return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * math.log(2.0 * math.pi)


def beta_log_prob(x: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Element-wise Beta log-density matching ``Beta.log_prob``."""
    log_normalizer = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return (alpha - 1.0) * jnp.log(x) + (beta - 1.0) * jnp.log1p(-x) - log_normalizer


def ordered_logistic_log_prob(
    y: jnp.ndarray,
    eta: jnp.ndarray,
    cutpoints: jnp.ndarray,
) -> jnp.ndarray:
    """Element-wise zero-based ordered-logistic log mass."""
    cumulative = jax.nn.sigmoid(cutpoints[:, None] - eta[None, :])
    probabilities = jnp.stack(
        (
            cumulative[0],
            cumulative[1] - cumulative[0],
            1.0 - cumulative[1],
        ),
        axis=-1,
    )
    selected = jnp.take_along_axis(probabilities, y[:, None], axis=-1)[..., 0]
    return jnp.log(selected)


def test_compiled_log_density_for_simple_unconstrained_model() -> None:
    bound = bind_model(SimpleNormal, y=jnp.array(2.0))
    log_prob = compile_log_density(bound)

    q = jnp.array([0.5])
    lp = log_prob(q)

    expected_prior = normal_log_prob(jnp.array(0.5), jnp.array(0.0), jnp.array(1.0))
    expected_obs = normal_log_prob(jnp.array(2.0), jnp.array(0.5), jnp.array(1.0))
    expected = expected_prior + expected_obs

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_for_constrained_model_includes_jacobian() -> None:
    bound = bind_model(ConstrainedNormal, y=jnp.array(3.0))
    log_prob = compile_log_density(bound)

    q = jnp.array([math.log(2.0)])
    lp = log_prob(q)

    sigma_constrained = jnp.exp(q[0])
    expected_prior = normal_log_prob(sigma_constrained, jnp.array(0.0), jnp.array(1.0))
    expected_obs = normal_log_prob(jnp.array(3.0), jnp.array(0.0), sigma_constrained)
    expected = expected_prior + expected_obs + q[0]

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_for_interval_constrained_model_includes_jacobian() -> None:
    bound = bind_model(IntervalConstrainedNormal)
    log_prob = compile_log_density(bound)

    q = jnp.array([0.4])
    lp = log_prob(q)

    width = jnp.array(5.0)
    constrained = -2.0 + width * jax.nn.sigmoid(q[0])
    expected_prior = normal_log_prob(constrained, jnp.array(0.0), jnp.array(1.0))
    expected_log_jacobian = jnp.log(width) - jax.nn.softplus(-q[0]) - jax.nn.softplus(q[0])
    expected = expected_prior + expected_log_jacobian

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_for_unit_interval_beta_includes_jacobian() -> None:
    bound = bind_model(UnitIntervalConstrainedBeta)
    log_prob = compile_log_density(bound)

    q = jnp.array([0.4])
    lp = log_prob(q)

    theta = jax.nn.sigmoid(q[0])
    expected_prior = beta_log_prob(theta, jnp.array(2.0), jnp.array(3.0))
    expected_log_jacobian = -jax.nn.softplus(-q[0]) - jax.nn.softplus(q[0])
    expected = expected_prior + expected_log_jacobian

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_unary_expression_likelihood_fields() -> None:
    y_data = jnp.array([0.0, 2.0, 3.0])
    bound = bind_model(PoissonLogDensity, y=y_data)
    log_prob = compile_log_density(bound)

    eta = jnp.array(0.4)
    lp = log_prob(jnp.array([eta]))

    expected = normal_log_prob(eta, jnp.array(0.0), jnp.array(1.0))
    rate = jnp.exp(eta)
    expected += jnp.sum(
        y_data * jnp.log(rate) - rate - jnp.asarray([0.0, math.log(2.0), math.log(6.0)])
    )
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_sigmoid_binomial_likelihood_fields() -> None:
    trials = jnp.array([4.0, 5.0, 6.0])
    y_data = jnp.array([1.0, 3.0, 4.0])
    bound = bind_model(BinomialLogisticDensity, trials=trials, y=y_data)
    log_prob = compile_log_density(bound)

    eta = jnp.array(0.25)
    lp = log_prob(jnp.array([eta]))

    expected = normal_log_prob(eta, jnp.array(0.0), jnp.array(1.0))
    probs = 1.0 / (1.0 + jnp.exp(-eta))
    expected += jnp.sum(
        gammaln(trials + 1.0)
        - gammaln(y_data + 1.0)
        - gammaln(trials - y_data + 1.0)
        + y_data * jnp.log(probs)
        + (trials - y_data) * jnp.log1p(-probs)
    )
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_beta_binomial_likelihood_fields() -> None:
    trials = jnp.array([4.0, 5.0, 6.0])
    y_data = jnp.array([1.0, 3.0, 4.0])
    bound = bind_model(BetaBinomialLogisticDensity, trials=trials, y=y_data)
    log_prob = compile_log_density(bound)

    eta = jnp.array(0.25)
    log_concentration = jnp.array(math.log(15.0))
    lp = log_prob(jnp.array([eta, log_concentration]))

    expected = normal_log_prob(eta, jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(log_concentration, jnp.array(math.log(20.0)), jnp.array(0.5))
    p = 1.0 / (1.0 + jnp.exp(-eta))
    concentration = jnp.exp(log_concentration)
    a = p * concentration
    b = (1.0 - p) * concentration
    expected += jnp.sum(
        gammaln(trials + 1.0)
        - gammaln(y_data + 1.0)
        - gammaln(trials - y_data + 1.0)
        + gammaln(y_data + a)
        + gammaln(trials - y_data + b)
        - gammaln(trials + a + b)
        - gammaln(a)
        - gammaln(b)
        + gammaln(a + b)
    )
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_beta_likelihood_fields() -> None:
    y_data = jnp.array([0.2, 0.4, 0.7])
    bound = bind_model(BetaLogisticDensity, y=y_data)
    log_prob = compile_log_density(bound)

    eta = jnp.array(0.25)
    log_concentration = jnp.array(math.log(15.0))
    lp = log_prob(jnp.array([eta, log_concentration]))

    expected = normal_log_prob(eta, jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(log_concentration, jnp.array(math.log(20.0)), jnp.array(0.5))
    p = 1.0 / (1.0 + jnp.exp(-eta))
    concentration = jnp.exp(log_concentration)
    a = p * concentration
    b = (1.0 - p) * concentration
    expected += jnp.sum(beta_log_prob(y_data, a, b))
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_includes_ordered_constraint_jacobian_and_likelihood() -> None:
    x_data = jnp.asarray([-1.0, 0.0, 1.0])
    y_data = jnp.asarray([0, 1, 2])
    bound = bind_model(OrderedLogisticDensity, n_cutpoints=2, x=x_data, y=y_data)
    log_prob = compile_log_density(bound)

    beta = jnp.asarray(0.5)
    raw_cutpoints = jnp.asarray([-0.4, 0.3])
    constrained_cutpoints = jnp.asarray(Ordered().inverse_transform(raw_cutpoints))
    q = jnp.concatenate((jnp.asarray([beta]), raw_cutpoints))

    lp = log_prob(q)

    expected = normal_log_prob(beta, jnp.asarray(0.0), jnp.asarray(1.0))
    expected += jnp.sum(normal_log_prob(constrained_cutpoints, jnp.asarray(0.0), jnp.asarray(2.0)))
    expected += jnp.sum(ordered_logistic_log_prob(y_data, beta * x_data, constrained_cutpoints))
    expected += raw_cutpoints[1]
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_negative_binomial_likelihood_fields() -> None:
    y_data = jnp.array([0.0, 2.0, 3.0])
    bound = bind_model(NegativeBinomialLogRateDensity, y=y_data)
    log_prob = compile_log_density(bound)

    eta = jnp.array(0.25)
    log_overdispersion = jnp.array(math.log(4.0))
    lp = log_prob(jnp.array([eta, log_overdispersion]))

    expected = normal_log_prob(eta, jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(log_overdispersion, jnp.array(math.log(5.0)), jnp.array(0.5))
    mean = jnp.exp(eta)
    overdispersion = jnp.exp(log_overdispersion)
    total = mean + overdispersion
    expected += jnp.sum(
        gammaln(y_data + overdispersion)
        - gammaln(overdispersion)
        - gammaln(y_data + 1.0)
        + overdispersion * jnp.log(overdispersion / total)
        + y_data * jnp.log(mean / total)
    )
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_includes_measurement_error_observed_sites() -> None:
    age = jnp.array([-1.0, 1.0])
    marriage_sd = jnp.array([0.1, 0.2])
    divorce_sd = jnp.array([0.3, 0.4])
    marriage_obs = jnp.array([0.25, -0.75])
    divorce_obs = jnp.array([0.1, -0.2])
    bound = bind_model(
        MeasurementErrorLogDensity,
        n_states=2,
        age=age,
        marriage_sd=marriage_sd,
        divorce_sd=divorce_sd,
        marriage_obs=marriage_obs,
        divorce_obs=divorce_obs,
    )
    log_prob = compile_log_density(bound)

    alpha = jnp.array(0.2)
    b_age = jnp.array(-0.4)
    b_marriage = jnp.array(0.3)
    log_sigma = jnp.array(-0.2)
    sigma = jnp.exp(log_sigma)
    marriage_true = jnp.array([0.4, -0.6])
    divorce_true = jnp.array([0.05, -0.35])
    q = jnp.concatenate(
        (
            jnp.array([alpha, b_age, b_marriage, log_sigma]),
            marriage_true,
            divorce_true,
        )
    )
    lp = log_prob(q)

    divorce_mu = alpha + b_age * age + b_marriage * marriage_true
    expected = normal_log_prob(alpha, jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(b_age, jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(b_marriage, jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(sigma, jnp.array(0.0), jnp.array(1.0)) + log_sigma
    expected += jnp.sum(normal_log_prob(marriage_true, jnp.array(0.0), jnp.array(1.0)))
    expected += jnp.sum(normal_log_prob(divorce_true, divorce_mu, sigma))
    expected += jnp.sum(normal_log_prob(marriage_obs, marriage_true, marriage_sd))
    expected += jnp.sum(normal_log_prob(divorce_obs, divorce_true, divorce_sd))

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_data_expressions() -> None:
    x_data = jnp.array([1.0, 2.0, 3.0])
    y_data = jnp.array([2.1, 2.9, 4.1])
    bound = bind_model(LinearPredictor, x=x_data, y=y_data)
    log_prob = compile_log_density(bound)

    q = jnp.array([2.0, 0.5])
    lp = log_prob(q)

    prior_lp = normal_log_prob(jnp.array(2.0), jnp.array(0.0), jnp.array(1.0))
    prior_lp += normal_log_prob(jnp.array(0.5), jnp.array(0.0), jnp.array(1.0))

    mu_computed = 2.0 + 0.5 * x_data
    obs_lp = jnp.sum(normal_log_prob(y_data, mu_computed, jnp.array(1.0)))

    expected = prior_lp + obs_lp
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_evaluates_matrix_column_indexing() -> None:
    x_data = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    y_data = jnp.array([2.0, 2.7, 4.1])
    bound = bind_model(MatrixColumnPredictor, x=x_data, y=y_data)
    log_prob = compile_log_density(bound)

    q = jnp.array([1.0, 0.75])
    actual = log_prob(q)

    expected = normal_log_prob(jnp.array(1.0), jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(jnp.array(0.75), jnp.array(0.0), jnp.array(1.0))
    expected += jnp.sum(normal_log_prob(y_data, 1.0 + 0.75 * x_data[:, 0], jnp.array(1.0)))

    assert jnp.allclose(actual, expected, atol=1e-6)


def test_compiled_log_density_evaluates_row_and_data_tuple_indexing() -> None:
    x_data = jnp.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    row_idx = jnp.array([0, 1, 0])
    y_data = jnp.array([2.5, 22.5, 101.5])
    bound = bind_model(MatrixRowAndDataIndexPredictor, n_obs=3, x=x_data, row_idx=row_idx, y=y_data)
    log_prob = compile_log_density(bound)

    beta = jnp.array([0.5, -0.25, 0.75])
    actual = log_prob(beta)

    expected = jnp.sum(normal_log_prob(beta, jnp.array(0.0), jnp.array(1.0)))
    expected += jnp.sum(
        normal_log_prob(y_data, x_data[0, :] + x_data[row_idx, 0] + beta, jnp.array(1.0))
    )

    assert jnp.allclose(actual, expected, atol=1e-6)


def test_bind_rejects_out_of_bounds_second_axis_tuple_indexing() -> None:
    @model
    class BadColumnIndex:
        x = Data.matrix()
        theta = Param(Normal(0, 1))
        y = Observed(Normal(theta + x[:, 2], 1))

    with pytest.raises(ValueError, match="axis 1"):
        bind_model(BadColumnIndex, x=jnp.ones((3, 2)), y=jnp.ones(3))


def test_compiled_log_density_evaluates_separated_scalar_slice_data_indexing() -> None:
    x_data = jnp.arange(30.0).reshape((2, 3, 5))
    group_idx = jnp.array([0, 2, 4, 1])
    y_data = jnp.array([1.0, 4.0, 7.0])
    bound = bind_model(SeparatedAdvancedIndexPredictor, x=x_data, group_idx=group_idx, y=y_data)
    log_prob = compile_log_density(bound)

    theta = jnp.array(0.25)
    actual = log_prob(jnp.array([theta]))

    expected = normal_log_prob(theta, jnp.array(0.0), jnp.array(1.0))
    expected += jnp.sum(
        normal_log_prob(y_data, theta + x_data[0, :, group_idx][3, :], jnp.array(1.0))
    )
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_compiled_log_density_evaluates_unary_negation_expression() -> None:
    @model
    class NegatedLinearPredictor:
        alpha = Param(Normal(0, 1))
        beta = Param(Normal(0, 1))
        x = Data.vector()
        mu = -(alpha + beta * x)
        y = Observed(Normal(mu, 1))

    x_data = jnp.array([1.0, 2.0, 3.0])
    y_data = jnp.array([-2.4, -3.1, -3.6])
    bound = bind_model(NegatedLinearPredictor, x=x_data, y=y_data)
    log_prob = compile_log_density(bound)

    q = jnp.array([2.0, 0.5])
    actual = log_prob(q)

    expected = normal_log_prob(jnp.array(2.0), jnp.array(0.0), jnp.array(1.0))
    expected += normal_log_prob(jnp.array(0.5), jnp.array(0.0), jnp.array(1.0))
    expected += jnp.sum(normal_log_prob(y_data, -(2.0 + 0.5 * x_data), jnp.array(1.0)))

    assert jnp.allclose(actual, expected, atol=1e-6)


def test_compiled_log_density_allows_valid_data_indexing() -> None:
    group_idx = jnp.array([0, 1, 0])
    y = jnp.array([0.25, -0.5, 0.5])
    bound = bind_model(IndexedNormal, n_groups=2, group_idx=group_idx, y=y)
    log_prob = compile_log_density(bound)

    theta = jnp.array([1.0, 2.0])
    actual = log_prob(theta)

    expected = jnp.sum(normal_log_prob(theta, jnp.array(0.0), jnp.array(1.0)))
    expected += jnp.sum(normal_log_prob(y, theta[group_idx], jnp.array(1.0)))
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_bind_rejects_out_of_bounds_data_indexing() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        bind_model(
            IndexedNormal,
            n_groups=2,
            group_idx=jnp.array([0, 2]),
            y=jnp.array([0.0, 0.0]),
        )


def test_bind_rejects_negative_data_indexing() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        bind_model(
            IndexedNormal,
            n_groups=2,
            group_idx=jnp.array([0, -1]),
            y=jnp.array([0.0, 0.0]),
        )


def test_bind_rejects_non_integer_data_indexing() -> None:
    with pytest.raises(TypeError, match="Index data must be integer"):
        bind_model(
            IndexedNormal,
            n_groups=2,
            group_idx=jnp.array([0.0, 1.0]),
            y=jnp.array([0.0, 0.0]),
        )
