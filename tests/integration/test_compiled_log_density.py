"""Integration tests for compiling declared models to log-density functions."""

from __future__ import annotations

import math

import jax.numpy as jnp
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.constraints import Positive
from jax.scipy.special import gammaln

from jaxstanv5.distributions import Binomial, Normal, Poisson
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
class LinearPredictor:
    """One-predictor linear model with known scale."""

    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, 1))


@model
class PoissonLogDensity:
    """Poisson count model using symbolic exponential rate."""

    eta = Param(Normal(0, 1))
    rate = exp(eta)
    y = Observed(Poisson(rate))


@model
class BinomialLogisticDensity:
    """Binomial count model using symbolic logistic success probabilities."""

    eta = Param(Normal(0, 1))
    trials = Data()
    y = Observed(Binomial(trials, sigmoid(eta)))


@model
class MeasurementErrorLogDensity:
    """Measurement-error model with latent vectors and two observed sites."""

    n_states = Data()
    age = Data()
    marriage_sd = Data()
    divorce_sd = Data()

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
