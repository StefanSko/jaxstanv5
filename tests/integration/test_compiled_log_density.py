"""Integration tests for compiling declared models to log-density functions."""

from __future__ import annotations

import math

import jax.numpy as jnp
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import Normal


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
