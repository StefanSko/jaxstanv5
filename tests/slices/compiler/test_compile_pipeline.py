"""Slice test — compile a bound model to a log-density function.

Verifies the full compilation path:
  BoundModel → compile_log_density → callable log_prob(unconstrained_params) → scalar
"""

from __future__ import annotations

import math
from typing import Protocol, cast

import jax.numpy as jnp

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import Normal
from jaxstanv5.model.bound import BoundModel


class BindableModel(Protocol):
    """Model class after the runtime ``@model`` decorator attaches ``bind``."""

    def bind(self, **values: object) -> BoundModel:
        """Bind concrete model data."""
        ...


def bind_model(model_cls: object, **values: object) -> BoundModel:
    """Call runtime-attached ``bind`` through one explicit typed boundary."""
    return cast(BindableModel, model_cls).bind(**values)


@model
class SimpleNormal:
    """Scalar normal with known scale — no constraints."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normal_log_prob(x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Element-wise normal log-density (matches Normal.log_prob)."""
    standardized = (x - loc) / scale
    return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compile_simple_unconstrained() -> None:
    bound = bind_model(SimpleNormal, y=jnp.array(2.0))
    log_prob = compile_log_density(bound)

    # mu = 0.5 (unconstrained, no transform needed)
    q = jnp.array([0.5])
    lp = log_prob(q)

    expected_prior = normal_log_prob(jnp.array(0.5), jnp.array(0.0), jnp.array(1.0))
    expected_obs = normal_log_prob(jnp.array(2.0), jnp.array(0.5), jnp.array(1.0))
    expected = expected_prior + expected_obs

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compile_constrained() -> None:
    bound = bind_model(ConstrainedNormal, y=jnp.array(3.0))
    log_prob = compile_log_density(bound)

    # sigma in unconstrained space: log(2.0)
    q = jnp.array([math.log(2.0)])
    lp = log_prob(q)

    sigma_constrained = jnp.exp(q[0])  # = 2.0
    expected_prior = normal_log_prob(sigma_constrained, jnp.array(0.0), jnp.array(1.0))
    expected_obs = normal_log_prob(jnp.array(3.0), jnp.array(0.0), sigma_constrained)
    # Jacobian: log_abs_det_jacobian(y) = y for exp transform
    expected_log_jac = q[0]  # log(2.0)
    expected = expected_prior + expected_obs + expected_log_jac

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compile_with_data_expression() -> None:
    x_data = jnp.array([1.0, 2.0, 3.0])
    y_data = jnp.array([2.1, 2.9, 4.1])
    bound = bind_model(LinearPredictor, x=x_data, y=y_data)
    log_prob = compile_log_density(bound)

    # alpha = 2.0, beta = 0.5
    q = jnp.array([2.0, 0.5])
    lp = log_prob(q)

    prior_lp = normal_log_prob(jnp.array(2.0), jnp.array(0.0), jnp.array(1.0))
    prior_lp += normal_log_prob(jnp.array(0.5), jnp.array(0.0), jnp.array(1.0))

    mu_computed = 2.0 + 0.5 * x_data
    obs_lp = jnp.sum(normal_log_prob(y_data, mu_computed, jnp.array(1.0)))

    expected = prior_lp + obs_lp
    assert jnp.allclose(lp, expected, atol=1e-6)
