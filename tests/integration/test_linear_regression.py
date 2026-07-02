"""Integration test for simple linear regression."""

import jax.numpy as jnp
import jax.random
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal, Truncated
from jaxstanv5.inference import sample


@model
class LinearRegression:
    """Flat linear regression with scalar intercept, slope, and scale."""

    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    sigma = Param(Truncated(Normal(0, 1), lower=0.0), constraint=Positive())
    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


def test_linear_regression_workflow_samples_with_expected_diagnostics() -> None:
    key = jax.random.PRNGKey(42)

    a_true = 2.0
    b_true = 0.5
    sigma_true = 1.0

    n = 50
    x_data = jnp.linspace(-3, 3, n)
    y_data = a_true + b_true * x_data + sigma_true * jax.random.normal(key, (n,))

    bound = bind_model(LinearRegression, x=x_data, y=y_data)
    result = sample(bound, seed=42, num_warmup=200, num_samples=500)

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)

    assert set(result.samples) == {"alpha", "beta", "sigma"}
    for param_name in ("alpha", "beta", "sigma"):
        assert result.samples[param_name].shape == (1, 500)
        assert jnp.all(jnp.isfinite(result.samples[param_name]))
        assert rhat_vals[param_name] < 1.05, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 50, f"ESS too low for {param_name}"

    assert jnp.all(result.samples["sigma"] > 0.0)
    assert result.diagnostics.warmup.is_divergent.shape == (1, 200)
    assert result.diagnostics.sampling.is_divergent.shape == (1, 500)
    assert result.diagnostics.sampling.acceptance_rate.shape == (1, 500)
    assert not jnp.any(result.diagnostics.sampling.is_divergent)
