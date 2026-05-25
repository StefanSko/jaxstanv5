"""Integration test for simple linear regression."""

import jax.numpy as jnp
import jax.random
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import sample


@model
class LinearRegression:
    """Flat linear regression with scalar intercept, slope, and scale."""

    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    sigma = Param(Normal(0, 1), constraint=Positive())
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


def test_linear_regression_recovers_parameters() -> None:
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

    alpha_est = float(jnp.mean(result.samples["alpha"]))
    beta_est = float(jnp.mean(result.samples["beta"]))
    sigma_est = float(jnp.mean(result.samples["sigma"]))
    print(
        "linear estimates: "
        f"alpha={alpha_est:.3f} true={a_true:.3f}, "
        f"beta={beta_est:.3f} true={b_true:.3f}, "
        f"sigma={sigma_est:.3f} true={sigma_true:.3f}"
    )

    for param_name in ("alpha", "beta", "sigma"):
        assert rhat_vals[param_name] < 1.05, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 50, f"ESS too low for {param_name}"

    assert jnp.isclose(jnp.mean(result.samples["alpha"]), a_true, atol=0.35)
    assert jnp.isclose(jnp.mean(result.samples["beta"]), b_true, atol=0.20)
    assert jnp.isclose(jnp.mean(result.samples["sigma"]), sigma_true, atol=0.35)
