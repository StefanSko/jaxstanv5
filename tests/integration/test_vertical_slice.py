"""Vertical-slice integration test — model → bind → sample → diagnostics.

This test documents the intended public API. It is expected to fail (red)
until the implementation is wired up.
"""

import jax.numpy as jnp
import jax.random

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import sample

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@model
class LinearRegression:
    """Flat linear regression with conjugate-like priors."""

    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    sigma = Param(Normal(0, 1), constraint=Positive())
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


@model
class HierarchicalLinearRegression:
    """Two-level hierarchical linear regression with varying intercept and slope."""

    # --- hyperparameters ---------------------------------------------------
    alpha_pop = Param(Normal(0, 1))
    beta_pop = Param(Normal(0, 1))
    sigma_alpha = Param(Normal(0, 1), constraint=Positive())
    sigma_beta = Param(Normal(0, 1), constraint=Positive())
    sigma = Param(Normal(0, 1), constraint=Positive())

    # --- data --------------------------------------------------------------
    n_groups = Data()
    x = Data()
    group_idx = Data()

    # --- group-level parameters (one per group) -----------------------------
    alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
    beta = Param(Normal(beta_pop, sigma_beta), size=n_groups)

    mu = alpha[group_idx] + beta[group_idx] * x
    y = Observed(Normal(mu, sigma))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_linear_regression_vertical_slice() -> None:
    key = jax.random.PRNGKey(42)

    # True parameters
    a_true = 2.0
    b_true = 0.5
    sigma_true = 1.0

    n = 50
    x_data = jnp.linspace(-3, 3, n)
    y_data = a_true + b_true * x_data + sigma_true * jax.random.normal(key, (n,))

    # Bind data
    key, subkey = jax.random.split(key)
    bound = LinearRegression.bind(x=x_data, y=y_data)

    # Sample
    result = sample(bound, seed=42, num_warmup=200, num_samples=500)

    # Diagnostics
    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)

    # Convergence and efficiency checks
    for param_name in ("alpha", "beta", "sigma"):
        assert rhat_vals[param_name] < 1.05, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 50, f"ESS too low for {param_name}"


def test_hierarchical_regression_vertical_slice() -> None:
    key = jax.random.PRNGKey(99)

    n_groups_val = 5
    obs_per_group = 10
    n = n_groups_val * obs_per_group

    # True hyperparameters
    alpha_pop_true = 0.0
    beta_pop_true = 2.0
    sigma_alpha_true = 0.5
    sigma_beta_true = 0.3
    sigma_true = 0.2

    key, *group_keys = jax.random.split(key, num=n_groups_val + 1)
    alpha_true = alpha_pop_true + sigma_alpha_true * jnp.array(
        [jax.random.normal(k, ()) for k in group_keys]
    )
    beta_true = beta_pop_true + sigma_beta_true * jnp.array(
        [jax.random.normal(k, ()) for k in group_keys]
    )

    group_idx_val = jnp.repeat(jnp.arange(n_groups_val), obs_per_group)
    x_data = jnp.linspace(-5, 5, n)
    key, subkey = jax.random.split(key)
    mu_true = alpha_true[group_idx_val] + beta_true[group_idx_val] * x_data
    y_data = mu_true + sigma_true * jax.random.normal(subkey, (n,))

    # Bind
    bound = HierarchicalLinearRegression.bind(
        n_groups=n_groups_val, x=x_data, group_idx=group_idx_val, y=y_data
    )

    # Sample
    result = sample(bound, seed=123, num_warmup=200, num_samples=500)

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)

    for param_name in ("alpha_pop", "beta_pop", "alpha", "beta", "sigma"):
        assert rhat_vals[param_name] < 1.1, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 40, f"ESS too low for {param_name}"
