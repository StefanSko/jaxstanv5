"""Integration test for hierarchical linear regression."""

import jax.numpy as jnp
import jax.random
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import sample


@model
class HierarchicalLinearRegression:
    """Two-level hierarchical linear regression with varying intercept and slope."""

    alpha_pop = Param(Normal(0, 1))
    beta_pop = Param(Normal(0, 1))
    sigma_alpha = Param(Normal(0, 1), constraint=Positive())
    sigma_beta = Param(Normal(0, 1), constraint=Positive())
    sigma = Param(Normal(0, 1), constraint=Positive())

    n_groups = Data()
    x = Data()
    group_idx = Data()

    alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
    beta = Param(Normal(beta_pop, sigma_beta), size=n_groups)

    mu = alpha[group_idx] + beta[group_idx] * x
    y = Observed(Normal(mu, sigma))


def test_hierarchical_regression_recovers_population_parameters() -> None:
    key = jax.random.PRNGKey(99)

    n_groups_val = 5
    obs_per_group = 50
    n = n_groups_val * obs_per_group

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

    bound = bind_model(
        HierarchicalLinearRegression,
        n_groups=n_groups_val,
        x=x_data,
        group_idx=group_idx_val,
        y=y_data,
    )
    result = sample(bound, seed=123, num_warmup=500, num_samples=2000)

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)

    alpha_pop_est = float(jnp.mean(result.samples["alpha_pop"]))
    beta_pop_est = float(jnp.mean(result.samples["beta_pop"]))
    sigma_est = float(jnp.mean(result.samples["sigma"]))
    print(
        "hierarchical estimates: "
        f"alpha_pop={alpha_pop_est:.3f} true={alpha_pop_true:.3f}, "
        f"beta_pop={beta_pop_est:.3f} true={beta_pop_true:.3f}, "
        f"sigma={sigma_est:.3f} true={sigma_true:.3f}"
    )

    for param_name in ("alpha_pop", "beta_pop", "alpha", "beta", "sigma"):
        assert rhat_vals[param_name] < 1.1, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 40, f"ESS too low for {param_name}"

    assert jnp.isclose(jnp.mean(result.samples["alpha_pop"]), alpha_pop_true, atol=0.75)
    assert jnp.isclose(jnp.mean(result.samples["beta_pop"]), beta_pop_true, atol=0.35)
    assert jnp.isclose(jnp.mean(result.samples["sigma"]), sigma_true, atol=0.25)
