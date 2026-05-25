"""Vertical-slice integration test — model → bind → sample → diagnostics.

This test documents the intended public API. It is expected to fail (red)
until the implementation is wired up.
"""

from typing import Protocol, cast

import jax.numpy as jnp
import jax.random

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import sample
from jaxstanv5.model.bound import BoundModel


class BindableModel(Protocol):
    """Model class after the runtime ``@model`` decorator attaches ``bind``."""

    def bind(self, **values: object) -> BoundModel:
        """Bind concrete model data."""
        ...


def bind_model(model_cls: object, **values: object) -> BoundModel:
    """Call runtime-attached ``bind`` through one explicit typed boundary."""
    return cast(BindableModel, model_cls).bind(**values)


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
    bound = bind_model(LinearRegression, x=x_data, y=y_data)

    # Sample
    result = sample(bound, seed=42, num_warmup=200, num_samples=500)

    # Diagnostics
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

    # Convergence and efficiency checks
    for param_name in ("alpha", "beta", "sigma"):
        assert rhat_vals[param_name] < 1.05, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 50, f"ESS too low for {param_name}"

    assert jnp.isclose(jnp.mean(result.samples["alpha"]), a_true, atol=0.35)
    assert jnp.isclose(jnp.mean(result.samples["beta"]), b_true, atol=0.20)
    assert jnp.isclose(jnp.mean(result.samples["sigma"]), sigma_true, atol=0.35)


def test_marriage_measurement_error_vertical_slice() -> None:
    key = jax.random.PRNGKey(2024)

    n_states_val = 60
    age_data = jnp.linspace(-1.5, 1.5, n_states_val)
    marriage_sd_data = jnp.full((n_states_val,), 0.05)
    divorce_sd_data = jnp.full((n_states_val,), 0.08)

    alpha_true = 0.15
    b_age_true = -0.65
    b_marriage_true = -0.45
    sigma_true = 0.25

    marriage_key, divorce_key, marriage_obs_key, divorce_obs_key = jax.random.split(key, 4)
    marriage_true = jax.random.normal(marriage_key, (n_states_val,))
    divorce_mu = alpha_true + b_age_true * age_data + b_marriage_true * marriage_true
    divorce_true = divorce_mu + sigma_true * jax.random.normal(divorce_key, (n_states_val,))

    marriage_obs_data = marriage_true + marriage_sd_data * jax.random.normal(
        marriage_obs_key,
        (n_states_val,),
    )
    divorce_obs_data = divorce_true + divorce_sd_data * jax.random.normal(
        divorce_obs_key,
        (n_states_val,),
    )

    @model
    class MarriageMeasurementError:
        """Measurement-error model with two observed likelihood sites."""

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

    bound = bind_model(
        MarriageMeasurementError,
        n_states=n_states_val,
        age=age_data,
        marriage_sd=marriage_sd_data,
        divorce_sd=divorce_sd_data,
        marriage_obs=marriage_obs_data,
        divorce_obs=divorce_obs_data,
    )

    result = sample(bound, seed=2025, num_warmup=500, num_samples=1000)

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)

    alpha_est = float(jnp.mean(result.samples["alpha"]))
    b_age_est = float(jnp.mean(result.samples["b_age"]))
    b_marriage_est = float(jnp.mean(result.samples["b_marriage"]))
    sigma_est = float(jnp.mean(result.samples["sigma"]))
    print(
        "measurement-error estimates: "
        f"alpha={alpha_est:.3f} true={alpha_true:.3f}, "
        f"b_age={b_age_est:.3f} true={b_age_true:.3f}, "
        f"b_marriage={b_marriage_est:.3f} true={b_marriage_true:.3f}, "
        f"sigma={sigma_est:.3f} true={sigma_true:.3f}"
    )

    for param_name in ("alpha", "b_age", "b_marriage", "sigma"):
        assert rhat_vals[param_name] < 1.1, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 40, f"ESS too low for {param_name}"

    assert jnp.isclose(jnp.mean(result.samples["alpha"]), alpha_true, atol=0.25)
    assert jnp.isclose(jnp.mean(result.samples["b_age"]), b_age_true, atol=0.25)
    assert jnp.isclose(jnp.mean(result.samples["b_marriage"]), b_marriage_true, atol=0.30)
    assert jnp.isclose(jnp.mean(result.samples["sigma"]), sigma_true, atol=0.20)


def test_hierarchical_regression_vertical_slice() -> None:
    key = jax.random.PRNGKey(99)

    n_groups_val = 5
    obs_per_group = 50
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
    bound = bind_model(
        HierarchicalLinearRegression,
        n_groups=n_groups_val,
        x=x_data,
        group_idx=group_idx_val,
        y=y_data,
    )

    # Sample
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
