"""Integration test for measurement-error regression."""

import jax.numpy as jnp
import jax.random
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import sample


@model
class MarriageMeasurementError:
    """Measurement-error model with two observed likelihood sites."""

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


def test_measurement_error_workflow_samples_with_expected_diagnostics() -> None:
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

    assert set(result.samples) == {
        "alpha",
        "b_age",
        "b_marriage",
        "sigma",
        "marriage_true",
        "divorce_true",
    }
    for param_name in ("alpha", "b_age", "b_marriage", "sigma"):
        assert result.samples[param_name].shape == (1, 1000)
        assert jnp.all(jnp.isfinite(result.samples[param_name]))
        assert rhat_vals[param_name] < 1.1, f"R-hat too high for {param_name}"
        assert ess_vals[param_name] > 40, f"ESS too low for {param_name}"
    for param_name in ("marriage_true", "divorce_true"):
        assert result.samples[param_name].shape == (1, 1000, n_states_val)
        assert jnp.all(jnp.isfinite(result.samples[param_name]))

    assert jnp.all(result.samples["sigma"] > 0.0)
    assert result.diagnostics.warmup.is_divergent.shape == (1, 500)
    assert result.diagnostics.sampling.is_divergent.shape == (1, 1000)
    assert result.diagnostics.sampling.acceptance_rate.shape == (1, 1000)
    assert not jnp.any(result.diagnostics.sampling.is_divergent)
