"""Calibrated validation for Binomial logistic likelihood models."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from _reference_models import binomial_logistic_fixture
from _validation import (
    ChainRunSpec,
    ScalarReference,
    assert_scalar_mean_matches_reference,
    scalar_grid_reference,
)
from jax.scipy.special import gammaln


def _binomial_logistic_reference(
    *,
    parameter: str,
    y: jax.Array,
    trials: jax.Array,
    prior_loc: float,
    prior_scale: float,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> ScalarReference:
    y_values = jnp.ravel(jnp.asarray(y))
    trial_values = jnp.ravel(jnp.asarray(trials))
    successes = jnp.sum(y_values)
    trial_count = jnp.sum(trial_values)
    log_choose_sum = jnp.sum(
        gammaln(trial_values + 1.0)
        - gammaln(y_values + 1.0)
        - gammaln(trial_values - y_values + 1.0)
    )
    log_two_pi = math.log(2.0 * math.pi)

    def log_unnormalized(eta: jax.Array) -> jax.Array:
        standardized_prior = (eta - prior_loc) / prior_scale
        log_prior = -jnp.log(prior_scale) - 0.5 * log_two_pi - 0.5 * standardized_prior**2
        log_likelihood = successes * eta - trial_count * jnp.logaddexp(0.0, eta)
        return log_prior + log_likelihood + log_choose_sum

    return scalar_grid_reference(
        parameter=parameter,
        log_unnormalized=log_unnormalized,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
    )


def test_binomial_logistic_matches_grid_reference_within_mcse() -> None:
    fixture = binomial_logistic_fixture()
    reference = _binomial_logistic_reference(
        parameter=fixture.parameter,
        y=fixture.y,
        trials=fixture.trials,
        prior_loc=fixture.prior_loc,
        prior_scale=fixture.prior_scale,
        grid_min=-4.0,
        grid_max=4.0,
        grid_size=20_000,
    )

    results = assert_scalar_mean_matches_reference(
        fixture.bound,
        reference=reference,
        run=ChainRunSpec(
            seed=8181,
            num_chains=4,
            num_warmup=400,
            num_samples=800,
            target_acceptance_rate=0.9,
        ),
        max_k=4.0,
        max_rhat=1.05,
        min_ess=150.0,
    )

    assert len(results) == 1
    assert results[0].parameter == fixture.parameter
    assert results[0].summary_name == "mean"
    assert results[0].k_min <= 4.0
