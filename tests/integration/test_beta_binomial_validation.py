"""Calibrated validation for Beta-binomial logistic likelihood models."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from _reference_models import beta_binomial_logistic_fixture
from _validation import (
    ChainRunSpec,
    ScalarReference,
    assert_scalar_mean_matches_reference,
    scalar_grid_reference,
)
from jax.scipy.special import gammaln


def _log_beta(a: jax.Array, b: jax.Array) -> jax.Array:
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def _beta_binomial_logistic_reference(
    *,
    parameter: str,
    y: jax.Array,
    trials: jax.Array,
    concentration: float,
    prior_loc: float,
    prior_scale: float,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> ScalarReference:
    y_values = jnp.ravel(jnp.asarray(y))
    trial_values = jnp.ravel(jnp.asarray(trials))
    log_choose_sum = jnp.sum(
        gammaln(trial_values + 1.0)
        - gammaln(y_values + 1.0)
        - gammaln(trial_values - y_values + 1.0)
    )
    concentration_value = jnp.asarray(concentration)
    log_two_pi = math.log(2.0 * math.pi)

    def log_unnormalized(eta: jax.Array) -> jax.Array:
        standardized_prior = (eta - prior_loc) / prior_scale
        log_prior = -jnp.log(prior_scale) - 0.5 * log_two_pi - 0.5 * standardized_prior**2
        probability = jax.nn.sigmoid(eta)[:, None]
        alpha = probability * concentration_value
        beta = (1.0 - probability) * concentration_value
        log_likelihood = log_choose_sum
        log_likelihood += jnp.sum(
            _log_beta(y_values[None, :] + alpha, trial_values[None, :] - y_values[None, :] + beta),
            axis=1,
        )
        log_likelihood -= y_values.size * _log_beta(alpha[:, 0], beta[:, 0])
        return log_prior + log_likelihood

    return scalar_grid_reference(
        parameter=parameter,
        log_unnormalized=log_unnormalized,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
    )


def test_beta_binomial_logistic_matches_grid_reference_within_mcse() -> None:
    fixture = beta_binomial_logistic_fixture()
    reference = _beta_binomial_logistic_reference(
        parameter=fixture.parameter,
        y=fixture.y,
        trials=fixture.trials,
        concentration=fixture.concentration,
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
            seed=8282,
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
