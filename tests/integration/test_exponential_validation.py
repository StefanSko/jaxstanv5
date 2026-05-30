"""Calibrated validation for Exponential likelihood models."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from _reference_models import exponential_rate_fixture
from _validation import (
    ChainRunSpec,
    ScalarReference,
    assert_scalar_mean_matches_reference,
    scalar_grid_reference,
)


def _exponential_rate_reference(
    *,
    parameter: str,
    y: jax.Array,
    prior_scale: float,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> ScalarReference:
    log_two_over_pi = math.log(2.0 / math.pi)
    y_values = jnp.ravel(jnp.asarray(y))
    y_count = y_values.size
    y_sum = jnp.sum(y_values)

    def log_unnormalized(rate: jax.Array) -> jax.Array:
        log_prior = 0.5 * log_two_over_pi - jnp.log(prior_scale) - 0.5 * (rate / prior_scale) ** 2
        log_likelihood = y_count * jnp.log(rate) - rate * y_sum
        return log_prior + log_likelihood

    return scalar_grid_reference(
        parameter=parameter,
        log_unnormalized=log_unnormalized,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
    )


def test_exponential_rate_matches_grid_reference_within_mcse() -> None:
    fixture = exponential_rate_fixture()
    reference = _exponential_rate_reference(
        parameter=fixture.parameter,
        y=fixture.y,
        prior_scale=fixture.prior_scale,
        grid_min=0.01,
        grid_max=8.0,
        grid_size=20_000,
    )

    results = assert_scalar_mean_matches_reference(
        fixture.bound,
        reference=reference,
        run=ChainRunSpec(seed=5050, num_chains=4, num_warmup=400, num_samples=800),
        max_k=4.0,
        max_rhat=1.05,
        min_ess=150.0,
    )

    assert len(results) == 1
    assert results[0].parameter == fixture.parameter
    assert results[0].summary_name == "mean"
    assert results[0].k_min <= 4.0
