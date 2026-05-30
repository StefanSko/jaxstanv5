"""Calibrated validation for Student-t likelihood models."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from _reference_models import student_t_location_fixture
from _validation import (
    ChainRunSpec,
    ScalarReference,
    assert_scalar_mean_matches_reference,
    scalar_grid_reference,
)
from jax.scipy.special import gammaln


def _student_t_location_reference(
    *,
    parameter: str,
    y: jax.Array,
    nu: float,
    prior_loc: float,
    prior_scale: float,
    obs_scale: float,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> ScalarReference:
    y_values = jnp.ravel(jnp.asarray(y))
    log_two_pi = math.log(2.0 * math.pi)

    def log_unnormalized(mu: jax.Array) -> jax.Array:
        standardized_prior = (mu - prior_loc) / prior_scale
        log_prior = -jnp.log(prior_scale) - 0.5 * log_two_pi - 0.5 * standardized_prior**2
        standardized = (y_values[:, None] - mu[None, :]) / obs_scale
        log_likelihood_terms = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * jnp.log(nu * math.pi)
            - jnp.log(obs_scale)
            - 0.5 * (nu + 1.0) * jnp.log1p(standardized**2 / nu)
        )
        return log_prior + jnp.sum(log_likelihood_terms, axis=0)

    return scalar_grid_reference(
        parameter=parameter,
        log_unnormalized=log_unnormalized,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
    )


def test_student_t_location_matches_grid_reference_within_mcse() -> None:
    fixture = student_t_location_fixture()
    reference = _student_t_location_reference(
        parameter=fixture.parameter,
        y=fixture.y,
        nu=fixture.nu,
        prior_loc=fixture.prior_loc,
        prior_scale=fixture.prior_scale,
        obs_scale=fixture.obs_scale,
        grid_min=-4.0,
        grid_max=5.0,
        grid_size=20_000,
    )

    results = assert_scalar_mean_matches_reference(
        fixture.bound,
        reference=reference,
        run=ChainRunSpec(seed=7070, num_chains=4, num_warmup=400, num_samples=800),
        max_k=4.0,
        max_rhat=1.05,
        min_ess=150.0,
    )

    assert len(results) == 1
    assert results[0].parameter == fixture.parameter
    assert results[0].summary_name == "mean"
    assert results[0].k_min <= 4.0
