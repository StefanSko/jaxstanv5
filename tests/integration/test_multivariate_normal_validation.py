"""Calibrated validation for multivariate-normal likelihood models."""

from __future__ import annotations

import jax.numpy as jnp
from _reference_models import multivariate_normal_likelihood_fixture
from _validation import (
    ChainRunSpec,
    assert_vector_mean_matches_reference,
    multivariate_normal_known_covariance_reference,
)


def test_multivariate_normal_likelihood_matches_conjugate_reference() -> None:
    fixture = multivariate_normal_likelihood_fixture()
    dimension = fixture.y.shape[0]
    reference = multivariate_normal_known_covariance_reference(
        parameter=fixture.parameter,
        y=fixture.y,
        prior_mean=fixture.prior_mean,
        prior_covariance=fixture.prior_scale**2 * jnp.eye(dimension),
        obs_covariance=fixture.covariance,
    )

    results = assert_vector_mean_matches_reference(
        fixture.bound,
        reference=reference,
        run=ChainRunSpec(seed=8080, num_chains=4, num_warmup=600, num_samples=1000),
        max_k=4.0,
        max_rhat=1.05,
        min_ess=150.0,
    )

    assert len(results) == dimension
    assert all(result.k_min <= 4.0 for result in results)
