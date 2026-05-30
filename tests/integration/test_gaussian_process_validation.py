"""Calibrated validation for fixed-kernel Gaussian-process regression."""

from __future__ import annotations

import jax.numpy as jnp
from _reference_models import fixed_kernel_gp_fixture
from _validation import (
    ChainRunSpec,
    assert_vector_mean_matches_reference,
    multivariate_normal_known_covariance_reference,
)


def test_fixed_kernel_gaussian_process_matches_conjugate_reference() -> None:
    fixture = fixed_kernel_gp_fixture(n=8, lengthscale=0.8, obs_sd=0.3)
    dimension = fixture.y.shape[0]
    reference = multivariate_normal_known_covariance_reference(
        parameter=fixture.parameter,
        y=fixture.y,
        prior_mean=jnp.zeros((dimension,)),
        prior_covariance=fixture.covariance,
        obs_covariance=fixture.obs_sd**2 * jnp.eye(dimension),
    )

    results = assert_vector_mean_matches_reference(
        fixture.bound,
        reference=reference,
        run=ChainRunSpec(seed=9090, num_chains=4, num_warmup=800, num_samples=1000),
        max_k=5.0,
        max_rhat=1.08,
        min_ess=100.0,
    )

    assert len(results) == dimension
    assert all(result.k_min <= 5.0 for result in results)
