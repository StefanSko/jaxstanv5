"""Aspirational workflow coverage across target Bayesian model families.

This module is a smoke test for the public workflow only:
``@model -> bind -> sample -> diagnostics``.  It intentionally avoids calibrated
posterior correctness claims; family-specific validation lives in dedicated
``test_*_validation.py`` modules and uses ``_validation.py`` references, SBC, or
Stan.

Shared fixtures import target distributions lazily so missing capabilities fail
as individual tests instead of aborting collection.
"""

from __future__ import annotations

import jax.numpy as jnp
from _reference_models import (
    eight_schools_fixture,
    exponential_rate_fixture,
    fixed_kernel_gp_fixture,
    multivariate_normal_likelihood_fixture,
    robust_regression_fixture,
)

from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.inference import sample


def test_eight_schools_non_centered_workflow_smoke() -> None:
    fixture = eight_schools_fixture()

    result = sample(fixture.bound, seed=11, num_chains=4, num_warmup=1000, num_samples=1000)

    rhat_vals = rhat(result.samples)
    for name in ("mu", "tau", "z"):
        assert rhat_vals[name] < 1.1, f"R-hat too high for {name}"
    assert jnp.all(jnp.isfinite(result.samples["mu"]))
    assert jnp.all(result.samples["tau"] > 0.0)


def test_exponential_rate_workflow_smoke() -> None:
    fixture = exponential_rate_fixture()

    result = sample(fixture.bound, seed=5, num_chains=4, num_warmup=500, num_samples=1000)

    assert rhat(result.samples)["rate"] < 1.1
    assert ess(result.samples)["rate"] > 200
    assert jnp.all(jnp.isfinite(result.samples["rate"]))
    assert jnp.all(result.samples["rate"] > 0.0)


def test_robust_regression_workflow_smoke() -> None:
    fixture = robust_regression_fixture()

    result = sample(fixture.bound, seed=9, num_chains=4, num_warmup=800, num_samples=1000)

    rhat_vals = rhat(result.samples)
    for name in ("alpha", "beta", "sigma"):
        assert rhat_vals[name] < 1.1, f"R-hat too high for {name}"
    assert jnp.all(result.samples["sigma"] > 0.0)


def test_multivariate_normal_likelihood_workflow_smoke() -> None:
    fixture = multivariate_normal_likelihood_fixture()

    result = sample(fixture.bound, seed=13, num_chains=4, num_warmup=500, num_samples=1000)

    assert rhat(result.samples)["mu"] < 1.1
    assert jnp.all(jnp.isfinite(result.samples["mu"]))


def test_gaussian_process_workflow_smoke() -> None:
    fixture = fixed_kernel_gp_fixture()

    result = sample(fixture.bound, seed=19, num_chains=4, num_warmup=800, num_samples=1000)

    assert rhat(result.samples)["f"] < 1.15
    assert jnp.all(jnp.isfinite(result.samples["f"]))
