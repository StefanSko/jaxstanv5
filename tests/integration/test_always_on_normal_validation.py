"""Always-on probabilistic validation against analytic Normal references."""

from __future__ import annotations

import jax.numpy as jnp
from _helpers import bind_model
from _validation import (
    ChainRunSpec,
    assert_normal_known_scale_matches_reference,
    normal_known_scale_reference,
    summarize_scalar_draws,
)
from bayeswire import Observed, Param, model
from bayeswire.distributions import Normal

from jaxstanv5.inference import sample
from jaxstanv5.validation import standardized_discrepancy


@model
class NormalKnownScaleValidationModel:
    """Scalar Normal model with known observation scale."""

    mu = Param(Normal(0.0, 1.0))
    y = Observed(Normal(mu, 1.0))


def test_summarize_scalar_draws_returns_mcse_rhat_and_ess() -> None:
    samples = {"mu": jnp.array([[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]])}

    summary = summarize_scalar_draws(samples, parameter="mu")

    assert summary.parameter == "mu"
    assert summary.mean == 1.75
    assert summary.sd > 0.0
    assert summary.ess > 0.0
    assert summary.rhat > 0.0
    assert summary.mcse_mean > 0.0
    assert summary.mcse_sd is None


def test_standardized_discrepancy_returns_signed_z_and_k_min() -> None:
    result = standardized_discrepancy(
        parameter="mu",
        summary_name="mean",
        estimate=2.0,
        reference=1.5,
        mcse=0.25,
    )

    assert result.parameter == "mu"
    assert result.summary_name == "mean"
    assert result.estimate == 2.0
    assert result.reference == 1.5
    assert result.mcse == 0.25
    assert result.signed_z == 2.0
    assert result.k_min == 2.0


def test_simple_normal_mean_matches_analytic_reference_within_mcse() -> None:
    y_data = jnp.array([1.0, 1.5, 2.0, 2.5])
    bound = bind_model(NormalKnownScaleValidationModel, y=y_data)
    reference = normal_known_scale_reference(
        parameter="mu",
        y=y_data,
        prior_loc=0.0,
        prior_scale=1.0,
        obs_scale=1.0,
    )

    results = assert_normal_known_scale_matches_reference(
        bound,
        reference=reference,
        run=ChainRunSpec(seed=2024, num_chains=4, num_warmup=300, num_samples=600),
        max_k=4.0,
    )

    assert len(results) == 1
    assert results[0].parameter == "mu"
    assert results[0].summary_name == "mean"
    assert results[0].k_min <= 4.0


def test_public_sampler_matches_reference_test_configuration_shape() -> None:
    y_data = jnp.array([1.0, 1.5, 2.0, 2.5])
    bound = bind_model(NormalKnownScaleValidationModel, y=y_data)

    result = sample(bound, seed=2024, num_chains=4, num_warmup=300, num_samples=600)

    assert result.samples["mu"].shape == (4, 600)
