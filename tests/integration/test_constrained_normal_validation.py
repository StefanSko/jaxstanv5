"""Posterior validation for a model with a constrained positive scale."""

from __future__ import annotations

import jax.numpy as jnp
from _helpers import bind_model
from _validation import (
    ChainRunSpec,
    assert_scalar_mean_matches_reference,
    positive_scale_grid_reference,
)

from jaxstanv5 import Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import Normal, Truncated


@model
class PositiveScaleNormalValidationModel:
    """Zero-location Normal model with unknown positive scale."""

    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())
    y = Observed(Normal(0.0, sigma))


def test_positive_scale_normal_mean_matches_grid_reference_within_mcse() -> None:
    y_data = jnp.array([-0.5, 0.25, 1.0, -1.25])
    bound = bind_model(PositiveScaleNormalValidationModel, y=y_data)
    reference = positive_scale_grid_reference(
        parameter="sigma",
        y=y_data,
        prior_loc=0.0,
        prior_scale=1.0,
        grid_min=0.01,
        grid_max=5.0,
        grid_size=20_000,
    )

    results = assert_scalar_mean_matches_reference(
        bound,
        reference=reference,
        run=ChainRunSpec(seed=3030, num_chains=4, num_warmup=300, num_samples=600),
        max_k=4.0,
        max_rhat=1.05,
        min_ess=100.0,
    )

    assert len(results) == 1
    assert results[0].parameter == "sigma"
    assert results[0].summary_name == "mean"
    assert results[0].k_min <= 4.0
