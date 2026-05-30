"""Calibrated validation for known-scale non-centered hierarchical Normals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from _reference_models import non_centered_known_scale_fixture
from _validation import (
    ChainRunSpec,
    ScalarReference,
    draw_validation_chains,
    hierarchical_normal_known_scale_reference,
    standardized_discrepancy,
    summarize_scalar_draws,
)


def _assert_draw_mean_matches_reference(
    samples: dict[str, jax.Array],
    *,
    reference: ScalarReference,
    max_k: float,
    max_rhat: float,
    min_ess: float,
) -> None:
    summary = summarize_scalar_draws(samples, parameter=reference.parameter)
    assert summary.rhat <= max_rhat
    assert summary.ess >= min_ess
    result = standardized_discrepancy(
        parameter=reference.parameter,
        summary_name="mean",
        estimate=summary.mean,
        reference=reference.mean,
        mcse=summary.mcse_mean,
    )
    assert result.k_min <= max_k


def test_non_centered_known_scale_hierarchy_matches_analytic_reference() -> None:
    fixture = non_centered_known_scale_fixture()
    reference = hierarchical_normal_known_scale_reference(
        population_parameter="mu_pop",
        group_parameter="theta",
        y=fixture.y_matrix,
        prior_loc=fixture.prior_loc,
        prior_scale=fixture.prior_scale,
        group_scale=fixture.group_scale,
        obs_scale=fixture.obs_scale,
    )

    samples = draw_validation_chains(
        fixture.bound,
        run=ChainRunSpec(seed=6060, num_chains=4, num_warmup=400, num_samples=800),
    )
    _assert_draw_mean_matches_reference(
        dict(samples),
        reference=reference.population,
        max_k=4.0,
        max_rhat=1.05,
        min_ess=150.0,
    )

    theta = samples["mu_pop"][..., None] + fixture.group_scale * samples["z"]
    for group_index, group_reference in enumerate(reference.groups):
        group_samples = {group_reference.parameter: theta[:, :, group_index]}
        _assert_draw_mean_matches_reference(
            group_samples,
            reference=group_reference,
            max_k=4.0,
            max_rhat=1.05,
            min_ess=150.0,
        )

    assert jnp.all(jnp.isfinite(theta))
