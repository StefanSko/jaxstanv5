"""Deterministic tests for public validation primitives."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5.validation import (
    ProjectionSpec,
    SbcValidationResult,
    project_vector_draws,
    project_vector_truth,
    projected_sbc_rank,
    scalar_sbc_rank,
    standardized_discrepancy,
    summarize_sbc_rank_uniformity,
)


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


def test_standardized_discrepancy_rejects_non_positive_mcse() -> None:
    with pytest.raises(ValueError, match="mcse must be positive"):
        standardized_discrepancy(
            parameter="mu",
            summary_name="mean",
            estimate=2.0,
            reference=1.5,
            mcse=0.0,
        )


def test_project_vector_draws_returns_scalar_draw_array() -> None:
    samples = {"f": jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])}
    projection = ProjectionSpec(name="f_sum", parameter="f", weights=jnp.asarray([1.0, 1.0]))

    projected = project_vector_draws(samples, projection=projection)

    assert projected.shape == (2, 2)
    assert jnp.allclose(projected, jnp.asarray([[3.0, 7.0], [11.0, 15.0]]))


def test_project_vector_truth_returns_scalar_value() -> None:
    projection = ProjectionSpec(name="f_contrast", parameter="f", weights=jnp.asarray([-1.0, 1.0]))

    projected = project_vector_truth(jnp.asarray([2.0, 5.0]), projection=projection)

    assert projected == 3.0


def test_projected_sbc_rank_counts_projected_draws_below_projected_truth() -> None:
    samples = {"f": jnp.asarray([[[0.0, 1.0], [1.0, 2.0]], [[2.0, 3.0], [3.0, 4.0]]])}
    projection = ProjectionSpec(name="f_first", parameter="f", weights=jnp.asarray([1.0, 0.0]))

    rank = projected_sbc_rank(samples, projection=projection, true_value=jnp.asarray([1.5, 0.0]))

    assert rank == 2


def test_project_vector_draws_rejects_missing_parameter() -> None:
    projection = ProjectionSpec(name="f_first", parameter="f", weights=jnp.asarray([1.0]))

    with pytest.raises(ValueError, match="Missing samples"):
        project_vector_draws({"theta": jnp.zeros((1, 1, 1))}, projection=projection)


def test_project_vector_draws_rejects_dimension_mismatch() -> None:
    projection = ProjectionSpec(name="f_first", parameter="f", weights=jnp.asarray([1.0, 0.0]))

    with pytest.raises(ValueError, match="Projection weights"):
        project_vector_draws({"f": jnp.zeros((1, 1, 3))}, projection=projection)


def test_scalar_sbc_rank_counts_draws_below_true_value() -> None:
    samples = {"mu": jnp.asarray([[0.0, 1.0, 2.0], [1.5, 2.5, 3.0]])}

    rank = scalar_sbc_rank(samples, parameter="mu", true_value=2.0)

    assert rank == 3


def test_scalar_sbc_rank_allows_edge_ranks() -> None:
    samples = {"mu": jnp.asarray([[1.0, 2.0], [3.0, 4.0]])}

    assert scalar_sbc_rank(samples, parameter="mu", true_value=0.5) == 0
    assert scalar_sbc_rank(samples, parameter="mu", true_value=5.0) == 4


def test_scalar_sbc_rank_rejects_missing_parameter() -> None:
    with pytest.raises(ValueError, match="Missing samples"):
        scalar_sbc_rank({"theta": jnp.asarray([[1.0]])}, parameter="mu", true_value=1.0)


def test_scalar_sbc_rank_rejects_non_scalar_draw_array() -> None:
    samples = {"theta": jnp.zeros((2, 3, 4))}

    with pytest.raises(ValueError, match="Scalar SBC draw arrays"):
        scalar_sbc_rank(samples, parameter="theta", true_value=1.0)


def test_summarize_sbc_rank_uniformity_bins_known_ranks() -> None:
    result = SbcValidationResult(
        parameter="mu",
        ranks=(0, 1, 2, 3, 4, 5, 6, 7),
        num_posterior_draws=7,
    )

    summary = summarize_sbc_rank_uniformity(result, num_rank_bins=4)

    assert summary.parameter == "mu"
    assert summary.bin_counts == (2, 2, 2, 2)
    assert summary.expected_bin_count == 2.0
    assert summary.max_abs_bin_z == 0.0
    assert summary.mean_rank_z == 0.0


def test_summarize_sbc_rank_uniformity_rejects_out_of_range_rank() -> None:
    result = SbcValidationResult(
        parameter="mu",
        ranks=(0, 8),
        num_posterior_draws=7,
    )

    with pytest.raises(ValueError, match="SBC ranks"):
        summarize_sbc_rank_uniformity(result, num_rank_bins=4)
