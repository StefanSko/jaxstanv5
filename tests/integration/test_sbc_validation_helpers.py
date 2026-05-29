"""Deterministic tests for private SBC validation helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from _validation import (
    VALIDATION_PLAN,
    SbcValidationResult,
    ValidationStage,
    ValidationStageStatus,
    assert_sbc_rank_uniformity,
    scalar_sbc_rank,
    summarize_sbc_rank_uniformity,
)


def test_validation_plan_marks_sbc_reference_completed() -> None:
    statuses = {item.stage: item.status for item in VALIDATION_PLAN}

    assert statuses[ValidationStage.SBC_REFERENCE] is ValidationStageStatus.COMPLETED


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


def test_assert_sbc_rank_uniformity_rejects_lopsided_ranks() -> None:
    result = SbcValidationResult(
        parameter="mu",
        ranks=(0, 0, 0, 0, 0, 0, 0, 0),
        num_posterior_draws=7,
    )

    with pytest.raises(AssertionError, match="SBC rank bin"):
        assert_sbc_rank_uniformity(
            result,
            num_rank_bins=4,
            max_abs_bin_z=1.0,
            max_abs_mean_rank_z=10.0,
        )
