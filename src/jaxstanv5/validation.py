"""Public validation primitives for posterior checks and SBC summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ScalarValidationResult:
    """Standardized comparison between a scalar estimate and a reference value."""

    parameter: str
    summary_name: str
    estimate: float
    reference: float
    mcse: float
    signed_z: float
    k_min: float


@dataclass(frozen=True)
class ProjectionSpec:
    """Fixed linear projection of a vector-valued parameter."""

    name: str
    parameter: str
    weights: jax.Array


@dataclass(frozen=True)
class SbcValidationResult:
    """Simulation-based calibration ranks for one scalar parameter."""

    parameter: str
    ranks: tuple[int, ...]
    num_posterior_draws: int


@dataclass(frozen=True)
class SbcRankUniformitySummary:
    """Binned rank-uniformity diagnostics for SBC ranks."""

    parameter: str
    ranks: tuple[int, ...]
    num_posterior_draws: int
    bin_counts: tuple[int, ...]
    expected_bin_count: float
    max_abs_bin_z: float
    mean_rank_z: float


def standardized_discrepancy(
    *,
    parameter: str,
    summary_name: str,
    estimate: float,
    reference: float,
    mcse: float,
) -> ScalarValidationResult:
    """Compute signed z and minimum MCSE discrepancy for one scalar summary."""
    if mcse <= 0.0:
        raise ValueError("mcse must be positive")

    signed_z = (estimate - reference) / mcse
    return ScalarValidationResult(
        parameter=parameter,
        summary_name=summary_name,
        estimate=estimate,
        reference=reference,
        mcse=mcse,
        signed_z=signed_z,
        k_min=abs(signed_z),
    )


def project_vector_draws(
    samples: Mapping[str, jax.Array],
    *,
    projection: ProjectionSpec,
) -> jax.Array:
    """Project vector posterior draws to scalar draws with shape ``(chains, samples)``."""
    if projection.parameter not in samples:
        raise ValueError(f"Missing samples for parameter: {projection.parameter}")

    draws = jnp.asarray(samples[projection.parameter])
    weights = jnp.asarray(projection.weights)
    if draws.ndim != 3:
        raise ValueError(
            "Projected vector draw arrays must have shape (num_chains, num_samples, n)"
        )
    if weights.ndim != 1:
        raise ValueError("Projection weights must be one-dimensional")
    if draws.shape[-1] != weights.shape[0]:
        raise ValueError("Projection weights must match the vector parameter dimension")
    return jnp.einsum("csn,n->cs", draws, weights)


def project_vector_truth(
    truth: jax.Array,
    *,
    projection: ProjectionSpec,
) -> float:
    """Project one true vector value to a scalar truth."""
    value = jnp.asarray(truth)
    weights = jnp.asarray(projection.weights)
    if value.ndim != 1:
        raise ValueError("Projected true values must be one-dimensional")
    if weights.ndim != 1:
        raise ValueError("Projection weights must be one-dimensional")
    if value.shape[0] != weights.shape[0]:
        raise ValueError("Projection weights must match the true vector dimension")
    return float(jnp.dot(value, weights))


def scalar_sbc_rank(
    samples: Mapping[str, jax.Array],
    *,
    parameter: str,
    true_value: float,
) -> int:
    """Return the count of scalar posterior draws strictly below ``true_value``."""
    if parameter not in samples:
        raise ValueError(f"Missing samples for parameter: {parameter}")

    draws = jnp.asarray(samples[parameter])
    if draws.ndim != 2:
        raise ValueError("Scalar SBC draw arrays must have shape (num_chains, num_samples)")
    if draws.size < 1:
        raise ValueError("At least one posterior draw is required")

    return int(jnp.sum(draws < true_value))


def projected_sbc_rank(
    samples: Mapping[str, jax.Array],
    *,
    projection: ProjectionSpec,
    true_value: jax.Array,
) -> int:
    """Return the SBC rank of a projected true vector among projected posterior draws."""
    projected_draws = project_vector_draws(samples, projection=projection)
    projected_truth = project_vector_truth(true_value, projection=projection)
    return scalar_sbc_rank(
        {projection.name: projected_draws}, parameter=projection.name, true_value=projected_truth
    )


def summarize_sbc_rank_uniformity(
    result: SbcValidationResult,
    *,
    num_rank_bins: int,
) -> SbcRankUniformitySummary:
    """Summarize SBC rank uniformity with binned and mean-rank z-scores."""
    if len(result.ranks) == 0:
        raise ValueError("At least one SBC rank is required")
    if result.num_posterior_draws < 1:
        raise ValueError("num_posterior_draws must be positive")
    if num_rank_bins < 1:
        raise ValueError("num_rank_bins must be at least 1")
    rank_categories = result.num_posterior_draws + 1
    if num_rank_bins > rank_categories:
        raise ValueError("num_rank_bins cannot exceed num_posterior_draws + 1")

    bin_counts = [0 for _ in range(num_rank_bins)]
    for rank in result.ranks:
        if rank < 0 or rank > result.num_posterior_draws:
            raise ValueError("SBC ranks must be between 0 and num_posterior_draws")
        bin_index = min((rank * num_rank_bins) // rank_categories, num_rank_bins - 1)
        bin_counts[bin_index] += 1

    num_simulations = len(result.ranks)
    expected_bin_count = num_simulations / num_rank_bins
    bin_probability = 1.0 / num_rank_bins
    bin_variance = num_simulations * bin_probability * (1.0 - bin_probability)
    if bin_variance == 0.0:
        max_abs_bin_z = 0.0
    else:
        bin_sd = math.sqrt(bin_variance)
        max_abs_bin_z = max(abs((count - expected_bin_count) / bin_sd) for count in bin_counts)

    rank_mean = sum(result.ranks) / num_simulations
    expected_rank_mean = result.num_posterior_draws / 2.0
    rank_variance = result.num_posterior_draws * (result.num_posterior_draws + 2.0) / 12.0
    mean_rank_se = math.sqrt(rank_variance / num_simulations)
    mean_rank_z = 0.0 if mean_rank_se == 0.0 else (rank_mean - expected_rank_mean) / mean_rank_se

    return SbcRankUniformitySummary(
        parameter=result.parameter,
        ranks=result.ranks,
        num_posterior_draws=result.num_posterior_draws,
        bin_counts=tuple(bin_counts),
        expected_bin_count=expected_bin_count,
        max_abs_bin_z=max_abs_bin_z,
        mean_rank_z=mean_rank_z,
    )


__all__ = [
    "ProjectionSpec",
    "ScalarValidationResult",
    "SbcRankUniformitySummary",
    "SbcValidationResult",
    "project_vector_draws",
    "project_vector_truth",
    "projected_sbc_rank",
    "scalar_sbc_rank",
    "standardized_discrepancy",
    "summarize_sbc_rank_uniformity",
]
