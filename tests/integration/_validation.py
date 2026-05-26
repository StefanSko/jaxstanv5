"""Planned posterior-validation harness for integration tests.

This module is private test infrastructure.  It records the staged validation
plan before the implementation exists, so future distribution work can turn one
stage green at a time without adding runtime API surface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp

from jaxstanv5.model.bound import BoundModel


class ValidationStage(Enum):
    """Ordered stages for the Normal-path validation harness."""

    ANALYTIC_SCALAR_NORMAL_REFERENCE = "analytic_scalar_normal_reference"
    ANALYTIC_HIERARCHICAL_NORMAL_REFERENCE = "analytic_hierarchical_normal_reference"
    PRIVATE_HELPERS = "private_helpers"
    PUBLIC_MULTI_CHAIN_DRAWS = "public_multi_chain_draws"
    ALWAYS_ON_NORMAL_TEST = "always_on_normal_test"
    STANDARDIZED_DISCREPANCIES = "standardized_discrepancies"
    CONSTRAINED_NORMAL_REFERENCE = "constrained_normal_reference"
    STAN_REFERENCE = "stan_reference"
    SBC_REFERENCE = "sbc_reference"


class ValidationStageStatus(Enum):
    """Implementation status for one validation-plan item."""

    IMPLEMENTED = "implemented"
    PLANNED = "planned"


@dataclass(frozen=True)
class ValidationPlanItem:
    """One planned validation stage."""

    stage: ValidationStage
    status: ValidationStageStatus
    description: str


VALIDATION_PLAN: tuple[ValidationPlanItem, ...] = (
    ValidationPlanItem(
        stage=ValidationStage.ANALYTIC_SCALAR_NORMAL_REFERENCE,
        status=ValidationStageStatus.IMPLEMENTED,
        description="Analytic posterior reference for scalar Normal known-scale models.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.ANALYTIC_HIERARCHICAL_NORMAL_REFERENCE,
        status=ValidationStageStatus.IMPLEMENTED,
        description="Analytic posterior reference for hierarchical Normal known-scale models.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.PRIVATE_HELPERS,
        status=ValidationStageStatus.PLANNED,
        description="Typed private helpers for references, draw summaries, and assertions.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.PUBLIC_MULTI_CHAIN_DRAWS,
        status=ValidationStageStatus.PLANNED,
        description="Use the public num_chains sampling API to obtain validation draws.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.ALWAYS_ON_NORMAL_TEST,
        status=ValidationStageStatus.PLANNED,
        description="Fast CI test for analytic Normal posterior mean, sd, ESS, and R-hat.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.STANDARDIZED_DISCREPANCIES,
        status=ValidationStageStatus.PLANNED,
        description="Record signed z-scores and k_min values for posterior summaries.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.CONSTRAINED_NORMAL_REFERENCE,
        status=ValidationStageStatus.PLANNED,
        description="Numerical reference for a positive-scale Normal model.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.STAN_REFERENCE,
        status=ValidationStageStatus.PLANNED,
        description="Slow fixed-data posterior summary comparisons against Stan.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.SBC_REFERENCE,
        status=ValidationStageStatus.PLANNED,
        description="Simulation-based calibration rank checks over generated datasets.",
    ),
)


@dataclass(frozen=True)
class ChainRunSpec:
    """Sampler settings for public multi-chain validation runs."""

    seed: int
    num_chains: int
    num_warmup: int
    num_samples: int


@dataclass(frozen=True)
class ScalarReference:
    """Reference posterior summary for one scalar parameter."""

    parameter: str
    mean: float
    sd: float


@dataclass(frozen=True)
class HierarchicalNormalReference:
    """Reference posterior summaries for a Normal-Normal hierarchy."""

    population: ScalarReference
    groups: tuple[ScalarReference, ...]


@dataclass(frozen=True)
class ScalarDrawSummary:
    """Posterior draw summary for one scalar parameter."""

    parameter: str
    mean: float
    sd: float
    ess: float
    rhat: float
    mcse_mean: float
    mcse_sd: float | None


@dataclass(frozen=True)
class ScalarValidationResult:
    """Standardized comparison between a draw summary and a reference summary."""

    parameter: str
    summary_name: str
    estimate: float
    reference: float
    mcse: float
    signed_z: float
    k_min: float


@dataclass(frozen=True)
class SbcValidationResult:
    """Simulation-based calibration ranks for one scalar parameter."""

    parameter: str
    ranks: tuple[int, ...]
    num_posterior_draws: int


def _not_implemented(stage: ValidationStage) -> NotImplementedError:
    return NotImplementedError(f"Validation stage is not implemented yet: {stage.value}")


def _require_positive_scale(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def normal_known_scale_reference(
    *,
    parameter: str,
    y: jax.Array,
    prior_loc: float,
    prior_scale: float,
    obs_scale: float,
) -> ScalarReference:
    """Implemented stage: return the analytic posterior for a Normal known-scale model.

    Model:
        ``mu ~ Normal(prior_loc, prior_scale)``
        ``y_i ~ Normal(mu, obs_scale)``
    """
    _require_positive_scale("prior_scale", prior_scale)
    _require_positive_scale("obs_scale", obs_scale)

    y_values = jnp.ravel(jnp.asarray(y))
    prior_precision = 1.0 / prior_scale**2
    obs_precision = 1.0 / obs_scale**2
    posterior_precision = prior_precision + float(y_values.size) * obs_precision
    posterior_variance = 1.0 / posterior_precision
    posterior_mean = posterior_variance * (
        prior_loc * prior_precision + float(jnp.sum(y_values)) * obs_precision
    )
    return ScalarReference(
        parameter=parameter,
        mean=posterior_mean,
        sd=math.sqrt(posterior_variance),
    )


def hierarchical_normal_known_scale_reference(
    *,
    population_parameter: str,
    group_parameter: str,
    y: jax.Array,
    prior_loc: float,
    prior_scale: float,
    group_scale: float,
    obs_scale: float,
) -> HierarchicalNormalReference:
    """Implemented stage: return analytic references for a Normal-Normal hierarchy.

    ``y`` must be rectangular with shape ``(num_groups, observations_per_group)``.

    Model:
        ``mu_pop ~ Normal(prior_loc, prior_scale)``
        ``theta_g ~ Normal(mu_pop, group_scale)``
        ``y_gi ~ Normal(theta_g, obs_scale)``
    """
    _require_positive_scale("prior_scale", prior_scale)
    _require_positive_scale("group_scale", group_scale)
    _require_positive_scale("obs_scale", obs_scale)

    y_values = jnp.asarray(y)
    if y_values.ndim != 2:
        raise ValueError("y must have shape (num_groups, observations_per_group)")

    num_groups = y_values.shape[0]
    observations_per_group = y_values.shape[1]
    dimension = num_groups + 1

    prior_precision = 1.0 / prior_scale**2
    group_precision = 1.0 / group_scale**2
    obs_precision = 1.0 / obs_scale**2

    precision = jnp.zeros((dimension, dimension))
    information = jnp.zeros((dimension,))

    precision = precision.at[0, 0].add(prior_precision)
    information = information.at[0].add(prior_loc * prior_precision)

    for group_index in range(num_groups):
        theta_index = group_index + 1
        precision = precision.at[0, 0].add(group_precision)
        precision = precision.at[theta_index, theta_index].add(group_precision)
        precision = precision.at[0, theta_index].add(-group_precision)
        precision = precision.at[theta_index, 0].add(-group_precision)
        precision = precision.at[theta_index, theta_index].add(
            observations_per_group * obs_precision
        )
        information = information.at[theta_index].add(
            jnp.sum(y_values[group_index]) * obs_precision
        )

    posterior_mean = jnp.linalg.solve(precision, information)
    posterior_covariance = jnp.linalg.inv(precision)
    posterior_sd = jnp.sqrt(jnp.diag(posterior_covariance))

    groups = tuple(
        ScalarReference(
            parameter=f"{group_parameter}[{group_index}]",
            mean=float(posterior_mean[group_index + 1]),
            sd=float(posterior_sd[group_index + 1]),
        )
        for group_index in range(num_groups)
    )
    return HierarchicalNormalReference(
        population=ScalarReference(
            parameter=population_parameter,
            mean=float(posterior_mean[0]),
            sd=float(posterior_sd[0]),
        ),
        groups=groups,
    )


def summarize_scalar_draws(
    samples: Mapping[str, jax.Array],
    *,
    parameter: str,
) -> ScalarDrawSummary:
    """Stage 2: summarize scalar posterior draws with ESS, R-hat, and MCSE."""
    raise _not_implemented(ValidationStage.PRIVATE_HELPERS)


def draw_validation_chains(
    bound: BoundModel,
    *,
    run: ChainRunSpec,
) -> Mapping[str, jax.Array]:
    """Stage 3: draw validation chains through the public sampling API."""
    raise _not_implemented(ValidationStage.PUBLIC_MULTI_CHAIN_DRAWS)


def assert_normal_known_scale_matches_reference(
    bound: BoundModel,
    *,
    reference: ScalarReference,
    run: ChainRunSpec,
    max_k: float,
) -> tuple[ScalarValidationResult, ...]:
    """Stage 4: fast always-on analytic Normal posterior validation."""
    raise _not_implemented(ValidationStage.ALWAYS_ON_NORMAL_TEST)


def standardized_discrepancy(
    *,
    parameter: str,
    summary_name: str,
    estimate: float,
    reference: float,
    mcse: float,
) -> ScalarValidationResult:
    """Stage 5: compute signed z and k_min for one scalar summary."""
    raise _not_implemented(ValidationStage.STANDARDIZED_DISCREPANCIES)


def positive_scale_grid_reference(
    *,
    parameter: str,
    y: jax.Array,
    prior_loc: float,
    prior_scale: float,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> ScalarReference:
    """Stage 6: compute a numerical reference for a positive-scale Normal model."""
    raise _not_implemented(ValidationStage.CONSTRAINED_NORMAL_REFERENCE)


def compare_against_stan_reference(
    *,
    jaxstan_summaries: tuple[ScalarDrawSummary, ...],
    stan_summaries: tuple[ScalarDrawSummary, ...],
    max_k: float,
) -> tuple[ScalarValidationResult, ...]:
    """Stage 7: compare jaxstan summaries against Stan within combined MCSE."""
    raise _not_implemented(ValidationStage.STAN_REFERENCE)


def run_sbc_rank_validation(
    *,
    parameter: str,
    num_simulations: int,
    num_posterior_draws: int,
) -> SbcValidationResult:
    """Stage 8: run simulation-based calibration for one scalar parameter."""
    raise _not_implemented(ValidationStage.SBC_REFERENCE)
