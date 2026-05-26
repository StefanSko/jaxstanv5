"""Planned posterior-validation harness for integration tests.

This module is private test infrastructure.  It records the staged validation
plan before the implementation exists, so future distribution work can turn one
stage green at a time without adding runtime API surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import jax

from jaxstanv5.model.bound import BoundModel


class ValidationStage(Enum):
    """Ordered stages for the Normal-path validation harness."""

    ANALYTIC_NORMAL_REFERENCE = "analytic_normal_reference"
    PRIVATE_HELPERS = "private_helpers"
    INDEPENDENT_CHAINS = "independent_chains"
    ALWAYS_ON_NORMAL_TEST = "always_on_normal_test"
    STANDARDIZED_DISCREPANCIES = "standardized_discrepancies"
    CONSTRAINED_NORMAL_REFERENCE = "constrained_normal_reference"
    STAN_REFERENCE = "stan_reference"
    SBC_REFERENCE = "sbc_reference"


@dataclass(frozen=True)
class ValidationPlanItem:
    """One planned validation stage."""

    stage: ValidationStage
    description: str


VALIDATION_PLAN: tuple[ValidationPlanItem, ...] = (
    ValidationPlanItem(
        stage=ValidationStage.ANALYTIC_NORMAL_REFERENCE,
        description="Analytic posterior reference for the Normal known-scale model.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.PRIVATE_HELPERS,
        description="Typed private helpers for references, draw summaries, and assertions.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.INDEPENDENT_CHAINS,
        description="Run a compiled sampler repeatedly to build explicit multi-chain draws.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.ALWAYS_ON_NORMAL_TEST,
        description="Fast CI test for analytic Normal posterior mean, sd, ESS, and R-hat.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.STANDARDIZED_DISCREPANCIES,
        description="Record signed z-scores and k_min values for posterior summaries.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.CONSTRAINED_NORMAL_REFERENCE,
        description="Numerical reference for a positive-scale Normal model.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.STAN_REFERENCE,
        description="Slow fixed-data posterior summary comparisons against Stan.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.SBC_REFERENCE,
        description="Simulation-based calibration rank checks over generated datasets.",
    ),
)


@dataclass(frozen=True)
class ChainRunSpec:
    """Sampler settings for repeated independent chains."""

    seeds: tuple[int, ...]
    num_warmup: int
    num_samples: int


@dataclass(frozen=True)
class ScalarReference:
    """Reference posterior summary for one scalar parameter."""

    parameter: str
    mean: float
    sd: float


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


def normal_known_scale_reference(
    *,
    parameter: str,
    y: jax.Array,
    prior_loc: float,
    prior_scale: float,
    obs_scale: float,
) -> ScalarReference:
    """Stage 1: return the analytic posterior for a Normal known-scale model."""
    raise _not_implemented(ValidationStage.ANALYTIC_NORMAL_REFERENCE)


def summarize_scalar_draws(
    samples: Mapping[str, jax.Array],
    *,
    parameter: str,
) -> ScalarDrawSummary:
    """Stage 2: summarize scalar posterior draws with ESS, R-hat, and MCSE."""
    raise _not_implemented(ValidationStage.PRIVATE_HELPERS)


def draw_independent_chains(
    bound: BoundModel,
    *,
    run: ChainRunSpec,
) -> Mapping[str, jax.Array]:
    """Stage 3: draw explicit independent chains for a bound model."""
    raise _not_implemented(ValidationStage.INDEPENDENT_CHAINS)


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
