"""Planned posterior-validation harness for integration tests.

This module is private test infrastructure.  It records the staged validation
plan before the implementation exists, so future distribution work can turn one
stage green at a time without adding runtime API surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp

from jaxstanv5.diagnostics import ess as effective_sample_size
from jaxstanv5.diagnostics import rhat as potential_scale_reduction
from jaxstanv5.inference import sample
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
    """Completion status for one validation-plan item."""

    COMPLETED = "completed"
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
        status=ValidationStageStatus.COMPLETED,
        description="Analytic posterior reference for scalar Normal known-scale models.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.ANALYTIC_HIERARCHICAL_NORMAL_REFERENCE,
        status=ValidationStageStatus.COMPLETED,
        description="Analytic posterior reference for hierarchical Normal known-scale models.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.PRIVATE_HELPERS,
        status=ValidationStageStatus.COMPLETED,
        description="Typed private helpers for references, draw summaries, and assertions.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.PUBLIC_MULTI_CHAIN_DRAWS,
        status=ValidationStageStatus.COMPLETED,
        description="Use the public num_chains sampling API to obtain validation draws.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.ALWAYS_ON_NORMAL_TEST,
        status=ValidationStageStatus.COMPLETED,
        description="Fast CI test for analytic Normal posterior mean, sd, ESS, and R-hat.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.STANDARDIZED_DISCREPANCIES,
        status=ValidationStageStatus.COMPLETED,
        description="Record signed z-scores and k_min values for posterior summaries.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.CONSTRAINED_NORMAL_REFERENCE,
        status=ValidationStageStatus.COMPLETED,
        description="Numerical reference for a positive-scale Normal model.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.STAN_REFERENCE,
        status=ValidationStageStatus.COMPLETED,
        description="Slow fixed-data log-density and posterior-summary comparisons against Stan.",
    ),
    ValidationPlanItem(
        stage=ValidationStage.SBC_REFERENCE,
        status=ValidationStageStatus.COMPLETED,
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
    target_acceptance_rate: float = 0.8


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


@dataclass(frozen=True)
class SbcSimulation:
    """One simulated truth and corresponding posterior model binding for SBC."""

    true_value: float
    bound: BoundModel


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
    """Implemented stage: summarize scalar posterior draws with ESS, R-hat, and MCSE."""
    if parameter not in samples:
        raise ValueError(f"Missing samples for parameter: {parameter}")

    draws = jnp.asarray(samples[parameter])
    if draws.ndim != 2:
        raise ValueError("Scalar draw arrays must have shape (num_chains, num_samples)")

    sample_count = draws.size
    if sample_count < 2:
        raise ValueError("At least two scalar draws are required")

    parameter_samples = {parameter: draws}
    ess_value = effective_sample_size(parameter_samples)[parameter]
    if ess_value <= 0.0:
        raise ValueError("ESS must be positive to compute MCSE")

    mean = float(jnp.mean(draws))
    centered = draws - mean
    sd = float(jnp.sqrt(jnp.sum(centered**2) / (sample_count - 1)))
    mcse_mean = sd / math.sqrt(ess_value)

    return ScalarDrawSummary(
        parameter=parameter,
        mean=mean,
        sd=sd,
        ess=ess_value,
        rhat=potential_scale_reduction(parameter_samples)[parameter],
        mcse_mean=mcse_mean,
        mcse_sd=None,
    )


def draw_validation_chains(
    bound: BoundModel,
    *,
    run: ChainRunSpec,
) -> Mapping[str, jax.Array]:
    """Implemented stage: draw validation chains through the public sampling API."""
    return sample(
        bound,
        seed=run.seed,
        num_chains=run.num_chains,
        num_warmup=run.num_warmup,
        num_samples=run.num_samples,
        target_acceptance_rate=run.target_acceptance_rate,
    ).samples


def assert_scalar_mean_matches_reference(
    bound: BoundModel,
    *,
    reference: ScalarReference,
    run: ChainRunSpec,
    max_k: float,
    max_rhat: float = 1.05,
    min_ess: float = 100.0,
) -> tuple[ScalarValidationResult, ...]:
    """Validate a scalar posterior mean against a reference within MCSE."""
    if max_k <= 0.0:
        raise ValueError("max_k must be positive")
    if max_rhat <= 0.0:
        raise ValueError("max_rhat must be positive")
    if min_ess <= 0.0:
        raise ValueError("min_ess must be positive")

    samples = draw_validation_chains(bound, run=run)
    summary = summarize_scalar_draws(samples, parameter=reference.parameter)

    if summary.rhat > max_rhat:
        raise AssertionError(
            f"R-hat for {summary.parameter} is {summary.rhat:.3f}; expected <= {max_rhat:.3f}"
        )
    if summary.ess < min_ess:
        raise AssertionError(
            f"ESS for {summary.parameter} is {summary.ess:.1f}; expected >= {min_ess:.1f}"
        )

    result = standardized_discrepancy(
        parameter=reference.parameter,
        summary_name="mean",
        estimate=summary.mean,
        reference=reference.mean,
        mcse=summary.mcse_mean,
    )
    if result.k_min > max_k:
        raise AssertionError(
            f"Posterior mean for {result.parameter} differs from reference by "
            f"{result.k_min:.2f} MCSEs; expected <= {max_k:.2f}"
        )
    return (result,)


def assert_normal_known_scale_matches_reference(
    bound: BoundModel,
    *,
    reference: ScalarReference,
    run: ChainRunSpec,
    max_k: float,
) -> tuple[ScalarValidationResult, ...]:
    """Implemented stage: validate a scalar Normal posterior mean within MCSE."""
    return assert_scalar_mean_matches_reference(
        bound,
        reference=reference,
        run=run,
        max_k=max_k,
    )


def standardized_discrepancy(
    *,
    parameter: str,
    summary_name: str,
    estimate: float,
    reference: float,
    mcse: float,
) -> ScalarValidationResult:
    """Implemented stage: compute signed z and k_min for one scalar summary."""
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


def scalar_grid_reference(
    *,
    parameter: str,
    log_unnormalized: Callable[[jax.Array], jax.Array],
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> ScalarReference:
    """Return a numerical scalar posterior reference from an unnormalized density.

    The density callable receives a one-dimensional grid and must return one
    log-density value per grid point.  The integral is approximated with a
    trapezoidal rule on an equally spaced grid.
    """
    if grid_max <= grid_min:
        raise ValueError("grid_max must be greater than grid_min")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    grid = jnp.linspace(grid_min, grid_max, grid_size)
    log_density = jnp.asarray(log_unnormalized(grid))
    if log_density.shape != grid.shape:
        raise ValueError("log_unnormalized must return one value per grid point")

    trapezoid_weights = jnp.ones_like(grid)
    trapezoid_weights = trapezoid_weights.at[0].set(0.5)
    trapezoid_weights = trapezoid_weights.at[-1].set(0.5)
    normalized_weights = jax.nn.softmax(log_density + jnp.log(trapezoid_weights))

    mean = float(jnp.sum(normalized_weights * grid))
    second_moment = float(jnp.sum(normalized_weights * grid**2))
    variance = max(second_moment - mean**2, 0.0)
    return ScalarReference(parameter=parameter, mean=mean, sd=math.sqrt(variance))


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
    """Implemented stage: numerical posterior reference for a positive scale.

    Model:
        ``sigma ~ Normal(prior_loc, prior_scale), sigma > 0``
        ``y_i ~ Normal(0, sigma)``

    The returned summaries are expectations over the constrained scale ``sigma``.
    A uniform grid and trapezoidal weights approximate the one-dimensional
    posterior integral.
    """
    _require_positive_scale("prior_scale", prior_scale)
    _require_positive_scale("grid_min", grid_min)

    y_values = jnp.ravel(jnp.asarray(y))
    log_two_pi = math.log(2.0 * math.pi)

    def log_unnormalized(sigma: jax.Array) -> jax.Array:
        standardized_prior = (sigma - prior_loc) / prior_scale
        log_prior = -jnp.log(prior_scale) - 0.5 * log_two_pi - 0.5 * standardized_prior**2
        log_likelihood_terms = (
            -jnp.log(sigma) - 0.5 * log_two_pi - 0.5 * (y_values[:, None] / sigma) ** 2
        )
        return log_prior + jnp.sum(log_likelihood_terms, axis=0)

    return scalar_grid_reference(
        parameter=parameter,
        log_unnormalized=log_unnormalized,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
    )


def compare_against_stan_reference(
    *,
    jaxstan_summaries: tuple[ScalarDrawSummary, ...],
    stan_summaries: tuple[ScalarDrawSummary, ...],
    max_k: float,
    max_rhat: float = 1.05,
    min_ess: float = 100.0,
) -> tuple[ScalarValidationResult, ...]:
    """Implemented stage: compare jaxstan summaries against Stan.

    Means are compared using the combined Monte Carlo standard error:
    ``sqrt(mcse_jaxstan**2 + mcse_stan**2)``.
    """
    if max_k <= 0.0:
        raise ValueError("max_k must be positive")
    if max_rhat <= 0.0:
        raise ValueError("max_rhat must be positive")
    if min_ess <= 0.0:
        raise ValueError("min_ess must be positive")

    stan_by_parameter = {summary.parameter: summary for summary in stan_summaries}
    results: list[ScalarValidationResult] = []
    for jaxstan_summary in jaxstan_summaries:
        if jaxstan_summary.rhat > max_rhat:
            raise AssertionError(
                f"jaxstan R-hat for {jaxstan_summary.parameter} is "
                f"{jaxstan_summary.rhat:.3f}; expected <= {max_rhat:.3f}"
            )
        if jaxstan_summary.ess < min_ess:
            raise AssertionError(
                f"jaxstan ESS for {jaxstan_summary.parameter} is "
                f"{jaxstan_summary.ess:.1f}; expected >= {min_ess:.1f}"
            )

        stan_summary = stan_by_parameter.get(jaxstan_summary.parameter)
        if stan_summary is None:
            raise ValueError(f"Missing Stan summary for parameter: {jaxstan_summary.parameter}")
        if stan_summary.rhat > max_rhat:
            raise AssertionError(
                f"Stan R-hat for {stan_summary.parameter} is "
                f"{stan_summary.rhat:.3f}; expected <= {max_rhat:.3f}"
            )
        if stan_summary.ess < min_ess:
            raise AssertionError(
                f"Stan ESS for {stan_summary.parameter} is "
                f"{stan_summary.ess:.1f}; expected >= {min_ess:.1f}"
            )

        combined_mcse = math.sqrt(jaxstan_summary.mcse_mean**2 + stan_summary.mcse_mean**2)
        result = standardized_discrepancy(
            parameter=jaxstan_summary.parameter,
            summary_name="mean",
            estimate=jaxstan_summary.mean,
            reference=stan_summary.mean,
            mcse=combined_mcse,
        )
        if result.k_min > max_k:
            raise AssertionError(
                f"jaxstan posterior mean for {result.parameter} differs from Stan by "
                f"{result.k_min:.2f} combined MCSEs; expected <= {max_k:.2f}"
            )
        results.append(result)

    return tuple(results)


def scalar_sbc_rank(
    samples: Mapping[str, jax.Array],
    *,
    parameter: str,
    true_value: float,
) -> int:
    """Return the SBC rank of ``true_value`` among scalar posterior draws."""
    if parameter not in samples:
        raise ValueError(f"Missing samples for parameter: {parameter}")

    draws = jnp.asarray(samples[parameter])
    if draws.ndim != 2:
        raise ValueError("Scalar SBC draw arrays must have shape (num_chains, num_samples)")
    if draws.size < 1:
        raise ValueError("At least one posterior draw is required")

    return int(jnp.sum(draws < true_value))


def run_sbc_rank_validation(
    *,
    parameter: str,
    simulations: tuple[SbcSimulation, ...],
    run: ChainRunSpec,
) -> SbcValidationResult:
    """Run simulation-based calibration ranks for one scalar parameter."""
    if len(simulations) == 0:
        raise ValueError("At least one SBC simulation is required")

    ranks: list[int] = []
    num_posterior_draws: int | None = None
    for simulation in simulations:
        samples = draw_validation_chains(simulation.bound, run=run)
        draws = jnp.asarray(samples[parameter])
        posterior_draw_count = int(draws.size)
        if num_posterior_draws is None:
            num_posterior_draws = posterior_draw_count
        elif posterior_draw_count != num_posterior_draws:
            raise ValueError("All SBC simulations must use the same number of posterior draws")
        ranks.append(
            scalar_sbc_rank(samples, parameter=parameter, true_value=simulation.true_value)
        )

    if num_posterior_draws is None:
        raise ValueError("At least one posterior draw is required")
    return SbcValidationResult(
        parameter=parameter,
        ranks=tuple(ranks),
        num_posterior_draws=num_posterior_draws,
    )


def summarize_sbc_rank_uniformity(
    result: SbcValidationResult,
    *,
    num_rank_bins: int,
) -> SbcRankUniformitySummary:
    """Summarize rank uniformity with binned z-scores and mean-rank z-score."""
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


def assert_sbc_rank_uniformity(
    result: SbcValidationResult,
    *,
    num_rank_bins: int,
    max_abs_bin_z: float,
    max_abs_mean_rank_z: float,
) -> SbcRankUniformitySummary:
    """Assert that SBC ranks are acceptably close to uniform."""
    if max_abs_bin_z <= 0.0:
        raise ValueError("max_abs_bin_z must be positive")
    if max_abs_mean_rank_z <= 0.0:
        raise ValueError("max_abs_mean_rank_z must be positive")

    summary = summarize_sbc_rank_uniformity(result, num_rank_bins=num_rank_bins)
    if summary.max_abs_bin_z > max_abs_bin_z:
        raise AssertionError(
            f"SBC rank bin z-score for {summary.parameter} is "
            f"{summary.max_abs_bin_z:.2f}; expected <= {max_abs_bin_z:.2f}"
        )
    if abs(summary.mean_rank_z) > max_abs_mean_rank_z:
        raise AssertionError(
            f"SBC mean-rank z-score for {summary.parameter} is "
            f"{summary.mean_rank_z:.2f}; expected absolute value <= {max_abs_mean_rank_z:.2f}"
        )
    return summary
