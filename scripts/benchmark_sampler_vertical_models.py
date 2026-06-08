#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Vertical sampler benchmarks using realistic in-memory model/data slices."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from jaxstanv5.inference import NutsDiagnosticTrace, SamplerResult
    from jaxstanv5.model.bound import BoundModel


class BoundModelBuilder(Protocol):
    """Build one fixed-shape benchmark bound model."""

    def __call__(self) -> BoundModel:
        """Return a bound model."""
        ...


class LinearRegressionModel(Protocol):
    """Decorated linear-regression benchmark model."""

    def bind(self, *, x: jax.Array, y: jax.Array) -> BoundModel:
        """Bind generated data."""
        ...


class HierarchicalRegressionModel(Protocol):
    """Decorated hierarchical benchmark model."""

    def bind(
        self,
        *,
        n_groups: int,
        group_idx: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ) -> BoundModel:
        """Bind generated data."""
        ...


class FixedKernelGpModel(Protocol):
    """Decorated fixed-kernel GP benchmark model."""

    def bind(
        self,
        *,
        n: int,
        chol: jax.Array,
        obs_sd: float,
        y: jax.Array,
    ) -> BoundModel:
        """Bind generated data."""
        ...


@dataclass(frozen=True)
class VerticalCase:
    """One realistic vertical sampler benchmark case."""

    name: str
    description: str
    num_chains: int
    num_warmup: int
    num_samples: int
    target_acceptance_rate: float
    default_runs: int
    build_bound: BoundModelBuilder


@dataclass(frozen=True)
class RunTiming:
    """One measured end-to-end sampler call."""

    seed: int
    seconds: float
    warmup_divergences: int
    sampling_divergences: int
    mean_sampling_acceptance: float
    max_sampling_integration_steps: int
    max_rhat: float
    min_ess: float


@dataclass(frozen=True)
class SummaryStats:
    """Summary statistics for repeated runtimes."""

    count: int
    mean_seconds: float
    stdev_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    ci95_half_width_seconds: float


@dataclass(frozen=True)
class CaseResult:
    """Benchmark result for one vertical case."""

    case: VerticalCase
    n_params: int
    compile_sampler_seconds: float
    cold_sample_seconds: float
    cached_stats: SummaryStats
    timings: tuple[RunTiming, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete benchmark result."""

    label: str
    branch: str
    commit: str
    runs_per_case: int | None
    cache_warmup_runs: int
    cases: tuple[CaseResult, ...]


class RunTimingJson(TypedDict):
    """JSON row for one measured sampler call."""

    seed: int
    seconds: float
    warmup_divergences: int
    sampling_divergences: int
    mean_sampling_acceptance: float
    max_sampling_integration_steps: int
    max_rhat: float
    min_ess: float


class SummaryStatsJson(TypedDict):
    """JSON row for summary stats."""

    count: int
    mean_seconds: float
    stdev_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    ci95_half_width_seconds: float


class CaseResultJson(TypedDict):
    """JSON object for one case result."""

    name: str
    description: str
    n_params: int
    num_chains: int
    num_warmup: int
    num_samples: int
    target_acceptance_rate: float
    compile_sampler_seconds: float
    cold_sample_seconds: float
    cached_stats: SummaryStatsJson
    timings: list[RunTimingJson]


class BenchmarkResultJson(TypedDict):
    """JSON object for complete benchmark result."""

    label: str
    branch: str
    commit: str
    runs_per_case: int | None
    cache_warmup_runs: int
    cases: list[CaseResultJson]


def _repo_root() -> Path:
    override = os.environ.get("JAXSTANV5_REPO_ROOT")
    if override is not None:
        return Path(override)
    return Path(__file__).resolve().parents[1]


def _add_repo_src() -> None:
    sys.path.insert(0, str(_repo_root() / "src"))


def _git_value(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _trace_arrays(trace: NutsDiagnosticTrace) -> tuple[jax.Array, ...]:
    return (
        trace.is_divergent,
        trace.acceptance_rate,
        trace.num_integration_steps,
        trace.num_trajectory_expansions,
        trace.energy,
    )


def _block_result(result: SamplerResult) -> None:
    for values in result.samples.values():
        values.block_until_ready()
    for values in _trace_arrays(result.diagnostics.warmup):
        values.block_until_ready()
    for values in _trace_arrays(result.diagnostics.sampling):
        values.block_until_ready()


def _sample_once(
    compiled_sampler: object, seed: int, case: VerticalCase
) -> tuple[SamplerResult, float]:
    from jaxstanv5.inference import CompiledSampler

    sampler = cast(CompiledSampler, compiled_sampler)
    start = time.perf_counter()
    result = sampler.sample(
        seed=seed,
        num_warmup=case.num_warmup,
        num_samples=case.num_samples,
        num_chains=case.num_chains,
    )
    _block_result(result)
    return result, time.perf_counter() - start


def _max_mapping_value(values: Mapping[str, float]) -> float:
    if not values:
        return float("nan")
    return max(values.values())


def _min_mapping_value(values: Mapping[str, float]) -> float:
    if not values:
        return float("nan")
    return min(values.values())


def _timing_from_result(result: SamplerResult, seed: int, seconds: float) -> RunTiming:
    from jaxstanv5.diagnostics import ess, rhat

    warmup_divergences = int(jnp.sum(result.diagnostics.warmup.is_divergent).item())
    sampling_divergences = int(jnp.sum(result.diagnostics.sampling.is_divergent).item())
    mean_acceptance = float(jnp.mean(result.diagnostics.sampling.acceptance_rate).item())
    max_steps = int(jnp.max(result.diagnostics.sampling.num_integration_steps).item())
    return RunTiming(
        seed=seed,
        seconds=seconds,
        warmup_divergences=warmup_divergences,
        sampling_divergences=sampling_divergences,
        mean_sampling_acceptance=mean_acceptance,
        max_sampling_integration_steps=max_steps,
        max_rhat=_max_mapping_value(rhat(result.samples)),
        min_ess=_min_mapping_value(ess(result.samples)),
    )


def _summary(values: Sequence[float]) -> SummaryStats:
    count = len(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if count > 1 else 0.0
    return SummaryStats(
        count=count,
        mean_seconds=mean,
        stdev_seconds=stdev,
        median_seconds=statistics.median(values),
        min_seconds=min(values),
        max_seconds=max(values),
        ci95_half_width_seconds=1.96 * stdev / (count**0.5),
    )


def _benchmark_case(
    case: VerticalCase,
    *,
    runs_override: int | None,
    cache_warmup_runs: int,
    seed_start: int,
) -> CaseResult:
    from jaxstanv5.inference import compile_sampler

    bound = case.build_bound()
    compile_start = time.perf_counter()
    compiled = compile_sampler(bound, target_acceptance_rate=case.target_acceptance_rate)
    compile_seconds = time.perf_counter() - compile_start

    cold_result, cold_seconds = _sample_once(compiled, seed_start, case)
    _ = cold_result

    for offset in range(cache_warmup_runs):
        warm_seed = seed_start + 1 + offset
        warm_result, _warm_seconds = _sample_once(compiled, warm_seed, case)
        _ = warm_result

    runs = runs_override if runs_override is not None else case.default_runs
    timings: list[RunTiming] = []
    first_measured_seed = seed_start + 1 + cache_warmup_runs
    for offset in range(runs):
        seed = first_measured_seed + offset
        result, seconds = _sample_once(compiled, seed, case)
        timings.append(_timing_from_result(result, seed, seconds))

    return CaseResult(
        case=case,
        n_params=bound.n_params,
        compile_sampler_seconds=compile_seconds,
        cold_sample_seconds=cold_seconds,
        cached_stats=_summary([timing.seconds for timing in timings]),
        timings=tuple(timings),
    )


def _build_linear_regression_bound() -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Normal

    @model
    class LinearRegressionVerticalBenchmark:
        """Simple real-scale Gaussian regression."""

        x = Data.vector()
        alpha = Param(Normal(0.0, 5.0))
        beta = Param(Normal(0.0, 2.0))
        sigma = Param(Normal(0.0, 1.0), constraint=Positive())
        mu = alpha + beta * x
        y = Observed(Normal(mu, sigma))

    n_obs = 5_000
    x = jnp.linspace(-2.0, 2.0, n_obs)
    residual = 0.25 * jnp.sin(jnp.arange(n_obs, dtype=jnp.float64) * 0.017)
    y = 1.5 - 0.7 * x + residual
    bindable = cast(LinearRegressionModel, LinearRegressionVerticalBenchmark)
    return bindable.bind(x=x, y=y)


def _build_hierarchical_regression_bound() -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Normal

    @model
    class HierarchicalRegressionVerticalBenchmark:
        """Non-centered varying-intercept/slope Gaussian regression."""

        n_groups = Data.scalar()
        group_idx = Data.vector()
        x = Data.vector()

        alpha_pop = Param(Normal(0.0, 2.0))
        beta_pop = Param(Normal(0.0, 1.0))
        sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
        sigma_beta = Param(Normal(0.0, 1.0), constraint=Positive())
        sigma = Param(Normal(0.0, 1.0), constraint=Positive())
        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)
        z_beta = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        beta = beta_pop + sigma_beta * z_beta
        mu = alpha[group_idx] + beta[group_idx] * x
        y = Observed(Normal(mu, sigma))

    n_groups = 80
    n_obs = 5_000
    group_idx = jnp.arange(n_obs, dtype=jnp.int32) % n_groups
    x = jnp.linspace(-1.5, 1.5, n_obs)
    groups = jnp.arange(n_groups, dtype=jnp.float64)
    alpha_by_group = 0.7 + 0.45 * jnp.sin(groups * 0.17)
    beta_by_group = -0.4 + 0.20 * jnp.cos(groups * 0.11)
    residual = 0.20 * jnp.sin(jnp.arange(n_obs, dtype=jnp.float64) * 0.013)
    y = alpha_by_group[group_idx] + beta_by_group[group_idx] * x + residual
    bindable = cast(HierarchicalRegressionModel, HierarchicalRegressionVerticalBenchmark)
    return bindable.bind(n_groups=n_groups, group_idx=group_idx, x=x, y=y)


def _build_fixed_kernel_gp_bound() -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import MultivariateNormal, Normal

    @model
    class FixedKernelGpVerticalBenchmark:
        """Latent GP with fixed covariance and Gaussian observations."""

        n = Data.scalar()
        chol = Data.matrix(n, n)
        obs_sd = Data.scalar()

        f = Param(MultivariateNormal(0.0, chol), size=n)
        y = Observed(Normal(f, obs_sd))

    n = 48
    x = jnp.linspace(-3.0, 3.0, n)
    distances = x[:, None] - x[None, :]
    covariance = jnp.exp(-0.5 * (distances / 0.9) ** 2) + 1.0e-4 * jnp.eye(n)
    chol = jnp.linalg.cholesky(covariance)
    obs_sd = 0.25
    f_true = jnp.sin(x) + 0.25 * jnp.cos(2.0 * x)
    y = f_true + 0.10 * jnp.sin(jnp.arange(n, dtype=jnp.float64) * 0.37)
    bindable = cast(FixedKernelGpModel, FixedKernelGpVerticalBenchmark)
    return bindable.bind(n=n, chol=chol, obs_sd=obs_sd, y=y)


def _all_cases() -> tuple[VerticalCase, ...]:
    return (
        VerticalCase(
            name="linear_regression_5000n_4c_500w_2000s",
            description="Gaussian regression with 5,000 observations and 3 parameters",
            num_chains=4,
            num_warmup=500,
            num_samples=2_000,
            target_acceptance_rate=0.9,
            default_runs=8,
            build_bound=_build_linear_regression_bound,
        ),
        VerticalCase(
            name="hierarchical_regression_80g_5000n_4c_600w_2000s",
            description=(
                "Non-centered varying intercept/slope regression; "
                "80 groups, 5,000 observations, 165 parameters"
            ),
            num_chains=4,
            num_warmup=600,
            num_samples=2_000,
            target_acceptance_rate=0.9,
            default_runs=6,
            build_bound=_build_hierarchical_regression_bound,
        ),
        VerticalCase(
            name="fixed_kernel_gp_48n_4c_600w_2000s",
            description="Latent fixed-kernel Gaussian process; 48-dimensional MVN parameter",
            num_chains=4,
            num_warmup=600,
            num_samples=2_000,
            target_acceptance_rate=0.9,
            default_runs=6,
            build_bound=_build_fixed_kernel_gp_bound,
        ),
    )


def _select_cases(names: Sequence[str]) -> tuple[VerticalCase, ...]:
    cases = _all_cases()
    if not names:
        return cases
    by_name = {case.name: case for case in cases}
    missing = tuple(name for name in names if name not in by_name)
    if missing:
        available = ", ".join(sorted(by_name))
        raise ValueError(f"unknown benchmark case(s): {', '.join(missing)}; available: {available}")
    return tuple(by_name[name] for name in names)


def _run_timing_json(timing: RunTiming) -> RunTimingJson:
    return {
        "seed": timing.seed,
        "seconds": timing.seconds,
        "warmup_divergences": timing.warmup_divergences,
        "sampling_divergences": timing.sampling_divergences,
        "mean_sampling_acceptance": timing.mean_sampling_acceptance,
        "max_sampling_integration_steps": timing.max_sampling_integration_steps,
        "max_rhat": timing.max_rhat,
        "min_ess": timing.min_ess,
    }


def _summary_json(stats: SummaryStats) -> SummaryStatsJson:
    return {
        "count": stats.count,
        "mean_seconds": stats.mean_seconds,
        "stdev_seconds": stats.stdev_seconds,
        "median_seconds": stats.median_seconds,
        "min_seconds": stats.min_seconds,
        "max_seconds": stats.max_seconds,
        "ci95_half_width_seconds": stats.ci95_half_width_seconds,
    }


def _case_result_json(result: CaseResult) -> CaseResultJson:
    return {
        "name": result.case.name,
        "description": result.case.description,
        "n_params": result.n_params,
        "num_chains": result.case.num_chains,
        "num_warmup": result.case.num_warmup,
        "num_samples": result.case.num_samples,
        "target_acceptance_rate": result.case.target_acceptance_rate,
        "compile_sampler_seconds": result.compile_sampler_seconds,
        "cold_sample_seconds": result.cold_sample_seconds,
        "cached_stats": _summary_json(result.cached_stats),
        "timings": [_run_timing_json(timing) for timing in result.timings],
    }


def _benchmark_result_json(result: BenchmarkResult) -> BenchmarkResultJson:
    return {
        "label": result.label,
        "branch": result.branch,
        "commit": result.commit,
        "runs_per_case": result.runs_per_case,
        "cache_warmup_runs": result.cache_warmup_runs,
        "cases": [_case_result_json(case_result) for case_result in result.cases],
    }


def _format_seconds(value: float) -> str:
    return f"{value:.3f}"


def _print_markdown(result: BenchmarkResult) -> None:
    print(f"### Vertical sampler benchmark: {result.label}")
    print()
    print(f"Branch: `{result.branch}`  ")
    print(f"Commit: `{result.commit}`  ")
    print("JAX x64: enabled  ")
    print(f"Cache warmup runs per case: {result.cache_warmup_runs}")
    print()
    print(
        "| case | params | chains | warmup | samples/chain | cached runs | cold sample s | "
        "cached mean s | cached median s | cached sd s | cached 95% CI s | "
        "sampling divergences | max R-hat | min ESS |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case_result in result.cases:
        stats = case_result.cached_stats
        sampling_divergences = sum(timing.sampling_divergences for timing in case_result.timings)
        max_rhat = max(timing.max_rhat for timing in case_result.timings)
        min_ess = min(timing.min_ess for timing in case_result.timings)
        print(
            f"| {case_result.case.name} | {case_result.n_params} | "
            f"{case_result.case.num_chains} | {case_result.case.num_warmup} | "
            f"{case_result.case.num_samples} | {stats.count} | "
            f"{_format_seconds(case_result.cold_sample_seconds)} | "
            f"{_format_seconds(stats.mean_seconds)} | {_format_seconds(stats.median_seconds)} | "
            f"{_format_seconds(stats.stdev_seconds)} | "
            f"±{_format_seconds(stats.ci95_half_width_seconds)} | "
            f"{sampling_divergences} | {max_rhat:.3f} | {min_ess:.1f} |"
        )
    print()
    print("Per-run cached timings:")
    for case_result in result.cases:
        timings = ", ".join(
            f"{timing.seed}:{timing.seconds:.3f}s" for timing in case_result.timings
        )
        total_warmup_divergences = sum(timing.warmup_divergences for timing in case_result.timings)
        total_sampling_divergences = sum(
            timing.sampling_divergences for timing in case_result.timings
        )
        mean_acceptance = statistics.fmean(
            timing.mean_sampling_acceptance for timing in case_result.timings
        )
        max_steps = max(timing.max_sampling_integration_steps for timing in case_result.timings)
        print(
            f"- `{case_result.case.name}`: {timings}. "
            f"Warmup/sampling divergences={total_warmup_divergences}/"
            f"{total_sampling_divergences}; "
            f"mean sampling acceptance={mean_acceptance:.3f}; "
            f"max integration steps={max_steps}."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="local", help="label to include in output")
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=[],
        help="benchmark case name; repeat to select multiple cases; defaults to all",
    )
    parser.add_argument("--runs", type=int, default=None, help="override measured runs per case")
    parser.add_argument(
        "--cache-warmup-runs",
        type=int,
        default=1,
        help="unmeasured cached runs after the cold sample and before measured cached runs",
    )
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _add_repo_src()
    cases = _select_cases(cast(Sequence[str], args.cases))
    runs_override = cast(int | None, args.runs)
    cache_warmup_runs = cast(int, args.cache_warmup_runs)
    seed_start = cast(int, args.seed_start)
    if runs_override is not None and runs_override < 2:
        raise ValueError("--runs must be at least 2 when provided")
    if cache_warmup_runs < 0:
        raise ValueError("--cache-warmup-runs must be non-negative")

    case_results: list[CaseResult] = []
    for index, case in enumerate(cases):
        case_seed_start = seed_start + 10_000 * index
        case_results.append(
            _benchmark_case(
                case,
                runs_override=runs_override,
                cache_warmup_runs=cache_warmup_runs,
                seed_start=case_seed_start,
            )
        )

    result = BenchmarkResult(
        label=cast(str, args.label),
        branch=_git_value(("rev-parse", "--abbrev-ref", "HEAD")),
        commit=_git_value(("rev-parse", "--short", "HEAD")),
        runs_per_case=runs_override,
        cache_warmup_runs=cache_warmup_runs,
        cases=tuple(case_results),
    )
    _print_markdown(result)
    if args.json_output is not None:
        json_output = cast(Path, args.json_output)
        json_output.write_text(json.dumps(_benchmark_result_json(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
