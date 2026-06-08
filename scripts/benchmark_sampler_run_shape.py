#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Benchmark repeated compiled sampler runs for issue #13."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxstanv5.inference import NutsDiagnosticTrace, SamplerResult
    from jaxstanv5.model.bound import BoundModel


class BoundModelBuilder(Protocol):
    """Build one fixed-shape benchmark bound model."""

    def __call__(self) -> BoundModel:
        """Return a bound model."""
        ...


class ScalarBenchmarkModel(Protocol):
    """Decorated scalar benchmark model class."""

    def bind(self, *, y: jax.Array) -> BoundModel:
        """Bind observed data."""
        ...


class IndexedRegressionBenchmarkModel(Protocol):
    """Decorated indexed-regression benchmark model class."""

    def bind(
        self,
        *,
        n_groups: int,
        group_idx: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ) -> BoundModel:
        """Bind observed data."""
        ...


class VectorBenchmarkModel(Protocol):
    """Decorated vector-parameter benchmark model class."""

    def bind(self, *, n: int, y: jax.Array) -> BoundModel:
        """Bind observed data."""
        ...


@dataclass(frozen=True)
class BenchmarkCase:
    """One sampler runtime benchmark case."""

    name: str
    description: str
    num_chains: int
    num_warmup: int
    num_samples: int
    default_runs: int
    build_bound: BoundModelBuilder


@dataclass(frozen=True)
class RunTiming:
    """One measured sampler call."""

    seed: int
    seconds: float
    warmup_divergences: int
    sampling_divergences: int
    mean_sampling_acceptance: float
    max_sampling_integration_steps: int


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
    """Benchmark result for one case."""

    case: BenchmarkCase
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
    num_chains: int
    num_warmup: int
    num_samples: int
    compile_sampler_seconds: float
    cold_sample_seconds: float
    cached_stats: SummaryStatsJson
    timings: list[RunTimingJson]


class BenchmarkResultJson(TypedDict):
    """JSON object for the complete benchmark result."""

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


def _run_once(
    compiled_sampler: object, seed: int, case: BenchmarkCase
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


def _timing_from_result(result: SamplerResult, seed: int, seconds: float) -> RunTiming:
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
    case: BenchmarkCase,
    *,
    runs_override: int | None,
    cache_warmup_runs: int,
    seed_start: int,
) -> CaseResult:
    from jaxstanv5.inference import compile_sampler

    bound = case.build_bound()
    compile_start = time.perf_counter()
    compiled = compile_sampler(bound)
    compile_seconds = time.perf_counter() - compile_start

    cold_result, cold_seconds = _run_once(compiled, seed_start, case)
    _ = cold_result

    for offset in range(cache_warmup_runs):
        warm_seed = seed_start + 1 + offset
        warm_result, _warm_seconds = _run_once(compiled, warm_seed, case)
        _ = warm_result

    runs = runs_override if runs_override is not None else case.default_runs
    timings: list[RunTiming] = []
    first_measured_seed = seed_start + 1 + cache_warmup_runs
    for offset in range(runs):
        seed = first_measured_seed + offset
        result, seconds = _run_once(compiled, seed, case)
        timings.append(_timing_from_result(result, seed, seconds))

    return CaseResult(
        case=case,
        compile_sampler_seconds=compile_seconds,
        cold_sample_seconds=cold_seconds,
        cached_stats=_summary([timing.seconds for timing in timings]),
        timings=tuple(timings),
    )


def _build_scalar_normal_bound() -> BoundModel:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.distributions import Normal

    @model
    class ScalarNormalBenchmark:
        """Scalar normal posterior benchmark."""

        mu = Param(Normal(0.0, 1.0))
        y = Observed(Normal(mu, 1.0))

    bindable = cast(ScalarBenchmarkModel, ScalarNormalBenchmark)
    return bindable.bind(y=jnp.array(2.0))


def _build_indexed_regression_bound(*, n_groups: int, n_obs: int) -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import Normal

    @model
    class IndexedRegressionBenchmark:
        """Group-indexed regression benchmark without funnel geometry."""

        n_groups = Data.scalar()
        group_idx = Data.vector()
        x = Data.vector()

        alpha = Param(Normal(0.0, 1.0), size=n_groups)
        beta = Param(Normal(0.0, 1.0), size=n_groups)

        mu = alpha[group_idx] + beta[group_idx] * x
        y = Observed(Normal(mu, 0.5))

    group_idx = jnp.arange(n_obs, dtype=jnp.int32) % n_groups
    x = jnp.linspace(-1.0, 1.0, n_obs)
    group_float = group_idx.astype(jnp.float32)
    true_alpha = 0.15 * jnp.sin(group_float)
    true_beta = 0.5 + 0.05 * jnp.cos(group_float)
    y = true_alpha + true_beta * x + 0.1 * jnp.sin(jnp.arange(n_obs, dtype=jnp.float32))
    bindable = cast(IndexedRegressionBenchmarkModel, IndexedRegressionBenchmark)
    return bindable.bind(
        n_groups=n_groups,
        group_idx=group_idx,
        x=x,
        y=y,
    )


def _build_medium_indexed_regression_bound() -> BoundModel:
    return _build_indexed_regression_bound(n_groups=16, n_obs=256)


def _build_large_indexed_regression_bound() -> BoundModel:
    return _build_indexed_regression_bound(n_groups=64, n_obs=1024)


def _build_positive_vector_bound() -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Normal

    @model
    class PositiveVectorBenchmark:
        """Constrained vector benchmark for compiled output transforms."""

        n = Data.scalar()
        theta = Param(Normal(0.0, 1.0), constraint=Positive(), size=n)
        y = Observed(Normal(theta, 1.0))

    n = 64
    y = 1.0 + 0.1 * jnp.sin(jnp.arange(n, dtype=jnp.float32))
    bindable = cast(VectorBenchmarkModel, PositiveVectorBenchmark)
    return bindable.bind(n=n, y=y)


def _all_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            name="scalar_normal_4c_200w_500s",
            description="1D normal posterior; overhead-sensitive repeated sampling",
            num_chains=4,
            num_warmup=200,
            num_samples=500,
            default_runs=16,
            build_bound=_build_scalar_normal_bound,
        ),
        BenchmarkCase(
            name="indexed_regression_16g_256n_4c_250w_500s",
            description="32D group-indexed regression; medium likelihood workload",
            num_chains=4,
            num_warmup=250,
            num_samples=500,
            default_runs=12,
            build_bound=_build_medium_indexed_regression_bound,
        ),
        BenchmarkCase(
            name="positive_vector_64d_4c_300w_600s",
            description="64D positive-constrained vector; stresses compiled output transforms",
            num_chains=4,
            num_warmup=300,
            num_samples=600,
            default_runs=10,
            build_bound=_build_positive_vector_bound,
        ),
        BenchmarkCase(
            name="indexed_regression_64g_1024n_4c_300w_600s",
            description="128D group-indexed regression stress case",
            num_chains=4,
            num_warmup=300,
            num_samples=600,
            default_runs=8,
            build_bound=_build_large_indexed_regression_bound,
        ),
    )


def _select_cases(names: Sequence[str]) -> tuple[BenchmarkCase, ...]:
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
        "num_chains": result.case.num_chains,
        "num_warmup": result.case.num_warmup,
        "num_samples": result.case.num_samples,
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
    print(f"### Sampler run-shape benchmark: {result.label}")
    print()
    print(f"Branch: `{result.branch}`  ")
    print(f"Commit: `{result.commit}`  ")
    print(f"Cache warmup runs per case: {result.cache_warmup_runs}")
    print()
    print(
        "| case | chains | warmup | samples | cached runs | compile_sampler s | "
        "cold sample s | cached mean s | cached median s | cached sd s | "
        "cached 95% CI s | min s | max s |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case_result in result.cases:
        stats = case_result.cached_stats
        print(
            f"| {case_result.case.name} | {case_result.case.num_chains} | "
            f"{case_result.case.num_warmup} | {case_result.case.num_samples} | "
            f"{stats.count} | {_format_seconds(case_result.compile_sampler_seconds)} | "
            f"{_format_seconds(case_result.cold_sample_seconds)} | "
            f"{_format_seconds(stats.mean_seconds)} | {_format_seconds(stats.median_seconds)} | "
            f"{_format_seconds(stats.stdev_seconds)} | "
            f"±{_format_seconds(stats.ci95_half_width_seconds)} | "
            f"{_format_seconds(stats.min_seconds)} | {_format_seconds(stats.max_seconds)} |"
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
            f"Divergences warmup/sampling={total_warmup_divergences}/{total_sampling_divergences}; "
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
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="override measured cached runs per case",
    )
    parser.add_argument(
        "--cache-warmup-runs",
        type=int,
        default=1,
        help="unmeasured cached runs after the cold sample and before measured cached runs",
    )
    parser.add_argument("--seed-start", type=int, default=10_000)
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
