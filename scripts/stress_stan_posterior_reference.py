#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Stress-test Stan posterior reference comparisons over many seeds."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from jaxstanv5.inference import CompiledSampler
    from jaxstanv5.model.bound import BoundModel


@dataclass(frozen=True)
class StressConfig:
    """Configuration for repeated Stan posterior comparisons."""

    runs: int
    seed_start: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float
    stan_adapt_delta: float


@dataclass(frozen=True)
class PosteriorCase:
    """One fixed-data posterior comparison case."""

    name: str
    parameter: str
    stan_model: Path
    stan_data: Path


@dataclass(frozen=True)
class CaseRuntime:
    """Prepared runtime objects for one posterior case."""

    case: PosteriorCase
    compiled_sampler: CompiledSampler
    stan_model: StanPosteriorModel


@dataclass(frozen=True)
class CaseRunSuccess:
    """Successful result for one case and seed."""

    case_name: str
    parameter: str
    seed: int
    signed_z: float
    k_min: float
    jaxstan_mean: float
    stan_mean: float
    combined_mcse: float
    jaxstan_rhat: float
    stan_rhat: float
    jaxstan_ess: float
    stan_ess: float
    jaxstan_sampling_seconds: float
    stan_sampling_seconds: float


@dataclass(frozen=True)
class CaseRunFailure:
    """Failed result for one case and seed."""

    case_name: str
    parameter: str
    seed: int
    reason: str
    jaxstan_sampling_seconds: float
    stan_sampling_seconds: float


@dataclass(frozen=True)
class StressResult:
    """Aggregate stress-test result."""

    successes: tuple[CaseRunSuccess, ...]
    failures: tuple[CaseRunFailure, ...]
    elapsed_seconds: float


class StanFit(Protocol):
    """Minimal CmdStanMCMC protocol needed by this script."""

    @property
    def column_names(self) -> Sequence[str]:
        """Draw column names."""
        ...

    def draws(self, *, inc_warmup: bool, concat_chains: bool) -> object:
        """Return posterior draws."""
        ...


class StanPosteriorModel(Protocol):
    """Minimal CmdStanModel protocol needed for sampling."""

    def sample(
        self,
        *,
        data: str,
        seed: int,
        chains: int,
        parallel_chains: int,
        iter_warmup: int,
        iter_sampling: int,
        show_progress: bool,
        output_dir: str,
        adapt_delta: float,
    ) -> StanFit:
        """Run Stan NUTS."""
        ...


class StanModelFactory(Protocol):
    """Callable constructor protocol for CmdStanModel."""

    def __call__(self, *, stan_file: str) -> StanPosteriorModel:
        """Construct a Stan model."""
        ...


class CmdStanPyModule(Protocol):
    """Minimal cmdstanpy module protocol."""

    @property
    def CmdStanModel(self) -> StanModelFactory:
        """CmdStanModel constructor."""
        ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _load_json(path: Path) -> Mapping[str, object]:
    return cast(Mapping[str, object], json.loads(path.read_text()))


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{name} must be numeric")


def _float_sequence(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_float(item, name=f"{name}[]") for item in value)


def _cmdstan_model(stan_file: Path) -> StanPosteriorModel:
    module = cast(CmdStanPyModule, importlib.import_module("cmdstanpy"))
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    return module.CmdStanModel(stan_file=str(stan_file))


def _cases(root: Path) -> tuple[PosteriorCase, ...]:
    stan_root = root / "reference" / "stan"
    return (
        PosteriorCase(
            name="normal_known_scale",
            parameter="mu",
            stan_model=stan_root / "models" / "normal_known_scale.stan",
            stan_data=stan_root / "data" / "normal_known_scale.json",
        ),
        PosteriorCase(
            name="positive_scale_normal",
            parameter="sigma",
            stan_model=stan_root / "models" / "positive_scale_normal.stan",
            stan_data=stan_root / "data" / "positive_scale_normal.json",
        ),
    )


def _build_normal_known_scale_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.distributions import Normal

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    prior_loc = _as_float(data["prior_loc"], name="prior_loc")
    prior_scale = _as_float(data["prior_scale"], name="prior_scale")
    obs_scale = _as_float(data["obs_scale"], name="obs_scale")

    @model
    class NormalKnownScaleStanReferenceModel:
        """Scalar Normal model matching the Stan fixture."""

        mu = Param(Normal(prior_loc, prior_scale))
        y = Observed(Normal(mu, obs_scale))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    return cast(BindableModel, NormalKnownScaleStanReferenceModel).bind(y=y)


def _build_positive_scale_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Normal

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    prior_loc = _as_float(data["prior_loc"], name="prior_loc")
    prior_scale = _as_float(data["prior_scale"], name="prior_scale")

    @model
    class PositiveScaleStanReferenceModel:
        """Positive-scale Normal model matching the Stan fixture."""

        sigma = Param(Normal(prior_loc, prior_scale), constraint=Positive())
        y = Observed(Normal(0.0, sigma))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    return cast(BindableModel, PositiveScaleStanReferenceModel).bind(y=y)


def _build_bound(case: PosteriorCase, data: Mapping[str, object]) -> BoundModel:
    if case.name == "normal_known_scale":
        return _build_normal_known_scale_bound(data)
    if case.name == "positive_scale_normal":
        return _build_positive_scale_bound(data)
    raise ValueError(f"Unknown posterior case: {case.name}")


def _prepare_runtime(case: PosteriorCase) -> CaseRuntime:
    from jaxstanv5.inference import compile_sampler

    data = _load_json(case.stan_data)
    bound = _build_bound(case, data)
    return CaseRuntime(
        case=case,
        compiled_sampler=compile_sampler(bound),
        stan_model=_cmdstan_model(case.stan_model),
    )


def _block_until_ready(samples: Mapping[str, jax.Array]) -> None:
    for value in samples.values():
        value.block_until_ready()


def _stan_scalar_samples(fit: StanFit, *, parameter: str) -> Mapping[str, jax.Array]:
    column_names = tuple(fit.column_names)
    if parameter not in column_names:
        raise ValueError(f"Missing Stan posterior draws for parameter: {parameter}")
    column_index = column_names.index(parameter)
    draws = jnp.asarray(fit.draws(inc_warmup=False, concat_chains=False))
    parameter_draws = draws[:, :, column_index].T
    return {parameter: jnp.asarray(parameter_draws)}


def _warm_jaxstan(runtimes: tuple[CaseRuntime, ...], config: StressConfig) -> None:
    for case_index, runtime in enumerate(runtimes):
        result = runtime.compiled_sampler.sample(
            seed=config.seed_start - len(runtimes) + case_index,
            num_chains=config.num_chains,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
        )
        _block_until_ready(result.samples)


def _run_case(
    runtime: CaseRuntime,
    config: StressConfig,
    *,
    seed: int,
) -> CaseRunSuccess:
    from integration._validation import (
        compare_against_stan_reference,
        summarize_scalar_draws,
    )

    case = runtime.case

    jaxstan_start = time.perf_counter()
    jaxstan_result = runtime.compiled_sampler.sample(
        seed=seed,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_until_ready(jaxstan_result.samples)
    jaxstan_sampling_seconds = time.perf_counter() - jaxstan_start
    jaxstan_summary = summarize_scalar_draws(jaxstan_result.samples, parameter=case.parameter)

    with tempfile.TemporaryDirectory(prefix=f"{case.name}-stan-") as output_dir:
        stan_start = time.perf_counter()
        stan_fit = runtime.stan_model.sample(
            data=str(case.stan_data),
            seed=seed,
            chains=config.num_chains,
            parallel_chains=config.num_chains,
            iter_warmup=config.num_warmup,
            iter_sampling=config.num_samples,
            show_progress=False,
            output_dir=output_dir,
            adapt_delta=config.stan_adapt_delta,
        )
        stan_sampling_seconds = time.perf_counter() - stan_start
        stan_samples = _stan_scalar_samples(stan_fit, parameter=case.parameter)
    stan_summary = summarize_scalar_draws(stan_samples, parameter=case.parameter)

    results = compare_against_stan_reference(
        jaxstan_summaries=(jaxstan_summary,),
        stan_summaries=(stan_summary,),
        max_k=config.max_k,
        max_rhat=config.max_rhat,
        min_ess=config.min_ess,
    )
    comparison = results[0]
    return CaseRunSuccess(
        case_name=case.name,
        parameter=case.parameter,
        seed=seed,
        signed_z=comparison.signed_z,
        k_min=comparison.k_min,
        jaxstan_mean=jaxstan_summary.mean,
        stan_mean=stan_summary.mean,
        combined_mcse=comparison.mcse,
        jaxstan_rhat=jaxstan_summary.rhat,
        stan_rhat=stan_summary.rhat,
        jaxstan_ess=jaxstan_summary.ess,
        stan_ess=stan_summary.ess,
        jaxstan_sampling_seconds=jaxstan_sampling_seconds,
        stan_sampling_seconds=stan_sampling_seconds,
    )


def _run_stress(config: StressConfig) -> StressResult:
    runtimes = tuple(_prepare_runtime(case) for case in _cases(_repo_root()))
    _warm_jaxstan(runtimes, config)

    successes: list[CaseRunSuccess] = []
    failures: list[CaseRunFailure] = []
    start = time.perf_counter()
    for run_index in range(config.runs):
        for case_index, runtime in enumerate(runtimes):
            seed = config.seed_start + run_index * len(runtimes) + case_index
            try:
                successes.append(_run_case(runtime, config, seed=seed))
            except Exception as exc:  # noqa: BLE001 - this helper reports stress failures.
                failures.append(
                    CaseRunFailure(
                        case_name=runtime.case.name,
                        parameter=runtime.case.parameter,
                        seed=seed,
                        reason=type(exc).__name__,
                        jaxstan_sampling_seconds=math.nan,
                        stan_sampling_seconds=math.nan,
                    )
                )
                print(
                    f"FAIL case={runtime.case.name} seed={seed} error={type(exc).__name__}: {exc}"
                )

    return StressResult(
        successes=tuple(successes),
        failures=tuple(failures),
        elapsed_seconds=time.perf_counter() - start,
    )


def _values_by_case(
    successes: tuple[CaseRunSuccess, ...],
    *,
    case_name: str,
    selector: str,
) -> tuple[float, ...]:
    return tuple(
        float(getattr(success, selector)) for success in successes if success.case_name == case_name
    )


def _print_distribution(name: str, values: tuple[float, ...]) -> None:
    if len(values) == 0:
        print(f"  {name}: no successful values")
        return

    array = jnp.asarray(values)
    print(
        f"  {name}: "
        f"mean={float(jnp.mean(array)):.3f} "
        f"sd={float(jnp.std(array, ddof=1)):.3f} "
        f"min={float(jnp.min(array)):.3f} "
        f"p95={float(jnp.quantile(array, 0.95)):.3f} "
        f"max={float(jnp.max(array)):.3f}"
    )


def _print_timing(
    successes: tuple[CaseRunSuccess, ...],
    *,
    case_name: str,
) -> None:
    jaxstan_times = _values_by_case(
        successes,
        case_name=case_name,
        selector="jaxstan_sampling_seconds",
    )
    stan_times = _values_by_case(
        successes,
        case_name=case_name,
        selector="stan_sampling_seconds",
    )
    _print_distribution("jaxstan_sampling_seconds", jaxstan_times)
    _print_distribution("stan_sampling_seconds", stan_times)
    if len(jaxstan_times) > 0 and len(stan_times) > 0:
        jaxstan_mean = float(jnp.mean(jnp.asarray(jaxstan_times)))
        stan_mean = float(jnp.mean(jnp.asarray(stan_times)))
        print(f"  stan_to_jaxstan_time_ratio_mean={stan_mean / jaxstan_mean:.2f}")


def _print_result(config: StressConfig, result: StressResult) -> None:
    total = len(result.successes) + len(result.failures)
    print(f"case_runs={total} passes={len(result.successes)} failures={len(result.failures)}")
    print(
        f"runs_per_case={config.runs} thresholds: max_k={config.max_k:.2f} "
        f"max_rhat={config.max_rhat:.3f} min_ess={config.min_ess:.1f} "
        f"stan_adapt_delta={config.stan_adapt_delta:.3f}"
    )
    print(
        f"elapsed={result.elapsed_seconds:.2f}s avg={result.elapsed_seconds / total:.3f}s/case-run"
    )

    case_names = tuple(dict.fromkeys(success.case_name for success in result.successes))
    for case_name in case_names:
        case_successes = tuple(
            success for success in result.successes if success.case_name == case_name
        )
        case_failures = tuple(
            failure for failure in result.failures if failure.case_name == case_name
        )
        print(f"case={case_name} passes={len(case_successes)} failures={len(case_failures)}")
        _print_distribution(
            "z",
            _values_by_case(result.successes, case_name=case_name, selector="signed_z"),
        )
        _print_distribution(
            "k",
            _values_by_case(result.successes, case_name=case_name, selector="k_min"),
        )
        _print_distribution(
            "jaxstan_rhat",
            _values_by_case(result.successes, case_name=case_name, selector="jaxstan_rhat"),
        )
        _print_distribution(
            "stan_rhat",
            _values_by_case(result.successes, case_name=case_name, selector="stan_rhat"),
        )
        _print_distribution(
            "jaxstan_ess",
            _values_by_case(result.successes, case_name=case_name, selector="jaxstan_ess"),
        )
        _print_distribution(
            "stan_ess",
            _values_by_case(result.successes, case_name=case_name, selector="stan_ess"),
        )
        _print_timing(result.successes, case_name=case_name)

    if result.failures:
        print("first_failures:")
        for failure in result.failures[:20]:
            print(
                f"  case={failure.case_name} seed={failure.seed} reason={failure.reason} "
                f"jaxstan_seconds={failure.jaxstan_sampling_seconds:.3f} "
                f"stan_seconds={failure.stan_sampling_seconds:.3f}"
            )


def _parse_args() -> StressConfig:
    parser = argparse.ArgumentParser(
        description="Stress-test jaxstan posterior summaries against Stan references."
    )
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=40_000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--max-k", type=float, default=4.0)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--stan-adapt-delta", type=float, default=0.95)
    args = parser.parse_args()
    return StressConfig(
        runs=args.runs,
        seed_start=args.seed_start,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_k=args.max_k,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        stan_adapt_delta=args.stan_adapt_delta,
    )


def main() -> int:
    config = _parse_args()
    if config.runs < 1:
        raise ValueError("--runs must be at least 1")
    if config.num_chains < 1:
        raise ValueError("--num-chains must be at least 1")
    if config.num_warmup < 1:
        raise ValueError("--num-warmup must be at least 1")
    if config.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if config.max_k <= 0.0:
        raise ValueError("--max-k must be positive")
    if config.max_rhat <= 0.0:
        raise ValueError("--max-rhat must be positive")
    if config.min_ess <= 0.0:
        raise ValueError("--min-ess must be positive")
    if not 0.0 < config.stan_adapt_delta < 1.0:
        raise ValueError("--stan-adapt-delta must be in (0, 1)")

    _add_repo_paths()
    result = _run_stress(config)
    _print_result(config, result)
    if result.failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
