#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Compare jaxstan posterior summaries against Stan fixed-data references."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from jaxstanv5.inference import NutsDiagnosticTrace, SamplerResult
    from jaxstanv5.model.bound import BoundModel


@dataclass(frozen=True)
class PosteriorConfig:
    """Configuration for Stan posterior comparisons."""

    seed: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float
    target_acceptance_rate: float


@dataclass(frozen=True)
class PosteriorCase:
    """One fixed-data posterior comparison case."""

    name: str
    parameter: str
    stan_model: Path
    stan_data: Path


@dataclass(frozen=True)
class PosteriorCaseResult:
    """Result for one posterior comparison case."""

    case_name: str
    parameter: str
    signed_z: float
    k_min: float
    jaxstan_mean: float
    stan_mean: float
    combined_mcse: float
    jaxstan_rhat: float
    stan_rhat: float
    jaxstan_ess: float
    stan_ess: float
    jaxstan_warmup_divergences: int
    jaxstan_sampling_divergences: int
    stan_sampling_divergences: int
    jaxstan_acceptance_rate_mean: float
    stan_acceptance_rate_mean: float


class StanDiagnosticTrace(NamedTuple):
    """Selected Stan sampler diagnostics."""

    is_divergent: jax.Array
    acceptance_rate: jax.Array


class StanDrawResult(NamedTuple):
    """Stan scalar samples and sampler diagnostics."""

    samples: Mapping[str, jax.Array]
    diagnostics: StanDiagnosticTrace


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
        PosteriorCase(
            name="exponential_rate",
            parameter="rate",
            stan_model=stan_root / "models" / "exponential_rate.stan",
            stan_data=stan_root / "data" / "exponential_rate.json",
        ),
        PosteriorCase(
            name="student_t_location",
            parameter="mu",
            stan_model=stan_root / "models" / "student_t_location.stan",
            stan_data=stan_root / "data" / "student_t_location.json",
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


def _build_exponential_rate_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Exponential, HalfNormal

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    prior_scale = _as_float(data["prior_scale"], name="prior_scale")

    @model
    class ExponentialRateStanReferenceModel:
        """Exponential-rate model matching the Stan fixture up to constants."""

        rate = Param(HalfNormal(prior_scale), constraint=Positive())
        y = Observed(Exponential(rate))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    return cast(BindableModel, ExponentialRateStanReferenceModel).bind(y=y)



def _build_student_t_location_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.distributions import Normal, StudentT

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    nu = _as_float(data["nu"], name="nu")
    prior_loc = _as_float(data["prior_loc"], name="prior_loc")
    prior_scale = _as_float(data["prior_scale"], name="prior_scale")
    obs_scale = _as_float(data["obs_scale"], name="obs_scale")

    @model
    class StudentTLocationStanReferenceModel:
        """Student-t location model matching the Stan fixture."""

        mu = Param(Normal(prior_loc, prior_scale))
        y = Observed(StudentT(nu, mu, obs_scale))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    return cast(BindableModel, StudentTLocationStanReferenceModel).bind(y=y)



def _build_bound(case: PosteriorCase, data: Mapping[str, object]) -> BoundModel:
    if case.name == "normal_known_scale":
        return _build_normal_known_scale_bound(data)
    if case.name == "positive_scale_normal":
        return _build_positive_scale_bound(data)
    if case.name == "exponential_rate":
        return _build_exponential_rate_bound(data)
    if case.name == "student_t_location":
        return _build_student_t_location_bound(data)
    raise ValueError(f"Unknown posterior case: {case.name}")


def _block_until_ready(samples: Mapping[str, jax.Array]) -> None:
    for value in samples.values():
        value.block_until_ready()


def _block_trace(trace: NutsDiagnosticTrace) -> None:
    trace.is_divergent.block_until_ready()
    trace.acceptance_rate.block_until_ready()
    trace.num_integration_steps.block_until_ready()
    trace.num_trajectory_expansions.block_until_ready()
    trace.energy.block_until_ready()


def _block_result(result: SamplerResult) -> None:
    _block_until_ready(result.samples)
    _block_trace(result.diagnostics.warmup)
    _block_trace(result.diagnostics.sampling)


def _draw_column(
    draws: jax.Array,
    column_names: tuple[str, ...],
    *,
    name: str,
) -> jax.Array:
    if name not in column_names:
        raise ValueError(f"Missing Stan posterior draws for column: {name}")
    column_index = column_names.index(name)
    return draws[:, :, column_index].T


def _stan_draw_result(fit: StanFit, *, parameter: str) -> StanDrawResult:
    column_names = tuple(fit.column_names)
    draws = jnp.asarray(fit.draws(inc_warmup=False, concat_chains=False))
    return StanDrawResult(
        samples={parameter: _draw_column(draws, column_names, name=parameter)},
        diagnostics=StanDiagnosticTrace(
            is_divergent=_draw_column(draws, column_names, name="divergent__").astype(bool),
            acceptance_rate=_draw_column(draws, column_names, name="accept_stat__"),
        ),
    )


def _run_case(case: PosteriorCase, config: PosteriorConfig, *, seed: int) -> PosteriorCaseResult:
    from integration._validation import (
        compare_against_stan_reference,
        summarize_scalar_draws,
    )
    from jaxstanv5.inference import compile_sampler

    data = _load_json(case.stan_data)
    bound = _build_bound(case, data)
    compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
    jaxstan_result = compiled.sample(
        seed=seed,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_result(jaxstan_result)
    jaxstan_summary = summarize_scalar_draws(jaxstan_result.samples, parameter=case.parameter)

    stan_model = _cmdstan_model(case.stan_model)
    with tempfile.TemporaryDirectory(prefix=f"{case.name}-stan-") as output_dir:
        stan_fit = stan_model.sample(
            data=str(case.stan_data),
            seed=seed,
            chains=config.num_chains,
            parallel_chains=config.num_chains,
            iter_warmup=config.num_warmup,
            iter_sampling=config.num_samples,
            show_progress=False,
            output_dir=output_dir,
            adapt_delta=config.target_acceptance_rate,
        )
        stan_draw_result = _stan_draw_result(stan_fit, parameter=case.parameter)
    stan_summary = summarize_scalar_draws(stan_draw_result.samples, parameter=case.parameter)

    results = compare_against_stan_reference(
        jaxstan_summaries=(jaxstan_summary,),
        stan_summaries=(stan_summary,),
        max_k=config.max_k,
        max_rhat=config.max_rhat,
        min_ess=config.min_ess,
    )
    result = results[0]
    return PosteriorCaseResult(
        case_name=case.name,
        parameter=case.parameter,
        signed_z=result.signed_z,
        k_min=result.k_min,
        jaxstan_mean=jaxstan_summary.mean,
        stan_mean=stan_summary.mean,
        combined_mcse=result.mcse,
        jaxstan_rhat=jaxstan_summary.rhat,
        stan_rhat=stan_summary.rhat,
        jaxstan_ess=jaxstan_summary.ess,
        stan_ess=stan_summary.ess,
        jaxstan_warmup_divergences=int(jnp.sum(jaxstan_result.diagnostics.warmup.is_divergent)),
        jaxstan_sampling_divergences=int(jnp.sum(jaxstan_result.diagnostics.sampling.is_divergent)),
        stan_sampling_divergences=int(jnp.sum(stan_draw_result.diagnostics.is_divergent)),
        jaxstan_acceptance_rate_mean=float(
            jnp.mean(jaxstan_result.diagnostics.sampling.acceptance_rate)
        ),
        stan_acceptance_rate_mean=float(jnp.mean(stan_draw_result.diagnostics.acceptance_rate)),
    )


def _parse_args() -> PosteriorConfig:
    parser = argparse.ArgumentParser(
        description="Compare jaxstan posterior summaries against Stan fixed-data references."
    )
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--max-k", type=float, default=4.0)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.95)
    args = parser.parse_args()
    return PosteriorConfig(
        seed=args.seed,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_k=args.max_k,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        target_acceptance_rate=args.target_acceptance_rate,
    )


def main() -> int:
    config = _parse_args()
    if config.num_chains < 1:
        raise ValueError("--num-chains must be at least 1")
    if config.num_warmup < 1:
        raise ValueError("--num-warmup must be at least 1")
    if config.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if config.max_k <= 0.0:
        raise ValueError("--max-k must be positive")
    if not 0.0 < config.target_acceptance_rate < 1.0:
        raise ValueError("--target-acceptance-rate must be in (0, 1)")

    _add_repo_paths()
    failures = 0
    start = time.perf_counter()
    for case_index, case in enumerate(_cases(_repo_root())):
        try:
            result = _run_case(case, config, seed=config.seed + case_index)
            print(
                f"PASS case={result.case_name} parameter={result.parameter} "
                f"z={result.signed_z:.3f} k={result.k_min:.3f} "
                f"jaxstan_mean={result.jaxstan_mean:.6f} stan_mean={result.stan_mean:.6f} "
                f"combined_mcse={result.combined_mcse:.6f} "
                f"jaxstan_rhat={result.jaxstan_rhat:.4f} stan_rhat={result.stan_rhat:.4f} "
                f"jaxstan_ess={result.jaxstan_ess:.1f} stan_ess={result.stan_ess:.1f} "
                f"jaxstan_warmup_divergences={result.jaxstan_warmup_divergences} "
                f"jaxstan_sampling_divergences={result.jaxstan_sampling_divergences} "
                f"stan_sampling_divergences={result.stan_sampling_divergences} "
                f"jaxstan_acceptance={result.jaxstan_acceptance_rate_mean:.3f} "
                f"stan_acceptance={result.stan_acceptance_rate_mean:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - this helper reports reference failures.
            failures += 1
            print(f"FAIL case={case.name} error={type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - start
    print(f"target_acceptance_rate={config.target_acceptance_rate:.3f}")
    print(f"elapsed={elapsed:.2f}s")
    if failures:
        print(f"failures={failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
