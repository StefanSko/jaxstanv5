#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Compare projected fixed-kernel GP posterior summaries against Stan."""

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
    from integration._validation import ProjectionSpec
    from jaxstanv5.inference import NutsDiagnosticTrace, SamplerResult
    from jaxstanv5.model.bound import BoundModel


type FloatVector = tuple[float, ...]
type FloatMatrix = tuple[FloatVector, ...]


@dataclass(frozen=True)
class GpStanConfig:
    """Configuration for GP Stan posterior projection comparisons."""

    seed: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float
    target_acceptance_rate: float
    stan_data: Path


@dataclass(frozen=True)
class GpStanProjectionResult:
    """Comparison result for one GP projection."""

    projection_name: str
    signed_z: float
    k_min: float
    jaxstan_mean: float
    stan_mean: float
    combined_mcse: float
    jaxstan_rhat: float
    stan_rhat: float
    jaxstan_ess: float
    stan_ess: float


class StanDiagnosticTrace(NamedTuple):
    """Selected Stan sampler diagnostics."""

    is_divergent: jax.Array
    acceptance_rate: jax.Array


class StanVectorDrawResult(NamedTuple):
    """Stan vector samples and sampler diagnostics."""

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


class BindableModel(Protocol):
    """Runtime model class with decorator-attached bind method."""

    def bind(self, **values: object) -> BoundModel:
        """Bind concrete model data."""
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


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    raise ValueError(f"{name} must be an integer")


def _float_sequence(value: object, *, name: str) -> FloatVector:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_float(item, name=f"{name}[]") for item in value)


def _float_matrix(value: object, *, name: str) -> FloatMatrix:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON matrix")
    rows: list[FloatVector] = []
    width: int | None = None
    for row_index, row_value in enumerate(value):
        row = _float_sequence(row_value, name=f"{name}[{row_index}]")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} rows must have equal length")
        rows.append(row)
    return tuple(rows)


def _cmdstan_model(stan_file: Path) -> StanPosteriorModel:
    module = cast(CmdStanPyModule, importlib.import_module("cmdstanpy"))
    return module.CmdStanModel(stan_file=str(stan_file))


def _projection_specs(n: int) -> tuple[ProjectionSpec, ...]:
    from integration._validation import ProjectionSpec

    first = jnp.zeros((n,), dtype=jnp.float64).at[0].set(1.0)
    middle = jnp.zeros((n,), dtype=jnp.float64).at[n // 2].set(1.0)
    mean = jnp.ones((n,), dtype=jnp.float64) / n
    contrast = jnp.zeros((n,), dtype=jnp.float64).at[0].set(-1.0).at[n - 1].set(1.0)
    return (
        ProjectionSpec(name="f_first", parameter="f", weights=first),
        ProjectionSpec(name="f_middle", parameter="f", weights=middle),
        ProjectionSpec(name="f_mean", parameter="f", weights=mean),
        ProjectionSpec(name="f_last_minus_first", parameter="f", weights=contrast),
    )


def _build_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import MultivariateNormal, Normal

    @model
    class FixedKernelGpStanProjectionModel:
        """Fixed-kernel GP model matching the Stan fixture."""

        n = Data.scalar()
        chol = Data.matrix(n, n)
        obs_sd = Data.scalar()

        f = Param(MultivariateNormal(0.0, chol), size=n)
        y = Observed(Normal(f, obs_sd))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    chol = jnp.array(_float_matrix(data["chol"], name="chol"), dtype=jnp.float64)
    obs_sd = _as_float(data["obs_sd"], name="obs_sd")
    n = _as_int(data["N"], name="N")
    return cast(BindableModel, FixedKernelGpStanProjectionModel).bind(
        n=n, chol=chol, obs_sd=obs_sd, y=y
    )


def _block_trace(trace: NutsDiagnosticTrace) -> None:
    trace.is_divergent.block_until_ready()
    trace.acceptance_rate.block_until_ready()
    trace.num_integration_steps.block_until_ready()
    trace.num_trajectory_expansions.block_until_ready()
    trace.energy.block_until_ready()


def _block_result(result: SamplerResult) -> None:
    for value in result.samples.values():
        value.block_until_ready()
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


def _stan_vector_column_name(column_names: tuple[str, ...], *, parameter: str, index: int) -> str:
    bracket_name = f"{parameter}[{index + 1}]"
    dotted_name = f"{parameter}.{index + 1}"
    if bracket_name in column_names:
        return bracket_name
    if dotted_name in column_names:
        return dotted_name
    raise ValueError(f"Missing Stan posterior draws for vector column: {bracket_name}")


def _stan_draw_result(fit: StanFit, *, parameter: str, dimension: int) -> StanVectorDrawResult:
    column_names = tuple(fit.column_names)
    draws = jnp.asarray(fit.draws(inc_warmup=False, concat_chains=False))
    components = tuple(
        _draw_column(
            draws,
            column_names,
            name=_stan_vector_column_name(column_names, parameter=parameter, index=index),
        )
        for index in range(dimension)
    )
    return StanVectorDrawResult(
        samples={parameter: jnp.stack(components, axis=-1)},
        diagnostics=StanDiagnosticTrace(
            is_divergent=_draw_column(draws, column_names, name="divergent__").astype(bool),
            acceptance_rate=_draw_column(draws, column_names, name="accept_stat__"),
        ),
    )


def _run(config: GpStanConfig) -> tuple[GpStanProjectionResult, ...]:
    from integration._validation import (
        compare_against_stan_reference,
        summarize_projected_draws,
    )
    from jaxstanv5.inference import compile_sampler

    data = _load_json(config.stan_data)
    dimension = len(_float_sequence(data["y"], name="y"))
    projections = _projection_specs(dimension)
    bound = _build_bound(data)
    compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
    jaxstan_result = compiled.sample(
        seed=config.seed,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_result(jaxstan_result)

    stan_model = _cmdstan_model(
        _repo_root() / "reference" / "stan" / "models" / "fixed_kernel_gp.stan"
    )
    with tempfile.TemporaryDirectory(prefix="fixed-kernel-gp-stan-") as output_dir:
        stan_fit = stan_model.sample(
            data=str(config.stan_data),
            seed=config.seed,
            chains=config.num_chains,
            parallel_chains=config.num_chains,
            iter_warmup=config.num_warmup,
            iter_sampling=config.num_samples,
            show_progress=False,
            output_dir=output_dir,
            adapt_delta=config.target_acceptance_rate,
        )
        stan_draw_result = _stan_draw_result(stan_fit, parameter="f", dimension=dimension)

    jaxstan_summaries = tuple(
        summarize_projected_draws(jaxstan_result.samples, projection=projection)
        for projection in projections
    )
    stan_summaries = tuple(
        summarize_projected_draws(stan_draw_result.samples, projection=projection)
        for projection in projections
    )
    comparisons = compare_against_stan_reference(
        jaxstan_summaries=jaxstan_summaries,
        stan_summaries=stan_summaries,
        max_k=config.max_k,
        max_rhat=config.max_rhat,
        min_ess=config.min_ess,
    )
    stan_by_projection = {summary.parameter: summary for summary in stan_summaries}
    jaxstan_by_projection = {summary.parameter: summary for summary in jaxstan_summaries}
    return tuple(
        GpStanProjectionResult(
            projection_name=comparison.parameter,
            signed_z=comparison.signed_z,
            k_min=comparison.k_min,
            jaxstan_mean=jaxstan_by_projection[comparison.parameter].mean,
            stan_mean=stan_by_projection[comparison.parameter].mean,
            combined_mcse=comparison.mcse,
            jaxstan_rhat=jaxstan_by_projection[comparison.parameter].rhat,
            stan_rhat=stan_by_projection[comparison.parameter].rhat,
            jaxstan_ess=jaxstan_by_projection[comparison.parameter].ess,
            stan_ess=stan_by_projection[comparison.parameter].ess,
        )
        for comparison in comparisons
    )


def _parse_args() -> GpStanConfig:
    parser = argparse.ArgumentParser(
        description="Compare projected fixed-kernel GP posteriors against Stan."
    )
    parser.add_argument("--seed", type=int, default=80_000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--max-k", type=float, default=4.0)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    parser.add_argument(
        "--stan-data",
        type=Path,
        default=_repo_root() / "reference" / "stan" / "data" / "fixed_kernel_gp_n8.json",
    )
    args = parser.parse_args()
    return GpStanConfig(
        seed=args.seed,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_k=args.max_k,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        target_acceptance_rate=args.target_acceptance_rate,
        stan_data=args.stan_data,
    )


def _validate_config(config: GpStanConfig) -> None:
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
    if not 0.0 < config.target_acceptance_rate < 1.0:
        raise ValueError("--target-acceptance-rate must be in (0, 1)")


def main() -> int:
    config = _parse_args()
    _validate_config(config)
    _add_repo_paths()
    start = time.perf_counter()
    try:
        results = _run(config)
    except Exception as exc:  # noqa: BLE001 - script reports reference failures.
        print(f"FAIL error={type(exc).__name__}: {exc}")
        return 1

    for result in results:
        print(
            f"PASS projection={result.projection_name} "
            f"z={result.signed_z:.3f} k={result.k_min:.3f} "
            f"jaxstan_mean={result.jaxstan_mean:.6f} "
            f"stan_mean={result.stan_mean:.6f} "
            f"combined_mcse={result.combined_mcse:.6f} "
            f"jaxstan_rhat={result.jaxstan_rhat:.4f} "
            f"stan_rhat={result.stan_rhat:.4f} "
            f"jaxstan_ess={result.jaxstan_ess:.1f} "
            f"stan_ess={result.stan_ess:.1f}"
        )
    print(f"target_acceptance_rate={config.target_acceptance_rate:.3f}")
    print(f"elapsed={time.perf_counter() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
