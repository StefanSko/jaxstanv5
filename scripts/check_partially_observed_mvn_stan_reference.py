#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Compare partially observed MVN imputation against an equivalent Stan model."""

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


type FloatVector = tuple[float, ...]
type FloatMatrix = tuple[FloatVector, ...]
type IntVector = tuple[int, ...]


@dataclass(frozen=True)
class PartialMvnConfig:
    """Configuration for the partial-observation Stan comparison."""

    seed: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float
    target_acceptance_rate: float
    stan_model: Path
    stan_data: Path


@dataclass(frozen=True)
class PartialMvnResult:
    """Comparison result for the missing MVN coordinate."""

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
    jaxstan_sampling_divergences: int
    stan_sampling_divergences: int


class StanDiagnosticTrace(NamedTuple):
    """Selected Stan sampler diagnostics."""

    is_divergent: jax.Array
    acceptance_rate: jax.Array


class StanDrawResult(NamedTuple):
    """Stan scalar draws and sampler diagnostics."""

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


def _int_sequence(value: object, *, name: str) -> IntVector:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_int(item, name=f"{name}[]") for item in value)


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


def _build_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Data, PartiallyObserved, model
    from jaxstanv5.distributions import MultivariateNormal

    @model
    class PartiallyObservedMvnStanReference:
        """Partially observed MVN matching the Stan reference model."""

        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        chol = Data.matrix(n, n)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)
        y = PartiallyObserved.vector(
            MultivariateNormal(0.0, chol),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    n = _as_int(data["N"], name="N")
    n_obs = _as_int(data["N_obs"], name="N_obs")
    n_mis = _as_int(data["N_mis"], name="N_mis")
    chol = jnp.asarray(_float_matrix(data["chol"], name="chol"), dtype=jnp.float64)
    observed_idx = jnp.asarray(_int_sequence(data["observed_idx"], name="observed_idx")) - 1
    missing_idx = jnp.asarray(_int_sequence(data["missing_idx"], name="missing_idx")) - 1
    observed_values = jnp.asarray(
        _float_sequence(data["observed_values"], name="observed_values"),
        dtype=jnp.float64,
    )
    return cast(BindableModel, PartiallyObservedMvnStanReference).bind(
        n=n,
        n_obs=n_obs,
        n_mis=n_mis,
        chol=chol,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        observed_values=observed_values,
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


def _run(config: PartialMvnConfig) -> PartialMvnResult:
    from integration._validation import compare_against_stan_reference, summarize_scalar_draws
    from jaxstanv5.inference import compile_sampler

    parameter = "y[1]"
    data = _load_json(config.stan_data)
    bound = _build_bound(data)
    compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
    jaxstan_result = compiled.sample(
        seed=config.seed,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_result(jaxstan_result)
    jaxstan_summary = summarize_scalar_draws(
        {parameter: jaxstan_result.samples["y"][..., 0]},
        parameter=parameter,
    )

    stan_model = _cmdstan_model(config.stan_model)
    with tempfile.TemporaryDirectory(prefix="partial-mvn-stan-") as output_dir:
        fit = stan_model.sample(
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
        stan_draws = _stan_draw_result(fit, parameter=parameter)
    stan_summary = summarize_scalar_draws(stan_draws.samples, parameter=parameter)

    comparison = compare_against_stan_reference(
        jaxstan_summaries=(jaxstan_summary,),
        stan_summaries=(stan_summary,),
        max_k=config.max_k,
        max_rhat=config.max_rhat,
        min_ess=config.min_ess,
    )[0]
    return PartialMvnResult(
        parameter=parameter,
        signed_z=comparison.signed_z,
        k_min=comparison.k_min,
        jaxstan_mean=jaxstan_summary.mean,
        stan_mean=stan_summary.mean,
        combined_mcse=comparison.mcse,
        jaxstan_rhat=jaxstan_summary.rhat,
        stan_rhat=stan_summary.rhat,
        jaxstan_ess=jaxstan_summary.ess,
        stan_ess=stan_summary.ess,
        jaxstan_sampling_divergences=int(jnp.sum(jaxstan_result.diagnostics.sampling.is_divergent)),
        stan_sampling_divergences=int(jnp.sum(stan_draws.diagnostics.is_divergent)),
    )


def _parse_args() -> PartialMvnConfig:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Compare partially observed MVN imputation against Stan."
    )
    parser.add_argument("--seed", type=int, default=42_000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--max-k", type=float, default=4.0)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    parser.add_argument(
        "--stan-model",
        type=Path,
        default=root / "reference" / "stan" / "models" / "partially_observed_mvn.stan",
    )
    parser.add_argument(
        "--stan-data",
        type=Path,
        default=root / "reference" / "stan" / "data" / "partially_observed_mvn.json",
    )
    args = parser.parse_args()
    return PartialMvnConfig(
        seed=args.seed,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_k=args.max_k,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        target_acceptance_rate=args.target_acceptance_rate,
        stan_model=args.stan_model,
        stan_data=args.stan_data,
    )


def main() -> int:
    _add_repo_paths()
    config = _parse_args()
    start = time.time()
    result = _run(config)
    elapsed = time.time() - start
    print(
        f"PASS parameter={result.parameter} signed_z={result.signed_z:.3f} "
        f"k_min={result.k_min:.3f} jaxstan_mean={result.jaxstan_mean:.6f} "
        f"stan_mean={result.stan_mean:.6f} combined_mcse={result.combined_mcse:.6f} "
        f"jaxstan_rhat={result.jaxstan_rhat:.4f} stan_rhat={result.stan_rhat:.4f} "
        f"jaxstan_ess={result.jaxstan_ess:.1f} stan_ess={result.stan_ess:.1f} "
        f"jaxstan_sampling_divergences={result.jaxstan_sampling_divergences} "
        f"stan_sampling_divergences={result.stan_sampling_divergences} elapsed={elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
