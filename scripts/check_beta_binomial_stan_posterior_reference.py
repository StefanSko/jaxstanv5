#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Compare hierarchical Beta-binomial logistic posterior summaries against Stan."""

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
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from jaxstanv5.inference import SamplerResult
    from jaxstanv5.model.bound import BoundModel


type FloatVector = tuple[float, ...]
type IntVector = tuple[int, ...]


@dataclass(frozen=True)
class BetaBinomialStanConfig:
    """Configuration for hierarchical Beta-binomial Stan posterior comparisons."""

    seed: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float
    target_acceptance_rate: float


@dataclass(frozen=True)
class BetaBinomialStanResult:
    """Printable result for one compared scalar parameter."""

    parameter: str
    signed_z: float
    k_min: float
    jaxstan_mean: float
    stan_mean: float
    combined_mcse: float
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


class StanDrawResult(NamedTuple):
    """Stan scalar samples."""

    samples: Mapping[str, jax.Array]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _load_json(path: Path) -> Mapping[str, object]:
    return cast(Mapping[str, object], json.loads(path.read_text()))


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    raise ValueError(f"{name} must be an integer")


def _int_sequence(value: object, *, name: str) -> IntVector:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_int(item, name=f"{name}[]") for item in value)


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{name} must be numeric")


def _float_sequence(value: object, *, name: str) -> FloatVector:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_float(item, name=f"{name}[]") for item in value)


def _cmdstan_model(stan_file: Path) -> StanPosteriorModel:
    module = cast(CmdStanPyModule, importlib.import_module("cmdstanpy"))
    return module.CmdStanModel(stan_file=str(stan_file))


def _build_bound(data: Mapping[str, object]) -> BoundModel:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import BetaBinomial, HalfNormal, Normal
    from jaxstanv5.math import exp, sigmoid
    from jaxstanv5.model import bind_model

    @model
    class HierarchicalBetaBinomialStanReferenceModel:
        """Hierarchical Beta-binomial model matching the Stan fixture."""

        n_groups = Data.scalar()
        group_idx = Data.vector()
        x = Data.vector()
        trials = Data.vector()

        alpha_pop = Param(Normal(0.0, 1.0))
        beta_pop = Param(Normal(0.0, 1.0))
        sigma_alpha = Param(HalfNormal(0.5), constraint=Positive())
        sigma_beta = Param(HalfNormal(0.5), constraint=Positive())
        log_concentration = Param(Normal(math.log(20.0), 0.5))
        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)
        z_beta = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        beta = beta_pop + sigma_beta * z_beta
        eta = alpha[group_idx] + beta[group_idx] * x
        p = sigmoid(eta)
        concentration = exp(log_concentration)
        a = p * concentration
        b = (1.0 - p) * concentration
        y = Observed(BetaBinomial(trials, a, b))

    n_groups = _as_int(data["G"], name="G")
    group_idx = jnp.array(_int_sequence(data["group_idx"], name="group_idx"), dtype=jnp.int32) - 1
    x = jnp.array(_float_sequence(data["x"], name="x"), dtype=jnp.float64)
    trials = jnp.array(_int_sequence(data["trials"], name="trials"), dtype=jnp.int32)
    y = jnp.array(_int_sequence(data["y"], name="y"), dtype=jnp.int32)
    return bind_model(
        HierarchicalBetaBinomialStanReferenceModel,
        dict(
            n_groups=n_groups,
            group_idx=group_idx,
            x=x,
            trials=trials,
            y=y,
        ),
    )


def _block_result(result: SamplerResult) -> None:
    for value in result.samples.values():
        value.block_until_ready()
    result.diagnostics.sampling.is_divergent.block_until_ready()


def _draw_column(draws: jax.Array, column_names: tuple[str, ...], *, name: str) -> jax.Array:
    if name not in column_names:
        raise ValueError(f"Missing Stan posterior draws for column: {name}")
    column_index = column_names.index(name)
    return draws[:, :, column_index].T


def _stan_draw_result(fit: StanFit, *, parameters: tuple[str, ...]) -> StanDrawResult:
    column_names = tuple(fit.column_names)
    draws = jnp.asarray(fit.draws(inc_warmup=False, concat_chains=False))
    return StanDrawResult(
        samples={name: _draw_column(draws, column_names, name=name) for name in parameters},
    )


def _run(config: BetaBinomialStanConfig) -> tuple[BetaBinomialStanResult, ...]:
    from integration._validation import compare_against_stan_reference, summarize_scalar_draws
    from jaxstanv5.inference import compile_sampler

    start = time.perf_counter()
    root = _repo_root()
    stan_root = root / "reference" / "stan"
    data_path = stan_root / "data" / "hierarchical_beta_binomial_logistic_varying_slopes.json"
    model_path = stan_root / "models" / "hierarchical_beta_binomial_logistic_varying_slopes.stan"
    parameters = ("alpha_pop", "beta_pop", "sigma_alpha", "sigma_beta", "log_concentration")

    bound = _build_bound(_load_json(data_path))
    compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
    jaxstan_result = compiled.sample(
        seed=config.seed,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_result(jaxstan_result)
    jaxstan_summaries = tuple(
        summarize_scalar_draws(jaxstan_result.samples, parameter=parameter)
        for parameter in parameters
    )

    stan_model = _cmdstan_model(model_path)
    with tempfile.TemporaryDirectory(prefix="hierarchical-beta-binomial-stan-") as output_dir:
        stan_fit = stan_model.sample(
            data=str(data_path),
            seed=config.seed,
            chains=config.num_chains,
            parallel_chains=config.num_chains,
            iter_warmup=config.num_warmup,
            iter_sampling=config.num_samples,
            show_progress=False,
            output_dir=output_dir,
            adapt_delta=config.target_acceptance_rate,
        )
        stan_draw_result = _stan_draw_result(stan_fit, parameters=parameters)
    stan_summaries = tuple(
        summarize_scalar_draws(stan_draw_result.samples, parameter=parameter)
        for parameter in parameters
    )

    comparisons = compare_against_stan_reference(
        jaxstan_summaries=jaxstan_summaries,
        stan_summaries=stan_summaries,
        max_k=config.max_k,
        max_rhat=config.max_rhat,
        min_ess=config.min_ess,
    )
    elapsed = time.perf_counter() - start
    stan_by_parameter = {summary.parameter: summary for summary in stan_summaries}
    jaxstan_by_parameter = {summary.parameter: summary for summary in jaxstan_summaries}
    return tuple(
        BetaBinomialStanResult(
            parameter=result.parameter,
            signed_z=result.signed_z,
            k_min=result.k_min,
            jaxstan_mean=jaxstan_by_parameter[result.parameter].mean,
            stan_mean=stan_by_parameter[result.parameter].mean,
            combined_mcse=result.mcse,
            elapsed_seconds=elapsed,
        )
        for result in comparisons
    )


def _parse_args() -> BetaBinomialStanConfig:
    parser = argparse.ArgumentParser(
        description="Compare hierarchical Beta-binomial logistic posteriors against Stan."
    )
    parser.add_argument("--seed", type=int, default=83_000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=800)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-k", type=float, default=5.0)
    parser.add_argument("--max-rhat", type=float, default=1.08)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    args = parser.parse_args()
    return BetaBinomialStanConfig(
        seed=args.seed,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_k=args.max_k,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        target_acceptance_rate=args.target_acceptance_rate,
    )


def _validate_config(config: BetaBinomialStanConfig) -> None:
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
    try:
        results = _run(config)
    except Exception as exc:  # noqa: BLE001 - optional reference script reports failures.
        print(f"FAIL error={type(exc).__name__}: {exc}")
        return 1

    for result in results:
        print(
            f"PASS parameter={result.parameter} signed_z={result.signed_z:.3f} "
            f"k_min={result.k_min:.3f} jaxstan_mean={result.jaxstan_mean:.6f} "
            f"stan_mean={result.stan_mean:.6f} combined_mcse={result.combined_mcse:.6f} "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
