#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Run SBC for the hierarchical Poisson varying-slopes model."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PoissonSbcConfig:
    """Configuration for hierarchical Poisson SBC checks."""

    seed: int
    n_groups: int
    observations_per_group: int
    num_simulations: int
    num_chains: int
    num_warmup: int
    num_samples: int
    num_rank_bins: int
    max_abs_bin_z: float
    max_abs_mean_rank_z: float
    target_acceptance_rate: float


@dataclass(frozen=True)
class PoissonSbcResult:
    """Printable SBC result for one scalar parameter."""

    parameter: str
    num_simulations: int
    num_posterior_draws: int
    bin_counts: tuple[int, ...]
    max_abs_bin_z: float
    mean_rank_z: float
    elapsed_seconds: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _build_model() -> object:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import HalfNormal, Normal, Poisson
    from jaxstanv5.math import exp

    @model
    class HierarchicalPoissonSbcModel:
        """Hierarchical Poisson varying-slopes SBC model."""

        n_groups = Data.scalar()
        group_idx = Data.vector()
        x = Data.vector()

        alpha_pop = Param(Normal(0.0, 0.5))
        beta_pop = Param(Normal(0.0, 0.5))
        sigma_alpha = Param(HalfNormal(0.4), constraint=Positive())
        sigma_beta = Param(HalfNormal(0.4), constraint=Positive())
        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)
        z_beta = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        beta = beta_pop + sigma_beta * z_beta
        eta = alpha[group_idx] + beta[group_idx] * x
        y = Observed(Poisson(exp(eta)))

    return HierarchicalPoissonSbcModel


def _data_values(config: PoissonSbcConfig) -> Mapping[str, object]:
    group_idx = jnp.repeat(jnp.arange(config.n_groups), config.observations_per_group)
    x = jnp.tile(
        jnp.linspace(-1.0, 1.0, config.observations_per_group, dtype=jnp.float64),
        config.n_groups,
    )
    return {"n_groups": config.n_groups, "group_idx": group_idx, "x": x}


def _run_sbc(config: PoissonSbcConfig) -> tuple[PoissonSbcResult, ...]:
    from integration._validation import assert_sbc_rank_uniformity
    from jaxstanv5.inference import compile_sampler
    from jaxstanv5.model import bind_model
    from jaxstanv5.simulation import simulate_prior_predictive
    from jaxstanv5.validation import SbcValidationResult, scalar_sbc_rank

    start = time.perf_counter()
    parameters = ("alpha_pop", "beta_pop", "sigma_alpha", "sigma_beta")
    model_cls = _build_model()
    data = _data_values(config)
    prior_predictive = simulate_prior_predictive(
        model_cls,
        seed=config.seed,
        num_samples=config.num_simulations,
        data=data,
    )
    truth_draws = {name: jnp.asarray(prior_predictive.parameters[name]) for name in parameters}
    y_draws = jnp.asarray(prior_predictive.observed["y"])
    ranks: dict[str, list[int]] = {name: [] for name in parameters}
    num_posterior_draws: int | None = None
    bindable = model_cls

    for simulation_index in range(config.num_simulations):
        bound = bind_model(bindable, dict(**prior_predictive.data, y=y_draws[simulation_index]))
        compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
        result = compiled.sample(
            seed=config.seed + 10_000 + simulation_index,
            num_chains=config.num_chains,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
        )
        for value in result.samples.values():
            value.block_until_ready()

        posterior_draw_count = int(config.num_chains * config.num_samples)
        if num_posterior_draws is None:
            num_posterior_draws = posterior_draw_count
        elif posterior_draw_count != num_posterior_draws:
            raise ValueError("All SBC simulations must use the same number of posterior draws")

        for parameter in parameters:
            rank = scalar_sbc_rank(
                result.samples,
                parameter=parameter,
                true_value=float(truth_draws[parameter][simulation_index]),
            )
            ranks[parameter].append(rank)
            print(
                f"DRAW simulation={simulation_index + 1}/{config.num_simulations} "
                f"parameter={parameter} true={float(truth_draws[parameter][simulation_index]):.6f} "
                f"rank={rank}"
            )

    if num_posterior_draws is None:
        raise ValueError("At least one posterior draw is required")

    elapsed = time.perf_counter() - start
    summaries: list[PoissonSbcResult] = []
    for parameter in parameters:
        validation_result = SbcValidationResult(
            parameter=parameter,
            ranks=tuple(ranks[parameter]),
            num_posterior_draws=num_posterior_draws,
        )
        uniformity = assert_sbc_rank_uniformity(
            validation_result,
            num_rank_bins=config.num_rank_bins,
            max_abs_bin_z=config.max_abs_bin_z,
            max_abs_mean_rank_z=config.max_abs_mean_rank_z,
        )
        summaries.append(
            PoissonSbcResult(
                parameter=parameter,
                num_simulations=len(validation_result.ranks),
                num_posterior_draws=validation_result.num_posterior_draws,
                bin_counts=uniformity.bin_counts,
                max_abs_bin_z=uniformity.max_abs_bin_z,
                mean_rank_z=uniformity.mean_rank_z,
                elapsed_seconds=elapsed,
            )
        )
    return tuple(summaries)


def _parse_args() -> PoissonSbcConfig:
    parser = argparse.ArgumentParser(
        description="Run hierarchical Poisson varying-slopes SBC checks."
    )
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--n-groups", type=int, default=4)
    parser.add_argument("--observations-per-group", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--num-rank-bins", type=int, default=5)
    parser.add_argument("--max-abs-bin-z", type=float, default=4.0)
    parser.add_argument("--max-abs-mean-rank-z", type=float, default=4.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    args = parser.parse_args()
    return PoissonSbcConfig(
        seed=args.seed,
        n_groups=args.n_groups,
        observations_per_group=args.observations_per_group,
        num_simulations=args.num_simulations,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_rank_bins=args.num_rank_bins,
        max_abs_bin_z=args.max_abs_bin_z,
        max_abs_mean_rank_z=args.max_abs_mean_rank_z,
        target_acceptance_rate=args.target_acceptance_rate,
    )


def _validate_config(config: PoissonSbcConfig) -> None:
    if config.n_groups < 1:
        raise ValueError("--n-groups must be at least 1")
    if config.observations_per_group < 1:
        raise ValueError("--observations-per-group must be at least 1")
    if config.num_simulations < 1:
        raise ValueError("--num-simulations must be at least 1")
    if config.num_chains < 1:
        raise ValueError("--num-chains must be at least 1")
    if config.num_warmup < 1:
        raise ValueError("--num-warmup must be at least 1")
    if config.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if config.num_rank_bins < 1:
        raise ValueError("--num-rank-bins must be at least 1")
    if config.max_abs_bin_z <= 0.0:
        raise ValueError("--max-abs-bin-z must be positive")
    if config.max_abs_mean_rank_z <= 0.0:
        raise ValueError("--max-abs-mean-rank-z must be positive")
    if not 0.0 < config.target_acceptance_rate < 1.0:
        raise ValueError("--target-acceptance-rate must be in (0, 1)")


def main() -> int:
    config = _parse_args()
    _validate_config(config)
    _add_repo_paths()
    try:
        results = _run_sbc(config)
    except Exception as exc:  # noqa: BLE001 - optional reference script reports failures.
        print(f"FAIL error={type(exc).__name__}: {exc}")
        return 1

    for result in results:
        print(
            f"PASS parameter={result.parameter} simulations={result.num_simulations} "
            f"posterior_draws={result.num_posterior_draws} bin_counts={result.bin_counts} "
            f"max_abs_bin_z={result.max_abs_bin_z:.3f} mean_rank_z={result.mean_rank_z:.3f} "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
