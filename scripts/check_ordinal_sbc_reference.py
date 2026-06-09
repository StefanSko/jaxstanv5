#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Run SBC for the ordinal-logistic regression model."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from jaxstanv5.inference import SamplerResult
    from jaxstanv5.model.bound import BoundModel


@dataclass(frozen=True)
class OrdinalSbcConfig:
    """Configuration for ordinal-logistic SBC checks."""

    seed: int
    num_observations: int
    num_simulations: int
    num_chains: int
    num_warmup: int
    num_samples: int
    num_rank_bins: int
    max_abs_bin_z: float
    max_abs_mean_rank_z: float
    target_acceptance_rate: float


@dataclass(frozen=True)
class OrdinalSbcResult:
    """Printable SBC result for one scalar parameter or projection."""

    parameter: str
    num_simulations: int
    num_posterior_draws: int
    bin_counts: tuple[int, ...]
    max_abs_bin_z: float
    mean_rank_z: float
    elapsed_seconds: float


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


def _build_model() -> object:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Ordered
    from jaxstanv5.distributions import Normal, OrderedLogistic

    @model
    class OrdinalLogisticSbcModel:
        """Ordinal-logistic SBC model with zero-based labels."""

        n_cutpoints = Data.scalar()
        x = Data.vector()

        beta = Param(Normal(0.0, 1.0))
        cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())

        eta = beta * x
        y = Observed(OrderedLogistic(eta, cutpoints))

    return OrdinalLogisticSbcModel


def _data_values(config: OrdinalSbcConfig) -> Mapping[str, object]:
    x = jnp.linspace(-1.5, 1.5, config.num_observations, dtype=jnp.float64)
    return {"n_cutpoints": 2, "x": x}


def _scalar_samples(result: SamplerResult) -> Mapping[str, jax.Array]:
    cutpoints = jnp.asarray(result.samples["cutpoints"])
    lower = cutpoints[:, :, 0]
    upper = cutpoints[:, :, 1]
    return {
        "beta": result.samples["beta"],
        "cutpoints[0]": lower,
        "cutpoints[1]": upper,
        "cutpoint_gap": upper - lower,
    }


def _truth_value(
    truth_draws: Mapping[str, jax.Array],
    *,
    parameter: str,
    simulation_index: int,
) -> float:
    if parameter == "beta":
        return float(truth_draws["beta"][simulation_index])
    cutpoints = truth_draws["cutpoints"]
    if parameter == "cutpoints[0]":
        return float(cutpoints[simulation_index, 0])
    if parameter == "cutpoints[1]":
        return float(cutpoints[simulation_index, 1])
    if parameter == "cutpoint_gap":
        return float(cutpoints[simulation_index, 1] - cutpoints[simulation_index, 0])
    raise ValueError(f"Unknown SBC parameter: {parameter}")


def _run_sbc(config: OrdinalSbcConfig) -> tuple[OrdinalSbcResult, ...]:
    from integration._validation import assert_sbc_rank_uniformity
    from jaxstanv5.inference import compile_sampler
    from jaxstanv5.simulation import simulate_prior_predictive
    from jaxstanv5.validation import SbcValidationResult, scalar_sbc_rank

    start = time.perf_counter()
    parameters = ("beta", "cutpoints[0]", "cutpoints[1]", "cutpoint_gap")
    model_cls = _build_model()
    data = _data_values(config)
    prior_predictive = simulate_prior_predictive(
        model_cls,
        seed=config.seed,
        num_samples=config.num_simulations,
        data=data,
    )
    truth_draws = {
        "beta": jnp.asarray(prior_predictive.parameters["beta"]),
        "cutpoints": jnp.asarray(prior_predictive.parameters["cutpoints"]),
    }
    y_draws = jnp.asarray(prior_predictive.observed["y"])
    ranks: dict[str, list[int]] = {name: [] for name in parameters}
    num_posterior_draws: int | None = None
    bindable = cast(BindableModel, model_cls)

    for simulation_index in range(config.num_simulations):
        bound = bindable.bind(**prior_predictive.data, y=y_draws[simulation_index])
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

        samples = _scalar_samples(result)
        for parameter in parameters:
            true_value = _truth_value(
                truth_draws,
                parameter=parameter,
                simulation_index=simulation_index,
            )
            rank = scalar_sbc_rank(samples, parameter=parameter, true_value=true_value)
            ranks[parameter].append(rank)
            print(
                f"DRAW simulation={simulation_index + 1}/{config.num_simulations} "
                f"parameter={parameter} true={true_value:.6f} rank={rank}"
            )

    if num_posterior_draws is None:
        raise ValueError("At least one posterior draw is required")

    elapsed = time.perf_counter() - start
    summaries: list[OrdinalSbcResult] = []
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
            OrdinalSbcResult(
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


def _parse_args() -> OrdinalSbcConfig:
    parser = argparse.ArgumentParser(description="Run ordinal-logistic SBC checks.")
    parser.add_argument("--seed", type=int, default=94_000)
    parser.add_argument("--num-observations", type=int, default=50)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--num-rank-bins", type=int, default=5)
    parser.add_argument("--max-abs-bin-z", type=float, default=4.0)
    parser.add_argument("--max-abs-mean-rank-z", type=float, default=4.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    args = parser.parse_args()
    return OrdinalSbcConfig(
        seed=args.seed,
        num_observations=args.num_observations,
        num_simulations=args.num_simulations,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_rank_bins=args.num_rank_bins,
        max_abs_bin_z=args.max_abs_bin_z,
        max_abs_mean_rank_z=args.max_abs_mean_rank_z,
        target_acceptance_rate=args.target_acceptance_rate,
    )


def _validate_config(config: OrdinalSbcConfig) -> None:
    if config.num_observations < 1:
        raise ValueError("--num-observations must be at least 1")
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
