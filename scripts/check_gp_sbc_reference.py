#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Run projected fixed-kernel GP simulation-based calibration checks."""

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
    from jaxstanv5.validation import ProjectionSpec


@dataclass(frozen=True)
class GpSbcConfig:
    """Configuration for projected GP SBC checks."""

    seed: int
    n: int
    lengthscale: float
    amplitude: float
    jitter: float
    obs_sd: float
    num_simulations: int
    num_chains: int
    num_warmup: int
    num_samples: int
    num_rank_bins: int
    max_abs_bin_z: float
    max_abs_mean_rank_z: float
    target_acceptance_rate: float


@dataclass(frozen=True)
class GpSbcProjectionResult:
    """Printable result for one GP SBC projection."""

    projection_name: str
    num_simulations: int
    num_posterior_draws: int
    bin_counts: tuple[int, ...]
    max_abs_bin_z: float
    mean_rank_z: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _rbf_covariance(
    x: jax.Array, *, lengthscale: float, amplitude: float, jitter: float
) -> jax.Array:
    diff = x[:, None] - x[None, :]
    covariance = amplitude**2 * jnp.exp(-0.5 * (diff / lengthscale) ** 2)
    return covariance + jitter * jnp.eye(x.shape[0], dtype=x.dtype)


def _projection_specs(n: int) -> tuple[ProjectionSpec, ...]:
    from jaxstanv5.validation import ProjectionSpec

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


def _build_model() -> object:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import MultivariateNormal, Normal

    @model
    class FixedKernelGpSbcModel:
        """Fixed-kernel GP model used for projected SBC."""

        n = Data.scalar()
        chol = Data.matrix(n, n)
        obs_sd = Data.scalar()

        f = Param(MultivariateNormal(0.0, chol), size=n)
        y = Observed(Normal(f, obs_sd))

    return FixedKernelGpSbcModel


def _simulated_values(
    model_cls: object,
    *,
    config: GpSbcConfig,
    chol: jax.Array,
) -> tuple[jax.Array, jax.Array, Mapping[str, jax.Array]]:
    from jaxstanv5.simulation import simulate_prior_predictive

    result = simulate_prior_predictive(
        model_cls,
        seed=config.seed,
        num_samples=config.num_simulations,
        data={"n": config.n, "chol": chol, "obs_sd": config.obs_sd},
    )
    return result.parameters["f"], result.observed["y"], result.data


def _run_projected_sbc(config: GpSbcConfig) -> tuple[GpSbcProjectionResult, ...]:
    from integration._validation import assert_sbc_rank_uniformity
    from jaxstanv5.inference import compile_sampler
    from jaxstanv5.model import bind_model
    from jaxstanv5.validation import (
        SbcValidationResult,
        project_vector_truth,
        projected_sbc_rank,
    )

    x = jnp.linspace(-3.0, 3.0, config.n, dtype=jnp.float64)
    covariance = _rbf_covariance(
        x,
        lengthscale=config.lengthscale,
        amplitude=config.amplitude,
        jitter=config.jitter,
    )
    chol = jnp.linalg.cholesky(covariance)
    model_cls = _build_model()
    f_truths, y_draws, data = _simulated_values(model_cls, config=config, chol=chol)
    projections = _projection_specs(config.n)
    ranks: dict[str, list[int]] = {projection.name: [] for projection in projections}
    num_posterior_draws: int | None = None
    bindable = model_cls

    for simulation_index in range(config.num_simulations):
        bound = bind_model(bindable, dict(**data, y=y_draws[simulation_index]))
        compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
        result = compiled.sample(
            seed=config.seed + 10_000 + simulation_index,
            num_chains=config.num_chains,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
        )
        result.samples["f"].block_until_ready()
        posterior_draw_count = int(
            jnp.asarray(result.samples["f"]).shape[0] * jnp.asarray(result.samples["f"]).shape[1]
        )
        if num_posterior_draws is None:
            num_posterior_draws = posterior_draw_count
        elif posterior_draw_count != num_posterior_draws:
            raise ValueError("All GP SBC simulations must use the same number of posterior draws")

        for projection in projections:
            true_projection = project_vector_truth(
                f_truths[simulation_index], projection=projection
            )
            rank = projected_sbc_rank(
                result.samples,
                projection=projection,
                true_value=f_truths[simulation_index],
            )
            ranks[projection.name].append(rank)
            print(
                f"DRAW simulation={simulation_index + 1}/{config.num_simulations} "
                f"projection={projection.name} true={true_projection:.6f} rank={rank}"
            )

    if num_posterior_draws is None:
        raise ValueError("At least one posterior draw is required")

    summaries: list[GpSbcProjectionResult] = []
    for projection in projections:
        validation_result = SbcValidationResult(
            parameter=projection.name,
            ranks=tuple(ranks[projection.name]),
            num_posterior_draws=num_posterior_draws,
        )
        uniformity = assert_sbc_rank_uniformity(
            validation_result,
            num_rank_bins=config.num_rank_bins,
            max_abs_bin_z=config.max_abs_bin_z,
            max_abs_mean_rank_z=config.max_abs_mean_rank_z,
        )
        summaries.append(
            GpSbcProjectionResult(
                projection_name=projection.name,
                num_simulations=len(validation_result.ranks),
                num_posterior_draws=validation_result.num_posterior_draws,
                bin_counts=uniformity.bin_counts,
                max_abs_bin_z=uniformity.max_abs_bin_z,
                mean_rank_z=uniformity.mean_rank_z,
            )
        )
    return tuple(summaries)


def _parse_args() -> GpSbcConfig:
    parser = argparse.ArgumentParser(description="Run projected fixed-kernel GP SBC checks.")
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--lengthscale", type=float, default=0.8)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--obs-sd", type=float, default=0.3)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--num-rank-bins", type=int, default=5)
    parser.add_argument("--max-abs-bin-z", type=float, default=4.0)
    parser.add_argument("--max-abs-mean-rank-z", type=float, default=4.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    args = parser.parse_args()
    return GpSbcConfig(
        seed=args.seed,
        n=args.n,
        lengthscale=args.lengthscale,
        amplitude=args.amplitude,
        jitter=args.jitter,
        obs_sd=args.obs_sd,
        num_simulations=args.num_simulations,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_rank_bins=args.num_rank_bins,
        max_abs_bin_z=args.max_abs_bin_z,
        max_abs_mean_rank_z=args.max_abs_mean_rank_z,
        target_acceptance_rate=args.target_acceptance_rate,
    )


def _validate_config(config: GpSbcConfig) -> None:
    if config.n < 2:
        raise ValueError("--n must be at least 2")
    if config.lengthscale <= 0.0:
        raise ValueError("--lengthscale must be positive")
    if config.amplitude <= 0.0:
        raise ValueError("--amplitude must be positive")
    if config.jitter <= 0.0:
        raise ValueError("--jitter must be positive")
    if config.obs_sd <= 0.0:
        raise ValueError("--obs-sd must be positive")
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
    start = time.perf_counter()
    try:
        results = _run_projected_sbc(config)
    except Exception as exc:  # noqa: BLE001 - script reports reference failures.
        print(f"FAIL error={type(exc).__name__}: {exc}")
        return 1

    for result in results:
        print(
            f"PASS projection={result.projection_name} "
            f"simulations={result.num_simulations} "
            f"posterior_draws={result.num_posterior_draws} "
            f"bin_counts={result.bin_counts} "
            f"max_abs_bin_z={result.max_abs_bin_z:.3f} "
            f"mean_rank_z={result.mean_rank_z:.3f}"
        )
    print(f"elapsed={time.perf_counter() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
