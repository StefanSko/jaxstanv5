#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Stress-test the always-on Normal validation thresholds.

Run from the repository root with either:

    uv run --script scripts/stress_always_on_normal_validation.py --runs 200

or, after making the file executable:

    ./scripts/stress_always_on_normal_validation.py --runs 200
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class StressConfig:
    """Configuration for repeated validation runs."""

    runs: int
    seed_start: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float


@dataclass(frozen=True)
class StressFailure:
    """One failed validation run."""

    seed: int
    reason: str
    k_min: float
    rhat: float
    ess: float


@dataclass(frozen=True)
class StressResult:
    """Aggregate stress-test result."""

    passes: int
    failures: tuple[StressFailure, ...]
    z_scores: tuple[float, ...]
    k_values: tuple[float, ...]
    rhat_values: tuple[float, ...]
    ess_values: tuple[float, ...]
    elapsed_seconds: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _block_until_ready(samples: dict[str, jax.Array]) -> None:
    for value in samples.values():
        value.block_until_ready()


def _run_stress(config: StressConfig) -> StressResult:
    _add_repo_paths()

    from integration._validation import (
        normal_known_scale_reference,
        standardized_discrepancy,
        summarize_scalar_draws,
    )
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.distributions import Normal
    from jaxstanv5.inference import compile_sampler
    from jaxstanv5.model.bound import BoundModel

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    @model
    class NormalKnownScaleValidationModel:
        """Scalar Normal model with known observation scale."""

        mu = Param(Normal(0.0, 1.0))
        y = Observed(Normal(mu, 1.0))

    y_data = jnp.array([1.0, 1.5, 2.0, 2.5])
    bound = cast(BindableModel, NormalKnownScaleValidationModel).bind(y=y_data)
    compiled = compile_sampler(bound)
    reference = normal_known_scale_reference(
        parameter="mu",
        y=y_data,
        prior_loc=0.0,
        prior_scale=1.0,
        obs_scale=1.0,
    )

    warmup_result = compiled.sample(
        seed=config.seed_start - 1,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_until_ready(warmup_result.samples)

    passes = 0
    failures: list[StressFailure] = []
    z_scores: list[float] = []
    k_values: list[float] = []
    rhat_values: list[float] = []
    ess_values: list[float] = []

    start = time.perf_counter()
    for run_index in range(config.runs):
        seed = config.seed_start + run_index
        try:
            result = compiled.sample(
                seed=seed,
                num_chains=config.num_chains,
                num_warmup=config.num_warmup,
                num_samples=config.num_samples,
            )
            _block_until_ready(result.samples)
            summary = summarize_scalar_draws(result.samples, parameter="mu")
            discrepancy = standardized_discrepancy(
                parameter="mu",
                summary_name="mean",
                estimate=summary.mean,
                reference=reference.mean,
                mcse=summary.mcse_mean,
            )

            z_scores.append(discrepancy.signed_z)
            k_values.append(discrepancy.k_min)
            rhat_values.append(summary.rhat)
            ess_values.append(summary.ess)

            reasons: list[str] = []
            if discrepancy.k_min > config.max_k:
                reasons.append("k")
            if summary.rhat > config.max_rhat:
                reasons.append("rhat")
            if summary.ess < config.min_ess:
                reasons.append("ess")

            if reasons:
                failures.append(
                    StressFailure(
                        seed=seed,
                        reason="+".join(reasons),
                        k_min=discrepancy.k_min,
                        rhat=summary.rhat,
                        ess=summary.ess,
                    )
                )
            else:
                passes += 1
        except Exception as exc:  # noqa: BLE001 - this helper reports stress failures.
            failures.append(
                StressFailure(
                    seed=seed,
                    reason=type(exc).__name__,
                    k_min=math.nan,
                    rhat=math.nan,
                    ess=math.nan,
                )
            )

    elapsed_seconds = time.perf_counter() - start
    return StressResult(
        passes=passes,
        failures=tuple(failures),
        z_scores=tuple(z_scores),
        k_values=tuple(k_values),
        rhat_values=tuple(rhat_values),
        ess_values=tuple(ess_values),
        elapsed_seconds=elapsed_seconds,
    )


def _print_distribution(name: str, values: tuple[float, ...]) -> None:
    if len(values) == 0:
        print(f"{name}: no successful values")
        return

    array = jnp.array(values)
    print(
        f"{name}: "
        f"mean={float(jnp.mean(array)):.3f} "
        f"sd={float(jnp.std(array, ddof=1)):.3f} "
        f"min={float(jnp.min(array)):.3f} "
        f"p95={float(jnp.quantile(array, 0.95)):.3f} "
        f"max={float(jnp.max(array)):.3f}"
    )


def _print_result(config: StressConfig, result: StressResult) -> None:
    total = result.passes + len(result.failures)
    print(f"runs={total} passes={result.passes} failures={len(result.failures)}")
    print(
        f"thresholds: max_k={config.max_k:.2f} "
        f"max_rhat={config.max_rhat:.3f} min_ess={config.min_ess:.1f}"
    )
    print(f"elapsed={result.elapsed_seconds:.2f}s avg={result.elapsed_seconds / total:.3f}s/run")
    _print_distribution("z", result.z_scores)
    _print_distribution("k", result.k_values)
    _print_distribution("rhat", result.rhat_values)
    _print_distribution("ess", result.ess_values)

    if result.failures:
        print("first_failures:")
        for failure in result.failures[:20]:
            print(
                f"  seed={failure.seed} reason={failure.reason} "
                f"k={failure.k_min:.3f} rhat={failure.rhat:.4f} ess={failure.ess:.1f}"
            )


def _parse_args() -> StressConfig:
    parser = argparse.ArgumentParser(
        description="Stress-test always-on scalar Normal posterior validation thresholds."
    )
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--max-k", type=float, default=4.0)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=100.0)
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
    )


def main() -> int:
    config = _parse_args()
    if config.runs < 1:
        raise ValueError("--runs must be at least 1")

    result = _run_stress(config)
    _print_result(config, result)
    if result.failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
