#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jaxstanv5",
# ]
#
# [tool.uv.sources]
# jaxstanv5 = { path = "..", editable = true }
# ///
"""Run optional simulation-based calibration reference checks."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp

if TYPE_CHECKING:
    from integration._validation import SbcSimulation


@dataclass(frozen=True)
class SbcScriptConfig:
    """Shared SBC script configuration."""

    seed: int
    num_simulations: int
    num_observations: int
    num_chains: int
    num_warmup: int
    num_samples: int
    num_rank_bins: int
    max_abs_bin_z: float
    max_abs_mean_rank_z: float
    target_acceptance_rate: float


@dataclass(frozen=True)
class SbcCaseResult:
    """Printable result for one SBC case."""

    case_name: str
    parameter: str
    num_simulations: int
    num_posterior_draws: int
    bin_counts: tuple[int, ...]
    max_abs_bin_z: float
    mean_rank_z: float
    elapsed_seconds: float


class SbcCase(Protocol):
    """A complete SBC generative case."""

    name: str
    parameter: str

    def simulations(self, config: SbcScriptConfig) -> tuple[SbcSimulation, ...]:
        """Generate SBC simulations."""
        ...


@dataclass(frozen=True)
class PriorPredictiveSbcCase:
    """SBC case backed by public prior-predictive simulation."""

    name: str
    parameter: str
    model_cls: object
    data: Mapping[str, object]
    observed_shapes: Mapping[str, tuple[int, ...]]

    def simulations(self, config: SbcScriptConfig) -> tuple[SbcSimulation, ...]:
        from integration._validation import SbcSimulation
        from jaxstanv5.model import bind_model
        from jaxstanv5.simulation import simulate_prior_predictive

        result = simulate_prior_predictive(
            self.model_cls,
            seed=config.seed,
            num_samples=config.num_simulations,
            data=self.data,
            observed_shapes=self.observed_shapes,
        )
        truth_draws = jnp.asarray(result.parameters[self.parameter])
        if truth_draws.ndim != 1:
            raise ValueError("SBC ranked parameter must be scalar")

        bindable = self.model_cls
        simulations: list[SbcSimulation] = []
        for simulation_index in range(config.num_simulations):
            values: dict[str, object] = {name: value for name, value in result.data.items()}
            values.update(
                {name: draws[simulation_index] for name, draws in result.observed.items()}
            )
            simulations.append(
                SbcSimulation(
                    true_value=float(truth_draws[simulation_index]),
                    bound=bind_model(bindable, dict(**values)),
                )
            )
        return tuple(simulations)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _built_in_cases(config: SbcScriptConfig) -> Mapping[str, SbcCase]:
    from bayeswire import Observed, Param, model
    from bayeswire.constraints import Positive
    from bayeswire.distributions import Normal

    @model
    class NormalKnownScaleSbcModel:
        """Scalar Normal known-scale SBC model."""

        mu = Param(Normal(0.0, 1.0))
        y = Observed(Normal(mu, 1.0))

    @model
    class PositiveScaleNormalSbcModel:
        """Positive-scale Normal SBC model."""

        sigma = Param(Normal(0.0, 1.0), constraint=Positive())
        y = Observed(Normal(0.0, sigma))

    return {
        "normal_known_scale": PriorPredictiveSbcCase(
            name="normal_known_scale",
            parameter="mu",
            model_cls=NormalKnownScaleSbcModel,
            data={},
            observed_shapes={"y": (config.num_observations,)},
        ),
        "positive_scale_normal": PriorPredictiveSbcCase(
            name="positive_scale_normal",
            parameter="sigma",
            model_cls=PositiveScaleNormalSbcModel,
            data={},
            observed_shapes={"y": (config.num_observations,)},
        ),
    }


def _load_json(path: Path) -> Mapping[str, object]:
    return cast(Mapping[str, object], json.loads(path.read_text()))


def _parse_observed_shape(value: str) -> tuple[str, tuple[int, ...]]:
    if "=" not in value:
        raise ValueError("Observed shapes must have form name=dim[,dim...]")
    name, raw_shape = value.split("=", 1)
    if name == "":
        raise ValueError("Observed shape name cannot be empty")
    if raw_shape in {"", "scalar"}:
        return name, ()
    dims = tuple(int(part) for part in raw_shape.split(",") if part != "")
    return name, dims


def _parse_observed_shapes(values: Sequence[str]) -> Mapping[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for value in values:
        name, shape = _parse_observed_shape(value)
        shapes[name] = shape
    return shapes


def _load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _load_object(spec: str) -> object:
    if ":" not in spec:
        raise ValueError("Object specs must have form module_or_path:object_name")
    module_name, object_name = spec.split(":", 1)
    if module_name.endswith(".py") or "/" in module_name:
        module = _load_module_from_path(Path(module_name).resolve())
    else:
        module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ValueError(f"Missing object {object_name!r} in {module_name!r}") from exc


def _raw_model_case(
    *,
    model_spec: str,
    parameter: str,
    data_json: Path | None,
    observed_shapes: Mapping[str, tuple[int, ...]],
) -> SbcCase:
    data: Mapping[str, object] = {} if data_json is None else _load_json(data_json)
    model_cls = _load_object(model_spec)
    return PriorPredictiveSbcCase(
        name=f"raw_model:{model_spec}",
        parameter=parameter,
        model_cls=model_cls,
        data=data,
        observed_shapes=observed_shapes,
    )


def _run_case(case: SbcCase, config: SbcScriptConfig, *, case_index: int) -> SbcCaseResult:
    from integration._validation import (
        ChainRunSpec,
        assert_sbc_rank_uniformity,
        run_sbc_rank_validation,
    )

    start = time.perf_counter()
    case_config = SbcScriptConfig(
        seed=config.seed + case_index,
        num_simulations=config.num_simulations,
        num_observations=config.num_observations,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_rank_bins=config.num_rank_bins,
        max_abs_bin_z=config.max_abs_bin_z,
        max_abs_mean_rank_z=config.max_abs_mean_rank_z,
        target_acceptance_rate=config.target_acceptance_rate,
    )
    simulations = case.simulations(case_config)
    result = run_sbc_rank_validation(
        parameter=case.parameter,
        simulations=simulations,
        run=ChainRunSpec(
            seed=config.seed + 10_000 + case_index,
            num_chains=config.num_chains,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            target_acceptance_rate=config.target_acceptance_rate,
        ),
    )
    summary = assert_sbc_rank_uniformity(
        result,
        num_rank_bins=config.num_rank_bins,
        max_abs_bin_z=config.max_abs_bin_z,
        max_abs_mean_rank_z=config.max_abs_mean_rank_z,
    )
    return SbcCaseResult(
        case_name=case.name,
        parameter=case.parameter,
        num_simulations=len(result.ranks),
        num_posterior_draws=result.num_posterior_draws,
        bin_counts=summary.bin_counts,
        max_abs_bin_z=summary.max_abs_bin_z,
        mean_rank_z=summary.mean_rank_z,
        elapsed_seconds=time.perf_counter() - start,
    )


def _parse_args() -> tuple[
    SbcScriptConfig, str, str | None, str | None, Path | None, Mapping[str, tuple[int, ...]]
]:
    parser = argparse.ArgumentParser(description="Run optional SBC reference checks.")
    parser.add_argument(
        "--case",
        choices=("normal_known_scale", "positive_scale_normal", "all"),
        default="all",
    )
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--parameter", default=None)
    parser.add_argument("--data-json", type=Path, default=None)
    parser.add_argument("--observed-shape", action="append", default=[])
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--num-rank-bins", type=int, default=10)
    parser.add_argument("--max-abs-bin-z", type=float, default=3.0)
    parser.add_argument("--max-abs-mean-rank-z", type=float, default=3.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.95)
    args = parser.parse_args()
    config = SbcScriptConfig(
        seed=args.seed,
        num_simulations=args.num_simulations,
        num_observations=args.num_observations,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_rank_bins=args.num_rank_bins,
        max_abs_bin_z=args.max_abs_bin_z,
        max_abs_mean_rank_z=args.max_abs_mean_rank_z,
        target_acceptance_rate=args.target_acceptance_rate,
    )
    return (
        config,
        cast(str, args.case),
        cast(str | None, args.model_file),
        cast(str | None, args.parameter),
        cast(Path | None, args.data_json),
        _parse_observed_shapes(cast(Sequence[str], args.observed_shape)),
    )


def _validate_config(config: SbcScriptConfig) -> None:
    if config.num_simulations < 1:
        raise ValueError("--num-simulations must be at least 1")
    if config.num_observations < 1:
        raise ValueError("--num-observations must be at least 1")
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
    config, case_name, model_file, parameter, data_json, observed_shapes = _parse_args()
    _validate_config(config)
    if model_file is not None and parameter is None:
        raise ValueError("--parameter is required with --model-file")

    _add_repo_paths()
    built_ins = _built_in_cases(config)
    cases: list[SbcCase] = (
        list(built_ins.values()) if case_name == "all" else [built_ins[case_name]]
    )
    if model_file is not None:
        cases.append(
            _raw_model_case(
                model_spec=model_file,
                parameter=cast(str, parameter),
                data_json=data_json,
                observed_shapes=observed_shapes,
            )
        )

    failures = 0
    for case_index, case in enumerate(cases):
        try:
            result = _run_case(case, config, case_index=case_index)
            print(
                f"PASS case={result.case_name} parameter={result.parameter} "
                f"simulations={result.num_simulations} "
                f"posterior_draws={result.num_posterior_draws} "
                f"bin_counts={result.bin_counts} "
                f"max_abs_bin_z={result.max_abs_bin_z:.3f} "
                f"mean_rank_z={result.mean_rank_z:.3f} "
                f"elapsed={result.elapsed_seconds:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001 - optional reference script reports failures.
            failures += 1
            print(f"FAIL case={case.name} error={type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
