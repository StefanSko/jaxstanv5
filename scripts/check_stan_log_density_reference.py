#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Compare jaxstan compiled log densities against Stan fixed-point references."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


@dataclass(frozen=True)
class LogDensityConfig:
    """Configuration for Stan log-density comparisons."""

    tolerance: float


@dataclass(frozen=True)
class LogDensityPoint:
    """One fixed parameter point to compare."""

    name: str
    unconstrained: tuple[float, ...]
    stan_parameters: Mapping[str, float]


@dataclass(frozen=True)
class LogDensityCase:
    """One model/data pair for log-density comparison."""

    name: str
    stan_model: Path
    stan_data: Path
    points: tuple[LogDensityPoint, ...]
    build_jaxstan_log_density: Callable[[Mapping[str, object]], Callable[[jax.Array], jax.Array]]
    constant_adjustment: Callable[[Mapping[str, object]], float]


@dataclass(frozen=True)
class LogDensityComparison:
    """One log-density comparison result."""

    case_name: str
    point_name: str
    jaxstan_log_density: float
    stan_log_density: float
    abs_error: float


class StanColumnIloc(Protocol):
    """Minimal iloc protocol for CmdStanPy's log_prob result."""

    def __getitem__(self, index: int) -> object:
        """Return one scalar cell."""
        ...


class StanLogProbColumn(Protocol):
    """Minimal column protocol for CmdStanPy's log_prob result."""

    @property
    def iloc(self) -> StanColumnIloc:
        """Position-based accessor."""
        ...


class StanLogProbFrame(Protocol):
    """Minimal DataFrame protocol for CmdStanPy's log_prob result."""

    def __getitem__(self, key: str) -> StanLogProbColumn:
        """Return one named column."""
        ...


class StanLogDensityModel(Protocol):
    """Minimal CmdStanModel protocol needed for log_prob."""

    def log_prob(
        self,
        params: Mapping[str, float],
        data: str,
        *,
        jacobian: bool,
        sig_figs: int,
    ) -> StanLogProbFrame:
        """Evaluate Stan's log probability."""
        ...


class StanModelFactory(Protocol):
    """Callable constructor protocol for CmdStanModel."""

    def __call__(self, *, stan_file: str) -> StanLogDensityModel:
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


def _cmdstan_model(stan_file: Path) -> StanLogDensityModel:
    module = cast(CmdStanPyModule, importlib.import_module("cmdstanpy"))
    return module.CmdStanModel(stan_file=str(stan_file))


def _normal_known_scale_log_density(data: Mapping[str, object]) -> Callable[[jax.Array], jax.Array]:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.compiler.core import compile_log_density
    from jaxstanv5.distributions import Normal
    from jaxstanv5.model.bound import BoundModel

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
    bound = cast(BindableModel, NormalKnownScaleStanReferenceModel).bind(y=y)
    return compile_log_density(bound)


def _normal_known_scale_constant_adjustment(data: Mapping[str, object]) -> float:
    sample_count = len(_float_sequence(data["y"], name="y"))
    prior_scale = _as_float(data["prior_scale"], name="prior_scale")
    obs_scale = _as_float(data["obs_scale"], name="obs_scale")
    half_log_two_pi = 0.5 * math.log(2.0 * math.pi)
    return (
        -(sample_count + 1) * half_log_two_pi
        - math.log(prior_scale)
        - sample_count * math.log(obs_scale)
    )


def _positive_scale_constant_adjustment(data: Mapping[str, object]) -> float:
    sample_count = len(_float_sequence(data["y"], name="y"))
    prior_scale = _as_float(data["prior_scale"], name="prior_scale")
    half_log_two_pi = 0.5 * math.log(2.0 * math.pi)
    return -(sample_count + 1) * half_log_two_pi - math.log(prior_scale)


def _positive_scale_log_density(data: Mapping[str, object]) -> Callable[[jax.Array], jax.Array]:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.compiler.core import compile_log_density
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Normal
    from jaxstanv5.model.bound import BoundModel

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
    bound = cast(BindableModel, PositiveScaleStanReferenceModel).bind(y=y)
    return compile_log_density(bound)


def _cases(root: Path) -> tuple[LogDensityCase, ...]:
    stan_root = root / "reference" / "stan"
    return (
        LogDensityCase(
            name="normal_known_scale",
            stan_model=stan_root / "models" / "normal_known_scale.stan",
            stan_data=stan_root / "data" / "normal_known_scale.json",
            points=(
                LogDensityPoint("low", (-1.25,), {"mu": -1.25}),
                LogDensityPoint("center", (0.0,), {"mu": 0.0}),
                LogDensityPoint("high", (2.0,), {"mu": 2.0}),
            ),
            build_jaxstan_log_density=_normal_known_scale_log_density,
            constant_adjustment=_normal_known_scale_constant_adjustment,
        ),
        LogDensityCase(
            name="positive_scale_normal",
            stan_model=stan_root / "models" / "positive_scale_normal.stan",
            stan_data=stan_root / "data" / "positive_scale_normal.json",
            points=(
                LogDensityPoint("tiny", (-2.0,), {"sigma": math.exp(-2.0)}),
                LogDensityPoint("small", (-0.5,), {"sigma": math.exp(-0.5)}),
                LogDensityPoint("unit", (0.0,), {"sigma": 1.0}),
                LogDensityPoint("large", (1.5,), {"sigma": math.exp(1.5)}),
            ),
            build_jaxstan_log_density=_positive_scale_log_density,
            constant_adjustment=_positive_scale_constant_adjustment,
        ),
    )


def _compare_case(case: LogDensityCase) -> tuple[LogDensityComparison, ...]:
    data = _load_json(case.stan_data)
    jaxstan_log_density = case.build_jaxstan_log_density(data)
    constant_adjustment = case.constant_adjustment(data)
    stan_model = _cmdstan_model(case.stan_model)

    comparisons: list[LogDensityComparison] = []
    for point in case.points:
        q = jnp.array(point.unconstrained, dtype=jnp.float64)
        jaxstan_value = float(jaxstan_log_density(q))
        stan_frame = stan_model.log_prob(
            params=dict(point.stan_parameters),
            data=str(case.stan_data),
            jacobian=True,
            sig_figs=18,
        )
        stan_value = _as_float(stan_frame["lp__"].iloc[0], name="lp__") + constant_adjustment
        comparisons.append(
            LogDensityComparison(
                case_name=case.name,
                point_name=point.name,
                jaxstan_log_density=jaxstan_value,
                stan_log_density=stan_value,
                abs_error=abs(jaxstan_value - stan_value),
            )
        )
    return tuple(comparisons)


def _parse_args() -> LogDensityConfig:
    parser = argparse.ArgumentParser(
        description="Compare jaxstan log densities against Stan fixed-point references."
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    return LogDensityConfig(tolerance=args.tolerance)


def main() -> int:
    config = _parse_args()
    if config.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")

    _add_repo_paths()
    failures = 0
    for case in _cases(_repo_root()):
        for comparison in _compare_case(case):
            passed = comparison.abs_error <= config.tolerance
            if not passed:
                failures += 1
            status = "PASS" if passed else "FAIL"
            print(
                f"{status} case={comparison.case_name} point={comparison.point_name} "
                f"jaxstan={comparison.jaxstan_log_density:.12f} "
                f"stan={comparison.stan_log_density:.12f} "
                f"abs_err={comparison.abs_error:.3e}"
            )

    if failures:
        print(f"failures={failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
