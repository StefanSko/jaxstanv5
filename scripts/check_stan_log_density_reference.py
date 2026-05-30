#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cmdstanpy>=1.3.0",
#   "jax>=0.6.0",
# ]
# ///
"""Compare jaxstan compiled log-density differences against Stan references.

CmdStanPy's ``log_prob`` reports densities up to parameter-independent
constants.  This script therefore compares log-density differences from one
reference point per case.  Distribution unit tests pin exact constants.
"""

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

type FloatVector = tuple[float, ...]
type FloatMatrix = tuple[FloatVector, ...]
type StanParameterValue = float | FloatVector | FloatMatrix


@dataclass(frozen=True)
class LogDensityConfig:
    """Configuration for Stan log-density comparisons."""

    tolerance: float


@dataclass(frozen=True)
class LogDensityPoint:
    """One fixed parameter point to compare."""

    name: str
    unconstrained: tuple[float, ...]
    stan_parameters: Mapping[str, StanParameterValue]


@dataclass(frozen=True)
class LogDensityCase:
    """One model/data pair for log-density-difference comparison."""

    name: str
    stan_model: Path
    stan_data: Path
    reference_point: LogDensityPoint
    points: tuple[LogDensityPoint, ...]
    build_jaxstan_log_density: Callable[[Mapping[str, object]], Callable[[jax.Array], jax.Array]]


@dataclass(frozen=True)
class LogDensityComparison:
    """One log-density-difference comparison result."""

    case_name: str
    point_name: str
    jaxstan_log_density_difference: float
    stan_log_density_difference: float
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
        params: Mapping[str, StanParameterValue],
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


def _exponential_rate_log_density(data: Mapping[str, object]) -> Callable[[jax.Array], jax.Array]:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.compiler.core import compile_log_density
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Exponential, HalfNormal
    from jaxstanv5.model.bound import BoundModel

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
    bound = cast(BindableModel, ExponentialRateStanReferenceModel).bind(y=y)
    return compile_log_density(bound)


def _student_t_location_log_density(data: Mapping[str, object]) -> Callable[[jax.Array], jax.Array]:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.compiler.core import compile_log_density
    from jaxstanv5.distributions import Normal, StudentT
    from jaxstanv5.model.bound import BoundModel

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
    bound = cast(BindableModel, StudentTLocationStanReferenceModel).bind(y=y)
    return compile_log_density(bound)


def _multivariate_normal_likelihood_log_density(
    data: Mapping[str, object],
) -> Callable[[jax.Array], jax.Array]:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.compiler.core import compile_log_density
    from jaxstanv5.distributions import MultivariateNormal, Normal
    from jaxstanv5.model.bound import BoundModel

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    prior_scale = _as_float(data["prior_scale"], name="prior_scale")

    @model
    class MultivariateNormalLikelihoodStanReferenceModel:
        """Multivariate Normal likelihood model matching the Stan fixture."""

        n_dim = Data()
        chol = Data()

        mu = Param(Normal(0.0, prior_scale), size=n_dim)
        y = Observed(MultivariateNormal(mu, chol))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    chol = jnp.array(_float_matrix(data["chol"], name="chol"), dtype=jnp.float64)
    bound = cast(BindableModel, MultivariateNormalLikelihoodStanReferenceModel).bind(
        n_dim=y.shape[0], chol=chol, y=y
    )
    return compile_log_density(bound)


def _fixed_kernel_gp_log_density(data: Mapping[str, object]) -> Callable[[jax.Array], jax.Array]:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.compiler.core import compile_log_density
    from jaxstanv5.distributions import MultivariateNormal, Normal
    from jaxstanv5.model.bound import BoundModel

    class BindableModel(Protocol):
        """Runtime model class with decorator-attached bind method."""

        def bind(self, **values: object) -> BoundModel:
            """Bind concrete model data."""
            ...

    @model
    class FixedKernelGpStanReferenceModel:
        """Fixed-kernel GP model matching the Stan fixture."""

        n = Data()
        chol = Data()
        obs_sd = Data()

        f = Param(MultivariateNormal(0.0, chol), size=n)
        y = Observed(Normal(f, obs_sd))

    y = jnp.array(_float_sequence(data["y"], name="y"), dtype=jnp.float64)
    chol = jnp.array(_float_matrix(data["chol"], name="chol"), dtype=jnp.float64)
    obs_sd = _as_float(data["obs_sd"], name="obs_sd")
    bound = cast(BindableModel, FixedKernelGpStanReferenceModel).bind(
        n=y.shape[0], chol=chol, obs_sd=obs_sd, y=y
    )
    return compile_log_density(bound)


def _cases(root: Path) -> tuple[LogDensityCase, ...]:
    stan_root = root / "reference" / "stan"
    return (
        LogDensityCase(
            name="normal_known_scale",
            stan_model=stan_root / "models" / "normal_known_scale.stan",
            stan_data=stan_root / "data" / "normal_known_scale.json",
            reference_point=LogDensityPoint("center", (0.0,), {"mu": 0.0}),
            points=(
                LogDensityPoint("low", (-1.25,), {"mu": -1.25}),
                LogDensityPoint("high", (2.0,), {"mu": 2.0}),
            ),
            build_jaxstan_log_density=_normal_known_scale_log_density,
        ),
        LogDensityCase(
            name="positive_scale_normal",
            stan_model=stan_root / "models" / "positive_scale_normal.stan",
            stan_data=stan_root / "data" / "positive_scale_normal.json",
            reference_point=LogDensityPoint("unit", (0.0,), {"sigma": 1.0}),
            points=(
                LogDensityPoint("tiny", (-2.0,), {"sigma": math.exp(-2.0)}),
                LogDensityPoint("small", (-0.5,), {"sigma": math.exp(-0.5)}),
                LogDensityPoint("large", (1.5,), {"sigma": math.exp(1.5)}),
            ),
            build_jaxstan_log_density=_positive_scale_log_density,
        ),
        LogDensityCase(
            name="exponential_rate",
            stan_model=stan_root / "models" / "exponential_rate.stan",
            stan_data=stan_root / "data" / "exponential_rate.json",
            reference_point=LogDensityPoint("reference", (math.log(1.5),), {"rate": 1.5}),
            points=(
                LogDensityPoint("low", (math.log(0.5),), {"rate": 0.5}),
                LogDensityPoint("high", (math.log(3.0),), {"rate": 3.0}),
            ),
            build_jaxstan_log_density=_exponential_rate_log_density,
        ),
        LogDensityCase(
            name="student_t_location",
            stan_model=stan_root / "models" / "student_t_location.stan",
            stan_data=stan_root / "data" / "student_t_location.json",
            reference_point=LogDensityPoint("center", (0.0,), {"mu": 0.0}),
            points=(
                LogDensityPoint("low", (-1.0,), {"mu": -1.0}),
                LogDensityPoint("high", (2.0,), {"mu": 2.0}),
            ),
            build_jaxstan_log_density=_student_t_location_log_density,
        ),
        LogDensityCase(
            name="multivariate_normal_likelihood",
            stan_model=stan_root / "models" / "multivariate_normal_likelihood.stan",
            stan_data=stan_root / "data" / "multivariate_normal_likelihood.json",
            reference_point=LogDensityPoint("zero", (0.0, 0.0, 0.0), {"mu": (0.0, 0.0, 0.0)}),
            points=(
                LogDensityPoint("mixed", (1.0, -0.5, 2.0), {"mu": (1.0, -0.5, 2.0)}),
                LogDensityPoint("negative", (-1.0, 0.5, -0.25), {"mu": (-1.0, 0.5, -0.25)}),
            ),
            build_jaxstan_log_density=_multivariate_normal_likelihood_log_density,
        ),
        LogDensityCase(
            name="fixed_kernel_gp",
            stan_model=stan_root / "models" / "fixed_kernel_gp.stan",
            stan_data=stan_root / "data" / "fixed_kernel_gp.json",
            reference_point=LogDensityPoint("zero", (0.0, 0.0, 0.0), {"f": (0.0, 0.0, 0.0)}),
            points=(
                LogDensityPoint("smooth", (0.5, 0.25, -0.1), {"f": (0.5, 0.25, -0.1)}),
                LogDensityPoint("rough", (-0.5, 0.75, 0.25), {"f": (-0.5, 0.75, 0.25)}),
            ),
            build_jaxstan_log_density=_fixed_kernel_gp_log_density,
        ),
    )


def _stan_log_density(
    model: StanLogDensityModel,
    *,
    point: LogDensityPoint,
    data_path: Path,
) -> float:
    stan_frame = model.log_prob(
        params=point.stan_parameters,
        data=str(data_path),
        jacobian=True,
        sig_figs=18,
    )
    return _as_float(stan_frame["lp__"].iloc[0], name="lp__")


def _compare_case(case: LogDensityCase) -> tuple[LogDensityComparison, ...]:
    data = _load_json(case.stan_data)
    jaxstan_log_density = case.build_jaxstan_log_density(data)
    stan_model = _cmdstan_model(case.stan_model)

    q_reference = jnp.array(case.reference_point.unconstrained, dtype=jnp.float64)
    jaxstan_reference = float(jaxstan_log_density(q_reference))
    stan_reference = _stan_log_density(
        stan_model,
        point=case.reference_point,
        data_path=case.stan_data,
    )

    comparisons: list[LogDensityComparison] = []
    for point in case.points:
        q = jnp.array(point.unconstrained, dtype=jnp.float64)
        jaxstan_difference = float(jaxstan_log_density(q)) - jaxstan_reference
        stan_difference = (
            _stan_log_density(stan_model, point=point, data_path=case.stan_data)
            - stan_reference
        )
        comparisons.append(
            LogDensityComparison(
                case_name=case.name,
                point_name=point.name,
                jaxstan_log_density_difference=jaxstan_difference,
                stan_log_density_difference=stan_difference,
                abs_error=abs(jaxstan_difference - stan_difference),
            )
        )
    return tuple(comparisons)


def _parse_args() -> LogDensityConfig:
    parser = argparse.ArgumentParser(
        description="Compare jaxstan log-density differences against Stan fixed-point references."
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
                f"jaxstan_delta={comparison.jaxstan_log_density_difference:.12f} "
                f"stan_delta={comparison.stan_log_density_difference:.12f} "
                f"abs_err={comparison.abs_error:.3e}"
            )

    if failures:
        print(f"failures={failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
