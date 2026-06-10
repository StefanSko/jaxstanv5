"""Deterministic golden models pinning the IR v1 wire format.

Shared by the golden-file tests and ``scripts/regenerate_ir_golden.py``.
Any change to a golden document is a wire-format change and requires a
deliberate golden-file diff plus a version decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from jaxstanv5 import Data, Observed, Param, PartiallyObserved, model
from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval
from jaxstanv5.distributions import (
    Bernoulli,
    Beta,
    HalfNormal,
    MultivariateNormal,
    Normal,
    OrderedLogistic,
    Poisson,
)
from jaxstanv5.math import exp
from jaxstanv5.model.decorator import ModelMeta


@dataclass(frozen=True)
class GoldenIRCase:
    """One golden model with deterministic bind values."""

    name: str
    meta: ModelMeta
    bind_values: dict[str, object]


def _meta(cls: type) -> ModelMeta:
    return cast(ModelMeta, getattr(cls, "_model_meta"))  # noqa: B009


def _linear_regression() -> GoldenIRCase:
    @model
    class LinearRegression:
        alpha = Param(Normal(0.0, 1.0))
        beta = Param(Normal(0.0, 1.0))
        sigma = Param(Normal(0.0, 1.0), constraint=Positive())
        x = Data.vector()
        mu = alpha + beta * x
        y = Observed(Normal(mu, sigma))

    return GoldenIRCase(
        name="linear_regression",
        meta=_meta(LinearRegression),
        bind_values={
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "y": [-1.6, -0.3, 0.4, 1.1, 2.2],
        },
    )


def _eight_schools_non_centered() -> GoldenIRCase:
    @model
    class EightSchoolsNonCentered:
        n_schools = Data.scalar()
        sigma = Data.vector(n_schools)

        mu = Param(Normal(0.0, 5.0))
        tau = Param(HalfNormal(5.0), constraint=Positive())
        z = Param(Normal(0.0, 1.0), size=n_schools)
        theta = mu + tau * z
        y = Observed(Normal(theta, sigma))

    return GoldenIRCase(
        name="eight_schools_non_centered",
        meta=_meta(EightSchoolsNonCentered),
        bind_values={
            "n_schools": 8,
            "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
            "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        },
    )


def _varying_intercepts_poisson() -> GoldenIRCase:
    @model
    class VaryingInterceptsPoisson:
        n_groups = Data.scalar()
        group_idx = Data.vector()
        x = Data.vector()

        alpha_pop = Param(Normal(0.0, 0.5))
        sigma_alpha = Param(HalfNormal(0.4), constraint=Positive())
        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        eta = alpha[group_idx] + 0.25 * x
        y = Observed(Poisson(exp(eta)))

    return GoldenIRCase(
        name="varying_intercepts_poisson",
        meta=_meta(VaryingInterceptsPoisson),
        bind_values={
            "n_groups": 3,
            "group_idx": [0, 0, 1, 1, 2, 2],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "y": [0, 1, 2, 1, 3, 2],
        },
    )


def _ordinal_regression() -> GoldenIRCase:
    @model
    class OrdinalRegression:
        n_cutpoints = Data.scalar()
        x = Data.vector()

        beta = Param(Normal(0.0, 1.0))
        cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())

        eta = beta * x
        y = Observed(OrderedLogistic(eta, cutpoints))

    return GoldenIRCase(
        name="ordinal_regression",
        meta=_meta(OrdinalRegression),
        bind_values={
            "n_cutpoints": 2,
            "x": [-1.5, -0.75, 0.0, 0.75, 1.5],
            "y": [0, 0, 1, 2, 2],
        },
    )


def _partially_observed_mvn() -> GoldenIRCase:
    @model
    class PartiallyObservedMvn:
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

    return GoldenIRCase(
        name="partially_observed_mvn",
        meta=_meta(PartiallyObservedMvn),
        bind_values={
            "n": 3,
            "n_obs": 2,
            "n_mis": 1,
            "chol": [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.25, 0.4375, 0.85]],
            "observed_idx": [0, 2],
            "missing_idx": [1],
            "observed_values": [0.7, -0.4],
        },
    )


def _bounded_rates() -> GoldenIRCase:
    @model
    class BoundedRates:
        p = Param(Beta(2.0, 2.0), constraint=UnitInterval())
        level = Param(Normal(1.0, 1.0), constraint=Interval(-1.0, 3.0))
        y = Observed(Bernoulli(p))

    return GoldenIRCase(
        name="bounded_rates",
        meta=_meta(BoundedRates),
        bind_values={"y": [0, 1, 1, 0, 1]},
    )


def golden_ir_cases() -> tuple[GoldenIRCase, ...]:
    """Return all golden IR cases in their pinned order."""
    return (
        _linear_regression(),
        _eight_schools_non_centered(),
        _varying_intercepts_poisson(),
        _ordinal_regression(),
        _partially_observed_mvn(),
        _bounded_rates(),
    )
