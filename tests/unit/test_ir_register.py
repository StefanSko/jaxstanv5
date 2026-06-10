"""Unit tests for registering user distributions as IR extension tags."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from jaxstanv5.distributions.core import (
    Distribution,
    DistributionParameter,
    DistributionValue,
    LogProbability,
)
from jaxstanv5.ir import (
    UnserializableDistribution,
    meta_from_dict,
    meta_to_dict,
    register_distribution,
    register_node,
)
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedFreeValue,
    ResolvedParam,
    ResolvedStochasticSite,
)
from jaxstanv5.model.expr import DataRef, ParamRef


@dataclass(frozen=True)
class _Gumbel:
    """Custom location-scale distribution used as a registry extension."""

    loc: DistributionParameter
    scale: DistributionParameter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


@dataclass(frozen=True)
class _RenamedLaplace:
    """Distribution registered under an explicit wire tag."""

    loc: DistributionParameter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


class _PlainDistribution:
    """Distribution without dataclass fields."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


def _meta_with(distribution: Distribution) -> ModelMeta:
    return ModelMeta(
        params={"alpha": ResolvedParam(distribution, constraint=None, size=None)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"alpha": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(ResolvedStochasticSite("alpha", distribution, ParamRef("alpha")),),
    )


def test_registered_distribution_round_trips_with_symbolic_fields() -> None:
    register_distribution(_Gumbel)
    meta = _meta_with(_Gumbel(DataRef("mu0"), 2.0))

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    entry = params[0]
    assert isinstance(entry, dict)
    value = entry["value"]
    assert isinstance(value, dict)
    assert value["distribution"] == {
        "node": "_Gumbel",
        "loc": {"node": "DataRef", "name": "mu0"},
        "scale": 2.0,
    }
    assert meta_from_dict(document) == meta


def test_register_distribution_is_idempotent() -> None:
    register_distribution(_Gumbel)
    register_distribution(_Gumbel)

    assert meta_from_dict(meta_to_dict(_meta_with(_Gumbel(0.0, 1.0)))) is not None


def test_register_distribution_rejects_non_dataclass_with_repair_instruction() -> None:
    with pytest.raises(
        UnserializableDistribution,
        match=r"_PlainDistribution.*dataclass",
    ):
        register_distribution(_PlainDistribution)


def test_register_node_supports_explicit_tag_override() -> None:
    register_node(_RenamedLaplace, tag="Laplace")
    meta = _meta_with(_RenamedLaplace(0.5))

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    entry = params[0]
    assert isinstance(entry, dict)
    value = entry["value"]
    assert isinstance(value, dict)
    assert value["distribution"] == {"node": "Laplace", "loc": 0.5}
    assert meta_from_dict(document) == meta


def test_register_node_rejects_tag_collisions() -> None:
    @dataclass(frozen=True)
    class Normal:
        rate: float

    with pytest.raises(ValueError, match="'Normal' is already registered"):
        register_node(Normal)


def test_registered_distribution_still_rejects_non_finite_fields() -> None:
    register_distribution(_Gumbel)

    from jaxstanv5.ir import NonFiniteConstant

    with pytest.raises(NonFiniteConstant):
        meta_to_dict(_meta_with(_Gumbel(math.inf, 1.0)))
