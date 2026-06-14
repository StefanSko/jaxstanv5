"""Unit tests for IR document encoding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

import pytest
from _ir_meta import minimal_meta

from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval
from jaxstanv5.distributions import Normal
from jaxstanv5.distributions.core import DistributionParameter, DistributionValue, LogProbability
from jaxstanv5.ir import (
    NonFiniteConstant,
    UnserializableDistribution,
    UnserializableValue,
    meta_to_dict,
)
from jaxstanv5.model._data_schema import (
    DataDimRef,
    ResolvedDataShapeSchema,
)
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedData,
    ResolvedFreeValue,
    ResolvedParam,
    ResolvedStochasticSite,
)
from jaxstanv5.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    FullSlice,
    IndexOp,
    IndexTuple,
    ParamRef,
    ScalarIndex,
    UnaryOp,
    VectorScatterOp,
)


def test_encodes_minimal_meta_into_versioned_envelope() -> None:
    document = meta_to_dict(minimal_meta())

    mu = {
        "node": "BinOp",
        "op": "+",
        "left": {"node": "ParamRef", "name": "alpha"},
        "right": {"node": "DataRef", "name": "x"},
    }
    likelihood = {"node": "Normal", "loc": mu, "scale": 1.0}
    assert document == {
        "jaxstanv5_ir": 1,
        "model": {
            "node": "ModelMeta",
            "params": [
                {
                    "name": "alpha",
                    "value": {
                        "node": "ResolvedParam",
                        "distribution": {"node": "Normal", "loc": 0.0, "scale": 1.0},
                        "constraint": None,
                        "size": None,
                    },
                }
            ],
            "data": [
                {
                    "name": "x",
                    "value": {
                        "node": "ResolvedData",
                        "schema": {"node": "ResolvedDataRankSchema", "rank": 1},
                    },
                }
            ],
            "observed_nodes": [
                {"node": "ResolvedObserved", "name": "y", "distribution": likelihood}
            ],
            "expressions": [{"name": "mu", "value": mu}],
            "free_values": [
                {
                    "name": "alpha",
                    "value": {"node": "ResolvedFreeValue", "constraint": None, "size": None},
                }
            ],
            "stochastic_sites": [
                {
                    "node": "ResolvedStochasticSite",
                    "name": "alpha",
                    "distribution": {"node": "Normal", "loc": 0.0, "scale": 1.0},
                    "value": {"node": "ParamRef", "name": "alpha"},
                },
                {
                    "node": "ResolvedStochasticSite",
                    "name": "y",
                    "distribution": likelihood,
                    "value": {"node": "DataRef", "name": "y"},
                },
            ],
        },
    }


def test_map_entries_preserve_insertion_order() -> None:
    meta = ModelMeta(
        params={
            "zeta": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None),
            "alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None),
        },
        data={},
        observed_nodes=(),
        expressions={},
        free_values={
            "zeta": ResolvedFreeValue(constraint=None, size=None),
            "alpha": ResolvedFreeValue(constraint=None, size=None),
        },
        stochastic_sites=(
            ResolvedStochasticSite("zeta", Normal(0.0, 1.0), ParamRef("zeta")),
            ResolvedStochasticSite("alpha", Normal(0.0, 1.0), ParamRef("alpha")),
        ),
    )

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    assert [entry["name"] for entry in params] == ["zeta", "alpha"]


def test_preserves_int_float_lexical_identity() -> None:
    meta = minimal_meta(mu=BinOp("*", ConstNode(1), ConstNode(1.0)))

    serialized = json.dumps(meta_to_dict(meta))

    assert '"left": {"node": "ConstNode", "value": 1}' in serialized
    assert '"right": {"node": "ConstNode", "value": 1.0}' in serialized


def test_encodes_constraints_sizes_and_schemas() -> None:
    meta = ModelMeta(
        params={
            "sigma": ResolvedParam(Normal(0.0, 1.0), constraint=Positive(), size=None),
            "cutpoints": ResolvedParam(Normal(0.0, 2.0), constraint=Ordered(), size=DataRef("n")),
            "p": ResolvedParam(Normal(0.0, 1.0), constraint=UnitInterval(), size=2),
            "level": ResolvedParam(Normal(0.0, 1.0), constraint=Interval(-1.0, 3.0), size=None),
        },
        data={
            "n": ResolvedData(ResolvedDataShapeSchema(())),
            "chol": ResolvedData(ResolvedDataShapeSchema((DataDimRef("n"), 4))),
        },
        observed_nodes=(),
        expressions={},
        free_values={"sigma": ResolvedFreeValue(constraint=Positive(), size=None)},
        stochastic_sites=(ResolvedStochasticSite("sigma", Normal(0.0, 1.0), ParamRef("sigma")),),
    )

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    encoded = {entry["name"]: entry["value"] for entry in params}
    assert encoded["sigma"]["constraint"] == {"node": "Positive"}
    assert encoded["cutpoints"]["constraint"] == {"node": "Ordered"}
    assert encoded["cutpoints"]["size"] == {"node": "DataRef", "name": "n"}
    assert encoded["p"]["constraint"] == {"node": "UnitInterval"}
    assert encoded["p"]["size"] == 2
    assert encoded["level"]["constraint"] == {"node": "Interval", "lower": -1.0, "upper": 3.0}
    data = model["data"]
    assert isinstance(data, list)
    assert data[1]["value"]["schema"] == {
        "node": "ResolvedDataShapeSchema",
        "dims": [{"node": "DataDimRef", "name": "n"}, 4],
    }


def test_encodes_index_and_scatter_nodes() -> None:
    indexed = IndexOp(
        DataRef("x"),
        IndexTuple((ScalarIndex(DataRef("idx")), FullSlice())),
    )
    scatter = VectorScatterOp(
        length=DataRef("n"),
        observed_idx=DataRef("obs_idx"),
        observed_values=DataRef("y_obs"),
        missing_idx=DataRef("mis_idx"),
        missing_values=ParamRef("y"),
    )
    meta = minimal_meta(mu=BinOp("+", UnaryOp("exp", indexed), scatter))

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    expressions = model["expressions"]
    assert isinstance(expressions, list)
    encoded = expressions[0]["value"]
    assert encoded["left"] == {
        "node": "UnaryOp",
        "function": "exp",
        "operand": {
            "node": "IndexOp",
            "base": {"node": "DataRef", "name": "x"},
            "index": {
                "node": "IndexTuple",
                "items": [
                    {"node": "ScalarIndex", "expr": {"node": "DataRef", "name": "idx"}},
                    {"node": "FullSlice"},
                ],
            },
        },
    }
    assert encoded["right"] == {
        "node": "VectorScatterOp",
        "length": {"node": "DataRef", "name": "n"},
        "observed_idx": {"node": "DataRef", "name": "obs_idx"},
        "observed_values": {"node": "DataRef", "name": "y_obs"},
        "missing_idx": {"node": "DataRef", "name": "mis_idx"},
        "missing_values": {"node": "ParamRef", "name": "y"},
    }


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_rejects_non_finite_constants(value: float) -> None:
    meta = minimal_meta(mu=ConstNode(value))

    with pytest.raises(NonFiniteConstant, match="finite"):
        meta_to_dict(meta)


class _OpaqueDistribution:
    """Concrete-only distribution without dataclass fields."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


@dataclass(frozen=True)
class _UnregisteredDistribution:
    """Dataclass distribution that is not registered for IR serialization."""

    loc: DistributionParameter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


def test_rejects_opaque_distribution_with_repair_instruction() -> None:
    meta = ModelMeta(
        params={"alpha": ResolvedParam(_OpaqueDistribution(), constraint=None, size=None)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"alpha": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(
            ResolvedStochasticSite("alpha", _OpaqueDistribution(), ParamRef("alpha")),
        ),
    )

    with pytest.raises(
        UnserializableDistribution,
        match=r"_OpaqueDistribution.*register_distribution",
    ):
        meta_to_dict(meta)


def test_rejects_unregistered_dataclass_distribution() -> None:
    meta = ModelMeta(
        params={"alpha": ResolvedParam(_UnregisteredDistribution(0.0), constraint=None, size=None)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"alpha": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(
            ResolvedStochasticSite("alpha", _UnregisteredDistribution(0.0), ParamRef("alpha")),
        ),
    )

    with pytest.raises(
        UnserializableDistribution,
        match=r"_UnregisteredDistribution.*register_distribution",
    ):
        meta_to_dict(meta)


def test_rejects_unserializable_leaf_values() -> None:
    meta = minimal_meta()
    loc = cast(DistributionParameter, [0.0])
    broken = ModelMeta(
        params={"alpha": ResolvedParam(Normal(loc, 1.0), constraint=None, size=None)},
        data=meta.data,
        observed_nodes=(),
        expressions={},
        free_values=meta.free_values,
        stochastic_sites=(ResolvedStochasticSite("alpha", Normal(0.0, 1.0), ParamRef("alpha")),),
    )

    with pytest.raises(UnserializableValue, match="list"):
        meta_to_dict(broken)
