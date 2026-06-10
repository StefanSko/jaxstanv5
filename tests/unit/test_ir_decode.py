"""Unit tests for IR document decoding."""

from __future__ import annotations

import json

import pytest
from _ir_meta import minimal_meta, rich_meta

from jaxstanv5.distributions import Normal
from jaxstanv5.ir import (
    MalformedIRDocument,
    UnknownNodeTag,
    UnsupportedIRVersion,
    meta_from_dict,
    meta_to_dict,
)
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedParam,
    ResolvedStochasticSite,
)
from jaxstanv5.model.expr import ParamRef


def test_round_trips_minimal_meta() -> None:
    meta = minimal_meta()

    assert meta_from_dict(meta_to_dict(meta)) == meta


def test_round_trips_rich_meta_through_json_text() -> None:
    meta = rich_meta()

    document = json.loads(json.dumps(meta_to_dict(meta)))

    assert meta_from_dict(document) == meta


def test_decodes_empty_containers_by_field_kind() -> None:
    meta = ModelMeta(
        params={"alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={},
        stochastic_sites=(),
    )

    decoded = meta_from_dict(meta_to_dict(meta))

    assert decoded.data == {}
    assert isinstance(decoded.data, dict)
    assert decoded.observed_nodes == ()
    assert isinstance(decoded.observed_nodes, tuple)
    assert decoded.free_values == {}
    assert isinstance(decoded.free_values, dict)
    assert decoded.stochastic_sites == ()
    assert isinstance(decoded.stochastic_sites, tuple)


def test_decoded_meta_preserves_map_insertion_order() -> None:
    meta = ModelMeta(
        params={
            "zeta": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None),
            "alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None),
        },
        data={},
        observed_nodes=(),
        expressions={},
        stochastic_sites=(ResolvedStochasticSite("zeta", Normal(0.0, 1.0), ParamRef("zeta")),),
    )

    decoded = meta_from_dict(meta_to_dict(meta))

    assert list(decoded.params) == ["zeta", "alpha"]


def test_rejects_missing_version() -> None:
    document = meta_to_dict(minimal_meta())
    del document["jaxstanv5_ir"]

    with pytest.raises(UnsupportedIRVersion, match="version"):
        meta_from_dict(document)


def test_rejects_unknown_version() -> None:
    document = meta_to_dict(minimal_meta())
    document["jaxstanv5_ir"] = 2

    with pytest.raises(UnsupportedIRVersion, match="2"):
        meta_from_dict(document)


def test_rejects_unknown_node_tag_with_repair_instruction() -> None:
    document = meta_to_dict(minimal_meta())
    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    entry = params[0]
    assert isinstance(entry, dict)
    value = entry["value"]
    assert isinstance(value, dict)
    distribution = value["distribution"]
    assert isinstance(distribution, dict)
    distribution["node"] = "MysteryDistribution"

    with pytest.raises(UnknownNodeTag, match=r"MysteryDistribution.*register"):
        meta_from_dict(document)


def test_rejects_non_dict_document() -> None:
    with pytest.raises(MalformedIRDocument, match="object"):
        meta_from_dict([1, 2, 3])


def test_rejects_missing_model_key() -> None:
    with pytest.raises(MalformedIRDocument, match="model"):
        meta_from_dict({"jaxstanv5_ir": 1})


def test_rejects_top_level_node_that_is_not_model_meta() -> None:
    with pytest.raises(MalformedIRDocument, match="ModelMeta"):
        meta_from_dict({"jaxstanv5_ir": 1, "model": {"node": "ParamRef", "name": "alpha"}})


def test_rejects_object_without_node_tag() -> None:
    document = meta_to_dict(minimal_meta())
    model = document["model"]
    assert isinstance(model, dict)
    expressions = model["expressions"]
    assert isinstance(expressions, list)
    entry = expressions[0]
    assert isinstance(entry, dict)
    entry["value"] = {"name": "oops"}

    with pytest.raises(MalformedIRDocument, match="node"):
        meta_from_dict(document)


def test_rejects_node_with_missing_field() -> None:
    document = meta_to_dict(minimal_meta())
    model = document["model"]
    assert isinstance(model, dict)
    expressions = model["expressions"]
    assert isinstance(expressions, list)
    entry = expressions[0]
    assert isinstance(entry, dict)
    entry["value"] = {"node": "BinOp", "op": "+", "left": {"node": "ParamRef", "name": "a"}}

    with pytest.raises(MalformedIRDocument, match="right"):
        meta_from_dict(document)


def test_rejects_node_with_unexpected_field() -> None:
    document = meta_to_dict(minimal_meta())
    model = document["model"]
    assert isinstance(model, dict)
    expressions = model["expressions"]
    assert isinstance(expressions, list)
    entry = expressions[0]
    assert isinstance(entry, dict)
    entry["value"] = {"node": "ParamRef", "name": "alpha", "extra": 1}

    with pytest.raises(MalformedIRDocument, match="extra"):
        meta_from_dict(document)


def test_rejects_map_entry_without_name_and_value() -> None:
    document = meta_to_dict(minimal_meta())
    model = document["model"]
    assert isinstance(model, dict)
    model["expressions"] = [{"key": "mu"}]

    with pytest.raises(MalformedIRDocument, match="name"):
        meta_from_dict(document)


def test_rejects_duplicate_map_entry_names() -> None:
    document = meta_to_dict(minimal_meta())
    model = document["model"]
    assert isinstance(model, dict)
    expressions = model["expressions"]
    assert isinstance(expressions, list)
    model["expressions"] = expressions + expressions

    with pytest.raises(MalformedIRDocument, match="mu"):
        meta_from_dict(document)
