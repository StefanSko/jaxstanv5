"""Unit tests for canonical IR bytes."""

from __future__ import annotations

import hashlib
import json

from _ir_meta import minimal_meta, rich_meta

from jaxstanv5.distributions import Normal
from jaxstanv5.ir import canonical_bytes, meta_from_dict, meta_to_dict
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedFreeValue,
    ResolvedParam,
    ResolvedStochasticSite,
)
from jaxstanv5.model.expr import ParamRef


def test_canonical_bytes_parse_back_to_the_document() -> None:
    meta = rich_meta()

    parsed = json.loads(canonical_bytes(meta).decode("utf-8"))

    assert parsed == meta_to_dict(meta)
    assert meta_from_dict(parsed) == meta


def test_canonical_bytes_are_compact() -> None:
    payload = canonical_bytes(minimal_meta())

    assert b": " not in payload
    assert b", " not in payload
    assert b"\n" not in payload


def test_canonical_bytes_keep_unicode_unescaped() -> None:
    meta = ModelMeta(
        params={"μ": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"μ": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(ResolvedStochasticSite("μ", Normal(0.0, 1.0), ParamRef("μ")),),
    )

    payload = canonical_bytes(meta)

    assert "μ".encode() in payload
    assert b"\\u" not in payload


def test_canonical_hash_is_stable_across_equal_metadata() -> None:
    first = hashlib.sha256(canonical_bytes(rich_meta())).hexdigest()
    second = hashlib.sha256(canonical_bytes(rich_meta())).hexdigest()

    assert first == second
