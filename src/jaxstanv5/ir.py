"""IR serialization for resolved model metadata (``jaxstanv5_ir`` version 1).

The wire format is a versioned JSON document over the closed inventory of
resolved model-metadata dataclasses. Registered dataclasses encode as tagged
objects, ordered ``dict[str, ...]`` fields encode as ``{"name", "value"}``
entry arrays in insertion order, and tuples encode as plain arrays. The tag
vocabulary, not Python class names, is the cross-language contract; see
``docs/ir-format-v1.md`` and the generated ``docs/ir-v1-tags.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import is_dataclass
from typing import cast

from jaxstanv5._ir_registry import (
    CORE_PROFILE_TAGS,
    NODE_KEY,
    NODE_SPECS_BY_CLASS,
    NODE_SPECS_BY_TAG,
    FieldKind,
    NodeSpec,
)
from jaxstanv5._ir_registry import (
    register_node as _register_node,
)
from jaxstanv5.model.decorator import ModelMeta, _make_bind
from jaxstanv5.model.dimensions import ResolvedModelDimensions

__all__ = [
    "IR_VERSION",
    "IRSerializationError",
    "MalformedIRDocument",
    "NonFiniteConstant",
    "UnknownNodeTag",
    "UnserializableDistribution",
    "UnserializableValue",
    "UnsupportedIRVersion",
    "bindable_from_meta",
    "canonical_bytes",
    "meta_from_dict",
    "meta_to_dict",
    "register_distribution",
    "render_ir_v1_tag_spec",
]

IR_VERSION = 1

_VERSION_KEY = "jaxstanv5_ir"
_MODEL_KEY = "model"

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class IRSerializationError(Exception):
    """Base error for IR serialization and deserialization failures."""


class NonFiniteConstant(IRSerializationError):
    """A non-finite float was found while encoding model metadata."""


class UnserializableDistribution(IRSerializationError):
    """A distribution is not registered for IR serialization."""


class UnserializableValue(IRSerializationError):
    """A leaf value has no IR encoding."""


class UnsupportedIRVersion(IRSerializationError):
    """The document version is missing or not readable by this build."""


class UnknownNodeTag(IRSerializationError):
    """A document node tag is not in the registry."""


class MalformedIRDocument(IRSerializationError):
    """The document structure violates the IR format."""


def register_distribution(cls: type, *, tag: str | None = None) -> None:
    """Register a user distribution dataclass for IR serialization.

    Documents using extension tags are consumable only by Python processes
    that imported the registering package; non-Python consumers reject them.
    """
    if not is_dataclass(cls):
        raise UnserializableDistribution(
            f"Distribution {cls.__name__!r} must be a dataclass to be IR-serializable. "
            "Decorate it with @dataclass(frozen=True) before calling "
            "jaxstanv5.ir.register_distribution, or replace it with a built-in distribution."
        )
    _register_node(cls, tag=tag)


def meta_to_dict(meta: ModelMeta) -> dict[str, JsonValue]:
    """Encode resolved model metadata into a versioned IR document."""
    return {_VERSION_KEY: IR_VERSION, _MODEL_KEY: _encode_value(meta)}


def _encode_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteConstant(
                f"Non-finite constant {value!r} cannot be encoded: strict JSON has no "
                "inf/nan tokens. Replace it with a finite constant in the model declaration."
            )
        return value
    if isinstance(value, dict):
        return _encode_map(cast("dict[object, object]", value))
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    spec = NODE_SPECS_BY_CLASS.get(type(value))
    if spec is not None:
        return _encode_node(value, spec)
    if hasattr(value, "log_prob"):
        raise UnserializableDistribution(
            f"Distribution {type(value).__name__!r} is not registered for IR serialization. "
            "Decorate it with @dataclass(frozen=True) and call "
            f"jaxstanv5.ir.register_distribution({type(value).__name__}), "
            "or replace it with a built-in distribution."
        )
    raise UnserializableValue(
        f"Values of type {type(value).__name__!r} have no IR encoding. Use Python "
        "int/float/str/bool scalars, expression nodes, or registered dataclass nodes."
    )


def _encode_map(value: dict[object, object]) -> list[JsonValue]:
    entries: list[JsonValue] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise UnserializableValue(
                f"IR map keys must be strings, got {type(key).__name__!r}. "
                "Use string-keyed dictionaries in resolved model metadata."
            )
        entries.append({"name": key, "value": _encode_value(item)})
    return entries


def _encode_node(value: object, spec: NodeSpec) -> dict[str, JsonValue]:
    encoded: dict[str, JsonValue] = {NODE_KEY: spec.tag}
    for name, _kind in spec.field_kinds:
        encoded[name] = _encode_value(getattr(value, name))
    return encoded


def canonical_bytes(meta: ModelMeta) -> bytes:
    """Return the canonical UTF-8 JSON bytes of the IR document.

    The model hash is ``sha256(canonical_bytes(meta))``, computed by the
    producer at write time. Consumers hash the file as received and never
    re-serialize to hash.
    """
    return json.dumps(
        meta_to_dict(meta),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def bindable_from_meta(
    meta: ModelMeta,
    *,
    dimensions: ResolvedModelDimensions | None = None,
) -> type[object]:
    """Return a bindable model class equivalent to one produced by ``@model``.

    Everything downstream of ``bind`` is unchanged and unaware of which path
    produced the metadata. Dimension labels are a separate sidecar document;
    pass the decoded ``dimensions`` (see ``dimension_metadata_from_dict``) to
    reconstruct a ``Dim(...)``-labeled model faithfully.
    """
    if dimensions is not None:
        _validate_dimension_variables(meta, dimensions)
    namespace: dict[str, object] = {
        "_model_meta": meta,
        "bind": classmethod(_make_bind(meta, dimensions=dimensions)),
    }
    if dimensions is not None:
        namespace["_model_dimensions"] = dimensions
    return type("IRModel", (object,), namespace)


def _validate_dimension_variables(meta: ModelMeta, dimensions: ResolvedModelDimensions) -> None:
    declared = set(meta.params) | set(meta.data) | {node.name for node in meta.observed_nodes}
    unknown = sorted(set(dimensions.variables) - declared)
    if unknown:
        raise ValueError(
            f"Dimension metadata references variables not declared by the model: {unknown}"
        )


def meta_from_dict(document: object) -> ModelMeta:
    """Decode a versioned IR document into resolved model metadata."""
    if not isinstance(document, dict):
        raise MalformedIRDocument(
            f"IR documents must be JSON objects, got {type(document).__name__!r}. "
            "Pass the parsed top-level object of a jaxstanv5_ir file."
        )
    envelope = cast("dict[str, object]", document)
    version = envelope.get(_VERSION_KEY)
    if version != IR_VERSION:
        raise UnsupportedIRVersion(
            f"Unsupported IR document version {version!r}; this build reads "
            f"{_VERSION_KEY} version {IR_VERSION}. Regenerate the document with "
            "meta_to_dict(...) or use a jaxstanv5 build that reads this version."
        )
    if _MODEL_KEY not in envelope:
        raise MalformedIRDocument(
            f"IR documents must contain a {_MODEL_KEY!r} key holding the encoded model."
        )
    decoded = _decode_node(envelope[_MODEL_KEY])
    if not isinstance(decoded, ModelMeta):
        raise MalformedIRDocument(
            f"The top-level {_MODEL_KEY!r} node must decode to ModelMeta, "
            f"got {type(decoded).__name__!r}."
        )
    return decoded


def _decode_node(value: object) -> object:
    if not isinstance(value, dict):
        raise MalformedIRDocument(
            f"IR nodes must be JSON objects with a {NODE_KEY!r} tag, got {type(value).__name__!r}."
        )
    encoded = cast("dict[str, object]", value)
    tag = encoded.get(NODE_KEY)
    if not isinstance(tag, str):
        raise MalformedIRDocument(
            f"IR node objects must carry a string {NODE_KEY!r} tag; got keys {sorted(encoded)}."
        )
    spec = NODE_SPECS_BY_TAG.get(tag)
    if spec is None:
        raise UnknownNodeTag(
            f"Unknown IR node tag {tag!r}. Import and register the package that defines it "
            "(jaxstanv5.ir.register_distribution(...)) before decoding, or restrict the "
            "document to the built-in tags listed in docs/ir-v1-tags.md."
        )
    expected = {name for name, _kind in spec.field_kinds}
    present = set(encoded) - {NODE_KEY}
    missing = expected - present
    extra = present - expected
    if missing:
        raise MalformedIRDocument(f"IR node {tag!r} is missing fields: {sorted(missing)}")
    if extra:
        raise MalformedIRDocument(f"IR node {tag!r} has unexpected fields: {sorted(extra)}")
    kwargs = {
        name: _decode_field(encoded[name], kind, tag=tag, field_name=name)
        for name, kind in spec.field_kinds
    }
    return spec.cls(**kwargs)


def _decode_field(value: object, kind: FieldKind, *, tag: str, field_name: str) -> object:
    if kind is FieldKind.MAP:
        return _decode_map(value, tag=tag, field_name=field_name)
    if kind is FieldKind.TUPLE:
        if not isinstance(value, list):
            raise MalformedIRDocument(
                f"Field {field_name!r} of IR node {tag!r} must be an array, "
                f"got {type(value).__name__!r}."
            )
        return tuple(_decode_value(item) for item in value)
    return _decode_value(value)


def _decode_map(value: object, *, tag: str, field_name: str) -> dict[str, object]:
    if not isinstance(value, list):
        raise MalformedIRDocument(
            f"Field {field_name!r} of IR node {tag!r} must be an array of "
            f"name/value entries, got {type(value).__name__!r}."
        )
    decoded: dict[str, object] = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"name", "value"}:
            raise MalformedIRDocument(
                f"Entries of field {field_name!r} of IR node {tag!r} must be objects "
                'with exactly the keys "name" and "value".'
            )
        pair = cast("dict[str, object]", entry)
        name = pair["name"]
        if not isinstance(name, str):
            raise MalformedIRDocument(
                f"Entry names in field {field_name!r} of IR node {tag!r} must be strings."
            )
        if name in decoded:
            raise MalformedIRDocument(
                f"Field {field_name!r} of IR node {tag!r} has a duplicate entry name {name!r}."
            )
        decoded[name] = _decode_value(pair["value"])
    return decoded


def _decode_value(value: object) -> object:
    if isinstance(value, dict):
        return _decode_node(value)
    if isinstance(value, list):
        raise MalformedIRDocument(
            "Arrays are only valid as tuple or map fields of IR nodes; "
            "a bare array has no IR meaning here."
        )
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise MalformedIRDocument(f"Values of type {type(value).__name__!r} have no IR decoding.")


def render_ir_v1_tag_spec() -> str:
    """Render the generated built-in tag and field spec for ``docs/ir-v1-tags.md``."""
    lines = [
        "# jaxstanv5 IR v1 — built-in node tags",
        "",
        "Generated by `scripts/regenerate_ir_golden.py` from the IR node registry;",
        "do not edit by hand. Documents confined to these tags form the core",
        "profile consumable by all backends and tools. Registry-extended documents",
        "are consumable only by Python processes that imported the registering",
        "package; other consumers reject their tags as unknown.",
        "",
        "Field kinds: `value` is a scalar, node, or null; `map` is an ordered",
        '`{"name", "value"}` entry array; `tuple` is a plain array.',
        "",
        "| Tag | Fields |",
        "|---|---|",
    ]
    for tag in CORE_PROFILE_TAGS:
        spec = NODE_SPECS_BY_TAG[tag]
        rendered = ", ".join(f"`{name}` ({kind.value})" for name, kind in spec.field_kinds)
        lines.append(f"| `{tag}` | {rendered if rendered else '—'} |")
    return "\n".join(lines) + "\n"
