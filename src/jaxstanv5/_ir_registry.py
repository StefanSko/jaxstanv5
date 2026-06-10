"""Private IR node registry: tag vocabulary and field-kind classification.

The registry maps the closed inventory of resolved model-metadata dataclasses
to wire tags and records, once per class, how each constructor field is
represented in the IR document. ``jaxstanv5.ir`` is the public surface.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import get_origin, get_type_hints

from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval
from jaxstanv5.distributions import (
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Exponential,
    HalfNormal,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    OrderedLogistic,
    Poisson,
    StudentT,
    Uniform,
)
from jaxstanv5.model._data_schema import (
    DataDimRef,
    ResolvedDataRankSchema,
    ResolvedDataShapeSchema,
)
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedData,
    ResolvedFreeValue,
    ResolvedObserved,
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

NODE_KEY = "node"


class FieldKind(Enum):
    """How a registered node field is represented in the IR document."""

    MAP = "map"
    TUPLE = "tuple"
    VALUE = "value"


@dataclass(frozen=True)
class NodeSpec:
    """Registered IR node class with its wire tag and field kinds."""

    cls: type
    tag: str
    field_kinds: tuple[tuple[str, FieldKind], ...]


NODE_SPECS_BY_TAG: dict[str, NodeSpec] = {}
NODE_SPECS_BY_CLASS: dict[type, NodeSpec] = {}


def register_node(cls: type, *, tag: str | None = None) -> None:
    """Register a dataclass as a tagged IR node.

    The tag defaults to the class name; pass ``tag=...`` to keep the wire
    vocabulary stable across Python class renames.
    """
    if not is_dataclass(cls):
        raise TypeError(
            f"IR node classes must be dataclasses; {cls.__name__} is not. "
            "Decorate it with @dataclass(frozen=True)."
        )
    node_tag = cls.__name__ if tag is None else tag
    existing = NODE_SPECS_BY_TAG.get(node_tag)
    if existing is not None:
        if existing.cls is cls:
            return
        raise ValueError(
            f"IR node tag {node_tag!r} is already registered to {existing.cls.__name__}. "
            "Pass register_node(cls, tag=...) with a unique tag."
        )
    registered = NODE_SPECS_BY_CLASS.get(cls)
    if registered is not None:
        raise ValueError(
            f"{cls.__name__} is already registered as IR tag {registered.tag!r}; "
            "one class maps to one tag."
        )
    spec = NodeSpec(cls=cls, tag=node_tag, field_kinds=_classify_fields(cls))
    NODE_SPECS_BY_TAG[node_tag] = spec
    NODE_SPECS_BY_CLASS[cls] = spec


def _classify_fields(cls: type) -> tuple[tuple[str, FieldKind], ...]:
    """Record each constructor field's wire representation once, at registration."""
    if not is_dataclass(cls):
        raise TypeError(f"Cannot classify fields of non-dataclass {cls.__name__}")
    hints = get_type_hints(cls)
    kinds: list[tuple[str, FieldKind]] = []
    for field in fields(cls):
        if not field.init:
            continue
        if field.name == NODE_KEY:
            raise ValueError(
                f"IR node classes cannot have a field named {NODE_KEY!r}: {cls.__name__}. "
                "Rename the field before registering the class."
            )
        kinds.append((field.name, _field_kind(hints[field.name])))
    return tuple(kinds)


def _field_kind(hint: object) -> FieldKind:
    origin = get_origin(hint)
    if origin is dict:
        return FieldKind.MAP
    if origin is tuple:
        return FieldKind.TUPLE
    return FieldKind.VALUE


_BUILTIN_NODE_CLASSES: tuple[type, ...] = (
    # Expression nodes
    ParamRef,
    DataRef,
    ConstNode,
    BinOp,
    UnaryOp,
    IndexOp,
    VectorScatterOp,
    # Index specifications
    ScalarIndex,
    FullSlice,
    IndexTuple,
    # Resolved data schemas
    DataDimRef,
    ResolvedDataShapeSchema,
    ResolvedDataRankSchema,
    # Resolved declarations
    ResolvedParam,
    ResolvedData,
    ResolvedObserved,
    ResolvedFreeValue,
    ResolvedStochasticSite,
    ModelMeta,
    # Constraints
    Positive,
    Interval,
    UnitInterval,
    Ordered,
    # Built-in distributions
    Normal,
    HalfNormal,
    StudentT,
    Exponential,
    Uniform,
    Beta,
    Bernoulli,
    Poisson,
    Binomial,
    BetaBinomial,
    NegativeBinomial,
    MultivariateNormal,
    OrderedLogistic,
)

for _builtin in _BUILTIN_NODE_CLASSES:
    register_node(_builtin)

CORE_PROFILE_TAGS: tuple[str, ...] = tuple(NODE_SPECS_BY_TAG)
