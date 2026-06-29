"""Authoring-side dimension labels and coordinate metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

type CoordValue = None | bool | int | float | str


class DimensionMetadataDocument(TypedDict):
    """JSON-ready dimension metadata document."""

    dims: dict[str, list[str]]
    coords: dict[str, list[CoordValue]]


@dataclass(frozen=True, init=False)
class Dim:
    """Semantic dimension label declared in model-authoring code."""

    name: str
    coords: tuple[CoordValue, ...] | None

    def __init__(self, name: str, *, coords: Sequence[object] | None = None) -> None:
        if not isinstance(name, str):
            raise TypeError("Dimension names must be strings")
        if name == "":
            raise ValueError("Dimension names must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coords", _normalize_coords(coords))


@dataclass(frozen=True)
class ResolvedVariableDims:
    """Resolved ordered dimension labels for one model variable."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedModelDimensions:
    """Resolved dimension labels and optional coordinates for a model."""

    variables: dict[str, ResolvedVariableDims]
    coords: dict[str, tuple[CoordValue, ...]]


def normalize_dims(dims: Sequence[Dim] | None) -> tuple[Dim, ...] | None:
    """Normalize an optional declaration-time dimension sequence."""
    if dims is None:
        return None
    normalized = tuple(dims)
    for dim in normalized:
        if not isinstance(dim, Dim):
            raise TypeError("Declaration dims must contain Dim instances")
    return normalized


def model_dimensions(model_cls: type[object]) -> ResolvedModelDimensions:
    """Return resolved dimension metadata attached by ``@model``."""
    metadata = model_cls.__dict__.get("_model_dimensions")
    if isinstance(metadata, ResolvedModelDimensions):
        return metadata
    raise TypeError("Model class has no resolved dimension metadata; decorate it with @model")


def dimension_metadata_to_dict(metadata: ResolvedModelDimensions) -> DimensionMetadataDocument:
    """Convert resolved dimension metadata to a JSON-ready document."""
    return {
        "dims": {
            name: list(variable_dims.names) for name, variable_dims in metadata.variables.items()
        },
        "coords": {name: list(coords) for name, coords in metadata.coords.items()},
    }


def _normalize_coords(coords: Sequence[object] | None) -> tuple[CoordValue, ...] | None:
    if coords is None:
        return None
    return tuple(_normalize_coord(value) for value in coords)


def _normalize_coord(value: object) -> CoordValue:
    if value is None or isinstance(value, bool | int | str):
        return cast(CoordValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Dimension coordinate floats must be finite")
        return value
    raise TypeError(
        "Dimension coordinates must be JSON scalar values: str, int, float, bool, or None"
    )
