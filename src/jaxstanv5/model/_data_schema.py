"""Private data declaration schema types."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstanv5.model._deferred import DeclarationSymbol


@dataclass(frozen=True)
class DataDimSymbol:
    """Declaration-time reference to a scalar data dimension."""

    symbol: DeclarationSymbol


type DataShapeDim = int | DataDimSymbol


@dataclass(frozen=True)
class DataShapeSchema:
    """Exact data shape schema."""

    dims: tuple[DataShapeDim, ...]


@dataclass(frozen=True)
class DataRankSchema:
    """Rank-only data schema."""

    rank: int


type DataSchema = DataShapeSchema | DataRankSchema


@dataclass(frozen=True)
class DataDimRef:
    """Resolved reference to a scalar data dimension by name."""

    name: str


type ResolvedDataShapeDim = int | DataDimRef


@dataclass(frozen=True)
class ResolvedDataShapeSchema:
    """Resolved exact data shape schema."""

    dims: tuple[ResolvedDataShapeDim, ...]


@dataclass(frozen=True)
class ResolvedDataRankSchema:
    """Resolved rank-only data schema."""

    rank: int


type ResolvedDataSchema = ResolvedDataShapeSchema | ResolvedDataRankSchema


def is_scalar_data_schema(schema: DataSchema) -> bool:
    """Return whether a declaration schema describes scalar data."""
    if isinstance(schema, DataShapeSchema):
        return schema.dims == ()
    return schema.rank == 0
