"""Core model declaration types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution, SymbolicDistributionParameter
from jaxstanv5.model._data_schema import (
    DataDimSymbol,
    DataRankSchema,
    DataSchema,
    DataShapeDim,
    DataShapeSchema,
    is_scalar_data_schema,
)
from jaxstanv5.model._deferred import (
    DeclarationSymbol,
    DeferredBinOp,
    DeferredIndexOp,
    DeferredUnaryOp,
)
from jaxstanv5.model.dimensions import Dim, normalize_dims

_SYMBOL_IDS = count()


def _next_symbol() -> DeclarationSymbol:
    """Return a fresh declaration symbol."""
    return DeclarationSymbol(next(_SYMBOL_IDS))


def _validate_static_data_dimension(value: int, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _validate_data_rank(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("Data rank must be an integer, not bool")
    if value < 0:
        raise ValueError("Data rank must be non-negative")
    return value


@dataclass(frozen=True, init=False)
class Param(SymbolicDistributionParameter):
    """Parameter declaration used inside ``@model`` class bodies."""

    distribution: Distribution
    constraint: Constraint | None
    size: Data | int | None
    dims: tuple[Dim, ...] | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        distribution: Distribution,
        constraint: Constraint | None = None,
        size: Data | int | None = None,
        *,
        dims: Sequence[Dim] | None = None,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "dims", normalize_dims(dims))
        object.__setattr__(self, "symbol", _next_symbol())

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True, init=False)
class Data(SymbolicDistributionParameter):
    """Data declaration used inside ``@model`` class bodies."""

    schema: DataSchema
    dims: tuple[Dim, ...] | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        *,
        shape: Sequence[object] | None = None,
        rank: int | None = None,
        dims: Sequence[Dim] | None = None,
    ) -> None:
        if shape is None and rank is None:
            raise TypeError(
                "Data declarations require a schema. Use Data.scalar(), Data.vector(...), "
                "Data.matrix(...), or Data.array(...)."
            )
        if shape is not None and rank is not None:
            raise TypeError("Data declarations must specify either shape or rank, not both")

        if shape is not None:
            schema: DataSchema = DataShapeSchema(
                tuple(self._normalize_shape_dim(dim) for dim in shape)
            )
        else:
            if rank is None:
                raise TypeError("Data rank is required")
            schema = DataRankSchema(_validate_data_rank(rank))

        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "dims", normalize_dims(dims))
        object.__setattr__(self, "symbol", _next_symbol())

    @classmethod
    def scalar(cls, *, dims: Sequence[Dim] | None = None) -> Data:
        """Declare scalar data."""
        return cls(shape=(), dims=dims)

    @classmethod
    def vector(cls, length: object | None = None, *, dims: Sequence[Dim] | None = None) -> Data:
        """Declare rank-1 data, optionally with an exact length."""
        if length is None:
            return cls(rank=1, dims=dims)
        return cls(shape=(length,), dims=dims)

    @classmethod
    def matrix(
        cls,
        rows: object | None = None,
        cols: object | None = None,
        *,
        dims: Sequence[Dim] | None = None,
    ) -> Data:
        """Declare rank-2 data, optionally with an exact shape."""
        if rows is None and cols is None:
            return cls(rank=2, dims=dims)
        if rows is None or cols is None:
            raise TypeError("Data.matrix requires both rows and cols, or neither")
        return cls(shape=(rows, cols), dims=dims)

    @classmethod
    def array(
        cls,
        *,
        shape: Sequence[object] | None = None,
        rank: int | None = None,
        dims: Sequence[Dim] | None = None,
    ) -> Data:
        """Declare generic array data by exact shape or rank."""
        if shape is None and rank is None:
            raise TypeError("Data.array requires shape or rank")
        return cls(shape=shape, rank=rank, dims=dims)

    @staticmethod
    def _normalize_shape_dim(dim: object) -> DataShapeDim:
        if isinstance(dim, bool):
            raise TypeError("Data shape dimensions must be integers, not bool")
        if isinstance(dim, int):
            return _validate_static_data_dimension(dim, label="Data shape dimension")
        if isinstance(dim, Data):
            if not is_scalar_data_schema(dim.schema):
                raise TypeError("Data shape dimensions must reference scalar data declarations")
            return DataDimSymbol(dim.symbol)
        raise TypeError("Data shape dimensions must be integers or scalar data declarations")

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True, init=False)
class Observed:
    """Observed variable declaration."""

    distribution: Distribution
    dims: tuple[Dim, ...] | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        distribution: Distribution,
        *,
        dims: Sequence[Dim] | None = None,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "dims", normalize_dims(dims))
        object.__setattr__(self, "symbol", _next_symbol())


@dataclass(frozen=True, init=False)
class PartiallyObserved(SymbolicDistributionParameter):
    """Partially observed vector declaration.

    The declaration contributes one log-density factor evaluated at the full
    assembled vector. Coordinates listed by ``missing_idx`` are free NUTS
    coordinates; coordinates listed by ``observed_idx`` are fixed data.
    """

    distribution: Distribution
    length: Data | int
    observed: Data
    observed_idx: Data
    missing_idx: Data
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        distribution: Distribution,
        *,
        length: Data | int,
        observed: Data,
        observed_idx: Data,
        missing_idx: Data,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "length", _validate_partial_vector_length(length))
        object.__setattr__(
            self,
            "observed",
            _validate_exact_vector_data(observed, label="PartiallyObserved observed values"),
        )
        object.__setattr__(
            self,
            "observed_idx",
            _validate_exact_vector_data(observed_idx, label="PartiallyObserved observed_idx"),
        )
        object.__setattr__(
            self,
            "missing_idx",
            _validate_exact_vector_data(missing_idx, label="PartiallyObserved missing_idx"),
        )
        object.__setattr__(self, "symbol", _next_symbol())

    @classmethod
    def vector(
        cls,
        distribution: Distribution,
        *,
        length: Data | int,
        observed: Data,
        observed_idx: Data,
        missing_idx: Data,
    ) -> PartiallyObserved:
        """Declare a rank-1 random vector with explicit observed/missing coordinates."""
        return cls(
            distribution,
            length=length,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


def _validate_partial_vector_length(length: Data | int) -> Data | int:
    if isinstance(length, bool):
        raise TypeError("PartiallyObserved.vector length must be an integer, not bool")
    if isinstance(length, int):
        return _validate_static_data_dimension(length, label="PartiallyObserved.vector length")
    if isinstance(length, Data):
        if not is_scalar_data_schema(length.schema):
            raise TypeError("PartiallyObserved.vector length must reference scalar data")
        return length
    raise TypeError("PartiallyObserved.vector length must be an integer or scalar data declaration")


def _validate_exact_vector_data(value: Data, *, label: str) -> Data:
    if not isinstance(value, Data):
        raise TypeError(f"{label} must be a Data.vector(length) declaration")
    if not isinstance(value.schema, DataShapeSchema) or len(value.schema.dims) != 1:
        raise TypeError(f"{label} must be a Data.vector(length) declaration")
    return value
