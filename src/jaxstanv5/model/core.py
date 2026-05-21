"""Core model declaration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.model.expr import (
    BinOp,
    DataRef,
    ExprNode,
    IndexOp,
    ParamRef,
    UnresolvedSymbol,
    to_expr,
)

_SYMBOL_IDS = count()


def next_symbol() -> UnresolvedSymbol:
    """Return a fresh unresolved declaration symbol."""
    return UnresolvedSymbol(next(_SYMBOL_IDS))


@dataclass(frozen=True)
class Param:
    """Parameter declaration used inside ``@model`` class bodies."""

    distribution: Distribution
    constraint: Constraint | None = None
    size: Data | DataRef | int | None = None
    symbol: UnresolvedSymbol = field(default_factory=next_symbol, init=False, repr=False)

    def ref(self) -> ParamRef:
        """Return a symbolic reference to this unresolved parameter."""
        return ParamRef(self.symbol)

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self.ref(), to_declaration_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_declaration_expr(other), self.ref())

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self.ref(), to_declaration_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_declaration_expr(other), self.ref())

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self.ref(), to_declaration_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_declaration_expr(other), self.ref())

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self.ref(), to_declaration_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_declaration_expr(other), self.ref())

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self.ref(), to_declaration_expr(index))


@dataclass(frozen=True)
class Data:
    """Data declaration used inside ``@model`` class bodies."""

    symbol: UnresolvedSymbol = field(default_factory=next_symbol, init=False, repr=False)

    def ref(self) -> DataRef:
        """Return a symbolic reference to this unresolved data slot."""
        return DataRef(self.symbol)

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self.ref(), to_declaration_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_declaration_expr(other), self.ref())

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self.ref(), to_declaration_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_declaration_expr(other), self.ref())

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self.ref(), to_declaration_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_declaration_expr(other), self.ref())

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self.ref(), to_declaration_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_declaration_expr(other), self.ref())

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self.ref(), to_declaration_expr(index))


@dataclass(frozen=True)
class Observed:
    """Observed variable declaration."""

    distribution: Distribution


def to_declaration_expr(value: object) -> ExprNode:
    """Convert declaration operands to expression nodes."""
    if isinstance(value, Param | Data):
        return value.ref()
    return to_expr(value)
