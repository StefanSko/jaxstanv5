"""Core model declaration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.model._pending import (
    PendingBinOp,
    PendingDataRef,
    PendingExprNode,
    PendingIndexOp,
    PendingParamRef,
    UnresolvedSymbol,
    to_pending_expr,
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
    size: Data | PendingDataRef | int | None = None
    symbol: UnresolvedSymbol = field(default_factory=next_symbol, init=False, repr=False)

    def ref(self) -> PendingParamRef:
        """Return a pending reference to this unresolved parameter."""
        return PendingParamRef(self.symbol)

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self.ref(), to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self.ref())

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self.ref(), to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self.ref())

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self.ref(), to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self.ref())

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self.ref(), to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self.ref())

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self.ref(), to_pending_expr(index))


@dataclass(frozen=True)
class Data:
    """Data declaration used inside ``@model`` class bodies."""

    symbol: UnresolvedSymbol = field(default_factory=next_symbol, init=False, repr=False)

    def ref(self) -> PendingDataRef:
        """Return a pending reference to this unresolved data slot."""
        return PendingDataRef(self.symbol)

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self.ref(), to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self.ref())

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self.ref(), to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self.ref())

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self.ref(), to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self.ref())

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self.ref(), to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self.ref())

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self.ref(), to_pending_expr(index))


@dataclass(frozen=True)
class Observed:
    """Observed variable declaration."""

    distribution: Distribution


def to_declaration_expr(value: object) -> PendingExprNode:
    """Convert declaration operands to pending expression nodes."""
    return to_pending_expr(value)
