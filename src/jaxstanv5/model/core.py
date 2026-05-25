"""Core model declaration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution, SymbolicDistributionParameter
from jaxstanv5.model._deferred import DeclarationSymbol, DeferredBinOp, DeferredIndexOp

_SYMBOL_IDS = count()


def _next_symbol() -> DeclarationSymbol:
    """Return a fresh declaration symbol."""
    return DeclarationSymbol(next(_SYMBOL_IDS))


@dataclass(frozen=True)
class Param(SymbolicDistributionParameter):
    """Parameter declaration used inside ``@model`` class bodies."""

    distribution: Distribution
    constraint: Constraint | None = None
    size: Data | int | None = None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

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

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True)
class Data(SymbolicDistributionParameter):
    """Data declaration used inside ``@model`` class bodies."""

    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

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

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True)
class Observed:
    """Observed variable declaration."""

    distribution: Distribution
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)
