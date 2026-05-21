"""Internal pending IR for ``@model`` class-body execution.

This module is package-private. It exists only to make the declaration phase
explicit before unresolved symbols are resolved into final model expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

type PendingRef = PendingParamRef | PendingDataRef
type PendingExprNode = PendingRef | PendingConst | PendingBinOp | PendingIndexOp
type PendingBinaryOperator = str


@dataclass(frozen=True)
class UnresolvedSymbol:
    """Declaration symbol before model metadata resolution."""

    id: int


@dataclass(frozen=True)
class PendingParamRef:
    """Unresolved reference to a parameter declaration."""

    name: UnresolvedSymbol

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self, to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self)

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self, to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self)

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self, to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self)

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self, to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self)

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self, to_pending_expr(index))


@dataclass(frozen=True)
class PendingDataRef:
    """Unresolved reference to a data declaration."""

    name: UnresolvedSymbol

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self, to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self)

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self, to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self)

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self, to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self)

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self, to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self)

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self, to_pending_expr(index))


@dataclass(frozen=True)
class PendingConst:
    """Literal scalar in a pending class-body expression."""

    value: int | float

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self, to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self)

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self, to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self)

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self, to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self)

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self, to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self)

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self, to_pending_expr(index))


@dataclass(frozen=True)
class PendingBinOp:
    """Pending binary operation built during class-body execution."""

    op: PendingBinaryOperator
    left: PendingExprNode
    right: PendingExprNode

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self, to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self)

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self, to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self)

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self, to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self)

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self, to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self)

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self, to_pending_expr(index))


@dataclass(frozen=True)
class PendingIndexOp:
    """Pending indexing operation built during class-body execution."""

    base: PendingExprNode
    index: PendingExprNode

    def __add__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", self, to_pending_expr(other))

    def __radd__(self, other: object) -> PendingBinOp:
        return PendingBinOp("+", to_pending_expr(other), self)

    def __sub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", self, to_pending_expr(other))

    def __rsub__(self, other: object) -> PendingBinOp:
        return PendingBinOp("-", to_pending_expr(other), self)

    def __mul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", self, to_pending_expr(other))

    def __rmul__(self, other: object) -> PendingBinOp:
        return PendingBinOp("*", to_pending_expr(other), self)

    def __truediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", self, to_pending_expr(other))

    def __rtruediv__(self, other: object) -> PendingBinOp:
        return PendingBinOp("/", to_pending_expr(other), self)

    def __getitem__(self, index: object) -> PendingIndexOp:
        return PendingIndexOp(self, to_pending_expr(index))


def is_pending_expr(value: object) -> bool:
    return isinstance(
        value,
        PendingParamRef | PendingDataRef | PendingConst | PendingBinOp | PendingIndexOp,
    )


def to_pending_expr(value: object) -> PendingExprNode:
    """Convert supported class-body values to pending expression nodes."""
    from jaxstanv5.model.core import Data, Param

    if isinstance(value, Param | Data):
        return value.ref()
    if isinstance(
        value, PendingParamRef | PendingDataRef | PendingConst | PendingBinOp | PendingIndexOp
    ):
        return value
    if isinstance(value, int | float):
        return PendingConst(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to a pending expression")
