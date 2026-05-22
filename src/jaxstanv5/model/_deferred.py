"""Internal deferred syntax captured during ``@model`` class-body execution.

These tokens are not semantic expression IR. They only remember what Python
operators evaluated before declaration names are available. The decorator turns
these tokens into final ``expr.py`` nodes in one explicit resolution step.
"""

from __future__ import annotations

from dataclasses import dataclass

type DeferredExpr = DeferredBinOp | DeferredIndexOp
type DeferredBinaryOperator = str


@dataclass(frozen=True)
class DeclarationSymbol:
    """Stable declaration identity before class attribute names are known."""

    id: int


@dataclass(frozen=True)
class DeferredBinOp:
    """Deferred binary operation from class-body syntax."""

    op: DeferredBinaryOperator
    left: object
    right: object

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
class DeferredIndexOp:
    """Deferred indexing operation from class-body syntax."""

    base: object
    index: object

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


def is_deferred_expr(value: object) -> bool:
    """Return whether ``value`` is captured class-body syntax."""
    return isinstance(value, DeferredBinOp | DeferredIndexOp)
