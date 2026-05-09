"""Symbolic expression nodes for model declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

type ExprNode = ParamRef | DataRef | ConstNode | BinOp | IndexOp
type BinaryOperator = str


class Expression(Protocol):
    """Symbolic expression protocol supporting model-body operations."""

    def __add__(self, other: object) -> BinOp: ...
    def __radd__(self, other: object) -> BinOp: ...
    def __sub__(self, other: object) -> BinOp: ...
    def __rsub__(self, other: object) -> BinOp: ...
    def __mul__(self, other: object) -> BinOp: ...
    def __rmul__(self, other: object) -> BinOp: ...
    def __truediv__(self, other: object) -> BinOp: ...
    def __rtruediv__(self, other: object) -> BinOp: ...
    def __getitem__(self, index: object) -> IndexOp: ...


@dataclass(frozen=True)
class ParamRef:
    """Reference to a model parameter by name."""

    name: str

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_expr(other), self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, to_expr(index))


@dataclass(frozen=True)
class DataRef:
    """Reference to bound model data by name."""

    name: str

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_expr(other), self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, to_expr(index))


@dataclass(frozen=True)
class ConstNode:
    """Literal scalar constant in a symbolic expression."""

    value: int | float

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_expr(other), self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, to_expr(index))


@dataclass(frozen=True)
class BinOp:
    """Binary operation between two expression nodes."""

    op: BinaryOperator
    left: ExprNode
    right: ExprNode

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_expr(other), self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, to_expr(index))


@dataclass(frozen=True)
class IndexOp:
    """Indexing operation over a base expression."""

    base: ExprNode
    index: ExprNode

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", to_expr(other), self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, to_expr(index))


def to_expr(value: object) -> ExprNode:
    """Convert supported Python values to expression nodes."""
    if isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp):
        return value
    if isinstance(value, int | float):
        return ConstNode(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to an expression node")
