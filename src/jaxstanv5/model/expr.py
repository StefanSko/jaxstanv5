"""Symbolic expression nodes for model declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jaxstanv5.distributions.core import SymbolicDistributionParameter
from jaxstanv5.model._expression_errors import array_like_constant_error, is_array_like_constant

type ExprNode = ParamRef | DataRef | ConstNode | BinOp | IndexOp | UnaryOp | VectorScatterOp
type IndexSpec = ScalarIndex | FullSlice | IndexTuple
type BinaryOperator = str
type UnaryFunction = str


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
    def __neg__(self) -> UnaryOp: ...
    def __getitem__(self, index: object) -> IndexOp: ...


@dataclass(frozen=True)
class ScalarIndex(SymbolicDistributionParameter):
    """Integer scalar or array index expression for one indexed axis."""

    expr: ExprNode


@dataclass(frozen=True)
class FullSlice(SymbolicDistributionParameter):
    """A full-axis ``:`` index item."""


@dataclass(frozen=True)
class IndexTuple(SymbolicDistributionParameter):
    """Multi-axis index specification."""

    items: tuple[IndexSpec, ...]


@dataclass(frozen=True)
class ParamRef(SymbolicDistributionParameter):
    """Reference to a model parameter by name."""

    name: str

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", _to_expr(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, _to_index_spec(index))


@dataclass(frozen=True)
class DataRef(SymbolicDistributionParameter):
    """Reference to bound model data by name."""

    name: str

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", _to_expr(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, _to_index_spec(index))


@dataclass(frozen=True)
class ConstNode(SymbolicDistributionParameter):
    """Literal scalar constant in a symbolic expression."""

    value: int | float

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", _to_expr(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, _to_index_spec(index))


@dataclass(frozen=True)
class BinOp(SymbolicDistributionParameter):
    """Binary operation between two expression nodes."""

    op: BinaryOperator
    left: ExprNode
    right: ExprNode

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", _to_expr(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, _to_index_spec(index))


@dataclass(frozen=True)
class UnaryOp(SymbolicDistributionParameter):
    """Unary function applied to one expression node."""

    function: UnaryFunction
    operand: ExprNode

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", _to_expr(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, _to_index_spec(index))


@dataclass(frozen=True)
class IndexOp(SymbolicDistributionParameter):
    """Indexing operation over a base expression."""

    base: ExprNode
    index: IndexSpec

    def __add__(self, other: object) -> BinOp:
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: object) -> BinOp:
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: object) -> BinOp:
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("/", _to_expr(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", self)

    def __getitem__(self, index: object) -> IndexOp:
        return IndexOp(self, _to_index_spec(index))


@dataclass(frozen=True)
class VectorScatterOp(SymbolicDistributionParameter):
    """Assemble one vector from fixed and free indexed coordinates."""

    length: ExprNode
    observed_idx: ExprNode
    observed_values: ExprNode
    missing_idx: ExprNode
    missing_values: ExprNode


def _to_expr(value: object) -> ExprNode:
    """Convert supported Python values to expression nodes."""
    if isinstance(
        value, ParamRef | DataRef | ConstNode | BinOp | IndexOp | UnaryOp | VectorScatterOp
    ):
        return value
    if isinstance(value, int | float):
        return ConstNode(value)
    if is_array_like_constant(value):
        raise array_like_constant_error()
    raise TypeError(f"Cannot convert {type(value).__name__} to an expression node")


def _to_index_spec(value: object) -> IndexSpec:
    """Normalize supported Python indexing syntax into explicit index IR."""
    if isinstance(value, ScalarIndex | FullSlice | IndexTuple):
        return value
    if isinstance(value, slice):
        return _to_slice_index_spec(value)
    if isinstance(value, tuple):
        if not value:
            raise TypeError("Empty index tuples are not supported in model declaration expressions")
        return IndexTuple(tuple(_to_index_tuple_item(item) for item in value))
    if isinstance(value, bool):
        raise TypeError("Index constants must be integers, not bool")
    return ScalarIndex(_to_expr(value))


def _to_index_tuple_item(value: object) -> IndexSpec:
    if isinstance(value, tuple):
        raise TypeError("Nested index tuples are not supported in model declaration expressions")
    return _to_index_spec(value)


def _to_slice_index_spec(value: slice) -> FullSlice:
    if value.start is None and value.stop is None and value.step is None:
        return FullSlice()
    raise TypeError("Only full slices ':' are supported in model declaration indexes")
