"""Unit tests for final/resolved expression IR primitives.

These nodes are used after ``@model`` resolves declaration symbols to string
names.
"""

import jax.numpy as jnp
import pytest

from jaxstanv5.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    FullSlice,
    IndexOp,
    IndexTuple,
    ParamRef,
    ScalarIndex,
    UnaryOp,
)


def test_param_and_data_arithmetic_builds_expression_tree() -> None:
    alpha = ParamRef("alpha")
    beta = ParamRef("beta")
    x = DataRef("x")

    expr = alpha + beta * x

    assert isinstance(expr, BinOp)
    assert expr.op == "+"
    assert expr.left == alpha
    assert isinstance(expr.right, BinOp)
    assert expr.right.op == "*"
    assert expr.right.left == beta
    assert expr.right.right == x


def test_scalar_constants_are_promoted_to_const_nodes() -> None:
    alpha = ParamRef("alpha")

    expr = 1.5 + alpha

    assert isinstance(expr, BinOp)
    assert expr.op == "+"
    assert expr.left == ConstNode(1.5)
    assert expr.right == alpha


def test_unary_negation_builds_expression_tree() -> None:
    alpha = ParamRef("alpha")
    beta = ParamRef("beta")

    expr = -(alpha + beta)

    assert isinstance(expr, UnaryOp)
    assert expr.function == "neg"
    assert expr.operand == BinOp("+", alpha, beta)


def test_indexing_builds_index_expression() -> None:
    alpha = ParamRef("alpha")
    group_idx = DataRef("group_idx")

    expr = alpha[group_idx]

    assert isinstance(expr, IndexOp)
    assert expr.base == alpha
    assert expr.index == ScalarIndex(group_idx)


def test_tuple_indexing_builds_explicit_index_ir() -> None:
    x = DataRef("x")

    expr = x[:, 0]

    assert isinstance(expr, IndexOp)
    assert expr.base == x
    assert expr.index == IndexTuple((FullSlice(), ScalarIndex(ConstNode(0))))


def test_final_expression_rejects_array_like_constants_with_guidance() -> None:
    alpha = ParamRef("alpha")

    with pytest.raises(TypeError) as exc_info:
        alpha + jnp.asarray([1.0, 2.0])

    message = str(exc_info.value)
    assert "Array-like constants are not supported in model declaration expressions" in message
    assert "Data.vector" in message
    assert "bind(...)" in message
    assert "ArrayImpl" not in message


def test_chained_arithmetic_supports_subtraction_and_division() -> None:
    alpha = ParamRef("alpha")
    beta = ParamRef("beta")
    scale = DataRef("scale")

    expr = (alpha - beta) / scale

    assert isinstance(expr, BinOp)
    assert expr.op == "/"
    assert isinstance(expr.left, BinOp)
    assert expr.left.op == "-"
    assert expr.right == scale
