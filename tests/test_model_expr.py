"""Tests for symbolic model expression nodes."""

from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, IndexOp, ParamRef


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


def test_indexing_builds_index_expression() -> None:
    alpha = ParamRef("alpha")
    group_idx = DataRef("group_idx")

    expr = alpha[group_idx]

    assert isinstance(expr, IndexOp)
    assert expr.base == alpha
    assert expr.index == group_idx


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
