"""Unit tests for class-body deferred expression capture.

Declaration operators capture Python class-body syntax only. They must not build
final resolved expression nodes; final expression IR is created by the explicit
model-declaration resolution step.
"""

from __future__ import annotations

import pytest

from jaxstanv5.distributions import Normal
from jaxstanv5.model._deferred import DeferredBinOp, DeferredExpr, DeferredIndexOp, DeferredUnaryOp
from jaxstanv5.model.core import Data, Param
from jaxstanv5.model.decorator import _resolve_declaration_expr
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, IndexOp, ParamRef, UnaryOp


def as_deferred_bin(value: object) -> DeferredBinOp:
    assert isinstance(value, DeferredBinOp)
    return value


def as_deferred_index(value: object) -> DeferredIndexOp:
    assert isinstance(value, DeferredIndexOp)
    return value


def as_deferred_unary(value: object) -> DeferredUnaryOp:
    assert isinstance(value, DeferredUnaryOp)
    return value


def assert_no_final_expr_nodes(value: object) -> None:
    """Deferred trees must not contain final/resolved expression nodes."""
    assert not isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp | UnaryOp)

    if isinstance(value, DeferredBinOp):
        assert_no_final_expr_nodes(value.left)
        assert_no_final_expr_nodes(value.right)
    elif isinstance(value, DeferredIndexOp):
        assert_no_final_expr_nodes(value.base)
        assert_no_final_expr_nodes(value.index)
    elif isinstance(value, DeferredUnaryOp):
        assert_no_final_expr_nodes(value.operand)
    else:
        assert isinstance(value, Param | Data | int | float)


def test_declaration_arithmetic_captures_deferred_syntax_not_final_expr_tree() -> None:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0))
    x = Data()

    expr = (alpha + beta) * (x - 1.0) + 2.0 / sigma

    root = as_deferred_bin(expr)
    assert root.op == "+"
    assert_no_final_expr_nodes(root)

    product = as_deferred_bin(root.left)
    assert product.op == "*"

    left_sum = as_deferred_bin(product.left)
    assert left_sum.op == "+"
    assert left_sum.left is alpha
    assert left_sum.right is beta

    right_difference = as_deferred_bin(product.right)
    assert right_difference.op == "-"
    assert right_difference.left is x
    assert right_difference.right == 1.0

    scale_term = as_deferred_bin(root.right)
    assert scale_term.op == "/"
    assert scale_term.left == 2.0
    assert scale_term.right is sigma


def test_declaration_unary_negation_captures_deferred_syntax_not_final_expr_tree() -> None:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    x = Data()

    direct = as_deferred_unary(-alpha)
    assert direct.function == "neg"
    assert direct.operand is alpha
    assert_no_final_expr_nodes(direct)

    nested = as_deferred_unary(-(alpha + beta * x))
    assert nested.function == "neg"
    assert_no_final_expr_nodes(nested)

    sum_expr = as_deferred_bin(nested.operand)
    assert sum_expr.op == "+"
    assert sum_expr.left is alpha

    product = as_deferred_bin(sum_expr.right)
    assert product.op == "*"
    assert product.left is beta
    assert product.right is x


def test_declaration_indexing_and_reverse_ops_stay_deferred() -> None:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    group_idx = Data()

    expr = 1.0 - alpha[group_idx] / (beta + 2.0)

    root = as_deferred_bin(expr)
    assert root.op == "-"
    assert_no_final_expr_nodes(root)

    assert root.left == 1.0

    quotient = as_deferred_bin(root.right)
    assert quotient.op == "/"

    indexed_alpha = as_deferred_index(quotient.left)
    assert indexed_alpha.base is alpha
    assert indexed_alpha.index is group_idx

    beta_plus_two = as_deferred_bin(quotient.right)
    assert beta_plus_two.op == "+"
    assert beta_plus_two.left is beta
    assert beta_plus_two.right == 2.0


def test_declaration_resolution_rejects_unsupported_deferred_operands() -> None:
    alpha = Param(Normal(0.0, 1.0))
    expr = alpha + "not an expression"

    with pytest.raises(TypeError, match="declaration expression"):
        _resolve_declaration_expr(expr, {alpha.symbol: "alpha"})


def test_deferred_expr_type_alias_covers_operator_results() -> None:
    alpha = Param(Normal(0.0, 1.0))

    expr: DeferredExpr = alpha + 1.0

    assert isinstance(expr, DeferredBinOp)
