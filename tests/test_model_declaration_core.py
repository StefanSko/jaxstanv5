"""Tests for the class-body pending expression phase."""

from __future__ import annotations

import pytest

from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model._pending import (
    PendingBinOp,
    PendingConst,
    PendingDataRef,
    PendingIndexOp,
    PendingParamRef,
    PendingRef,
    UnresolvedSymbol,
)
from jaxstanv5.model.core import Data, Param
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, IndexOp, ParamRef


def as_pending_bin(value: object) -> PendingBinOp:
    assert isinstance(value, PendingBinOp)
    return value


def as_pending_index(value: object) -> PendingIndexOp:
    assert isinstance(value, PendingIndexOp)
    return value


def as_pending_ref(value: object) -> PendingRef:
    assert isinstance(value, PendingParamRef | PendingDataRef)
    assert isinstance(value.name, UnresolvedSymbol)
    return value


def as_pending_const(value: object) -> PendingConst:
    assert isinstance(value, PendingConst)
    return value


def assert_no_final_expr_nodes(value: object) -> None:
    """Pending trees must not contain final/resolved expression nodes."""
    assert not isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp)

    if isinstance(value, PendingBinOp):
        assert_no_final_expr_nodes(value.left)
        assert_no_final_expr_nodes(value.right)
    elif isinstance(value, PendingIndexOp):
        assert_no_final_expr_nodes(value.base)
        assert_no_final_expr_nodes(value.index)
    else:
        assert isinstance(value, PendingParamRef | PendingDataRef | PendingConst)


def test_declaration_arithmetic_builds_pending_tree_not_final_expr_tree() -> None:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0))
    x = Data()

    expr = (alpha + beta) * (x - 1.0) + 2.0 / sigma

    root = as_pending_bin(expr)
    assert root.op == "+"
    assert_no_final_expr_nodes(root)

    product = as_pending_bin(root.left)
    assert product.op == "*"

    left_sum = as_pending_bin(product.left)
    assert left_sum.op == "+"
    assert as_pending_ref(left_sum.left).name == alpha.symbol
    assert as_pending_ref(left_sum.right).name == beta.symbol

    right_difference = as_pending_bin(product.right)
    assert right_difference.op == "-"
    assert as_pending_ref(right_difference.left).name == x.symbol
    assert as_pending_const(right_difference.right).value == 1.0

    scale_term = as_pending_bin(root.right)
    assert scale_term.op == "/"
    assert as_pending_const(scale_term.left).value == 2.0
    assert as_pending_ref(scale_term.right).name == sigma.symbol


def test_declaration_indexing_and_reverse_ops_stay_in_pending_phase() -> None:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    group_idx = Data()

    expr = 1.0 - alpha[group_idx] / (beta + 2.0)

    root = as_pending_bin(expr)
    assert root.op == "-"
    assert_no_final_expr_nodes(root)

    assert as_pending_const(root.left).value == 1.0

    quotient = as_pending_bin(root.right)
    assert quotient.op == "/"

    indexed_alpha = as_pending_index(quotient.left)
    assert as_pending_ref(indexed_alpha.base).name == alpha.symbol
    assert as_pending_ref(indexed_alpha.index).name == group_idx.symbol

    beta_plus_two = as_pending_bin(quotient.right)
    assert beta_plus_two.op == "+"
    assert as_pending_ref(beta_plus_two.left).name == beta.symbol
    assert as_pending_const(beta_plus_two.right).value == 2.0


def test_pending_phase_rejects_unsupported_operands() -> None:
    alpha = Param(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="expression"):
        _ = alpha + "not an expression"
