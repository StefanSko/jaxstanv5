"""Unit tests for evaluate_expr — ExprNode tree to jax.Array evaluation."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5.compiler.core import evaluate_expr
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, IndexOp, ParamRef


def test_param_ref_lookup() -> None:
    result = evaluate_expr(ParamRef("mu"), {"mu": jnp.array(3.0)})
    assert jnp.allclose(result, jnp.array(3.0))


def test_data_ref_lookup() -> None:
    result = evaluate_expr(DataRef("x"), {"x": jnp.array([1.0, 2.0])})
    assert jnp.allclose(result, jnp.array([1.0, 2.0]))


def test_const_node() -> None:
    result = evaluate_expr(ConstNode(5), {})
    assert jnp.allclose(result, jnp.array(5))


def test_const_node_float() -> None:
    result = evaluate_expr(ConstNode(3.14), {})
    assert jnp.allclose(result, jnp.array(3.14))


def test_binop_add() -> None:
    node = BinOp("+", ConstNode(2), ConstNode(3))
    result = evaluate_expr(node, {})
    assert jnp.allclose(result, jnp.array(5))


def test_binop_sub() -> None:
    node = BinOp("-", ConstNode(7), ConstNode(2))
    result = evaluate_expr(node, {})
    assert jnp.allclose(result, jnp.array(5))


def test_binop_mul() -> None:
    node = BinOp("*", ConstNode(3), ConstNode(4))
    result = evaluate_expr(node, {})
    assert jnp.allclose(result, jnp.array(12))


def test_binop_div() -> None:
    node = BinOp("/", ConstNode(6), ConstNode(2))
    result = evaluate_expr(node, {})
    assert jnp.allclose(result, jnp.array(3))


def test_binop_nested() -> None:
    # (a + b) * c  with a=1, b=2, c=3
    node = BinOp(
        "*",
        BinOp("+", ParamRef("a"), ParamRef("b")),
        ParamRef("c"),
    )
    values = {"a": jnp.array(1), "b": jnp.array(2), "c": jnp.array(3)}
    result = evaluate_expr(node, values)
    assert jnp.allclose(result, jnp.array(9))


def test_index_op() -> None:
    node = IndexOp(ParamRef("arr"), ConstNode(0))
    values = {"arr": jnp.array([10, 20, 30])}
    result = evaluate_expr(node, values)
    assert jnp.allclose(result, jnp.array(10))


def test_index_op_with_data_index() -> None:
    node = IndexOp(ParamRef("alpha"), DataRef("idx"))
    values = {
        "alpha": jnp.array([0.1, 0.2, 0.3]),
        "idx": jnp.array([0, 2]),
    }
    result = evaluate_expr(node, values)
    assert jnp.allclose(result, jnp.array([0.1, 0.3]))
