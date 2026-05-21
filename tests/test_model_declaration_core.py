"""Tests for model declaration expression delegation."""

from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model.core import Data, Param
from jaxstanv5.model.expr import BinOp


def test_declaration_arithmetic_accepts_declaration_operands() -> None:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    x = Data()

    expr = alpha + beta * x

    assert isinstance(expr, BinOp)
    assert expr.op == "+"
    assert expr.left == alpha.ref()
    assert isinstance(expr.right, BinOp)
    assert expr.right.op == "*"
    assert expr.right.left == beta.ref()
    assert expr.right.right == x.ref()
