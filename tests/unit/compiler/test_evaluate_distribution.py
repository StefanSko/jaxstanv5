"""Unit tests for _evaluate_distribution — Distribution field resolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxstanv5.compiler.core import _evaluate_distribution
from jaxstanv5.distributions import Binomial, Normal, Poisson
from jaxstanv5.distributions.core import _concrete_parameter
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ParamRef, UnaryOp


def test_scalar_fields_pass_through() -> None:
    """Distribution with only scalar fields returns identical distribution."""
    dist = Normal(loc=0.0, scale=1.0)
    result = _evaluate_distribution(dist, {})

    assert isinstance(result, Normal)
    assert jnp.allclose(_concrete_parameter(result.loc), jnp.array(0.0))
    assert jnp.allclose(_concrete_parameter(result.scale), jnp.array(1.0))


def test_const_node_fields() -> None:
    """ConstNode fields are evaluated to arrays."""
    dist = Normal(loc=ConstNode(3.0), scale=ConstNode(2.0))
    result = _evaluate_distribution(dist, {})

    assert jnp.allclose(_concrete_parameter(result.loc), jnp.array(3.0))
    assert jnp.allclose(_concrete_parameter(result.scale), jnp.array(2.0))


def test_param_ref_fields() -> None:
    """ParamRef fields are looked up in the value dict."""
    dist = Normal(loc=ParamRef("mu"), scale=ConstNode(1.0))
    result = _evaluate_distribution(dist, {"mu": jnp.array(2.5)})

    assert jnp.allclose(_concrete_parameter(result.loc), jnp.array(2.5))
    assert jnp.allclose(_concrete_parameter(result.scale), jnp.array(1.0))


def test_data_ref_fields() -> None:
    """DataRef fields are looked up in the value dict."""
    dist = Normal(loc=DataRef("x"), scale=ParamRef("sigma"))
    result = _evaluate_distribution(dist, {"x": jnp.array([1.0, 2.0]), "sigma": jnp.array(0.5)})

    assert jnp.allclose(_concrete_parameter(result.loc), jnp.array([1.0, 2.0]))
    assert jnp.allclose(_concrete_parameter(result.scale), jnp.array(0.5))


def test_binop_fields() -> None:
    """BinOp fields are recursively evaluated."""
    # loc = alpha + beta * x
    loc_expr = BinOp(
        "+",
        ParamRef("alpha"),
        BinOp("*", ParamRef("beta"), DataRef("x")),
    )
    dist = Normal(loc=loc_expr, scale=ConstNode(1.0))
    values = {"alpha": jnp.array(2.0), "beta": jnp.array(0.5), "x": jnp.array([1.0, 2.0])}
    result = _evaluate_distribution(dist, values)

    expected_loc = 2.0 + 0.5 * jnp.array([1.0, 2.0])
    assert jnp.allclose(_concrete_parameter(result.loc), expected_loc)
    assert jnp.allclose(_concrete_parameter(result.scale), jnp.array(1.0))


def test_unary_function_fields() -> None:
    """UnaryOp fields are recursively evaluated."""
    rate_expr = UnaryOp("exp", BinOp("+", ParamRef("alpha"), DataRef("x")))
    dist = Poisson(rate=rate_expr)
    values = {"alpha": jnp.array(1.0), "x": jnp.array([0.0, 1.0])}
    result = _evaluate_distribution(dist, values)

    expected_rate = jnp.exp(1.0 + jnp.array([0.0, 1.0]))
    assert jnp.allclose(_concrete_parameter(result.rate), expected_rate)


def test_sigmoid_unary_function_fields() -> None:
    """Sigmoid UnaryOp fields are recursively evaluated."""
    probs_expr = UnaryOp("sigmoid", BinOp("+", ParamRef("alpha"), DataRef("x")))
    dist = Binomial(total_count=10.0, probs=probs_expr)
    values = {"alpha": jnp.array(1.0), "x": jnp.array([0.0, 1.0])}
    result = _evaluate_distribution(dist, values)

    expected_probs = jax.nn.sigmoid(1.0 + jnp.array([0.0, 1.0]))
    assert jnp.allclose(_concrete_parameter(result.probs), expected_probs)


def test_log_prob_works_on_evaluated_distribution() -> None:
    """After evaluation, log_prob can be called with concrete arrays."""
    dist = Normal(loc=ParamRef("mu"), scale=ConstNode(1.0))
    evaluated = _evaluate_distribution(dist, {"mu": jnp.array(0.5)})
    lp = evaluated.log_prob(jnp.array(2.0))
    assert lp.shape == ()
    assert jnp.isfinite(lp)
