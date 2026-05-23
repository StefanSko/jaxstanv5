"""Unit tests for evaluate_distribution — Distribution field resolution."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5.compiler.core import evaluate_distribution
from jaxstanv5.distributions import Normal
from jaxstanv5.distributions.core import concrete_parameter
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ParamRef


def test_scalar_fields_pass_through() -> None:
    """Distribution with only scalar fields returns identical distribution."""
    dist = Normal(loc=0.0, scale=1.0)
    result = evaluate_distribution(dist, {})

    assert isinstance(result, Normal)
    assert jnp.allclose(concrete_parameter(result.loc), jnp.array(0.0))
    assert jnp.allclose(concrete_parameter(result.scale), jnp.array(1.0))


def test_const_node_fields() -> None:
    """ConstNode fields are evaluated to arrays."""
    dist = Normal(loc=ConstNode(3.0), scale=ConstNode(2.0))
    result = evaluate_distribution(dist, {})

    assert jnp.allclose(concrete_parameter(result.loc), jnp.array(3.0))
    assert jnp.allclose(concrete_parameter(result.scale), jnp.array(2.0))


def test_param_ref_fields() -> None:
    """ParamRef fields are looked up in the value dict."""
    dist = Normal(loc=ParamRef("mu"), scale=ConstNode(1.0))
    result = evaluate_distribution(dist, {"mu": jnp.array(2.5)})

    assert jnp.allclose(concrete_parameter(result.loc), jnp.array(2.5))
    assert jnp.allclose(concrete_parameter(result.scale), jnp.array(1.0))


def test_data_ref_fields() -> None:
    """DataRef fields are looked up in the value dict."""
    dist = Normal(loc=DataRef("x"), scale=ParamRef("sigma"))
    result = evaluate_distribution(dist, {"x": jnp.array([1.0, 2.0]), "sigma": jnp.array(0.5)})

    assert jnp.allclose(concrete_parameter(result.loc), jnp.array([1.0, 2.0]))
    assert jnp.allclose(concrete_parameter(result.scale), jnp.array(0.5))


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
    result = evaluate_distribution(dist, values)

    expected_loc = 2.0 + 0.5 * jnp.array([1.0, 2.0])
    assert jnp.allclose(concrete_parameter(result.loc), expected_loc)
    assert jnp.allclose(concrete_parameter(result.scale), jnp.array(1.0))


def test_log_prob_works_on_evaluated_distribution() -> None:
    """After evaluation, log_prob can be called with concrete arrays."""
    dist = Normal(loc=ParamRef("mu"), scale=ConstNode(1.0))
    evaluated = evaluate_distribution(dist, {"mu": jnp.array(0.5)})
    lp = evaluated.log_prob(jnp.array(2.0))
    assert lp.shape == ()
    assert jnp.isfinite(lp)
