"""Unit tests for compile_log_density — full log-density assembly."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.distributions import Normal
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta, ResolvedObserved, ResolvedParam
from jaxstanv5.model.expr import ConstNode, ParamRef


@dataclass(frozen=True)
class _FakeObserved:
    y: jnp.ndarray


def test_scalar_unconstrained() -> None:
    """Single scalar parameter, no constraint, observed has no expressions."""
    meta = ModelMeta(
        params={"mu": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), None, None)},
        data_slots=[],
        observed_name="y",
        observed=ResolvedObserved(Normal(ParamRef("mu"), ConstNode(1.0))),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(2.0)},
        param_shapes={"mu": ()},
        n_params=1,
    )

    log_prob = compile_log_density(bound)
    lp = log_prob(jnp.array([0.5]))

    # Manual: prior Normal(0,1).log_prob(0.5) + Normal(0.5,1).log_prob(2.0)
    std_normal = Normal(0.0, 1.0)
    expected = std_normal.log_prob(jnp.array(0.5)) + Normal(0.5, 1.0).log_prob(jnp.array(2.0))

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_no_params() -> None:
    """Model with only observed — no parameters, just data likelihood."""
    meta = ModelMeta(
        params={},
        data_slots=[],
        observed_name="y",
        observed=ResolvedObserved(Normal(ConstNode(0.0), ConstNode(1.0))),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(2.0)},
        param_shapes={},
        n_params=0,
    )

    log_prob = compile_log_density(bound)
    lp = log_prob(jnp.array([]))

    expected = Normal(0.0, 1.0).log_prob(jnp.array(2.0))
    assert jnp.allclose(lp, expected, atol=1e-6)
