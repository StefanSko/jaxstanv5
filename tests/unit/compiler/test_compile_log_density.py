"""Unit tests for compile_log_density — full log-density assembly."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.distributions import Normal
from jaxstanv5.distributions.core import DistributionValue, LogProbability
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta, ResolvedObserved, ResolvedParam
from jaxstanv5.model.expr import ConstNode, ParamRef


@dataclass(frozen=True)
class _FakeObserved:
    y: jnp.ndarray


class _TraceCounter:
    def __init__(self) -> None:
        self.calls = 0

    def record_call(self) -> None:
        self.calls += 1


@dataclass(frozen=True)
class _CountingLogProb:
    counter: _TraceCounter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        self.counter.record_call()
        value = jnp.asarray(x)
        return -0.5 * value**2


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


def test_compiled_log_density_does_not_reenter_python_for_same_shape_calls() -> None:
    """After the first trace, same-shape calls should use the compiled executable."""
    counter = _TraceCounter()
    meta = ModelMeta(
        params={"theta": ResolvedParam(_CountingLogProb(counter), None, None)},
        data_slots=[],
        observed_name="y",
        observed=ResolvedObserved(Normal(ParamRef("theta"), ConstNode(1.0))),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(1.0)},
        param_shapes={"theta": ()},
        n_params=1,
    )

    log_prob = compile_log_density(bound)

    log_prob(jnp.array([0.1], dtype=jnp.float32)).block_until_ready()
    calls_after_first_trace = counter.calls
    log_prob(jnp.array([0.2], dtype=jnp.float32)).block_until_ready()

    assert calls_after_first_trace > 0
    assert counter.calls == calls_after_first_trace


def test_compiled_log_density_remains_differentiable() -> None:
    """BlackJAX needs gradients through the compiled log-density."""
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

    grad = jax.grad(compile_log_density(bound))(jnp.array([0.5]))

    assert grad.shape == (1,)
    assert jnp.all(jnp.isfinite(grad))


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
