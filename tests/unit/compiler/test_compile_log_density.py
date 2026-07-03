"""Unit tests for compile_log_density — full log-density assembly."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from bayeswire.distributions import Bernoulli, Binomial, Normal
from bayeswire.distributions.core import DistributionValue, LogProbability
from bayeswire.model.decorator import ModelMeta, ResolvedObserved, ResolvedParam
from bayeswire.model.expr import ConstNode, ParamRef, UnaryOp

from jaxstanv5._backends.jax.distributions import log_prob as distribution_log_prob
from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.model.bound import BoundModel


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
        data={},
        observed_nodes=(ResolvedObserved("y", Normal(ParamRef("mu"), ConstNode(1.0))),),
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
    expected = distribution_log_prob(std_normal, jnp.array(0.5)) + distribution_log_prob(
        Normal(0.5, 1.0), jnp.array(2.0)
    )

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_bernoulli_sigmoid_likelihood_matches_binomial_one_trial() -> None:
    bernoulli_meta = ModelMeta(
        params={"eta": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), None, None)},
        data={},
        observed_nodes=(ResolvedObserved("y", Bernoulli(UnaryOp("sigmoid", ParamRef("eta")))),),
        expressions={},
    )
    binomial_meta = ModelMeta(
        params={"eta": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), None, None)},
        data={},
        observed_nodes=(
            ResolvedObserved("y", Binomial(ConstNode(1.0), UnaryOp("sigmoid", ParamRef("eta")))),
        ),
        expressions={},
    )
    y = jnp.asarray([0.0, 1.0, 1.0, 0.0])
    param_shapes: dict[str, tuple[int, ...]] = {"eta": ()}
    bernoulli_bound = BoundModel(
        meta=bernoulli_meta,
        data={"y": y},
        param_shapes=param_shapes,
        n_params=1,
    )
    binomial_bound = BoundModel(
        meta=binomial_meta,
        data={"y": y},
        param_shapes=param_shapes,
        n_params=1,
    )

    actual = compile_log_density(bernoulli_bound)(jnp.array([0.2]))
    expected = compile_log_density(binomial_bound)(jnp.array([0.2]))

    assert jnp.allclose(actual, expected, atol=1e-6)


def test_multiple_observed_nodes_contribute_to_log_density() -> None:
    meta = ModelMeta(
        params={"mu": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), None, None)},
        data={},
        observed_nodes=(
            ResolvedObserved("y", Normal(ParamRef("mu"), ConstNode(1.0))),
            ResolvedObserved("z", Normal(ParamRef("mu"), ConstNode(2.0))),
        ),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(2.0), "z": jnp.array(-1.0)},
        param_shapes={"mu": ()},
        n_params=1,
    )

    log_prob = compile_log_density(bound)
    lp = log_prob(jnp.array([0.5]))

    expected = distribution_log_prob(Normal(0.0, 1.0), jnp.array(0.5))
    expected += distribution_log_prob(Normal(0.5, 1.0), jnp.array(2.0))
    expected += distribution_log_prob(Normal(0.5, 2.0), jnp.array(-1.0))

    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compiled_log_density_does_not_reenter_python_for_same_shape_calls() -> None:
    """After the first trace, same-shape calls should use the compiled executable."""
    counter = _TraceCounter()
    meta = ModelMeta(
        params={"theta": ResolvedParam(_CountingLogProb(counter), None, None)},
        data={},
        observed_nodes=(ResolvedObserved("y", Normal(ParamRef("theta"), ConstNode(1.0))),),
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
        data={},
        observed_nodes=(ResolvedObserved("y", Normal(ParamRef("mu"), ConstNode(1.0))),),
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
        data={},
        observed_nodes=(ResolvedObserved("y", Normal(ConstNode(0.0), ConstNode(1.0))),),
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

    expected = distribution_log_prob(Normal(0.0, 1.0), jnp.array(2.0))
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compile_log_density_handles_prior_only_model() -> None:
    meta = ModelMeta(
        params={"mu": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), None, None)},
        data={},
        observed_nodes=(),
        expressions={},
    )
    bound = BoundModel(meta=meta, data={}, param_shapes={"mu": ()}, n_params=1)

    log_prob = compile_log_density(bound)
    lp = log_prob(jnp.array([0.5]))

    expected = distribution_log_prob(Normal(0.0, 1.0), jnp.array(0.5))
    assert jnp.allclose(lp, expected, atol=1e-6)


def test_compile_log_density_rejects_wrong_q_shape() -> None:
    meta = ModelMeta(
        params={"mu": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), None, None)},
        data={},
        observed_nodes=(ResolvedObserved("y", Normal(ParamRef("mu"), ConstNode(1.0))),),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(1.0)},
        param_shapes={"mu": ()},
        n_params=1,
    )
    log_prob = compile_log_density(bound)

    assert jnp.isfinite(log_prob(jnp.array([0.5])))
    with pytest.raises(ValueError, match="expected 1, got 0"):
        log_prob(jnp.array([]))
    with pytest.raises(ValueError, match="expected 1, got 2"):
        log_prob(jnp.array([0.5, 999.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        log_prob(jnp.array([[0.5]]))
