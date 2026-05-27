"""Unit tests for inference helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import Normal
from jaxstanv5.inference.core import (
    NutsDiagnosticTrace,
    SamplerDiagnostics,
    SamplerResult,
    _constrain_sample_values,
    _unflatten_samples,
    compile_sampler,
)
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta, ResolvedObserved, ResolvedParam
from jaxstanv5.model.expr import ConstNode


def test_unflatten_scalar_params() -> None:
    """Flat (N, 2) with two scalar params → dict of (1, N) each."""
    flat = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    shapes: dict[str, tuple[int, ...]] = {"a": (), "b": ()}
    result = _unflatten_samples(flat, shapes)

    assert set(result.keys()) == {"a", "b"}
    assert result["a"].shape == (1, 3)
    assert result["b"].shape == (1, 3)
    assert jnp.allclose(result["a"][0], jnp.array([1.0, 3.0, 5.0]))
    assert jnp.allclose(result["b"][0], jnp.array([2.0, 4.0, 6.0]))


def test_unflatten_mixed_shapes() -> None:
    """Scalar + 1D vector param → correct shapes with chain dim."""
    # scalar alpha (1 element), vector beta (2 elements) → 3 total
    flat = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    shapes: dict[str, tuple[int, ...]] = {"alpha": (), "beta": (2,)}
    result = _unflatten_samples(flat, shapes)

    assert result["alpha"].shape == (1, 2)
    assert result["beta"].shape == (1, 2, 2)
    assert jnp.allclose(result["alpha"][0], jnp.array([0.1, 0.4]))
    assert jnp.allclose(result["beta"][0, 0], jnp.array([0.2, 0.3]))
    assert jnp.allclose(result["beta"][0, 1], jnp.array([0.5, 0.6]))


def test_unflatten_single_sample() -> None:
    """Single draw with one scalar param."""
    flat = jnp.array([[7.0]])
    shapes: dict[str, tuple[int, ...]] = {"mu": ()}
    result = _unflatten_samples(flat, shapes)

    assert result["mu"].shape == (1, 1)
    assert jnp.allclose(result["mu"][0, 0], jnp.array(7.0))


def test_unflatten_zero_sized_parameter_does_not_advance_offset() -> None:
    """Zero-sized parameters consume no sampled-vector entries."""
    flat = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    shapes: dict[str, tuple[int, ...]] = {"empty": (0,), "first": (), "second": ()}

    result = _unflatten_samples(flat, shapes)

    assert result["empty"].shape == (1, 2, 0)
    assert jnp.allclose(result["first"][0], jnp.array([10.0, 30.0]))
    assert jnp.allclose(result["second"][0], jnp.array([20.0, 40.0]))


def test_constrain_sample_values_applies_parameter_constraints() -> None:
    meta = ModelMeta(
        params={"sigma": ResolvedParam(Normal(ConstNode(0.0), ConstNode(1.0)), Positive(), None)},
        data_slots=[],
        observed_nodes=(ResolvedObserved("y", Normal(ConstNode(0.0), ConstNode(1.0))),),
        expressions={},
    )
    samples = {"sigma": jnp.array([[0.0, jnp.log(2.0)]])}

    constrained = _constrain_sample_values(samples, meta)

    assert jnp.allclose(constrained["sigma"], jnp.array([[1.0, 2.0]]))


def test_sampler_result_dataclass() -> None:
    """SamplerResult is a frozen dataclass with samples and diagnostics."""
    samples = {"mu": jnp.array([[1.0, 2.0, 3.0]])}
    trace = NutsDiagnosticTrace(
        is_divergent=jnp.array([[False, False, False]]),
        acceptance_rate=jnp.array([[0.8, 0.9, 1.0]]),
        num_integration_steps=jnp.array([[1, 3, 7]]),
        num_trajectory_expansions=jnp.array([[1, 2, 3]]),
        energy=jnp.array([[1.0, 1.5, 2.0]]),
    )
    diagnostics = SamplerDiagnostics(warmup=trace, sampling=trace)
    r = SamplerResult(samples=samples, diagnostics=diagnostics)

    assert r.samples is samples
    assert r.samples["mu"].shape == (1, 3)
    assert r.diagnostics is diagnostics


def test_compile_sampler_rejects_invalid_target_acceptance_rate() -> None:
    meta = ModelMeta(
        params={},
        data_slots=[],
        observed_nodes=(ResolvedObserved("y", Normal(ConstNode(0.0), ConstNode(1.0))),),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(0.0)},
        param_shapes={},
        n_params=0,
    )

    with pytest.raises(ValueError, match="target_acceptance_rate"):
        compile_sampler(bound, target_acceptance_rate=1.0)


def test_sample_rejects_non_positive_chain_count() -> None:
    meta = ModelMeta(
        params={},
        data_slots=[],
        observed_nodes=(ResolvedObserved("y", Normal(ConstNode(0.0), ConstNode(1.0))),),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(0.0)},
        param_shapes={},
        n_params=0,
    )
    compiled = compile_sampler(bound)

    with pytest.raises(ValueError, match="num_chains"):
        compiled.sample(seed=0, num_warmup=10, num_samples=10, num_chains=0)


def test_compiled_sampler_returns_empty_result_for_parameterless_model() -> None:
    meta = ModelMeta(
        params={},
        data_slots=[],
        observed_nodes=(ResolvedObserved("y", Normal(ConstNode(0.0), ConstNode(1.0))),),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(0.0)},
        param_shapes={},
        n_params=0,
    )

    compiled = compile_sampler(bound)

    result = compiled.sample(seed=0, num_warmup=10, num_samples=10, num_chains=4)

    assert result.samples == {}
    assert result.diagnostics.warmup.is_divergent.shape == (4, 10)
    assert result.diagnostics.sampling.is_divergent.shape == (4, 10)
    assert not jnp.any(result.diagnostics.warmup.is_divergent)
    assert not jnp.any(result.diagnostics.sampling.is_divergent)
    assert jnp.all(jnp.isnan(result.diagnostics.warmup.acceptance_rate))
    assert jnp.all(jnp.isnan(result.diagnostics.sampling.acceptance_rate))
