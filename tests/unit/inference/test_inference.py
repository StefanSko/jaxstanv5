"""Unit tests for inference helpers."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5.distributions import Normal
from jaxstanv5.inference.core import SamplerResult, compile_sampler, sample, unflatten_samples
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta, ResolvedObserved
from jaxstanv5.model.expr import ConstNode


def test_unflatten_scalar_params() -> None:
    """Flat (N, 2) with two scalar params → dict of (1, N) each."""
    flat = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    shapes: dict[str, tuple[int, ...]] = {"a": (), "b": ()}
    result = unflatten_samples(flat, shapes)

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
    result = unflatten_samples(flat, shapes)

    assert result["alpha"].shape == (1, 2)
    assert result["beta"].shape == (1, 2, 2)
    assert jnp.allclose(result["alpha"][0], jnp.array([0.1, 0.4]))
    assert jnp.allclose(result["beta"][0, 0], jnp.array([0.2, 0.3]))
    assert jnp.allclose(result["beta"][0, 1], jnp.array([0.5, 0.6]))


def test_unflatten_single_sample() -> None:
    """Single draw with one scalar param."""
    flat = jnp.array([[7.0]])
    shapes: dict[str, tuple[int, ...]] = {"mu": ()}
    result = unflatten_samples(flat, shapes)

    assert result["mu"].shape == (1, 1)
    assert jnp.allclose(result["mu"][0, 0], jnp.array(7.0))


def test_unflatten_zero_sized_parameter_does_not_advance_offset() -> None:
    """Zero-sized parameters consume no sampled-vector entries."""
    flat = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    shapes: dict[str, tuple[int, ...]] = {"empty": (0,), "first": (), "second": ()}

    result = unflatten_samples(flat, shapes)

    assert result["empty"].shape == (1, 2, 0)
    assert jnp.allclose(result["first"][0], jnp.array([10.0, 30.0]))
    assert jnp.allclose(result["second"][0], jnp.array([20.0, 40.0]))


def test_sampler_result_dataclass() -> None:
    """SamplerResult is a frozen dataclass with a samples dict."""
    samples = {"mu": jnp.array([[1.0, 2.0, 3.0]])}
    r = SamplerResult(samples=samples)

    assert r.samples is samples
    assert r.samples["mu"].shape == (1, 3)


def test_compiled_sampler_returns_empty_result_for_parameterless_model() -> None:
    meta = ModelMeta(
        params={},
        data_slots=[],
        observed_name="y",
        observed=ResolvedObserved(Normal(ConstNode(0.0), ConstNode(1.0))),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(0.0)},
        param_shapes={},
        n_params=0,
    )

    compiled = compile_sampler(bound)

    assert compiled.sample(seed=0, num_warmup=10, num_samples=10) == SamplerResult(samples={})


def test_sample_uses_compiled_sampler(monkeypatch) -> None:
    meta = ModelMeta(
        params={},
        data_slots=[],
        observed_name="y",
        observed=ResolvedObserved(Normal(ConstNode(0.0), ConstNode(1.0))),
        expressions={},
    )
    bound = BoundModel(
        meta=meta,
        data={"y": jnp.array(0.0)},
        param_shapes={},
        n_params=0,
    )
    calls: list[BoundModel] = []

    class FakeCompiledSampler:
        def sample(self, seed: int, num_warmup: int, num_samples: int) -> SamplerResult:
            assert seed == 4
            assert num_warmup == 5
            assert num_samples == 6
            return SamplerResult(samples={"sentinel": jnp.array([[1.0]])})

    def fake_compile_sampler(received: BoundModel) -> FakeCompiledSampler:
        calls.append(received)
        return FakeCompiledSampler()

    monkeypatch.setattr("jaxstanv5.inference.core.compile_sampler", fake_compile_sampler)

    result = sample(bound, seed=4, num_warmup=5, num_samples=6)

    assert calls == [bound]
    assert set(result.samples) == {"sentinel"}
