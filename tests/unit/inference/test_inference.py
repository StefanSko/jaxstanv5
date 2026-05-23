"""Unit tests for inference helpers."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5.inference.core import SamplerResult, unflatten_samples


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
