"""Unit tests for diagnostics — rhat and ess."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxstanv5.diagnostics.core import ess, rhat


def test_rhat_two_chains_one_param() -> None:
    """Two well-mixed chains from same distribution → rhat ≈ 1.0."""
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    chain1 = jax.random.normal(key1, (200,))
    chain2 = jax.random.normal(key2, (200,))
    samples = {"mu": jnp.stack([chain1, chain2])}

    result = rhat(samples)
    assert "mu" in result
    assert 0.95 < result["mu"] < 1.10


def test_rhat_multi_param() -> None:
    """Multiple parameters all return values."""
    key = jax.random.PRNGKey(1)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    samples = {
        "alpha": jnp.stack([jax.random.normal(key1, (100,)), jax.random.normal(key2, (100,))]),
        "beta": jnp.stack([jax.random.normal(key3, (100,)), jax.random.normal(key4, (100,))]),
    }

    result = rhat(samples)
    assert set(result.keys()) == {"alpha", "beta"}
    assert all(isinstance(v, float) for v in result.values())


def test_ess_two_chains() -> None:
    """ESS is reasonable for 200 i.i.d. draws."""
    key = jax.random.PRNGKey(2)
    key1, key2 = jax.random.split(key)
    n = 200
    samples = {
        "mu": jnp.stack([jax.random.normal(key1, (n,)), jax.random.normal(key2, (n,))]),
    }

    result = ess(samples)
    assert "mu" in result
    assert result["mu"] > 50  # ESS should be a decent fraction of total draws


def test_ess_multi_param() -> None:
    """Multiple parameters all return ESS values."""
    key = jax.random.PRNGKey(3)
    keys = jax.random.split(key, 4)
    n = 100
    samples = {
        "a": jnp.stack([jax.random.normal(keys[0], (n,)), jax.random.normal(keys[1], (n,))]),
        "b": jnp.stack([jax.random.normal(keys[2], (n,)), jax.random.normal(keys[3], (n,))]),
    }

    result = ess(samples)
    assert set(result.keys()) == {"a", "b"}
    assert all(isinstance(v, float) for v in result.values())
    assert all(v > 0 for v in result.values())
