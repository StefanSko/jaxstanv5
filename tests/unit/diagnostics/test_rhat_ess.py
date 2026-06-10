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


def test_rhat_and_ess_skip_zero_sized_parameters() -> None:
    key = jax.random.PRNGKey(4)
    key1, key2 = jax.random.split(key)
    n = 100
    samples = {
        "empty": jnp.zeros((2, n, 0)),
        "mu": jnp.stack([jax.random.normal(key1, (n,)), jax.random.normal(key2, (n,))]),
    }

    assert set(rhat(samples)) == {"mu"}
    assert set(ess(samples)) == {"mu"}


def test_rhat_single_chain_splits_internally() -> None:
    """Single chain (1, N) is split in half before computing rhat."""
    key = jax.random.PRNGKey(5)
    n = 500
    chain = jax.random.normal(key, (n,))
    samples = {"mu": chain.reshape(1, n)}

    result = rhat(samples)
    assert "mu" in result
    assert 0.95 < result["mu"] < 1.10  # split rhat ≈ 1.0 for i.i.d. draws


def test_rhat_multi_chain_splits_every_chain() -> None:
    """Split R-hat detects identical within-chain trends across multiple chains."""
    key = jax.random.PRNGKey(6)
    keys = jax.random.split(key, 4)
    n = 1000
    trend = jnp.linspace(-3.0, 3.0, n)
    chains = [jax.random.normal(k, (n,)) + trend for k in keys]
    samples = {"mu": jnp.stack(chains)}

    result = rhat(samples)
    assert "mu" in result
    assert result["mu"] > 1.5
