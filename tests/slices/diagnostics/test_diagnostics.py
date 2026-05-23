"""Slice test — diagnostics on well-mixed chains from a real @model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxstanv5.diagnostics.core import ess, rhat


def test_diagnostics_on_well_mixed_chains() -> None:
    """rhat ≈ 1.0 and ess is reasonable for chains from a known distribution."""
    key = jax.random.PRNGKey(77)
    key1, key2 = jax.random.split(key)

    one_chain = 1000
    chain1 = jax.random.normal(key1, (one_chain,))
    chain2 = jax.random.normal(key2, (one_chain,))

    samples = {"mu": jnp.stack([chain1, chain2])}  # (2 chains, 1000 draws)

    rhat_vals = rhat(samples)
    ess_vals = ess(samples)

    assert "mu" in rhat_vals
    assert "mu" in ess_vals
    assert rhat_vals["mu"] < 1.05  # well-mixed
    assert ess_vals["mu"] > 100  # reasonable ESS for 1000 draws
